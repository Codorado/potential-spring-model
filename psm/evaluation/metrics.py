"""Region-restricted mass-effect evaluation metrics.

All metrics take label maps / displacement fields already in a common space and a
region mask from :func:`psm.evaluation.regions.build_regions`, and are computed
identically for every method, so comparisons are method-independent.

Convention
  A = registered atlas tissue labels   (baseline, no deformation)
  W = warped atlas tissue labels        (A + this method's displacement)
  P = patient tissue labels             (ground truth)
Tissue labels: ``1=CSF, 2=GM, 3=WM, 4=VT`` (background 0 is rare in-band).

Three metric families
  1. change_set_matrix : the 2x2 "did the warp move tissue the right way" table
                         -> recall, precision, net reclassification, F1.
  2. jacobian_metrics  : determinant of J = I + grad(u) -> volume change and
                         the fraction of folded (non-invertible) voxels.
  3. offtarget_fraction: fraction of total displacement magnitude OUTSIDE the
                         band (locality / physical plausibility).

NOTE on units: ``jacobian_metrics`` expects ``disp`` in *voxel* units (so that
J = I + du/dx is dimensionless on the voxel grid). Fields in mm must be divided
by the voxel size before being passed in. PSM outputs voxel units already.
"""

from __future__ import annotations

import numpy as np

TISSUE_LABELS = (1, 2, 3, 4)  # CSF, GM, WM, VT


def _safe(n, d):
    return float(n) / float(d) if d > 0 else 0.0


def change_set_matrix(A, W, P, region):
    """2x2 reclassification table on the change set, within ``region``.

      stayed_correct : A==P & W==P
      broke          : A==P & W!=P   (displacement introduced an error)
      fixed          : A!=P & W==P   (displacement corrected mass effect)
      stayed_wrong   : A!=P & W!=P

      recall = fixed / (fixed + stayed_wrong)
      precision = fixed / (fixed + broke)
      net_reclassification = (fixed - broke) / (fixed + stayed_wrong)
      f1 = 2*P*R / (P+R)
    """
    region = np.asarray(region, dtype=bool)
    A = np.rint(np.asarray(A)).astype(np.int32)[region]
    W = np.rint(np.asarray(W)).astype(np.int32)[region]
    P = np.rint(np.asarray(P)).astype(np.int32)[region]

    base_ok = (A == P)
    warp_ok = (W == P)
    stayed_correct = int(np.sum(base_ok & warp_ok))
    broke = int(np.sum(base_ok & ~warp_ok))
    fixed = int(np.sum(~base_ok & warp_ok))
    stayed_wrong = int(np.sum(~base_ok & ~warp_ok))

    recall = _safe(fixed, fixed + stayed_wrong)
    precision = _safe(fixed, fixed + broke)
    net = _safe(fixed - broke, fixed + stayed_wrong)
    f1 = _safe(2 * precision * recall, precision + recall)

    return {
        "stayed_correct": stayed_correct,
        "broke": broke,
        "fixed": fixed,
        "stayed_wrong": stayed_wrong,
        "n_changed_truth": fixed + stayed_wrong,
        "recall": recall,
        "precision": precision,
        "net_reclassification": net,
        "f1": f1,
    }


def per_tissue_dice(A, W, P, region, labels=TISSUE_LABELS):
    """Per-tissue Dice within ``region`` for atlas (A) and warped atlas (W)
    against the patient ground truth (P).

    Returns ``{label: {"dice_atlas", "dice_warped", "delta"}}`` where
    ``delta = dice_warped - dice_atlas`` (positive = the displacement improved
    overlap for that tissue).
    """
    region = np.asarray(region, dtype=bool)
    A = np.rint(np.asarray(A)).astype(np.int32)[region]
    W = np.rint(np.asarray(W)).astype(np.int32)[region]
    P = np.rint(np.asarray(P)).astype(np.int32)[region]

    def _dice(x, p):
        return _safe(2 * int((x & p).sum()), int(x.sum()) + int(p.sum()))

    out = {}
    for L in labels:
        p = (P == L)
        da = _dice(A == L, p)
        dw = _dice(W == L, p)
        out[int(L)] = {"dice_atlas": da, "dice_warped": dw, "delta": dw - da}
    return out


def jacobian_metrics(disp, region):
    """Volume-change and folding metrics from the deformation Jacobian.

    The deformation maps ``x -> x + u(x)``; its Jacobian is ``J = I + grad(u)``.
    ``det(J) < 1`` is compression, ``> 1`` expansion, ``<= 0`` folding
    (non-invertible, non-physical).

    disp   : (X, Y, Z, 3) displacement in *voxel* units.
    region : bool mask over which to summarise ``det(J)``.
    """
    disp = np.asarray(disp, dtype=np.float32)
    grads = [[np.gradient(disp[..., i], axis=j) for j in range(3)] for i in range(3)]
    J = np.empty(disp.shape[:3] + (3, 3), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            J[..., i, j] = grads[i][j] + (1.0 if i == j else 0.0)
    detJ = np.linalg.det(J)

    vals = detJ[np.asarray(region, dtype=bool)]
    if vals.size == 0:
        nan = float("nan")
        return {"median_detJ": nan, "p05_detJ": nan, "p95_detJ": nan, "frac_folded": nan}
    return {
        "median_detJ": float(np.median(vals)),
        "p05_detJ": float(np.percentile(vals, 5)),
        "p95_detJ": float(np.percentile(vals, 95)),
        "frac_folded": float(np.mean(vals <= 0.0)),
    }


def offtarget_fraction(disp, band_mask, brain_mask):
    """Fraction of total displacement magnitude OUTSIDE the band.

    Physics models concentrate deformation near the tumor; image registration
    spreads it across the brain. Lower = more local.
    """
    disp = np.asarray(disp, dtype=np.float32)
    mag = np.linalg.norm(disp, axis=-1)
    brain = np.asarray(brain_mask, dtype=bool)
    band = np.asarray(band_mask, dtype=bool) & brain
    total = float(mag[brain].sum())
    if total <= 0:
        return 0.0
    return float(mag[brain & ~band].sum()) / total


def score_method(A, W, P, disp, regions, band_for_offtarget="band_10mm",
                 brain_mask=None):
    """Compute all metrics for one method over every region in ``regions``.

    Returns a JSON-serialisable nested dict::

      { <region>: {"change_set": {...}, "per_tissue_dice": {...},
                   "jacobian": {...}}, ..., "offtarget_fraction": float }
    """
    out = {}
    for name, mask in regions.items():
        out[name] = {
            "change_set": change_set_matrix(A, W, P, mask),
            "per_tissue_dice": per_tissue_dice(A, W, P, mask),
            "jacobian": jacobian_metrics(disp, mask),
        }
    if brain_mask is None:
        brain_mask = regions.get("wholebrain")
    out["offtarget_fraction"] = offtarget_fraction(
        disp, regions[band_for_offtarget], brain_mask)
    return out
