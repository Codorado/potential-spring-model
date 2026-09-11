#!/usr/bin/env python3
"""Alpha ablation: choose the single global parameter ``alpha``.

Reproduces the paper's operating-point criterion: sweep ``alpha``, and for each
value solve PSM, warp the atlas, and report the peritumoral-band (10 mm) net
reclassification together with the folded-voxel fraction. The chosen operating
point is the ``alpha`` that maximises net reclassification while keeping the
deformation invertible (folded fraction ~ 0). The paper selects ``alpha=1000``.

Inputs (one patient; for a cohort, average the per-alpha rows across patients):
  --tumor       tumor-concentration NIfTI
  --atlas       registered atlas tissue NIfTI (1=CSF 2=GM 3=WM 4=VT)
  --patient-gt  patient tissue ground-truth NIfTI (same labels) for scoring
  --tumor-seg   patient BraTS tumor segmentation NIfTI (1=NCR 2=ED 4=ET) for bands

Example:
  python scripts/alpha_ablation.py --tumor t.nii.gz --atlas a.nii.gz \\
      --patient-gt gt.nii.gz --tumor-seg seg.nii.gz \\
      --alphas 100 300 1000 3000 10000 --band 10 --out ablation.json
"""

import argparse
import json
import os
import sys

# allow running from a fresh clone without `pip install -e .`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from psm import solve_mass_effect, apply_displacement_to_labels, load_volume
from psm.evaluation import build_regions, change_set_matrix, jacobian_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tumor", required=True)
    p.add_argument("--atlas", required=True)
    p.add_argument("--patient-gt", required=True, help="patient tissue ground truth")
    p.add_argument("--tumor-seg", required=True, help="BraTS tumor segmentation")
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[100, 300, 1000, 3000, 10000])
    p.add_argument("--band", type=float, default=10.0, help="band width in mm")
    p.add_argument("--n-iter", type=int, default=300)
    p.add_argument("--out", default=None, help="optional JSON output path")
    args = p.parse_args()

    tumor, _ = load_volume(args.tumor)
    atlas, _ = load_volume(args.atlas)
    patient_gt, _ = load_volume(args.patient_gt)
    tumor_seg, _ = load_volume(args.tumor_seg)

    brain = np.asarray(atlas) > 0
    regions = build_regions(tumor_seg, brain, band_widths_mm=(args.band,))
    band_mask = regions[f"band_{int(args.band)}mm"]

    rows = []
    for alpha in args.alphas:
        res = solve_mass_effect(tumor, atlas, alpha=alpha, n_iter=args.n_iter,
                                brain_mask=brain, verbose=False)
        warped = apply_displacement_to_labels(atlas, res.displacement)
        cs = change_set_matrix(atlas, warped, patient_gt, band_mask)
        jac = jacobian_metrics(res.displacement, band_mask)
        rows.append({
            "alpha": float(alpha),
            "net_reclassification": cs["net_reclassification"],
            "recall": cs["recall"],
            "frac_folded": jac["frac_folded"],
            "mean_disp_mag": float(np.linalg.norm(res.displacement, axis=-1)[brain].mean()),
        })
        print(f"alpha={alpha:>8.0f}  net={cs['net_reclassification']:+.4f}  "
              f"recall={cs['recall']:.4f}  folded={jac['frac_folded']:.2e}  "
              f"|u|={rows[-1]['mean_disp_mag']:.3f}")

    # operating point: max net reclassification among ~invertible fields
    invertible = [r for r in rows if r["frac_folded"] <= 1e-3] or rows
    best = max(invertible, key=lambda r: r["net_reclassification"])
    print(f"\noperating point: alpha={best['alpha']:.0f} "
          f"(net={best['net_reclassification']:+.4f}, folded={best['frac_folded']:.2e})")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"rows": rows, "operating_point": best}, f, indent=2)
        print(f"results -> {args.out}")


if __name__ == "__main__":
    main()
