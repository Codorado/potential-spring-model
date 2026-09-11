"""Evaluation regions for the mass-effect comparison.

Defines the spatial masks over which every method is scored *identically*, so the
comparison stays method-independent:

  - peritumoral band : a Euclidean shell of width ``r_mm`` around the tumor
                       (headline region; mass effect concentrates here)
  - whole brain      : tumor-masked whole-brain mask

The band is derived only from the patient's tumor segmentation and brain mask,
never from any solver's displacement, so the change-set denominator is identical
across methods.

Label conventions
  BraTS tumor seg : ``1=NCR, 2=ED, 4=ET``.
  tissue / atlas  : ``0=bg, 1=CSF, 2=GM, 3=WM, 4=VT``.
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi


def brain_mask_from_labels(label_map):
    """Brain = any non-background tissue label."""
    return np.asarray(label_map) > 0


def brats_tumor_masks(seg):
    """Split a BraTS segmentation into ``(whole, core, edema)`` boolean masks.

    ``core = NCR + ET`` is solid tumor that *replaces* tissue, where displacement
    is not physically meaningful and is excluded from every scoring region.
    """
    seg = np.rint(np.asarray(seg)).astype(np.int32)
    whole = seg > 0
    core = (seg == 1) | (seg == 4)
    edema = (seg == 2)
    return whole, core, edema


def peritumoral_band(tumor_mask, brain_mask, r_mm, voxel_mm=1.0, exclude_mask=None):
    """Geometric peritumoral shell ``{0 < dist_to_tumor <= r_mm} ∩ brain``.

    Distance is a true Euclidean distance transform (mm), so the band width is
    isotropic and independent of connectivity/iteration count.
    """
    tumor_mask = np.asarray(tumor_mask, dtype=bool)
    brain_mask = np.asarray(brain_mask, dtype=bool)
    dist = ndi.distance_transform_edt(~tumor_mask, sampling=voxel_mm)
    band = (dist > 0) & (dist <= r_mm) & brain_mask
    if exclude_mask is not None:
        band &= ~np.asarray(exclude_mask, dtype=bool)
    return band


def whole_brain_region(brain_mask, exclude_mask=None):
    """Whole-brain region, optionally with the tumor removed."""
    region = np.asarray(brain_mask, dtype=bool).copy()
    if exclude_mask is not None:
        region &= ~np.asarray(exclude_mask, dtype=bool)
    return region


def build_regions(patient_tumor_seg, brain_mask,
                  band_widths_mm=(5.0, 10.0, 15.0, 20.0), voxel_mm=1.0):
    """Build the full set of named evaluation regions for one subject.

    Returns ``{region_name: bool mask}`` with ``band_5mm … band_20mm`` and
    ``wholebrain``. All regions exclude the tumor (tissue is replaced there).
    """
    whole, _core, _edema = brats_tumor_masks(patient_tumor_seg)
    regions = {}
    for r in band_widths_mm:
        regions[f"band_{int(r)}mm"] = peritumoral_band(
            whole, brain_mask, r_mm=r, voxel_mm=voxel_mm, exclude_mask=whole)
    regions["wholebrain"] = whole_brain_region(brain_mask, exclude_mask=whole)
    return regions
