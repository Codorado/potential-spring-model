#!/usr/bin/env python3
"""Score a displacement field against a patient tissue ground truth.

Computes the full metric set (per-tissue Dice and Delta-Dice, change-set recall /
net reclassification, Jacobian folding, off-target fraction) over peritumoral
bands and the whole brain, for one method's field on one patient.

Inputs:
  --atlas       registered atlas tissue NIfTI (baseline A; 1=CSF 2=GM 3=WM 4=VT)
  --disp        displacement field NIfTI (X, Y, Z, 3), voxel units
  --patient-gt  patient tissue ground-truth NIfTI (P, same labels)
  --tumor-seg   BraTS tumor segmentation NIfTI (1=NCR 2=ED 4=ET) for the bands

Example:
  python scripts/evaluate.py --atlas a.nii.gz --disp disp.nii.gz \\
      --patient-gt gt.nii.gz --tumor-seg seg.nii.gz --out scores.json
"""

import argparse
import json
import os
import sys

# allow running from a fresh clone without `pip install -e .`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from psm import apply_displacement_to_labels, load_volume
from psm.evaluation import build_regions, score_method
from psm.evaluation.regions import brain_mask_from_labels


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--atlas", required=True)
    p.add_argument("--disp", required=True)
    p.add_argument("--patient-gt", required=True)
    p.add_argument("--tumor-seg", required=True)
    p.add_argument("--out", default=None, help="optional JSON output path")
    args = p.parse_args()

    atlas, _ = load_volume(args.atlas)
    disp, _ = load_volume(args.disp)
    patient_gt, _ = load_volume(args.patient_gt)
    tumor_seg, _ = load_volume(args.tumor_seg)

    brain = brain_mask_from_labels(atlas)
    regions = build_regions(tumor_seg, brain)
    warped = apply_displacement_to_labels(atlas, disp)

    scores = score_method(atlas, warped, patient_gt, disp, regions,
                          brain_mask=brain)

    # concise console summary at the 10 mm band
    b = scores["band_10mm"]
    print("band_10mm:")
    print(f"  net reclassification : {b['change_set']['net_reclassification']:+.4f}")
    print(f"  recall               : {b['change_set']['recall']:.4f}")
    for L, name in {1: "CSF", 2: "GM", 3: "WM", 4: "VT"}.items():
        d = b["per_tissue_dice"][L]
        print(f"  dDice {name:<3}           : {d['delta']:+.4f} "
              f"(atlas {d['dice_atlas']:.3f} -> warped {d['dice_warped']:.3f})")
    print(f"  folded fraction      : {b['jacobian']['frac_folded']:.2e}")
    print(f"off-target fraction    : {scores['offtarget_fraction']:.4f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(scores, f, indent=2)
        print(f"\nfull scores -> {args.out}")


if __name__ == "__main__":
    main()
