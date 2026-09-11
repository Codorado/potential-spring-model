#!/usr/bin/env python3
"""Run the Potential Spring Model on one patient.

Inputs:
  --tumor   tumor-concentration NIfTI (potential ``c``, in [0, 1])
  --atlas   registered healthy-atlas tissue-label NIfTI (1=CSF 2=GM 3=WM 4=VT)
Output:
  a 4D displacement field NIfTI (X, Y, Z, 3) in voxel units (= mm at 1 mm iso),
  and optionally the warped atlas tissue map.

Example:
  python scripts/run_psm.py \\
      --tumor tumor_concentration.nii.gz \\
      --atlas registered_atlas_tissue.nii.gz \\
      --out   out/disp.nii.gz \\
      --alpha 1000 --save-warped-atlas
"""

import argparse
import os
import sys

# allow running from a fresh clone without `pip install -e .`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psm import (
    solve_mass_effect,
    apply_displacement_to_labels,
    load_volume,
    save_displacement,
    save_volume,
)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tumor", required=True, help="tumor concentration NIfTI")
    p.add_argument("--atlas", required=True, help="registered atlas tissue NIfTI")
    p.add_argument("--out", required=True, help="output displacement NIfTI (.nii.gz)")
    p.add_argument("--alpha", type=float, default=1000.0,
                   help="tumor-potential weight (default 1000, the paper operating point)")
    p.add_argument("--beta", type=float, default=1.0, help="spring weight (default 1.0)")
    p.add_argument("--n-iter", type=int, default=300, help="Adam iterations (default 300)")
    p.add_argument("--grid-downsample", type=int, default=5,
                   help="node grid coarsening factor vs voxel grid (default 5)")
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--save-warped-atlas", action="store_true",
                   help="also write the atlas warped by the field W(x)=A(x-u(x))")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    tumor, affine = load_volume(args.tumor)
    atlas, _ = load_volume(args.atlas)

    result = solve_mass_effect(
        tumor, atlas,
        alpha=args.alpha, beta=args.beta, n_iter=args.n_iter,
        grid_downsample=args.grid_downsample, learning_rate=args.learning_rate,
        verbose=not args.quiet,
    )

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    save_displacement(result.displacement, affine, args.out)
    print(f"displacement field -> {args.out}")

    if args.save_warped_atlas:
        warped = apply_displacement_to_labels(atlas, result.displacement)
        warped_path = args.out.replace(".nii.gz", "").replace(".nii", "") + "_warped-atlas.nii.gz"
        save_volume(warped, affine, warped_path)
        print(f"warped atlas      -> {warped_path}")


if __name__ == "__main__":
    main()
