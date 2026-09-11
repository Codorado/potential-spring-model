#!/usr/bin/env python3
"""Self-contained PSM demo — no external data required.

Loads the bundled healthy atlas, synthesises a Gaussian tumor-concentration blob
inside the brain, runs the Potential Spring Model, applies the resulting field to
the atlas, and prints displacement statistics. Outputs are written to
``examples/out/``.

Run from the repository root:
    python examples/demo.py
"""

import os
import sys

# allow running from a fresh clone without `pip install -e .`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from psm import (
    solve_mass_effect,
    apply_displacement_to_labels,
    load_volume,
    save_displacement,
    save_volume,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ATLAS = os.path.join(ROOT, "data", "atlas", "atlas_seg_256.nii.gz")
OUT = os.path.join(HERE, "out")


def synth_tumor(shape, center, radius_vox=18.0):
    """A smooth Gaussian concentration blob, peak 1.0 at ``center``."""
    zz, yy, xx = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]),
        indexing="ij",
    )
    r2 = (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2
    return np.exp(-r2 / (2.0 * radius_vox ** 2)).astype(np.float32)


def main():
    os.makedirs(OUT, exist_ok=True)
    atlas, affine = load_volume(ATLAS)
    brain = atlas > 0
    print(f"atlas {atlas.shape} | brain voxels {int(brain.sum())}")

    # place the synthetic tumor at the brain centroid, offset to one hemisphere
    cz, cy, cx = np.array(np.where(brain)).mean(axis=1)
    center = (cz, cy + 25, cx)
    tumor = synth_tumor(atlas.shape, center) * brain
    print(f"synthetic tumor peak {tumor.max():.2f} at "
          f"({center[0]:.0f}, {center[1]:.0f}, {center[2]:.0f})")

    result = solve_mass_effect(tumor, atlas, alpha=1000.0, n_iter=300)

    mag = np.linalg.norm(result.displacement, axis=-1)
    print(f"\ndisplacement (voxel units = mm @1mm iso):")
    print(f"  mean |u| in brain : {mag[brain].mean():.3f}")
    print(f"  max  |u|          : {mag.max():.3f}")

    warped = apply_displacement_to_labels(atlas, result.displacement)
    changed = int(np.sum((warped != atlas) & brain))
    print(f"  voxels relabelled : {changed} ({100 * changed / int(brain.sum()):.2f}% of brain)")

    save_displacement(result.displacement, affine, os.path.join(OUT, "demo_disp.nii.gz"))
    save_volume(tumor, affine, os.path.join(OUT, "demo_tumor.nii.gz"))
    save_volume(warped, affine, os.path.join(OUT, "demo_warped_atlas.nii.gz"))
    print(f"\noutputs -> {OUT}/")


if __name__ == "__main__":
    main()
