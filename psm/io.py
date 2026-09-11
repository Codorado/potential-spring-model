"""NIfTI input/output helpers."""

from __future__ import annotations

import numpy as np
import nibabel as nib


def load_volume(path, dtype=np.float32):
    """Load a NIfTI volume as an array; return ``(data, affine)``."""
    img = nib.load(str(path))
    return np.asarray(img.get_fdata(), dtype=dtype), img.affine


def save_volume(data, affine, path):
    """Save an array as a NIfTI volume."""
    nib.save(nib.Nifti1Image(np.asarray(data), affine), str(path))


def save_displacement(disp, affine, path):
    """Save a dense displacement field ``(X, Y, Z, 3)`` as a 4D NIfTI."""
    disp = np.asarray(disp, dtype=np.float32)
    if disp.ndim != 4 or disp.shape[-1] != 3:
        raise ValueError(f"expected (X, Y, Z, 3) displacement, got {disp.shape}")
    nib.save(nib.Nifti1Image(disp, affine), str(path))
