"""Apply a displacement field to a label map or image."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


def apply_displacement_to_labels(label_map, disp, order=0):
    """Warp a volume with a dense forward displacement field.

    Uses a pull warp: each destination voxel reads from source coordinate
    ``src = dest - disp(dest)``. This realises the warped atlas

        W(x) = A(x - u(x))

    used in the paper to score tissue overlap after mass effect.

    Parameters
    ----------
    label_map : 3D volume to warp (integer tissue labels, or a scalar image).
    disp      : dense displacement field, shape ``(*label_map.shape, 3)``, in
                voxel units.
    order     : interpolation order. 0 (nearest) preserves integer labels; use
                1 (linear) for continuous images.

    Returns
    -------
    Warped volume, same shape and dtype as ``label_map``.
    """
    label_map = np.asarray(label_map)
    disp = np.asarray(disp, dtype=np.float32)
    if disp.shape[:3] != label_map.shape or disp.shape[-1] != 3:
        raise ValueError(
            f"disp {disp.shape} incompatible with label_map {label_map.shape}"
        )

    Mx, My, Mz = label_map.shape
    X, Y, Z = np.meshgrid(np.arange(Mx), np.arange(My), np.arange(Mz), indexing="ij")
    src = [
        (X - disp[..., 0]).ravel(),
        (Y - disp[..., 1]).ravel(),
        (Z - disp[..., 2]).ravel(),
    ]
    warped = map_coordinates(label_map, src, order=order, mode="nearest")
    return warped.reshape(label_map.shape).astype(label_map.dtype)
