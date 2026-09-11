"""Regular spring grid over the brain volume.

The Potential Spring Model lays a coarse regular grid over the image volume and
treats each node as a point mass connected to its neighbours by springs.
Interior nodes are free to move; a one-node-thick shell on the brain boundary is
held fixed (rigid-skull Dirichlet boundary condition).

All coordinates are in *voxel* units of the input volume, so a node at integer
position ``(i, j, k)`` maps directly onto voxel ``(i, j, k)``. For 1 mm isotropic
data (the setting used in the paper) voxel units equal millimetres.
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage


def generate_grid(shape, bounds):
    """Build a regular node grid and report its per-axis spacing.

    Parameters
    ----------
    shape  : (nx, ny, nz) number of grid nodes per axis.
    bounds : either a scalar ``(lo, hi)`` applied to every axis, or a triple of
             ``((xlo, xhi), (ylo, yhi), (zlo, zhi))`` in voxel coordinates.

    Returns
    -------
    X, Y, Z : node coordinate arrays, each of shape ``shape``.
    spacing : (sx, sy, sz) rest length between adjacent nodes per axis.
    """
    shape = tuple(int(s) for s in shape)
    if isinstance(bounds[0], (tuple, list, np.ndarray)):
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    else:
        xmin, xmax = ymin, ymax = zmin, zmax = bounds
    x = np.linspace(xmin, xmax, shape[0])
    y = np.linspace(ymin, ymax, shape[1])
    z = np.linspace(zmin, zmax, shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    sx = float(x[1] - x[0]) if shape[0] > 1 else 1.0
    sy = float(y[1] - y[0]) if shape[1] > 1 else 1.0
    sz = float(z[1] - z[0]) if shape[2] > 1 else 1.0
    return X, Y, Z, (sx, sy, sz)


def brain_grid_masks(X, Y, Z, brain_mask):
    """Classify grid nodes as interior (free) or boundary (fixed).

    A node is *inside* if its nearest voxel lies within ``brain_mask``. The
    fixed boundary is the one-node-thick shell of the inside set; the remaining
    inside nodes are free.

    Returns
    -------
    interior : bool array, shape of ``X`` — free nodes.
    boundary : bool array, shape of ``X`` — fixed nodes.
    """
    ms = np.array(brain_mask.shape)
    ix = np.clip(np.rint(X).astype(int), 0, ms[0] - 1)
    iy = np.clip(np.rint(Y).astype(int), 0, ms[1] - 1)
    iz = np.clip(np.rint(Z).astype(int), 0, ms[2] - 1)
    inside = brain_mask[ix, iy, iz]
    boundary = inside & ~scipy.ndimage.binary_erosion(inside, iterations=1)
    return inside & ~boundary, boundary
