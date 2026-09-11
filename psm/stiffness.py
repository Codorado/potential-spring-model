"""Tissue-specific spring stiffness.

Each grid node inherits a stiffness multiplier from the tissue it sits in. The
edge stiffness used in the spring energy is the average of its two endpoints,
``k_ij = (k_i + k_j) / 2``.

Default multipliers follow region-specific brain elasticity: gray matter is
~2.7x stiffer than white matter (white matter = reference 1.0), and CSF is ~10x
softer. These are the values used for the results reported in the paper.

Label convention of the tissue/atlas map: ``1=CSF, 2=GM, 3=WM, 4=VT`` (0 = bg).
"""

from __future__ import annotations

import numpy as np

# label -> stiffness multiplier (relative to white matter = 1.0).
# Any label not listed here (incl. WM=3 and background) defaults to 1.0.
DEFAULT_STIFFNESS = {
    1: 0.1,   # CSF  — ~10x softer than white matter
    2: 2.7,   # GM   — ~2.7x stiffer than white matter
    # 3: 1.0  # WM   — reference (implicit default)
    # 4: 0.1  # VT   — ventricles; left at 1.0 in the paper, set to 0.1 to soften
}


def stiffness_from_tissue(tissue_map, mapping=None):
    """Map an integer tissue-label volume to a per-voxel stiffness volume.

    Parameters
    ----------
    tissue_map : integer label volume (``1=CSF, 2=GM, 3=WM, 4=VT``).
    mapping    : optional ``{label: multiplier}`` overriding ``DEFAULT_STIFFNESS``.
                 Labels absent from the mapping get a multiplier of 1.0.

    Returns
    -------
    float32 volume, same shape as ``tissue_map``.
    """
    if mapping is None:
        mapping = DEFAULT_STIFFNESS
    labels = np.rint(np.asarray(tissue_map)).astype(np.int32)
    stiff = np.ones(labels.shape, dtype=np.float32)
    for label, value in mapping.items():
        stiff[labels == int(label)] = float(value)
    return stiff


def sample_node_stiffness(stiffness_volume, X, Y, Z):
    """Sample the stiffness volume at grid-node positions (nearest voxel)."""
    ms = np.array(stiffness_volume.shape)
    ix = np.clip(np.rint(X).astype(int), 0, ms[0] - 1)
    iy = np.clip(np.rint(Y).astype(int), 0, ms[1] - 1)
    iz = np.clip(np.rint(Z).astype(int), 0, ms[2] - 1)
    return np.asarray(stiffness_volume, dtype=np.float32)[ix, iy, iz]
