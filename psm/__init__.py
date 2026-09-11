"""Potential Spring Model (PSM): a differentiable brain-tumor mass-effect solver.

Given a tumor-concentration field and a registered healthy-brain atlas, PSM
computes the tumor-induced tissue displacement field by minimising an energy of
a tumor potential plus a tissue-weighted spring lattice (JAX autodiff + Adam).

Typical use::

    from psm import solve_mass_effect, apply_displacement_to_labels, load_volume

    tumor, aff = load_volume("tumor_concentration.nii.gz")
    atlas, _   = load_volume("registered_atlas_tissue.nii.gz")
    result     = solve_mass_effect(tumor, atlas, alpha=1000.0)
    warped     = apply_displacement_to_labels(atlas, result.displacement)
"""

from .model import solve_mass_effect, PSMResult
from .warp import apply_displacement_to_labels
from .stiffness import stiffness_from_tissue, DEFAULT_STIFFNESS
from .io import load_volume, save_volume, save_displacement

__all__ = [
    "solve_mass_effect",
    "PSMResult",
    "apply_displacement_to_labels",
    "stiffness_from_tissue",
    "DEFAULT_STIFFNESS",
    "load_volume",
    "save_volume",
    "save_displacement",
]

__version__ = "0.1.0"
