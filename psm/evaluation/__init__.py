"""Region-restricted evaluation metrics for mass-effect deformation fields."""

from .regions import (
    build_regions,
    peritumoral_band,
    whole_brain_region,
    brats_tumor_masks,
    brain_mask_from_labels,
)
from .metrics import (
    change_set_matrix,
    per_tissue_dice,
    jacobian_metrics,
    offtarget_fraction,
    score_method,
)

__all__ = [
    "build_regions",
    "peritumoral_band",
    "whole_brain_region",
    "brats_tumor_masks",
    "brain_mask_from_labels",
    "change_set_matrix",
    "per_tissue_dice",
    "jacobian_metrics",
    "offtarget_fraction",
    "score_method",
]
