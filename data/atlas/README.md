# Healthy-brain atlas (SRI24)

Shared healthy-tissue atlas used as the deformable template for the Potential
Spring Model and all benchmarked methods in the paper.

The atlas is **third-party data**, derived from the SRI24 atlas of normal adult
human brain structure (Rohlfing et al., 2010) and distributed under CC-BY-SA.
It is not covered by the repository's MIT license — see [`LICENSE`](LICENSE) in
this directory for the terms and the required citation.

## Files
- `atlas_seg_256.nii.gz` — 4-class tissue segmentation, `256^3` @ 1 mm isotropic,
  in SRI24 atlas space. **This is the input PSM needs** (it supplies the spring
  stiffness map and the brain mask, and is the volume that gets warped for
  tissue-overlap evaluation).
- `atlas_t1_256.nii.gz` — the matching T1 structural image (provided for
  visualisation and for image-based registration baselines; not required by PSM).

## Label convention
| label | tissue              |
|-------|---------------------|
| 0     | background          |
| 1     | CSF                 |
| 2     | gray matter (GM)    |
| 3     | white matter (WM)   |
| 4     | ventricles (VT)     |

## Provenance
The source atlas is **SRI24**, built by its authors via template-free nonrigid
registration of 24 healthy adult subjects:

> T. Rohlfing, N. M. Zahr, E. V. Sullivan, and A. Pfefferbaum, "The SRI24
> multichannel atlas of normal adult human brain structure," *Human Brain
> Mapping*, 31(5):798-819, 2010. doi:10.1002/hbm.20906
> Distributed at <https://www.nitrc.org/projects/sri24/> under CC-BY-SA.

The files here are derivative works of that atlas: the tissue labels were mapped
onto the 4-class convention above (SRI24 supplies a three-compartment CSF/GM/WM
maximum-likelihood tissue segmentation; the ventricle class is separated out),
and both volumes are resampled to 256^3 at 1 mm isotropic in SRI24 space. The T1
is brain-extracted — it carries no skull, face, or neck.

No patient data from this study is contained in, or derivable from, these files.

## Per-patient use
To run PSM on a patient, this atlas must first be brought into that patient's
image space — in the paper, by an affine registration of the atlas T1 to the patient T1 (the same warp is
applied to the labels with nearest-neighbour interpolation). Any standard
registration tool (e.g. ANTs) can produce this "registered atlas" input; the
registration itself is outside the scope of this repository.
