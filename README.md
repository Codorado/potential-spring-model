# Potential Spring Model (PSM) — a differentiable brain-tumor mass-effect solver

PSM computes the tissue **deformation field** induced by a growing brain tumor
from two inputs:

1. a **tumor-concentration** field $c$ (e.g. from a Fisher–Kolmogorov growth fit), and
2. a **registered healthy-brain atlas** (tissue labels in the patient's space).

It is a fast, fully **differentiable**, energy-based alternative to biophysical
PDE mass-effect solvers: a single global parameter, no per-patient inversion, one
forward solve (~15 s on a GPU).

> Reference implementation for the paper *"A Differentiable Brain Tumor Mass
> Effect Solver."* Author and affiliation details are anonymized here pending
> double-blind review.

## Method

A regular grid is laid over the brain; interior nodes are free to move while a
boundary shell is held fixed (rigid-skull / Dirichlet condition). The tumor acts
as a potential that drives nodes outward down its concentration gradient; a
tissue-weighted spring lattice — each node coupled to its **26 neighbours** in
the $3\times3\times3$ cube — resists deformation. We minimise

$$
E(\mathbf{u}) =
\underbrace{\alpha \sum_{i\in\mathcal{F}} c(\mathbf{x}_i+\mathbf{u}_i)}_{\text{tumor potential}} +
\underbrace{\beta \sum_{(i,j)\in\mathcal{E}} k_{ij}\left(\lVert(\mathbf{x}_i+\mathbf{u}_i)-(\mathbf{x}_j+\mathbf{u}_j)\rVert - r^{0}_{ij}\right)^2}_{\text{tissue-weighted spring lattice}}
$$

over the free-node displacements $\mathbf{u}$ by gradient descent (Adam), with
$\nabla_{\mathbf{u}}E$ from JAX autodiff. Edge stiffness $k_{ij}=\tfrac12(k_i+k_j)$
uses tissue-specific node stiffness (GM $\approx 2.7\times$ stiffer than WM, CSF
$\approx 10\times$ softer). The single free parameter is the force/stiffness
ratio $\alpha$ ($\beta$ fixed); the paper operating point is $\alpha=1000$,
chosen to maximise peritumoral net reclassification while keeping the deformation
invertible (folded-voxel fraction $\approx 0$). Node displacements are
interpolated to the voxel grid and returned as a dense field in voxel units
(= mm for 1 mm isotropic data).

## Installation

```bash
git clone <repo-url> potential-spring-model
cd potential-spring-model
python -m venv .venv && source .venv/bin/activate
pip install -e .
# For GPU execution, install the matching JAX CUDA wheel, e.g.:
#   pip install -U "jax[cuda12]"
```

## Quick start

Run the self-contained demo (synthesises a tumor on the bundled atlas — no
external data needed):

```bash
python examples/demo.py
```

### Command line

```bash
python scripts/run_psm.py \
    --tumor tumor_concentration.nii.gz \
    --atlas registered_atlas_tissue.nii.gz \
    --out   out/disp.nii.gz \
    --alpha 1000 --save-warped-atlas
```

### Python API

```python
from psm import solve_mass_effect, apply_displacement_to_labels, load_volume

tumor, affine = load_volume("tumor_concentration.nii.gz")   # c in [0, 1]
atlas, _      = load_volume("registered_atlas_tissue.nii.gz")  # 1=CSF 2=GM 3=WM 4=VT

result = solve_mass_effect(tumor, atlas, alpha=1000.0)       # PSMResult
disp   = result.displacement                                  # (X, Y, Z, 3), voxel units
warped = apply_displacement_to_labels(atlas, disp)            # W(x) = A(x - u(x))
```

## Inputs and outputs

| | description |
|---|---|
| **Input** `tumor` | tumor-concentration volume $c\in[0,1]$ (the driving potential) |
| **Input** `atlas_tissue` | registered healthy-atlas tissue labels (`1=CSF, 2=GM, 3=WM, 4=VT`); supplies the spring stiffness and, by default, the brain mask |
| **Output** `displacement` | dense field `(X, Y, Z, 3)` in voxel units (= mm at 1 mm isotropic) |

A bundled atlas is provided under [`data/atlas/`](data/atlas/) (see its README for
the label convention and how to register it to a patient). The tumor
concentration is produced upstream by a tumor-growth model and is not part of
this repository.

## Evaluation

The deformation is evaluated by warping the atlas tissue map and measuring tissue
overlap against the patient's own segmentation, in peritumoral bands (5/10/15/20
mm shells around the tumor) and over the whole brain. Implemented in
[`psm/evaluation/`](psm/evaluation/):

- **`regions`** — band and whole-brain masks (derived only from the patient tumor
  segmentation, so the comparison is method-independent).
- **`metrics`** — per-tissue Dice and $\Delta$Dice, change-set recall / net
  reclassification, Jacobian folding (invertibility), and off-target fraction.

```bash
python scripts/evaluate.py --atlas a.nii.gz --disp disp.nii.gz \
    --patient-gt gt.nii.gz --tumor-seg seg.nii.gz --out scores.json
```

## Choosing alpha

`scripts/alpha_ablation.py` sweeps $\alpha$ and reports, per value, the
peritumoral net reclassification and the folded-voxel fraction, then picks the
operating point (max net reclassification subject to an invertible field) — the
procedure used to select $\alpha=1000$ in the paper.

```bash
python scripts/alpha_ablation.py --tumor t.nii.gz --atlas a.nii.gz \
    --patient-gt gt.nii.gz --tumor-seg seg.nii.gz \
    --alphas 100 300 1000 3000 10000
```

## Repository layout

```
psm/
  model.py            core solver: energy (potential + spring lattice), Adam
  grid.py             regular node grid + interior/boundary classification
  stiffness.py        tissue-label -> per-node spring stiffness
  warp.py             apply a displacement field to a label map (pull warp)
  io.py               NIfTI load/save helpers
  evaluation/
    regions.py        peritumoral-band and whole-brain region builders
    metrics.py        change-set, per-tissue Dice, Jacobian, off-target
scripts/
  run_psm.py          CLI: tumor + atlas -> displacement field
  alpha_ablation.py   sweep alpha; report net reclassification vs folding
  evaluate.py         score a field against a patient ground truth
data/atlas/           bundled 4-class SRI24 healthy atlas (seg + T1)
examples/demo.py      self-contained end-to-end demo
```

## Baselines

The paper benchmarks PSM against two biophysical PDE solvers (a per-patient
inversion solver and a forward linear-elasticity solver) and an image-based
deformable-registration ceiling. Those are **external** tools and are not
redistributed here; please obtain and cite them from their own repositories. The
evaluation code in `psm/evaluation/` is method-agnostic and scores any
displacement field supplied in the common space.

## Citation

See [`CITATION.cff`](CITATION.cff). Citation details will be completed upon
publication.

## License

- **Code** (`psm/`, `scripts/`, `examples/`) — MIT, see [`LICENSE`](LICENSE).
  (The copyright holder is left generic for anonymous review; set it before
  public release.)
- **Bundled atlas** (`data/atlas/`) — third-party data derived from the SRI24
  atlas, under CC-BY-SA; see [`data/atlas/LICENSE`](data/atlas/LICENSE). Any
  publication using it must cite Rohlfing et al., *Human Brain Mapping*
  31(5):798-819, 2010.
