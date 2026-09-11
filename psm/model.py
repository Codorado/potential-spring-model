"""Potential Spring Model (PSM) — the differentiable mass-effect solver.

A regular grid is laid over the brain; interior nodes are free and a boundary
shell is fixed. The tumor concentration acts as a potential that drives nodes
outward down its gradient, while a tissue-weighted spring lattice (each node
coupled to its 26 neighbours in the 3x3x3 cube) resists deformation. The total
energy

    E(u) = alpha * sum_i  c(x_i + u_i)                          [tumor potential]
         + beta  * sum_ij k_ij ( ||(x_i+u_i)-(x_j+u_j)|| - r0_ij )^2  [springs]

is minimised over the free-node displacements u by gradient descent (Adam), with
gradients from JAX automatic differentiation. The single global parameter is the
force/stiffness ratio ``alpha`` (``beta`` fixed); no per-patient inversion.

The optimised node displacements are interpolated to the full voxel grid and
returned as a dense field in voxel units (= mm for 1 mm isotropic data).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.ndimage

import jax
import jax.numpy as jnp
import optax

from .grid import generate_grid, brain_grid_masks
from .stiffness import stiffness_from_tissue, sample_node_stiffness

_EPS = 1e-8  # inside sqrt to keep spring-length gradients finite at zero


@dataclass
class PSMResult:
    """Output of :func:`solve_mass_effect`."""

    displacement: np.ndarray            # dense field, (X, Y, Z, 3), voxel units
    energy_total: list = field(default_factory=list)
    energy_potential: list = field(default_factory=list)
    energy_spring: list = field(default_factory=list)
    n_free_nodes: int = 0


def _build_spring_energy(sx, sy, sz):
    """Spring energy over all 13 undirected edge directions (= 26 neighbours):
    3 face-axis + 6 in-plane diagonals + 4 body diagonals.

    Including the diagonal edges makes the discrete lattice stiffness
    approximately isotropic (a mass-spring discretisation of linear elasticity).
    """
    r_xy = jnp.sqrt(sx ** 2 + sy ** 2)
    r_xz = jnp.sqrt(sx ** 2 + sz ** 2)
    r_yz = jnp.sqrt(sy ** 2 + sz ** 2)
    r_xyz = jnp.sqrt(sx ** 2 + sy ** 2 + sz ** 2)

    def _axis(diff, r0, k):
        dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + _EPS)
        return jnp.sum(k * (dist - r0) ** 2)

    def _k(a, b):
        return 0.5 * (a + b)  # edge stiffness = mean of endpoint stiffnesses

    def spring_energy(pos, ks):
        # face neighbours
        E = _axis(pos[1:, :, :] - pos[:-1, :, :], sx, _k(ks[1:, :, :], ks[:-1, :, :]))
        E += _axis(pos[:, 1:, :] - pos[:, :-1, :], sy, _k(ks[:, 1:, :], ks[:, :-1, :]))
        E += _axis(pos[:, :, 1:] - pos[:, :, :-1], sz, _k(ks[:, :, 1:], ks[:, :, :-1]))
        # in-plane (edge) diagonals — both orientations per plane
        E += _axis(pos[1:, 1:, :] - pos[:-1, :-1, :], r_xy, _k(ks[1:, 1:, :], ks[:-1, :-1, :]))
        E += _axis(pos[1:, :, 1:] - pos[:-1, :, :-1], r_xz, _k(ks[1:, :, 1:], ks[:-1, :, :-1]))
        E += _axis(pos[:, 1:, 1:] - pos[:, :-1, :-1], r_yz, _k(ks[:, 1:, 1:], ks[:, :-1, :-1]))
        E += _axis(pos[1:, :-1, :] - pos[:-1, 1:, :], r_xy, _k(ks[1:, :-1, :], ks[:-1, 1:, :]))
        E += _axis(pos[1:, :, :-1] - pos[:-1, :, 1:], r_xz, _k(ks[1:, :, :-1], ks[:-1, :, 1:]))
        E += _axis(pos[:, 1:, :-1] - pos[:, :-1, 1:], r_yz, _k(ks[:, 1:, :-1], ks[:, :-1, 1:]))
        # body (corner) diagonals — all four space-diagonal orientations
        E += _axis(pos[1:, 1:, 1:] - pos[:-1, :-1, :-1], r_xyz, _k(ks[1:, 1:, 1:], ks[:-1, :-1, :-1]))
        E += _axis(pos[1:, 1:, :-1] - pos[:-1, :-1, 1:], r_xyz, _k(ks[1:, 1:, :-1], ks[:-1, :-1, 1:]))
        E += _axis(pos[1:, :-1, 1:] - pos[:-1, 1:, :-1], r_xyz, _k(ks[1:, :-1, 1:], ks[:-1, 1:, :-1]))
        E += _axis(pos[:-1, 1:, 1:] - pos[1:, :-1, :-1], r_xyz, _k(ks[:-1, 1:, 1:], ks[1:, :-1, :-1]))
        return E

    return spring_energy


def solve_mass_effect(
    tumor: np.ndarray,
    atlas_tissue: np.ndarray,
    *,
    alpha: float = 1000.0,
    beta: float = 1.0,
    n_iter: int = 300,
    grid_downsample: int = 5,
    learning_rate: float = 0.05,
    brain_mask: Optional[np.ndarray] = None,
    stiffness_mapping: Optional[dict] = None,
    record_energy: bool = False,
    verbose: bool = True,
) -> PSMResult:
    """Solve the tumor mass-effect deformation for one patient.

    Parameters
    ----------
    tumor         : tumor-concentration volume ``c`` in [0, 1], shape (X, Y, Z).
                    Acts as the driving potential.
    atlas_tissue  : registered healthy-atlas tissue labels, same shape, integer
                    (``1=CSF, 2=GM, 3=WM, 4=VT``). Supplies both the per-node
                    spring stiffness and (by default) the brain mask.
    alpha         : tumor-potential weight (the single global parameter; paper
                    operating point ``alpha=1000``).
    beta          : spring weight (held fixed at 1.0 in the paper).
    n_iter        : Adam iterations.
    grid_downsample : integer factor by which the node grid is coarser than the
                    voxel grid (paper uses 5 -> ~5 mm node spacing on 1 mm data).
    learning_rate : initial Adam learning rate (cosine-decayed to ~0).
    brain_mask    : optional explicit brain mask; defaults to ``atlas_tissue > 0``.
    stiffness_mapping : optional ``{label: multiplier}`` (see ``psm.stiffness``).
    record_energy : if True, store per-iteration energy history in the result.
    verbose       : print progress every 10 iterations.

    Returns
    -------
    PSMResult with the dense displacement field (X, Y, Z, 3) in voxel units.
    """
    tumor = np.asarray(tumor, dtype=np.float32)
    atlas_tissue = np.asarray(atlas_tissue)
    if tumor.shape != atlas_tissue.shape:
        raise ValueError(
            f"tumor {tumor.shape} and atlas_tissue {atlas_tissue.shape} must match"
        )

    # ── 1. Masks ─────────────────────────────────────────────────────────────
    if brain_mask is None:
        brain_mask = np.asarray(atlas_tissue) > 0
    else:
        brain_mask = np.asarray(brain_mask, dtype=bool)
    tumor = tumor.copy()
    tumor[~brain_mask] = 0.0

    # ── 2. Coarse node grid over the volume ──────────────────────────────────
    shape_reduced = tuple(int(s / grid_downsample) for s in tumor.shape)
    X0, Y0, Z0, (sx, sy, sz) = generate_grid(
        shape_reduced,
        bounds=((0, tumor.shape[0]), (0, tumor.shape[1]), (0, tumor.shape[2])),
    )
    interior, _boundary = brain_grid_masks(X0, Y0, Z0, brain_mask)

    # ── 3. Per-node stiffness from the atlas tissue labels ───────────────────
    stiff_vol = stiffness_from_tissue(atlas_tissue, mapping=stiffness_mapping)
    node_stiff = sample_node_stiffness(stiff_vol, X0, Y0, Z0)

    nx, ny, nz = X0.shape
    free_flat = interior.ravel()
    free_indices = jnp.array(np.where(free_flat)[0], dtype=jnp.int32)
    n_free = int(free_indices.shape[0])
    if verbose:
        print(f"grid {X0.shape} | free nodes {n_free} | spacing "
              f"({sx:.2f}, {sy:.2f}, {sz:.2f}) vox")

    # ── 4. JAX tensors / energy ───────────────────────────────────────────────
    tumor_jax = jnp.asarray(tumor)
    pos0 = jnp.asarray(np.stack([X0, Y0, Z0], axis=-1))   # (nx, ny, nz, 3)
    ks = jnp.asarray(node_stiff)
    spring_energy = _build_spring_energy(sx, sy, sz)

    def reconstruct(u_free):
        u_full = jnp.zeros((nx * ny * nz, 3)).at[free_indices].set(u_free)
        return pos0 + u_full.reshape(nx, ny, nz, 3)

    def potential_energy(pos):
        coords = [pos[..., d].ravel() for d in range(3)]
        c = jax.scipy.ndimage.map_coordinates(tumor_jax, coords, order=1, mode="nearest")
        return jnp.sum(c)

    @jax.jit
    def loss(u_free):
        pos = reconstruct(u_free)
        e_pot = alpha * potential_energy(pos)
        e_spr = beta * spring_energy(pos, ks)
        # normalise by node count so gradients are comparable across patients
        return (e_pot + e_spr) / n_free, (e_pot / n_free, e_spr / n_free)

    value_and_grad = jax.jit(jax.value_and_grad(loss, has_aux=True))

    # ── 5. Adam with cosine decay ─────────────────────────────────────────────
    schedule = optax.cosine_decay_schedule(init_value=learning_rate,
                                           decay_steps=n_iter, alpha=2e-3)
    optimizer = optax.adam(schedule)
    u_free = jnp.zeros((n_free, 3))
    opt_state = optimizer.init(u_free)

    e_tot_hist, e_pot_hist, e_spr_hist = [], [], []
    for i in range(n_iter):
        (e_tot, (e_pot, e_spr)), grads = value_and_grad(u_free)
        updates, opt_state = optimizer.update(grads, opt_state)
        u_free = optax.apply_updates(u_free, updates)
        if record_energy:
            e_tot_hist.append(float(e_tot))
            e_pot_hist.append(float(e_pot))
            e_spr_hist.append(float(e_spr))
        if verbose and i % 10 == 0:
            print(f"[{i:03d}] E={float(e_tot):.4f}  pot={float(e_pot):.4f}  "
                  f"spr={float(e_spr):.4f}")

    # ── 6. Interpolate sparse node displacements to the voxel grid ────────────
    u_np = np.zeros((nx * ny * nz, 3), dtype=np.float32)
    u_np[free_flat] = np.asarray(u_free)
    u_grid = u_np.reshape(nx, ny, nz, 3)
    zoom = np.array(tumor.shape) / np.array((nx, ny, nz))
    disp = np.zeros((*tumor.shape, 3), dtype=np.float32)
    for d in range(3):
        disp[..., d] = scipy.ndimage.zoom(u_grid[..., d], zoom, order=1)

    return PSMResult(
        displacement=disp,
        energy_total=e_tot_hist,
        energy_potential=e_pot_hist,
        energy_spring=e_spr_hist,
        n_free_nodes=n_free,
    )
