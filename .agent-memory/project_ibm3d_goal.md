---
name: Project — ibm3d goal and current status
description: Identifies the end goal of the ibm3d project (AMR + Taira–Colonius IBPM) and what's actually implemented today (NS+AMR only).
type: project
originSessionId: 12fb2afb-57e7-4a3b-acaf-2f8c91188f9d
---
**End goal**: a 3D, MPI-parallel, AMR-capable incompressible Navier–Stokes
solver implementing the **Taira–Colonius immersed-boundary projection method**
(JCP 2007, doi.org/10.1016/j.jcp.2007.03.005) on top of AMReX.

**Current status (as of 2026-05-15)**: the *substrate* — i.e. pure NS+AMR
without immersed bodies — is implemented and verified.

- Staggered (MAC) grid, second-order centred differencing.
- Explicit advection, Crank–Nicolson diffusion via truncated Neumann
  series `(I − εL)^{-1} ≈ Σ_{k=0}^{N} (εL)^k`.
- Composite cell-centred pressure Poisson via **AMReX MLMG** (`MLPoisson`),
  with `average_down_faces` of `u*` before forming the RHS so that the
  composite divergence is level-consistent.
- Periodic BCs only; FillPatch (`face_linear_interp`, `lincc_interp`) at
  every operator apply for ghost cells at patch + C/F boundaries.
- Verified on 3D Taylor–Green vortex: `|div u|_∞ ~ 10⁻¹⁴` per step at
  single level, `~ 10⁻¹⁴` per step at 2 AMR levels, monotonic energy
  decay, 8–9 MLMG iters to converge.

**Why:** the user wanted the NS+AMR foundation working independently
before adding the IB machinery, since the IB step changes the linear
operator into a saddle-point system and is invasive.

**How to apply:** treat the codebase as a *scaffold* the IB step will be
grafted onto.  Don't refactor away the per-level FillPatch / averaging
patterns — they're the load-bearing AMR machinery the IB code will reuse.
Trilinos has been temporarily removed from the build; it will return when
the IB is added (see `project_ib_matrixfree_plan.md`).
