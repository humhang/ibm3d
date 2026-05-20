---
name: Project — ibm3d goal and current status
description: Identifies the end goal of the ibm3d project (AMR + Taira–Colonius IBPM) and what's actually implemented today (NS+AMR only).
type: project
originSessionId: 12fb2afb-57e7-4a3b-acaf-2f8c91188f9d
---
**End goal**: a 3D, MPI-parallel, AMR-capable incompressible Navier–Stokes
solver implementing the **Taira–Colonius immersed-boundary projection method**
(JCP 2007, doi.org/10.1016/j.jcp.2007.03.005) on top of AMReX.

**Current status (as of 2026-05-19)**: the *substrate*
— pure NS+AMR without immersed bodies — is implemented and verified
using the **Perot 1997 / Taira–Colonius (without IB)** fractional-step
method.

- Staggered (MAC) grid, second-order centred differencing.
- Explicit AB2 advection, Crank–Nicolson diffusion via truncated Neumann
  series `B^N = Σ_{k=0}^{N} (εL)^k ≈ (I − εL)^{-1}`.  Default `N = 2`.
- Pressure step solves the **modified Poisson** `(D B^N G) p = D u*/dt`
  matrix-free via hand-rolled BiCGStab on the negated operator
  `−D B^N G`.  No preconditioner currently.
- Projection: `u^{n+1} = u* − dt B^N G p` (same `B^N` as the
  predictor — that's the Perot exact-factorisation consistency).
- `m_pressure` solved for directly each step; previous step value
  is the Krylov warm start.  `m_phi` is gone.
- Physical BCs are implemented (`periodic`, `noslip`, `inflow`, `slip`,
  `outflow`); pressure C/F interpolation uses `pc_interp`.
- Verified on 3D Taylor–Green vortex: `|div u|_∞ ~ 10⁻¹¹` per step at
  single level *and* at 2 AMR levels (Krylov tolerance dominated, not
  method error), monotonic energy decay matching the previous
  Chorin/MLMG numerical result to 6+ digits.

**What changed on 2026-05-15 (second pass)**: the first implementation
used a Chorin-style projection with MLMG on the standard Poisson
`∇²φ = (1/dt) D u*` plus an incremental pressure update `p ← p + φ`.
The user asked specifically for the Perot 1997 formulation as the
correct stepping stone to the eventual Taira–Colonius IBPM, since the
IB rows attach naturally to the modified-Poisson structure but not to
the standard one.  See `project_algorithmic_decisions.md` for the
algorithmic rationale and `reference_perot_operator_sign.md` for the
sign-convention bug that ate ~30 minutes during the rewrite.

**Why:** the user wanted the NS+AMR foundation working independently
before adding the IB machinery, since the IB step changes the linear
operator into a saddle-point system and is invasive.

**How to apply:** treat the codebase as a *scaffold* the IB step will be
grafted onto.  Don't refactor away the per-level FillPatch / averaging
patterns — they're the load-bearing AMR machinery the IB code will reuse.
Trilinos has been temporarily removed from the build; it will return when
the IB is added (see `project_ib_matrixfree_plan.md`).
