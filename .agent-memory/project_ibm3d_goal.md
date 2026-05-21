---
name: Project — ibm3d goal and current status
description: Identifies the end goal of the ibm3d project (AMR + Taira–Colonius IBPM) and what's implemented today.
type: project
originSessionId: 12fb2afb-57e7-4a3b-acaf-2f8c91188f9d
---
**End goal**: a 3D, MPI-parallel, AMR-capable incompressible Navier–Stokes
solver implementing the **Taira–Colonius immersed-boundary projection method**
(JCP 2007, doi.org/10.1016/j.jcp.2007.03.005) on top of AMReX.

**Current status (as of 2026-05-20)**: the NS+AMR substrate is
implemented and verified, and the first prescribed-velocity
Taira–Colonius IB projection path is in place on the finest AMR level.

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
- IB geometry loads from 2D ASCII curves or 3D ASCII/binary STL surfaces.
  `IBGeometry` owns host/device points, elements, and element-centroid
  markers using AMReX `GpuArray` records; marker weights are line length
  in 2D or triangle area in 3D.
- The coupled finest-level projection solves `[-D; E] B^N [G H] [p; f]`
  with the existing matrix-free BiCGStab path.  `H` spreads force
  components to matching MAC faces and `E` interpolates with
  `ParallelFor`, owner masks, and atomics to avoid double-counting shared
  patch faces.
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

**What changed on 2026-05-20**: the first IB pass added geometry loading
with device copies, element-centroid markers in `IBGeometry`, Peskin
4-point spread/interp, GPU-ready finest-level IB tagging, and the
coupled projection in `ProjectPerot`.  The Tpetra/Belos wrapper and
MLMG preconditioner are still future work; the initial path deliberately
reuses the local BiCGStab solver so the operator is executable
immediately.

**How to apply:** keep the Perot `B^N` consistency across predictor,
Schur operator, and projection.  IB coupling currently belongs on the
finest level only; coarser data is synchronized by average-down where
covered.  Trilinos remains out of the build until the Tpetra wrapper is
actually implemented (see `project_ib_matrixfree_plan.md`).
