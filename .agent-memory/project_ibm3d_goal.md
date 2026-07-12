---
name: Project — ibm3d goal and current status
description: Identifies the end goal of the ibm3d project (AMR + Taira–Colonius IBPM) and what's implemented today.
type: project
originSessionId: 12fb2afb-57e7-4a3b-acaf-2f8c91188f9d
---
**End goal**: a 3D, MPI-parallel, AMR-capable incompressible Navier–Stokes
solver implementing the **Taira–Colonius immersed-boundary projection method**
(JCP 2007, doi.org/10.1016/j.jcp.2007.03.005) on top of AMReX.

**Current status (updated 2026-07-11)**: the NS+AMR substrate is
implemented and verified, and the first prescribed-velocity
Taira–Colonius IB projection path uses a composite pressure hierarchy with
the Lagrangian coupling attached only to the finest AMR level.

- Staggered (MAC) grid, second-order centred differencing.
- Explicit AB2 advection, Crank–Nicolson diffusion via truncated Neumann
  series `B^N = Σ_{k=0}^{N} (εL)^k ≈ (I − εL)^{-1}`.  Default `N = 2`.
- Pressure step solves the **modified Poisson** `(D B^N G) p = D u*/dt`
  matrix-free via hand-rolled BiCGStab on the negated operator
  `−D B^N G`.  A checked AMReX composite-Poisson solve supplies the
  pressure-block initial guess for non-singular systems.
- Projection: `u^{n+1} = u* − dt B^N G p` (same `B^N` as the
  predictor — that's the Perot exact-factorisation consistency).
- `m_pressure` solved for directly each step; previous step value
  is the Krylov warm start.  `m_phi` is gone.
- Physical BCs are implemented (`periodic`, `noslip`, `inflow`, `slip`,
  `outflow`); pressure C/F interpolation uses `cell_cons_interp` with
  explicit physical-BC filling of AMReX interpolation scratch data.
- IB geometry loads from 2D ASCII curves or 3D ASCII/binary STL surfaces.
  `IBGeometry` owns host/device points, elements, and element-centroid
  markers using AMReX `GpuArray` records; marker weights are line length
  in 2D or triangle area in 3D.
- The coupled AMR IB projection eliminates pressure from
  `[-D; E] B^N [G H] [p; f]` and solves the resulting marker-space force
  Schur equation with fixed-work in-repo BiCGStab.  The approximate pressure
  inverse combines MLMG preconditioner cycles with modified-Poisson defect
  corrections; pressure recovery and final acceptance use the exact operator
  and full coupled residual.  The Krylov representation uses marker
  force-column factors `h_f^(d-1)/w_k`; stored forces and reported block
  residuals remain physical.  `H` spreads force components to matching MAC
  faces only on the finest level and `E` interpolates finest-level face
  corrections with marker-centred finite-support kernels, owner masks,
  and atomics to avoid double-counting shared patch faces.
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
coupled projection in `ProjectPerot`.

**What changed on 2026-07-11**: the main coupled path now eliminates pressure
and solves the matrix-free force Schur equation with in-repo BiCGStab.  It uses
a fixed MLMG/modified-Poisson approximate pressure inverse, exact pressure
recovery, and a true full-coupled residual check.  The Tpetra/Belos wrapper and
marker-block preconditioner remain future work.

**How to apply:** keep the Perot `B^N` consistency across predictor,
coupled projection operator, and projection.  IB rows and force columns belong
on the finest level only, while pressure unknowns and active equations span
the hierarchy.  Trilinos remains out of the build until the Tpetra wrapper is
actually implemented.  Do not restore the removed alternating block cleanup;
extend the implemented marker-space `M - B A^-1 C` strategy according to
`project_ib_matrixfree_plan.md`.
