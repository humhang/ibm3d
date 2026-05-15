---
name: User — CFD researcher building an AMR IBPM solver
description: Background and expertise level of the user, used to calibrate explanation depth and pacing on CFD / numerical-linear-algebra questions.
type: user
originSessionId: 12fb2afb-57e7-4a3b-acaf-2f8c91188f9d
---
The user is a CFD researcher implementing the **Taira–Colonius immersed-boundary
projection method** (doi.org/10.1016/j.jcp.2007.03.005) on an AMR Cartesian
grid.  They have deep working knowledge of:

- Incompressible Navier–Stokes, staggered/MAC discretisations, projection
  methods (approximate and exact), Crank–Nicolson, Neumann-series inversion of
  `(I − εL)`.
- Saddle-point systems, Schur complements, the Taira–Colonius "modified
  Poisson" formulation, IB regularised-delta spread/interpolate operators.
- AMR concepts: levels, refinement ratios, coarse–fine interfaces, refluxing,
  proper nesting.
- Krylov methods, multigrid (geometric and algebraic), block preconditioning.

They are comfortable with operator-theoretic notation (`Q = [G H]`, modified
Schur `Q^T A^{−1} Q`, the BN polynomial truncation) and reach for the right
vocabulary unprompted.  Calibrate explanations accordingly — skip the
introductory framing, jump to the substance and tradeoffs.

They use **Zed** as their editor (CLion legacy in path names is incidental — the
project directory is `~/CLionProjects/ibm3d` but they've moved off CLion).
