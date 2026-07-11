---
name: Feedback — matrix-free is the agreed design for the scalable IB Poisson
description: Records a settled architectural decision after a multi-turn back-and-forth, so future agents don't re-litigate it when replacing the hand-rolled IB solver.
type: feedback
originSessionId: 12fb2afb-57e7-4a3b-acaf-2f8c91188f9d
---
**Rule**: when replacing the current hand-rolled coupled BiCGStab solver,
implement the pressure-eliminated marker operator
`S_f = M - B A^-1 C` matrix-free and solve it with Belos FGMRES.  Use an
AMReX MLMG-based approximate composite pressure inverse and a local marker
preconditioner.  Do **not** assemble a `Tpetra::CrsMatrix` and hand it to AMG.

**Why**: settled with the user during the 2026-05-14 conversation
after a tradeoff discussion covering:

- assembly cost on every regrid (AMR makes this expensive);
- assembly cost on every time step (moving bodies make `H`,`E` rows
  change with Lagrangian-point positions);
- the natural reusability of AMReX's FillPatch + flux-match
  machinery for matrix-free operator apply;
- GPU portability (stencil + delta convolution map cleanly to GPU;
  irregular SpMV on IB rows does not).

The user explicitly asked "or do you have better solution, using
matrix free iteration anyway?" and accepted the matrix-free
recommendation in the follow-up.

**How to apply**:

- Expose matrix-free applications of `A`, `B`, `C`, and `M` through
  AMReX-side spread, gradient, `B^N`, divergence, interpolation, FillPatch,
  and average-down routines.
- Apply `A^-1` approximately with a fixed small MLMG cycle count when a
  linear Schur `apply()` is required.  If inner work varies, use FGMRES and
  retain a true full-coupled residual check.
- Preserve the current `h_l` pressure-row and `h_f^(d-1)/w_k` marker-column
  scaling at the Tpetra boundary.
- Start marker preconditioning with local block Jacobi built from
  `E B^N H`; do not restore the retired alternating force/pressure cleanup.
- If a future requirement forces matrix assembly (e.g. a direct solve
  for a very small Lagrangian-point count) — confirm with the user
  before going that route.  This decision is intentional, not default.

Full design detail: `project_ib_matrixfree_plan.md`.
