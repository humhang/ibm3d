---
name: Feedback — matrix-free is the agreed design for the IB Poisson
description: Records a settled architectural decision after a multi-turn back-and-forth, so future agents don't re-litigate it.
type: feedback
originSessionId: 12fb2afb-57e7-4a3b-acaf-2f8c91188f9d
---
**Rule**: for the Taira–Colonius IBPM modified-Poisson step, implement
the operator `S = Q^T B^N Q` matrix-free as a `Tpetra::Operator`.  Do
**not** assemble it as a `Tpetra::CrsMatrix` and hand to AMG.

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

- When the IB phase begins, build the operator as a `Tpetra::Operator`
  subclass that calls AMReX-side `apply()` routines (spread, grad,
  polynomial-in-L, divergence, interpolate).
- Use Belos CG for the outer Krylov solve (system is SPD after the
  BN approximation).
- Use AMReX `MLMG` wrapped as a `Tpetra::Operator` for the
  preconditioner — the standard Poisson is a good local approximation
  to the modified Poisson away from the IB.
- If a future requirement forces matrix assembly (e.g. a direct solve
  for a very small Lagrangian-point count) — confirm with the user
  before going that route.  This decision is intentional, not default.

Full design detail: `project_ib_matrixfree_plan.md`.
