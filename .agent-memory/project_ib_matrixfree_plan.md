---
name: Project — matrix-free Trilinos plan for the scalable IB solve
description: Architectural plan for replacing the hand-rolled IB BiCGStab implementation with a scalable Tpetra/Belos path; settled in the 2026-05-14 conversation.
type: project
originSessionId: 12fb2afb-57e7-4a3b-acaf-2f8c91188f9d
---
The first in-repo IB path already solves the composite AMR Taira-Colonius
projection with hand-rolled BiCGStab.  The scalable follow-up
should wrap the same operator structure in Trilinos.  The target saddle-point
structure is

```
[ I − dtL    G     H ] [u]   [r1]
[   D        0     0 ] [p] = [0 ]
[   E        0     0 ] [f̃]  [u_B]
```

with `G,D` the discrete grad/divergence (already implemented matrix-free
here), and `H,E` the Eulerian↔Lagrangian spread/interpolate operators
using a discrete-delta kernel (Roma 3-pt or Peskin 4-pt).  Following
Taira–Colonius's BN approximation, this formally reduces to a symmetric
"modified Poisson" `S = Q^T B^N Q` on `λ = (p; f̃)` when the discrete
spread/interpolation and divergence/gradient pairs are adjoints.  The
current composite AMR realization is not assumed symmetric because C/F
FillPatch interpolation and active-cell masking alter those adjoint pairs.

**Decision (settled with the user on 2026-05-14)**: implement the
future Trilinos-backed `S` **matrix-free**, *not* as an assembled
`Tpetra::CrsMatrix`.

**Reason**:

- **AMR regridding** changes `D, G, L` row/column structure every regrid
  step; assembling and triple-producting `Q^T B^N Q` is expensive.
- **Moving bodies** change `H, E` rows every time step (kernel support
  shifts with Lagrangian point positions).  Assembly would have to be
  redone *every step*.  Matrix-free has no setup cost beyond updating
  the Lagrangian point list.
- GPU portability (stencil sweeps + delta kernels are textbook GPU
  kernels; SpMV on irregular IB rows is not).

**How to apply**:

- Wrap `S` as a `Tpetra::Operator` subclass whose `apply(X, Y)` calls
  AMReX-side routines: spread (`H λ`) → grad (`G λ`) → polynomial in
  `L` (matrix-free face Laplacian sweeps) → divergence (`D`) →
  interpolate (`E`).  Use Belos GMRES initially.  Switch to CG only after
  symmetry and positive definiteness have been demonstrated for the actual
  weighted composite AMR operator, not only for the formal single-grid
  algebra.
- **Preconditioner**: wrap AMReX `MLMG` standard Poisson as a
  `Tpetra::Operator` and use as a left/right preconditioner.  Far from
  the IB, `S ≈ ∇²` and MLMG kills the high-frequency error; the IB
  cross-coupling rides along on the Krylov iterations.
- Keep AMReX's `FillPatch` and face average-down machinery for C/F
  interface handling, adding FAC/reflux synchronization only where the
  composite discretization requires it.  Don't encode C/F coupling rows in
  an explicit matrix.
- Static-body single-level is a fine starting point for verification;
  the matrix-free machinery is a strict superset and works there too.

**Reference codebase**: IBAMR (Bhalla, Griffith et al. — SAMRAI + PETSc).
JCP 2013 *"A unified mathematical framework and an adaptive numerical
method for fluid-structure interaction with rigid, deforming, and
elastic bodies"* describes the operator + preconditioner structure in
detail.  Treat it as the production reference; AMReX + Tpetra here is
roughly the equivalent stack.

**Current status note (2026-05-20, updated 2026-07-10)**: the first IB projection pass is
implemented without Trilinos: `INSSolver_IB.cpp` owns the geometry-facing
`H/E` kernels and marker tagging, while `INSSolver_Project.cpp` owns the
composite hierarchy operator and hand-rolled BiCGStab.  IB rows and force
columns are active only on the finest level, but pressure rows span all
active AMR cells.  The items below apply when replacing that hand-rolled
coupled solve with the planned Tpetra/Belos wrapper.

**What to bring back into the build when implementing the Tpetra path**:

- `find_package(Trilinos REQUIRED COMPONENTS Tpetra Belos Ifpack2 Teuchos)`
  in the top-level `CMakeLists.txt`.
- `Tpetra::all_libs`, `Belos::all_libs`, `Ifpack2::all_libs`,
  `Teuchos::all_libs` in `src/CMakeLists.txt`.
- Re-add `-DTrilinos_DIR=...` to the `.zed/tasks.json` configure tasks
  (path is `/Users/hang/opt/trilinos-17.0.0/install/lib/cmake/Trilinos`,
  see `reference_dep_paths.md`).
