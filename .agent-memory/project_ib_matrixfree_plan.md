---
name: Project — matrix-free Trilinos plan for the scalable IB solve
description: Architectural plan for replacing the local IB BiCGStab prototype with a scalable Tpetra/Belos path; settled in the 2026-05-14 conversation.
type: project
originSessionId: 12fb2afb-57e7-4a3b-acaf-2f8c91188f9d
---
The local first IB path already solves the finest-level Taira-Colonius
projection with hand-rolled BiCGStab.  The scalable follow-up should
wrap the same operator structure in Trilinos.  The target saddle-point
structure is

```
[ I − dtL    G     H ] [u]   [r1]
[   D        0     0 ] [p] = [0 ]
[   E        0     0 ] [f̃]  [u_B]
```

with `G,D` the discrete grad/divergence (already implemented matrix-free
here), and `H,E` the Eulerian↔Lagrangian spread/interpolate operators
using a discrete-delta kernel (Roma 3-pt or Peskin 4-pt).  Following
Taira–Colonius's BN approximation, this reduces to a single SPD
"modified Poisson" `S = Q^T B^N Q` on `λ = (p; f̃)`.

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
  interpolate (`E`).  Use Belos CG (system is SPD).
- **Preconditioner**: wrap AMReX `MLMG` standard Poisson as a
  `Tpetra::Operator` and use as a left/right preconditioner.  Far from
  the IB, `S ≈ ∇²` and MLMG kills the high-frequency error; the IB
  cross-coupling rides along on the Krylov iterations.
- Keep AMReX's `FillPatch` + `average_down_faces` + `FluxRegister`
  machinery for the C/F interface handling.  Don't try to encode C/F
  coupling rows in any explicit matrix.
- Static-body single-level is a fine starting point for verification;
  the matrix-free machinery is a strict superset and works there too.

**Reference codebase**: IBAMR (Bhalla, Griffith et al. — SAMRAI + PETSc).
JCP 2013 *"A unified mathematical framework and an adaptive numerical
method for fluid-structure interaction with rigid, deforming, and
elastic bodies"* describes the operator + preconditioner structure in
detail.  Treat it as the production reference; AMReX + Tpetra here is
roughly the equivalent stack.

**Current status note (2026-05-20)**: the first IB projection pass is
implemented without Trilinos in `INSSolver_IB.cpp`: `IBGeometry` owns
GPU-friendly host/device points, elements, and markers; `H/E` use a
Peskin 4-point kernel; `ErrorEst` tags marker neighborhoods; and
`ProjectPerot` calls the local coupled BiCGStab on the finest level.
The items below apply when replacing that local coupled solve with the
planned Tpetra/Belos wrapper.

**What to bring back into the build when implementing the Tpetra path**:

- `find_package(Trilinos REQUIRED COMPONENTS Tpetra Belos Ifpack2 Teuchos)`
  in the top-level `CMakeLists.txt`.
- `Tpetra::all_libs`, `Belos::all_libs`, `Ifpack2::all_libs`,
  `Teuchos::all_libs` in `src/CMakeLists.txt`.
- Re-add `-DTrilinos_DIR=...` to the `.zed/tasks.json` configure tasks
  (path is `/Users/hang/opt/trilinos-17.0.0/install/lib/cmake/Trilinos`,
  see `reference_dep_paths.md`).
