---
name: Project — matrix-free Trilinos plan for the IB stage
description: Architectural plan for the immersed-boundary phase, settled in the 2026-05-14 conversation. Read before starting IBPM work.
type: project
originSessionId: 12fb2afb-57e7-4a3b-acaf-2f8c91188f9d
---
The IB phase will turn the linear system into the Taira–Colonius
saddle-point structure

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

**Decision (settled with the user on 2026-05-14)**: implement `S`
**matrix-free**, *not* as an assembled `Tpetra::CrsMatrix`.

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

**What to bring back into the build when starting this phase**:

- `find_package(Trilinos REQUIRED COMPONENTS Tpetra Belos Ifpack2 Teuchos)`
  in the top-level `CMakeLists.txt`.
- `Tpetra::all_libs`, `Belos::all_libs`, `Ifpack2::all_libs`,
  `Teuchos::all_libs` in `src/CMakeLists.txt`.
- Re-add `-DTrilinos_DIR=...` to the `.zed/tasks.json` configure tasks
  (path is `/Users/hang/opt/trilinos-17.0.0/install/lib/cmake/Trilinos`,
  see `reference_dep_paths.md`).
