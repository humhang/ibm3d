---
name: Project — matrix-free force-Schur plan for the scalable IB solve
description: Settled plan for replacing coupled BiCGStab with a marker-space force Schur solve, using composite pressure inverses from AMReX MLMG and a flexible outer Krylov method.
type: project
originSessionId: 12fb2afb-57e7-4a3b-acaf-2f8c91188f9d
---
The first in-repo IB path solves the composite AMR Taira-Colonius projection
with hand-rolled BiCGStab.  Eliminating velocity gives the coupled system

```
[ A  C ] [p] = [r_p ]
[ B  M ] [f]   [r_ib]
```

where

```
A = -D B^N G       C = -D B^N H
B =  E B^N G       M =  E B^N H.
```

Pressure rows and unknowns span all active AMR levels.  `H`, `E`, and marker
force unknowns act only on the finest level.  The composite realization is
not assumed symmetric because C/F FillPatch interpolation and active-cell
masking alter the formal adjoint pairs.

**Settled decisions**:

1. Keep the coupled operator and all Schur blocks matrix-free.  Do not assemble
   a `Tpetra::CrsMatrix`; AMR regridding changes `D/G/L`, moving bodies change
   `H/E`, and the stencil/delta operations map better to portable GPU kernels.
2. Replace the eventual full coupled unpreconditioned iteration with a
   marker-space force Schur solve.  Do not restore alternating force-only and
   pressure-only cleanup iterations.
3. Use a flexible nonsymmetric outer method, initially Belos FGMRES.  The
   composite operator is not proven symmetric, and the pressure inverse may
   use variable inner work.

## Current coupled scaling

The hand-rolled solver now applies the same scaling that the future Tpetra
wrapper must preserve.  For level `l`, define the geometric-mean cell spacing

```
h_l = (product_d dx_l[d])^(1/d).
```

For marker `k` with line-length/triangle-area quadrature weight `w_k`, define

```
c_k = h_f^(d-1) / w_k
    = (product_d dx_f[d]) / (h_f w_k),
f_k = c_k fhat_k.
```

BiCGStab currently solves

```
[ h_l A       h_l C C_f ] [p   ] = [h_l r_p]
[ B               M C_f ] [fhat]   [r_ib   ],
```

with the pressure-row factor applied level by level and `C_f=diag(c_k)`
repeated for each force component.  Stored/output `m_ib_force` remains the
physical `f`, and diagnostics report physical pressure and marker residuals;
only the Krylov representation and combined convergence norm are scaled.

Why this scaling is fixed rather than tunable: spreading already contains
`w_k`.  On a uniform grid with marker spacing comparable to `h`, the
transformation makes all four blocks `O(1/h)` and recovers the expected
transpose scaling of the pressure/marker cross blocks.  It also prevents the
extra derivative in pressure rows and marker quadrature weights from setting
the Krylov balance accidentally.

## Future force-Schur solve

Eliminate pressure for a trial force:

```
p = A^-1 (r_p - C f),
(M - B A^-1 C) f = r_ib - B A^-1 r_p.
```

Thus the marker-space operator and right-hand side are

```
S_f f = g,
S_f = M - B A^-1 C,
g   = r_ib - B A^-1 r_p.
```

Apply `S_f` without assembly:

1. Spread a trial marker force and form `C f` with the existing composite
   hierarchy kernels.
2. Apply an approximate composite pressure inverse to `C f`.
3. Apply `B` to that pressure response and subtract it from `M f`.
4. Run outer FGMRES in marker space, then recover pressure once with
   `p = A^-1(r_p-Cf)` and check the true full coupled residual.

`A^-1` must not be a fully converged hand-rolled modified-Poisson solve on
every outer iteration.  Start with AMReX MLMG for the standard composite
Poisson as an approximate inverse, using a fixed small number of V-cycles and
a zero initial guess so each application is a reproducible linear map.  A
better approximation can wrap MLMG inside a short modified-Poisson correction.
If inner tolerances or cycle counts vary, keep FGMRES and tighten inner work as
the outer residual falls.  Always judge final acceptance with the original
matrix-free coupled operator, not with the approximate Schur operator.

The first marker preconditioner should be local and GPU portable: component
block Jacobi from a diagonal/local-neighborhood approximation of
`M = E B^N H`, including the same marker-weight scaling above.  Add overlap
between nearby marker supports only if block Jacobi is insufficient.  This
attacks the dense-marker near-null modes without introducing a global
assembled matrix.

An alternative implementation is full-system FGMRES with the same force-Schur
factorization used as a block triangular preconditioner.  Prefer that form if
variable inner pressure solves make a standalone Schur `apply()` insufficiently
linear.  The exact full coupled operator must remain the outer residual
operator in either formulation.

Keep AMReX FillPatch and face average-down machinery inside every block apply.
Static-body single-level tests are the first verification stage, followed by
the existing 2D/3D AMR IB cases and an MPI decomposition-invariance check.

**Reference codebase**: IBAMR (Bhalla, Griffith et al. — SAMRAI + PETSc).
JCP 2013 *"A unified mathematical framework and an adaptive numerical
method for fluid-structure interaction with rigid, deforming, and
elastic bodies"* describes the operator + preconditioner structure in
detail.  Treat it as the production reference; AMReX + Tpetra here is
roughly the equivalent stack.

**Current status note (updated 2026-07-11)**: `INSSolver_IB.cpp` owns the
geometry-facing `H/E` kernels and marker tagging, while
`INSSolver_Project.cpp` owns the composite hierarchy operator, scaling, and
hand-rolled BiCGStab.  The old post-BiCGStab alternating force/pressure
subiterations and pressure-only cleanup have been removed.  Trilinos remains
out of the current build.  The scaling above is only equilibration: small IB
cases still take O(100-700) iterations, and the non-singular 3D cylinder takes
2657 iterations at `1e-4` when allowed to run past its default 1000-iteration
cap.  Treat that cost as the baseline the force-Schur preconditioner must beat.

**What to bring back into the build when implementing the Tpetra path**:

- `find_package(Trilinos REQUIRED COMPONENTS Tpetra Belos Ifpack2 Teuchos)`
  in the top-level `CMakeLists.txt`.
- `Tpetra::all_libs`, `Belos::all_libs`, `Ifpack2::all_libs`,
  `Teuchos::all_libs` in `src/CMakeLists.txt`.
- Re-add `-DTrilinos_DIR=...` to the `.zed/tasks.json` configure tasks
  (path is `/Users/hang/opt/trilinos-17.0.0/install/lib/cmake/Trilinos`,
  see `reference_dep_paths.md`).
