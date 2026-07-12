---
name: Project — matrix-free force-Schur solve for IB projection
description: Current marker-space force Schur implementation and the settled path to Belos FGMRES, using composite pressure inverses from AMReX MLMG and local marker preconditioning.
type: project
originSessionId: 12fb2afb-57e7-4a3b-acaf-2f8c91188f9d
---
The composite AMR Taira-Colonius projection has the block system

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
2. Use a marker-space force Schur solve as the main coupled path.  Do not
   restore alternating force-only and pressure-only cleanup iterations.
3. The transitional in-repo implementation uses BiCGStab and therefore keeps
   every approximate pressure inverse application fixed and reproducible.
   Move to Belos FGMRES when pressure work becomes variable or a local marker
   preconditioner is added.  The composite operator is not proven symmetric.

## Current Schur scaling

The hand-rolled solver applies the same scaling that a future Tpetra wrapper
must preserve.  For level `l`, define the geometric-mean cell spacing

```
h_l = (product_d dx_l[d])^(1/d).
```

For marker `k` with line-length/triangle-area quadrature weight `w_k`, define

```
c_k = h_f^(d-1) / w_k
    = (product_d dx_f[d]) / (h_f w_k),
f_k = c_k fhat_k.
```

The equivalent scaled coupled system is

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

## Force-Schur solve

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
4. Run outer BiCGStab in marker space with fixed pressure work, then recover
   pressure once with
   `p = A^-1(r_p-Cf)` and check the true full coupled residual.

`A^-1` must not be a fully converged hand-rolled modified-Poisson solve on
every outer iteration.  Start with AMReX MLMG for the standard composite
Poisson as an approximate inverse, using a fixed small number of V-cycles and
a zero initial guess so each application is a reproducible linear map.  A
better approximation can wrap MLMG inside a short modified-Poisson correction.
If inner tolerances or cycle counts vary, replace BiCGStab with FGMRES and
tighten inner work as the outer residual falls.  Always judge final acceptance
with the original matrix-free coupled operator, not with the approximate Schur
operator.

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
`INSSolver_Project.cpp` owns the composite hierarchy blocks, scaling, and
marker-space BiCGStab.  Its pressure inverse starts from zero, uses
`ins.ib_schur_mg_iters` MLMG preconditioner cycles, and applies a fixed number
of modified-Poisson defect corrections controlled by
`ins.ib_schur_pressure_corrections`.  Pressure recovery uses the exact
matrix-free modified operator, and acceptance uses an independent true
coupled residual.  A failed true check invokes one exact coupled BiCGStab
fallback; it does not restore alternating block cleanup.

One-step measured results at the shipped tolerances are 1 marker iteration for
the 2D square cases, 4 for the 3D plane cases, 5 for the coarse 2D cylinder,
and 7 for the four-level 2D cylinder.  The latter formerly failed at 4000 full
coupled iterations.  The non-singular 3D cylinder channel is still the known
limitation: 254 Schur iterations reduce the approximate residual to `9.4e-5`
at an average factor of `0.964`, but the true coupled relative residual is
`8.6e-3`.  Its exact fallback still takes 2657 iterations when the cap is
raised to 4000.  This case requires a better pressure inverse and local marker
preconditioning; do not accept its approximate Schur residual as convergence.

**What to bring back into the build when implementing the Tpetra path**:

- `find_package(Trilinos REQUIRED COMPONENTS Tpetra Belos Ifpack2 Teuchos)`
  in the top-level `CMakeLists.txt`.
- `Tpetra::all_libs`, `Belos::all_libs`, `Ifpack2::all_libs`,
  `Teuchos::all_libs` in `src/CMakeLists.txt`.
- Re-add `-DTrilinos_DIR=...` to the `.zed/tasks.json` configure tasks
  (path is `/Users/hang/opt/trilinos-17.0.0/install/lib/cmake/Trilinos`,
  see `reference_dep_paths.md`).
