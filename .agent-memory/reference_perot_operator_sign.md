---
name: Reference — sign convention of the Perot modified-Poisson operator
description: `D B^N G` is negative semidefinite (like ∇²); the code negates it before the Krylov solve. Bit me on the Perot rewrite; document for future bites.
type: reference
---

The Perot modified-Poisson operator `D B^N G` (gradient → polynomial
in face Laplacian → divergence) is **negative** semidefinite when
discretised on a periodic cell-centred grid.  Spectrum on a Fourier
mode `cos(k·x)`:

  λ(D B^N G, k) = −K_h² · (1 − ε K² + (ε K²)² − …)

where `K_h² > 0` is the discrete `D G` eigenvalue and the bracket
is positive for small `ε K²` (which is enforced anyway by the
Neumann-series convergence requirement).  Null space: constants.

Unpreconditioned CG would require a positive semidefinite operator: it
takes `α = (r,r)/(d, A d)`, and `(d, A d) ≤ 0` for our `A = D B^N G`
makes the search direction step in the wrong direction.

The code's solution: in `ApplyModifiedPoissonOp`, **negate the result
at the end** (`r ← −r`).  In `ProjectPerot`, build the RHS with a
**negative** sign as well (`rhs = −(1/dt) D u*`).  Both negations
cancel, so the recovered `p` solves the original Perot equation.

The projection step itself — `u^{n+1} = u* − dt B^N G p` — keeps the
natural positive sign on `B^N G p`.  Only the *solve* uses the
negated operator+RHS pair.

This is the same sign story as AMReX `MLPoisson` (see
`reference_mlpoisson_sign.md`), but the current code path is
matrix-free.  Full-domain levels have the positive sign after the flip;
partial AMR levels with C/F Dirichlet ghosts are nonsymmetric, so the
active solver is BiCGStab rather than CG.

**How to apply**: if you ever see sign-sensitive Krylov failures on an
operator related to ∇²,
check the sign convention first.  The two convention pairs are
*(negative operator, positive RHS)* — fine for direct methods, broken
for CG — and *(negative operator, negative RHS)* — recovers the
positive-sign pressure equation used by the current Krylov solve.
