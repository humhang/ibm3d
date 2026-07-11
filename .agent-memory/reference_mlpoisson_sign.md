---
name: Reference — AMReX MLPoisson sign convention
description: MLPoisson's discrete Laplacian sign, verified against the 3D kernel source. Bit me once already, will bite again.
type: reference
originSessionId: 12fb2afb-57e7-4a3b-acaf-2f8c91188f9d
---
AMReX's `MLPoisson` Fapply computes the **positive** discrete Laplacian:

```cpp
// AMReX_MLPoisson_3D_K.H, mlpoisson_adotx
y(i,j,k) = dhx*(x(i-1) - 2x(i) + x(i+1))
         + dhy*(x(i,j-1) - 2x(i,j) + x(i,j+1))
         + dhz*(x(i,j,k-1) - 2x(i,j,k) + x(i,j,k+1));
```

This is `y = +∇²x`, not `-∇²x`.  The `mlpoisson_gsrb` smoother has
`res = rhs - ∇²φ`, so at convergence `∇²φ = rhs`.

Consequently: `mlmg.solve(phi, rhs, …)` solves `∇²φ = rhs` directly.

**For a standard projection step**:
```
0 = ∇·u^{n+1} = ∇·u* − dt ∇²φ   ⇒   ∇²φ = (1/dt) ∇·u*
```
So **rhs = +(1/dt) ∇·u*** (POSITIVE).  A negative sign makes that
standard projection go the *wrong* direction — divergence then grows ~3–5× per
step instead of being killed.  (This bug appeared during the initial
NS+AMR implementation and was caught only via end-to-end testing; the
MLMG convergence diagnostic looked clean because the wrong-sign system
also converges, it just doesn't enforce divergence-free.)

The current `BuildCompositePoissonInitialGuess` use is different: its
target matrix-free operator is `-D B^N G`, approximately `-∇²`, while
`MLPoisson` applies `+∇²`.  It therefore passes `-rhs` to MLMG.  The result
is only a candidate initial guess and is retained only when the true
modified-operator residual decreases.

The MLLinOp doc-comment language about `(αI − βL)` is misleading — the
final operator MLPoisson exposes is `+L = +∇²`, regardless of any
internal scalar storage.

**How to apply**: derive the sign from the operator being approximated.
For a direct standard projection use `+(1/dt) D u*`; for the checked
approximate inverse of the code's negated modified operator, use the
negative of that operator's RHS.
