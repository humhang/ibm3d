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

**For the projection step**:
```
0 = ∇·u^{n+1} = ∇·u* − dt ∇²φ   ⇒   ∇²φ = (1/dt) ∇·u*
```
So **rhs = +(1/dt) ∇·u*** (POSITIVE).  A negative sign here makes the
projection go the *wrong* direction — divergence then grows ~3–5× per
step instead of being killed.  (This bug appeared during the initial
NS+AMR implementation and was caught only via end-to-end testing; the
MLMG convergence diagnostic looked clean because the wrong-sign system
also converges, it just doesn't enforce divergence-free.)

The MLLinOp doc-comment language about `(αI − βL)` is misleading — the
final operator MLPoisson exposes is `+L = +∇²`, regardless of any
internal scalar storage.

**How to apply**: if you ever see `|div u|` growing across time steps
in an MLMG-based projection, *flip the sign of the RHS first* — that's
the most likely cause.
