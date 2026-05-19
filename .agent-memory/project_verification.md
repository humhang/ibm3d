---
name: Project — solver verification (convergence orders)
description: The NS+Perot solver is verified 2nd-order in space / 1st-order in time against the exact 2D Taylor–Green vortex. How to reproduce and why these orders.
type: project
---

**Verified 2026-05-18** against the exact 2D Taylor–Green decaying
vortex (an analytic solution of incompressible NS):

- **Spatial: 2nd order.**  Fixed small `dt=2.5e-4`, `nu=0.05`,
  `t=0.15`, single level, periodic.  L2 velocity error vs exact:
  N=16 → 9.72e-5, N=32 → 2.40e-5, N=64 → 5.93e-6.  Ratio 4.05 each
  doubling ⇒ order ≈ **2.02** (Linf ≈ 2.00).

- **Temporal: 1st order.**  Total-vs-exact error CANNOT show this —
  at fixed grid the spatial error doesn't vanish and is opposite-
  signed to the temporal error, so the total is non-monotone in dt
  (it sinks *below* the spatial floor near dt≈8e-3 then rises back).
  Use **fixed-grid self-convergence** instead: N=128, compare the
  final field at dt vs dt/2 (spatial error identical → cancels).
  ‖u(4e-3)−u(2e-3)‖=2.47e-7, ‖u(2e-3)−u(1e-3)‖=1.24e-7,
  ‖u(1e-3)−u(5e-4)‖=6.18e-8.  Ratios exactly 2.00 ⇒ order **1.00**.

**Why these orders** (by design, not a bug): centered 2nd-order
spatial stencils ⇒ 2nd-order space.  The advection term enters the
predictor as the explicit-Euler `dt·A^n` (A^n = −(u^n·∇)u^n at time n
only) ⇒ 1st-order in time.  CN diffusion + the `B^N` Perot factor are
2nd-order but cannot lift the overall order past the explicit-Euler
advection.  To get 2nd-order time one would need AB2 / RK / explicit-
midpoint advection — not implemented, and not required for the
current goals.

**How to reproduce**: `inputs.tg2d` (ic=tg2d).  Prints a
`TG2D-ERROR ... L2= Linf=` line at the end.  For the temporal study
add `ins.tg2d_dump=<file>` to write the final level-0 cell-centred
(u,v) and `ins.tg2d_cmp=<file>` to print `TG2D-SELFCONV ... L2diff=`
against a previous dump (same N).  Implementation: `ComputeTG2DError`
+ the tg2d branch in `InitFlowField` + the dump/cmp block at the end
of `Run` (`src/INSSolver.cpp`), `VisMF` for the field I/O.

**Takeaway for future agents**: the discretisation is correct and
converges at its design rate.  Don't "fix" the 1st-order temporal
result — it's expected.  When verifying any time-stepping change,
re-run the self-convergence sweep (the exact-error metric alone is
unreliable once spatial and temporal errors are comparable).
