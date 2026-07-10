---
name: Project — solver verification (convergence orders)
description: The NS+Perot solver is verified 2nd-order in space AND 2nd-order in time (AB2 advection + cn_order≥2 diffusion) against the exact 2D Taylor–Green vortex. How to reproduce; the subtle reason TG2D alone can't see the advection time order.
type: project
---

Verified against the exact 2D Taylor–Green vortex (an analytic
solution of incompressible NS).  Use **fixed-grid self-convergence**
for temporal order — the total-vs-exact error cannot isolate it (at
fixed grid the spatial error doesn't vanish, is opposite-signed, and
partially cancels the temporal error → non-monotone in dt).
`ins.tg2d_dump`/`ins.tg2d_cmp` write/compare the final level-0
cell-centred (u,v) so ‖u(dt)−u(dt/2)‖ cancels the (identical)
spatial error.

- **Spatial: 2nd order** (2026-05-18).  `dt=2.5e-4`, `nu=0.05`,
  `t=0.15`, single level, periodic.  L2 error vs exact:
  N=16→9.72e-5, 32→2.40e-5, 64→5.93e-6.  Ratio 4.05 ⇒ **2.02**
  (Linf ≈ 2.00).

- **Temporal: 2nd order** (2026-05-19, after the AB2 upgrade).
  Self-convergence, N=128, dt 4e-3→5e-4:
  - Stationary TG2D, `cn_order=2`: ‖Δu‖ 2.47e-11, 6.18e-12,
    1.55e-12 ⇒ ratio 4.0 ⇒ order **2.00**.
  - Convecting TG2D (`tg_uc=1, tg_vc=0.5`), AB2 + `cn_order=2`:
    ‖Δu‖ 4.77e-6, 1.19e-6, 2.98e-7 ⇒ ratio 4.0 ⇒ order **2.00**.

**The subtle point (don't relearn this the hard way).**  For the
*stationary* TG2D the nonlinear term (u·∇)u is an exact gradient, so
the pressure projection annihilates it entirely — the projected
velocity is **independent of the advection time discretisation**
(Euler and AB2 give byte-identical results).  Therefore:
  - stationary TG2D temporal order = `cn_order` (the `B^N`
    Neumann-truncation order; its error is O((εL)^{N+1}) → global
    O(dt^N)); it probes ONLY the diffusion time integration.
  - to see the advection time order you MUST use a flow with
    rotational advection — the **convecting** TG vortex (Galilean
    boost by (tg_uc,tg_vc); still an exact periodic NS solution).

This is why the pre-AB2 result looked "1st order": that was the
`cn_order=1` diffusion truncation, NOT explicit-Euler advection.
Earlier memory wrongly attributed it to advection — corrected here.

**Scheme orders (by design):** centered space ⇒ 2nd order space.
Time: advection = AB2 `dt(3/2 A^n − 1/2 A^{n-1})` ⇒ 2nd order;
diffusion = `B^N` truncated Neumann ⇒ order N.  Globally 2nd-order
in time **iff** AB2 is active AND `cn_order ≥ 2`.  Default
`m_cn_order` is now **2** for exactly this reason; all the shipped
`tests/*/inputs.*` set `cn_order=2`.  AB2 falls back to Euler on step 1 and
the first step after a regrid (`m_ab2_valid`) — one O(dt²) step,
harmless at fixed grid; with *frequent* regrids it can erode the
formal temporal order (the deferred fix is to FillPatch
`m_advect_old` through the regrid hooks, like the velocity).

**Reproduce:** `tests/3d/tg2d/inputs.tg2d` (ic=tg2d; set `tg_uc`/`tg_vc` for the
convecting case).  Prints `TG2D-ERROR … L2= Linf=`; with
`ins.tg2d_dump`/`ins.tg2d_cmp` prints `TG2D-SELFCONV … L2diff=`.
Code: `ComputeTG2DError` + the tg2d branch in `InitFlowField` +
the dump/cmp block at the end of `Run` (VisMF for field I/O); AB2 in
`Advance` (`m_advect_old`, `m_ab2_valid`).

**IB smoke tests (added 2026-05-20; coupled BiCGStab rerun 2026-07-10):**

- `tests/3d/ib_plane/inputs.ib_plane` — single-level two-triangle STL
  plane in the Taylor–Green field.  One step gives
  `|E u - U_ib|_inf ≈ 2.0e-10`, `|div u|_inf ≈ 6.6e-12`.
- `tests/3d/ib_plane_amr/inputs.ib_plane_amr` — same geometry with one
  refinement level.  Vorticity tags plus IB tags keep the body on the
  finest mesh.  One step gives combined relative residual `6.0e-11`,
  `|E u - U_ib|_inf ≈ 1.5e-10`, and `|div u|_inf ≈ 2.0e-11`.
- `tests/2d/ib_square_amr/inputs.ib_square_amr` — one refinement level,
  singular periodic pressure system.  One step converges in 821 iterations
  to combined relative residual `8.8e-11`, marker slip `2.0e-11`, and
  `|div u|_inf ≈ 1.2e-11`.
- `tests/3d/ib_cylinder_channel/inputs.ib_cylinder_channel` — channel
  flow past a stationary radius-0.125 cylinder, with STL panel sizes
  near `1.5 * dx` because coupled BiCGStab has no in-iteration IB block
  preconditioner.
  One step gives roughly O(100-300) composite IB BiCGStab iterations at `1e-4`
  relative residual, `|E u - U_ib|_inf ≈ 2e-3`, and
  `|div u|_inf ≈ 1.5e-4`.
- `tests/2d/ib_cylinder_re100/inputs.ib_cylinder_re100` remains the
  conditioning stress case.  A one-step four-level run reaches the 4000
  iteration limit at combined relative residual `1.4e-4` for a requested
  `1e-5`; marker slip is `4.5e-4`, much smaller than the former GMRES
  failure, but the remaining pressure residual leaves
  `|div u|_inf ≈ 1.1e-3`.
- `tests/2d/ib_cylinder_re100_coarse/inputs.ib_cylinder_re100_coarse` keeps
  the same physical problem and base grid but uses one refinement level and
  24 markers.  A one-step local run converges in 3189 iterations to combined
  relative residual `9.7e-6`, marker slip `1.5e-5`, and
  `|div u|_inf ≈ 9.3e-5`.

**Composite C/F regression (2026-07-09):**

- `tests/2d/tg2d_cf/inputs.tg2d_cf` convects Taylor–Green through a partial
  refinement boundary for five short steps.  Composite BiCGStab reaches a
  true relative residual of about `9e-13`; `|div u|_inf` remains between
  `2e-15` and `4e-15`.  This case specifically guards hierarchy application
  of every `B^N` term and conservative-linear pressure interpolation with
  physical-BC-filled scratch data.

**Takeaway:** discretisation is correct at its design rate
(2nd/2nd).  When verifying any time-stepping change, run the
self-convergence sweep AND use the convecting vortex if the change
touches advection — stationary TG2D is blind to it.
