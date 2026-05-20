---
name: Project — algorithmic design decisions
description: Records the numerical/algorithmic choices that aren't obvious from reading the code (why matrix-free modified Poisson, why truncated Neumann, what "approximate projection" looks like across AMR levels).
type: project
originSessionId: 12fb2afb-57e7-4a3b-acaf-2f8c91188f9d
---
Decisions baked into the current solver, with rationale:

1. **MAC staggered grid**.  u, v, w on face-centred MultiFabs; p and φ on
   cell-centred MultiFabs.  Standard for incompressible NS — pressure
   decoupled on the discrete grid, exact discrete projection possible.

2. **Explicit second-order centred advection**.  Trades stability margin
   for simplicity and isotropy.  No upwinding / Godunov machinery; relies
   on the projection + viscous regularisation to control dispersion.

3. **Crank–Nicolson diffusion, inverted via truncated Neumann series**:
   `(I − εL)^{-1} ≈ Σ_{k=0}^{N} (εL)^k` with `ε = ν dt / 2` and `N = 2`
   by default.  Convergence requires `‖εL‖ < 1`, i.e. `ν dt / h² < 1/(2d)`.
   Cheap, no inner solve, but **NOT** unconditionally stable — `ComputeDt`
   enforces a diffusive cap so the Neumann series remains in its
   convergence regime.  Increase `ins.cn_order` if `ν dt / h²` is large
   but still inside that cap.

4. **Modified Poisson is matrix-free, not MLMG**.  The pressure step
   solves `(D B^N G) p = (1/dt) D u*` directly by composing
   `ComputePressureGradient`, `ApplyBNFace`, and cell divergence.  We
   still use AMReX `MultiFab`/FillPatch/average-down machinery, but
   there is no current AMReX `MLMG` pressure solve and no `m_phi`
   incremental pressure update.  This keeps the pure-NS substrate aligned
   with the eventual Taira-Colonius IB saddle-point operator.

5. **Modified-Poisson sign and Krylov choice**.  `D B^N G` is negative
   semidefinite on full periodic/Neumann levels, so
   `ApplyModifiedPoissonOp` returns `-D B^N G` and `ProjectPerot` builds
   `rhs = -(1/dt) D u*`.  Full-domain levels then have the positive sign,
   but partial AMR levels with C/F Dirichlet ghosts are nonsymmetric; the
   current solver therefore uses matrix-free **BiCGStab**, not CG.  The
   old AMReX `MLPoisson` sign note remains useful historical context in
   `reference_mlpoisson_sign.md`, but MLMG is not in the active pressure
   path.

6. **Non-subcycled time stepping**.  Same `dt` on every AMR level, chosen
   from an unsplit advective CFL across all levels and the finest-level
   Neumann-series diffusive cap.  Less efficient than subcycling but
   dramatically simpler — no per-level temporal
   interpolation of ghost data, no nested time loops.  IAMR-style
   subcycling can be retrofitted later if needed.

7. **Approximate composite projection**.  Local `∇φ` is computed on each
   level and applied as `u^{n+1} = u* − dt ∇φ`.  After projection,
   `average_down_faces(m_vel)` and `average_down(m_pressure)` sync the
   coarse-level overlap with the fine averages, which absorbs the small
   divergence residual the local projection leaves on C/F-adjacent coarse
   cells.  Verified: `|div u|_∞ < 10⁻¹⁴` on the Taylor–Green AMR test.
   Full refluxing of `∇φ` (a "sync solve") is **not** implemented and is
   not needed for the current test suite — but would be required if the
   composite divergence had to be exactly zero (e.g. for very long-time
   stability or for the IBPM constraint).

8. **Vorticity-based refinement tagging**.  `ErrorEst` computes
   `|ω|` at cell centres from averaged face velocities and tags cells
   above `ins.refine_vort`.  Tunable; for 3D Taylor–Green peak `|ω|≈2`,
   so a threshold of 1.0 tags the upper half of the range.

9. **Physical boundary conditions** (added 2026-05-15, third pass —
   `src/INSSolver_BC.cpp`).  Per-domain-face, read from input:
   `periodic` / `noslip` / `inflow` / `slip` / `outflow`.
   - `noslip` and `inflow` are the SAME numerically: Dirichlet
     velocity = `ins.vel_<face>` (a non-zero value on `noslip` = a
     moving wall, e.g. the lid).  Pressure: homogeneous Neumann.
   - `slip`: `u·n = 0`, tangential zero-gradient.  Pressure Neumann.
   - `outflow`: velocity zero-gradient.  Pressure Dirichlet p = 0.
   This Dirichlet-vel⇒Neumann-p / outflow-vel⇒Dirichlet-p pairing is
   the standard projection-method choice the user specified.
   - **Singular-system gate**: `m_pressure_singular` is true unless
     some face is `outflow`.  `SolveModifiedPoisson` only subtracts
     the mean of RHS/solution when singular.  Subtracting it with an
     outflow present would remove the pressure level the outflow
     Dirichlet pins → wrong.
   - **Staggered handling**: normal component sits on the boundary
     face (set directly for Dirichlet, extrapolated for outflow);
     tangential components reflected about the wall value half a
     cell out.  AMReX FillPatch keeps `PhysBCFunctNoOp` (interior +
     C/F only); domain-boundary ghosts overwritten afterwards by
     `FillVelGhostPhys`/`FillPresGhostPhys`.
   - **Inhomogeneous data + Neumann series**: the iterated `(εL)^k`
     terms in `B^N` use HOMOGENEOUS wall data; the prescribed
     velocity is re-imposed on `u*` and `u^{n+1}` by
     `EnforceVelDirichlet` after the predictor and the projection.
     Standard low-truncation-order treatment; O(εL·boundary) error.
   - **Tested**: periodic Taylor–Green regression is stable; `inputs.lid`
     and `inputs.channel` have been run-verified.  The channel case is
     the outflow-Dirichlet pressure check.

10. **Per-level operator must zero its coarse–fine ghosts**
    (bug found + fixed 2026-05-18, `inputs.lid_amr`).  The per-level
    modified-Poisson Krylov path (`SolveModifiedPoisson` / `ApplyBNFace` /
    `ApplyModifiedPoissonOp`) only fills ghosts via `FillBoundary` +
    domain physical BC.  It does **not** interpolate C/F ghosts from
    coarse.  When a fine patch has an *interior* C/F interface (as in
    the lid cavity, where vorticity tagging hugs the walls) the
    working MultiFabs' C/F ghost cells were uninitialised → `G φ`
    read garbage → `dAd ~ −1e16` on step 1 → dt collapsed to ~1e-18
    → blow-up.  The periodic TG-AMR test never caught this because
    `refine_vort` tagged the whole domain, so the fine level was the
    full (periodic) domain with no interior C/F interface.
    The lid-cavity AMR debugging on 2026-05-18 turned out to be a
    **cascade of four distinct bugs**, all fixed; the final working
    scheme is recorded here so nobody re-derives it:

    1. **dt / Neumann-series stability** (`ComputeDt`).  `B^N` only
       approximates `(I−εL)^{-1}` when `ε‖L‖ = 2·dim·ν·dt/h² < 1`;
       beyond `ε‖L‖≥1` the operator `D B^N G` turns indefinite.  The
       old quiescent-IC fallback `dt = 0.25 h²/ν` gave `ε‖L‖ = 1.5`.
       Fix: cap dt every step at `dt_diff = α h²/(2·dim·ν)` with
       `α = 0.5` (so `ε‖L‖ ≤ 0.5`), using the *finest* level's h, and
       take `min(dt_adv, dt_diff)`.  `ins.fixed_dt` still must satisfy
       this cap; it only bypasses the advective CFL.

    2. **C/F pressure must be Dirichlet-from-coarse, not 0**.  The
       per-level "Dirichlet-0 at C/F" idea (zeroing the fine pressure
       C/F ghost) injects an O(p_true/dx) artificial gradient — the
       lid-cavity fine patches sit on the high-pressure corners, so it
       blew up.  Fix: solve coarse→fine and fill the fine C/F pressure
       ghost by interpolating the *already-solved coarser* pressure
       (`FillCellPatch` = FillPatchTwoLevels valid-from-self +
       C/F-from-coarse, then domain physical BC).  The search-direction
       *correction* stays homogeneous at C/F (its ghost = 0).  Use a
       SEPARATE ghost buffer `p_g` for the operator-on-p apply —
       `FillPatchTwoLevels` must not alias dst with the fine source
       (single-level lev 0 tolerates the alias; two-level NaNs).

    3. **Mean-pin only truly-singular levels**.  `m_pressure_singular`
       is global (no outflow).  A partial fine patch with
       Dirichlet-from-coarse C/F is NOT singular — pinning its mean
       corrupts it.  Gate: `level_singular = m_pressure_singular &&
       grids[lev].numPts()==Domain(lev).numPts()` (full-domain level
       only).  Preserves periodic-TG-AMR (fine = full periodic domain
       → still pinned).

    4. **C/F pressure interp must be piecewise-constant**, not
       conservative-linear.  `FillPatchTwoLevels` builds an internal
       coarse scratch, fills only its valid from the coarse source,
       then runs the coarse BC functor — which is `PhysBCFunctNoOp`,
       so the scratch's domain ghosts stay uninitialised.
       `lincc_interp` computes slopes that read them → NaN for fine
       patches hugging a wall.  `&pc_interp` uses no slopes → immune.
       (Filling the *real* coarse MF's ghosts does NOT help — FillPatch
       uses its own scratch.)  **This applies to THREE pressure
       interpolation sites**, all must use `pc_interp`:
       `FillCellPatch` (every step), AND the regrid hooks
       `MakeNewLevelFromCoarse` (InterpFromCoarseLevel) and
       `RemakeLevel` (FillPatchTwoLevels).  The regrid hooks were
       missed in the first pass → the lid cavity ran fine until the
       first regrid (step 10, `regrid_int=10`), then NaN'd in the
       pressure near the no-slip wall.  Velocity interp uses
       `face_linear_interp` which is slope-free → immune (leave it).
       If you ever add another coarse→fine pressure interpolation,
       use `pc_interp`.

    5. **Per-level operator is non-symmetric → use BiCGStab, not CG**.
       Once a fine patch has a C/F interface the staggered `B^N` plus
       ad-hoc C/F ghost breaks D/G adjointness, so `−D B^N G` is not
       SPD; CG diverges (`|Gp|2` 1e7→1e16, oscillating).  The coarse
       full-domain level has no C/F so CG worked there (masked it).
       `SolveModifiedPoisson` is now matrix-free **BiCGStab** with
       breakdown guards.  Converges ~80–190 iters/level to 1e-11 on
       all three lid-cavity AMR levels.

    **Verified 2026-05-18**: single-level lid `|div u|~3e-11`;
    1-level AMR lid stable, `|u|` tracks single-level, `|div u|~1e-2`
    at C/F; full 2-level `inputs.lid_amr` stable, `|u|` 0.07→0.23
    smooth, `|div u|~2e-2` (bounded, steady).  The residual
    `|div u|~1e-2` at C/F is the *expected* per-level-approximation
    error (Dirichlet-from-coarse, no reflux) — interior is
    divergence-free.  A proper composite `D B^N G` with reflux remains
    the deferred refactor for when C/F divergence must vanish.  Do NOT
    "simplify" away: the four `setVal(0)`/buffer choices, the
    `pc_interp`, the `level_singular` gate, and BiCGStab are all
    load-bearing.

11. **AB2 advection + cn_order=2 default → 2nd-order in time**
    (added 2026-05-19).  The advection term in the predictor is now
    2nd-order Adams–Bashforth: `dt·(3/2 A^n − 1/2 A^{n-1})`
    (`m_advect_old` stores A^{n-1}; `m_ab2_valid` falls back to Euler
    on step 1 and the first step after a regrid — one O(dt²) step,
    harmless at fixed grid).  Replaced explicit Euler `dt·A^n`.  The
    diffusion time order equals the `B^N` truncation order N, so
    `m_cn_order` default is now **2** (was 1) and every shipped
    `inputs.*` sets `cn_order=2`; the scheme is globally 2nd-order in
    time only with AB2 AND N≥2.  Verified by self-convergence on the
    stationary TG2D (diffusion order, =N) and the convecting TG2D
    (advection order, AB2) — both clean 2.00.  See
    `project_verification.md` for the crucial subtlety that
    stationary TG2D's advection is projected out (irrotational) and
    cannot probe AB2 — use the convecting vortex (`tg_uc`,`tg_vc`).
    Deferred: regridding `m_advect_old` through the regrid hooks (so
    AB2 survives regrids without the Euler fallback) — only matters
    for AMR runs that regrid very frequently.

12. **Pressure-gradient fields must retain outflow Dirichlet faces**
    (bug found + fixed 2026-05-19).  `FillPresGhostPhys` correctly
    applies outflow `p=0` by odd reflection, so `ComputePressureGradient`
    produces a nonzero normal boundary gradient for a constant pressure
    mode.  The bug was then calling `FillVelGhostPhys(...,
    homogeneous=true)` on that freshly computed `G p` field before
    `ApplyBNFace`; on a normal outflow component, the velocity outflow
    rule overwrote the valid boundary face by zero-gradient extrapolation,
    erasing the pressure Dirichlet pin and leaving constants in the
    operator nullspace.  Fix: do **not** pre-fill pressure-gradient
    fields with velocity BCs before `ApplyBNFace`.  `ApplyBNFace` copies
    the raw valid `G p` into the `k=0` term first, then fills the
    homogeneous velocity ghosts only for the iterated `k>=1`
    Neumann-series work term.  This preserves the outflow pressure pin
    while keeping the mobility operator's homogeneous velocity treatment.
    Regression: run an outflow input with `ins.check_pressure_pin=1`.
    It applies the real matrix-free operator to constant `p=1` on level
    0 and aborts unless `|-D B^N G 1|_inf` is nonzero.

13. **Advective CFL is unsplit and includes prescribed Dirichlet speeds**
    (fixed 2026-05-19).  The old `ComputeDt` used `max_d |u_d|/dx_d`,
    which underestimates the multidimensional CFL for diagonal flow, and
    it missed moving-lid tangential speeds on the first step because those
    live in ghost faces rather than valid face data.  The current bound is
    `max(Σ_d max|u_d|/dx_d)` per level, additionally taking the maximum
    over prescribed Dirichlet face velocity vectors from `ins.vel_*`.
    This makes quiescent moving-wall and inflow starts choose a sensible
    advective dt immediately; the diffusive `B^N` cap may still be the
    active limit.
