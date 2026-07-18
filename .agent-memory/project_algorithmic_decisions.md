---
name: Project — algorithmic design decisions
description: Records the numerical/algorithmic choices that aren't obvious from reading the code (why matrix-free modified Poisson, why truncated Neumann, and how the composite AMR projection is coupled).
type: project
originSessionId: 12fb2afb-57e7-4a3b-acaf-2f8c91188f9d
---
Decisions baked into the current solver, with rationale:

1. **MAC staggered grid**.  u, v, w on face-centred MultiFabs; p on
   cell-centred MultiFabs.  Standard for incompressible NS — pressure
   decoupled on the discrete grid, exact discrete projection possible.  The
   full stored viscous Cauchy stress uses
   `tau_ij = nu * (d(u_i)/d(x_j) + d(u_j)/d(x_i))`: `tau_ii` is
   cell-centred and each `tau_ij`, `i != j`, is nodal in directions i and j
   (the corresponding edge).  `tau_ij` and `tau_ji` are separate fields but
   are assigned from the same computed value so storage is exactly symmetric.
   Pressure remains separate; total stress is `sigma = -p I + tau`.

2. **Explicit second-order centred advection**.  Trades stability margin
   for simplicity and isotropy.  No upwinding / Godunov machinery; relies
   on the projection + viscous regularisation to control dispersion.

3. **Crank–Nicolson diffusion, inverted via truncated Neumann series**:
   `(I − εL)^{-1} ≈ Σ_{k=0}^{N} (εL)^k` with `ε = ν dt / 2` and `N = 2`
   by default.  Convergence requires `‖εL‖ < 1`, i.e. `ν dt / h² < 1/(2d)`.
   Cheap, no inner solve, but **NOT** unconditionally stable — `ComputeDt`
   enforces a diffusive cap so the Neumann series remains in its
   convergence regime.  Increase `ins.cn_order` if `ν dt / h²` is large
   but still inside that cap.  The explicit CN half-step now consumes the
   saved stress as `u^n + (dt/2) div(tau^n)` instead of applying `nu L`
   directly.  The compatible MAC composition gives
   `div(tau) = nu (L u + G D u)`, which reduces to the old face Laplacian for
   the discretely divergence-free saved velocity.  Stress is recomputed from
   the synchronized final velocity after initialization, every regrid, and
   every projection.  Fine velocity ghosts from `face_linear_interp` are not
   exactly divergence-free at partial C/F interfaces, so the symmetric form
   retains a small local `nu G D u` contribution there.

4. **The modified-Poisson operator and convergence test are matrix-free**.
   The pressure step
   solves `(D B^N G) p = (1/dt) D u*` directly by composing
   `ComputePressureGradient`, `ApplyCompositeBNFaces`, and cell divergence.
   AMReX `MLPoisson` is used only to propose a checked initial guess for
   non-singular pressure blocks; it is neither the modified operator nor an
   in-iteration preconditioner, and acceptance uses the true matrix-free
   residual.  There is no `m_phi` incremental pressure update.  This keeps
   the pure-NS substrate aligned with the eventual Taira-Colonius IB
   saddle-point operator.

5. **Modified-Poisson sign and Krylov choice**.  `D B^N G` is negative
   semidefinite on full periodic/Neumann levels, so the composite pressure
   operator applies `-D B^N G` and `ProjectPerot` builds
   `rhs = -(1/dt) D u*`.  Full-domain levels then have the positive sign,
   but composite C/F interpolation and active-cell masking are not assumed
   symmetric; the current solver therefore uses matrix-free **BiCGStab**,
   not CG.  The
   AMReX `MLPoisson` uses the opposite Laplacian sign, so the checked warm
   start solves against a negated standard-Poisson RHS; see
   `reference_mlpoisson_sign.md`.  It does not replace the active
   matrix-free operator or its residual test.

6. **Non-subcycled time stepping**.  Same `dt` on every AMR level, chosen
   from an unsplit advective CFL across all levels and the finest-level
   Neumann-series diffusive cap.  Less efficient than subcycling but
   dramatically simpler — no per-level temporal
   interpolation of ghost data, no nested time loops.  IAMR-style
   subcycling can be retrofitted later if needed.

7. **Composite AMR projection**.  Projection solves use a hierarchy-wide
   Krylov operator over the AMR pressure fields.  Fine pressure ghosts are
   filled from the current coarse iterate inside each operator apply;
   covered coarse cells are masked out of the active equations.  With IB
   enabled, the Lagrangian force vector is part of the same Krylov vector,
   but the IB rows/force columns are applied only on the finest level to
   avoid the over-dense coarse marker system.  Every intermediate term in
   `B^N(Gp [+ Hf])` is averaged fine-to-coarse and FillPatched at C/F before
   the next face Laplacian; the final face correction is synchronized before
   divergence.

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
     some face is `outflow`.  The composite pressure and coupled IB solves
     only subtract the mean of RHS/solution when singular.  Subtracting it
     with an outflow present would remove the pressure level the outflow
     Dirichlet pins → wrong.
   - **Staggered handling**: normal component sits on the boundary
     face (set directly for Dirichlet, extrapolated for outflow);
     tangential components reflected about the wall value half a
     cell out.  Velocity FillPatch keeps `PhysBCFunctNoOp` and explicitly
     applies `FillVelGhostPhys` afterwards.  Pressure FillPatch uses
     `FillPresGhostPhys` callbacks so AMReX's coarse interpolation scratch
     also receives valid physical ghosts.
   - **Inhomogeneous data + Neumann series**: the iterated `(εL)^k`
     terms in `B^N` use HOMOGENEOUS wall data; the prescribed
     velocity is re-imposed on `u*` and `u^{n+1}` by
     `EnforceVelDirichlet` after the predictor and the projection.
     Standard low-truncation-order treatment; O(εL·boundary) error.
   - **Tested**: periodic Taylor–Green regression is stable; `tests/3d/lid/inputs.lid`
     and `tests/3d/channel/inputs.channel` have been run-verified.  The channel case is
     the outflow-Dirichlet pressure check.

10. **Retired per-level modified-Poisson needed valid C/F ghosts**
    (bug found + fixed 2026-05-18, `tests/3d/lid_amr/inputs.lid_amr`;
    retained as historical context after the composite solve became the
    sole pressure path).  The retired per-level pressure Krylov path only
    filled ghosts via `FillBoundary` + domain physical BC.  It did **not**
    interpolate C/F ghosts from
    coarse.  When a fine patch has an *interior* C/F interface (as in
    the lid cavity, where vorticity tagging hugs the walls) the
    working MultiFabs' C/F ghost cells were uninitialised → `G φ`
    read garbage → `dAd ~ −1e16` on step 1 → dt collapsed to ~1e-18
    → blow-up.  The periodic TG-AMR test never caught this because
    `refine_vort` tagged the whole domain, so the fine level was the
    full (periodic) domain with no interior C/F interface.
    The lid-cavity AMR debugging on 2026-05-18 turned out to be a
    **cascade of five distinct bugs**, all fixed; the final working
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

    4. **C/F pressure interpolation is conservative-linear with an
       explicit physical-BC functor.** `FillPatchTwoLevels` builds an
       internal coarse scratch.  `PhysBCFunctNoOp` leaves the scratch's
       outside-domain cells as NaNs, so linear slopes fail when a fine
       patch touches a wall; filling only the real coarse MultiFab does
       not fix that scratch.  The correct repair is to call
       `FillPresGhostPhys` from the FillPatch coarse/fine BC functors and
       use `cell_cons_interp`.  This applies to `FillCellPatch`,
       `MakeNewLevelFromCoarse`, and `RemakeLevel`.  The former
       `pc_interp` workaround avoided NaNs but imposed an O(dx) pressure
       jump at every C/F interface.

    5. **The retired per-level operator was non-symmetric**.
       Once a fine patch has a C/F interface the staggered `B^N` plus
       ad-hoc C/F ghost breaks D/G adjointness, so `−D B^N G` is not
       SPD; CG diverges (`|Gp|2` 1e7→1e16, oscillating).  The coarse
       full-domain level had no C/F so CG worked there (masked it).  The
       current hierarchy operator keeps all active pressure equations in
       one composite BiCGStab solve.

    **Updated 2026-07-09**: the predictor, modified-Poisson/IB operator,
    and projection now apply every `B^N` term across the hierarchy.
    Before each face Laplacian, covered coarse faces are averaged down
    and fine C/F ghosts are FillPatched from the current coarse term.
    A partial-grid convecting Taylor–Green test converges to a true
    residual near 1e-12 with `|div u|` near machine precision; the old
    level-local zero C/F terms produced O(1e-2) velocity jumps despite
    the same reported Krylov tolerance.  BiCGStab periodically replaces
    its recursive residual with the true composite residual.  Keep the
    hierarchy `B^N`, explicit pressure BC functors, `level_singular`
    gate, and nonsymmetric Krylov treatment together.

11. **AB2 advection + cn_order=2 default → 2nd-order in time**
    (added 2026-05-19).  The advection term in the predictor is now
    2nd-order Adams–Bashforth: `dt·(3/2 A^n − 1/2 A^{n-1})`
    (`m_advect_old` stores A^{n-1}; `m_ab2_valid` falls back to Euler
    on step 1 and the first step after a regrid — one O(dt²) step,
    harmless at fixed grid).  Replaced explicit Euler `dt·A^n`.  The
    diffusion time order equals the `B^N` truncation order N, so
    `m_cn_order` default is now **2** (was 1) and every shipped
    `tests/*/inputs.*` sets `cn_order=2`; the scheme is globally 2nd-order in
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

14. **The IB projection uses a matrix-free force Schur solve**
    (initial path added 2026-05-20; force Schur path added 2026-07-11).
    `IBGeometry` loads 2D ASCII curves or 3D
    ASCII/binary STL surfaces, builds one marker per element centroid,
    stores host/device points/elements/markers as AMReX `GpuArray`
    records, and uploads device copies.  `INSSolver_IB.cpp` implements
    Peskin 4-point `H/E` as marker-centred finite-support kernels,
    owner-mask interpolation plus atomics to avoid double-counted shared
    faces, and GPU-ready IB refinement tagging.  The coupled AMR IB solve
    eliminates pressure and applies `M - B A^-1 C` matrix-free in marker
    space.  The in-repo outer BiCGStab uses fixed MLMG preconditioner cycles
    plus fixed modified-Poisson defect corrections for its approximate
    pressure inverse.  The scaling represents each physical marker force as
    `f_k = h_f^(d-1)/w_k * fhat_k`.  Because `H` already contains marker
    quadrature `w_k`, this makes all four uniform-grid blocks `O(1/h)` and
    restores their expected cross-block adjoint scaling.  Stored forces and
    block diagnostics remain physical.  Pressure is recovered with the exact
    modified operator, and the original scaled coupled residual controls
    acceptance.  IB
    coupling is applied only on the finest AMR level; coarser active
    pressure equations remain part of the same Krylov solve.  The supplied
    IB smoke cases are:
    `tests/3d/ib_plane`, `tests/3d/ib_plane_amr`, and
    `tests/3d/ib_cylinder_channel`.  The cylinder case intentionally uses an
    STL panel size near `1.5 * dx` because the solver still lacks a marker
    preconditioner and is sensitive to over-refined marker meshes.  The
    former alternating force/pressure subiterations and pressure-only cleanup
    remain removed.  Representative one-step Schur counts are 1 for the 2D
    AMR square and 7 for the four-level 2D cylinder.  The non-singular 3D
    cylinder still fails the recovered true residual and invokes the exact
    coupled fallback, which takes 2657 iterations at `1e-4` when its cap is
    raised to 4000.  The settled next step is Belos FGMRES with a local marker
    preconditioner.  See `project_ib_matrixfree_plan.md`.
