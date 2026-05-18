---
name: Project — algorithmic design decisions
description: Records the numerical/algorithmic choices that aren't obvious from reading the code (why MLMG now, why truncated Neumann, what "approximate projection" looks like across AMR levels).
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
   Cheap, no inner solve, but **NOT** unconditionally stable — the
   advective CFL is the constraint, and viscous stability is monitored
   implicitly by the series truncation tail.  Increase `ins.cn_order` if
   `ν dt / h²` is large.

4. **Pressure Poisson via AMReX MLMG composite solve**.  MLPoisson handles
   C/F flux matching internally; we pass per-level RHS = `(1/dt) ∇·u*`
   (after `average_down_faces` of `u*` so coarse face values at the C/F
   interface equal the averaged fine flux) and per-level solution into
   `m_phi[lev]`.  Periodic BCs are set via `setDomainBC(Periodic, …)`.
   This will be replaced by a matrix-free Trilinos `Tpetra::Operator`
   when the IB step lands (see `project_ib_matrixfree_plan.md`).

5. **AMReX MLPoisson sign**: `Lφ = +∇²φ`.  So `solve(phi, rhs)` solves
   `∇²φ = rhs`.  The RHS here is `+(1/dt) ∇·u*`, **not** negative.
   See `reference_mlpoisson_sign.md` — this is verified against the
   `mlpoisson_adotx` / `mlpoisson_gsrb` kernels in the installed
   AMReX headers, contradicting some MLLinOp doc wording about a
   `(αI − βL)` operator.

6. **Non-subcycled time stepping**.  Same `dt` on every AMR level, chosen
   from the advective CFL on the finest level.  Less efficient than
   subcycling but dramatically simpler — no per-level temporal
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
   - **Tested**: periodic Taylor–Green regression is bit-identical
     after the BC layer (periodic short-circuits the boundary loop).
     `inputs.lid` (cavity) and `inputs.channel` (in/outflow) were
     written and code-reviewed but NOT run-verified in the
     2026-05-15 session (the bash tool intermittently refused to
     launch the binary).  First action next session: run both and
     confirm stability + sane flow.
