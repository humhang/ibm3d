# ibm3d — incompressible Navier–Stokes solver on AMReX (AMR-capable)

Staggered-grid (MAC) projection-method solver for the incompressible
Navier–Stokes equations, built on AMReX with multi-level support
(`AmrCore`).  The intended end goal is the **Taira–Colonius
immersed-boundary projection method** ([JCP 2007](https://doi.org/10.1016/j.jcp.2007.03.005));
the current code implements the NS+AMR substrate on top of which the IB
machinery will be added.

## Equations

$$
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u}\cdot\nabla)\mathbf{u}
\;=\; -\nabla p + \nu\,\nabla^2\mathbf{u}, \qquad
\nabla \cdot \mathbf{u} \;=\; 0.
$$

## Grid

| Field | Location     | AMReX nodality |
| ----- | ------------ | -------------- |
| `u`   | x-faces      | `(1,0,0)`      |
| `v`   | y-faces      | `(0,1,0)`      |
| `w`   | z-faces      | `(0,0,1)`      |
| `p`   | cell centres | `(0,0,0)`      |

## Time stepping (per step, no temporal subcycling between levels)

```
A^n  = -(u^n · ∇) u^n                                            (explicit, centred)
u*   = (I − εL)^{-1} [ (I + εL) u^n + dt A^n − dt ∇p^n ]          (CN diffusion)
        with (I − εL)^{-1} ≈ Σ_{k=0}^{N} (εL)^k,  ε = ν dt / 2
∇²φ  = (1/dt) ∇·u*                                                (composite Poisson)
u^{n+1} = u* − dt ∇φ                                              (projection)
p^{n+1} = p^n + φ                                                 (incremental)
```

## AMR

- Composite cell-centred pressure Poisson via **AMReX `MLMG` + `MLPoisson`**.
- `average_down_faces` of `u*` before forming the RHS, so the coarse-side
  divergence at every coarse–fine interface uses the averaged fine flux —
  this makes the per-level RHS the *composite* divergence by construction.
- Ghost cells at intra-level patch boundaries: `FillBoundary` (via
  `FillPatchSingleLevel`).  Ghost cells at C/F boundaries: interpolation
  from coarse (`face_linear_interp` for velocity, `lincc_interp` for
  pressure / φ) via `FillPatchTwoLevels`.
- After projection, `average_down_faces(m_vel)` and
  `average_down(m_pressure)` sync the coarse representation with the fine
  averages, absorbing the small divergence residual the local projection
  leaves on C/F-adjacent coarse cells.
- Refinement is driven by `|ω|` (cell-centred vorticity magnitude) exceeding
  `ins.refine_vort`.

## Build / run

All standard operations are wired up as Zed tasks in `.zed/tasks.json`
(see `AGENTS.md`).  For a manual workflow:

```sh
# Debug
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DAMReX_DIR=/path/to/amrex/install/lib/cmake/AMReX
cmake --build build-debug --target ins_solver -j
ln -sf build-debug/compile_commands.json compile_commands.json

# Run single-level Taylor–Green vortex
./build-debug/src/ins_solver inputs

# Run 2-level AMR Taylor–Green vortex
./build-debug/src/ins_solver inputs.tg_amr
```

Plotfiles are written every `ins.plot_int` steps to `plt#####` (or
`plt_amr#####` for the AMR input) and can be viewed with VisIt or yt.

## Test inputs

| File             | Configuration                                                   |
| ---------------- | --------------------------------------------------------------- |
| `inputs`         | 32³ single-level Taylor–Green vortex, periodic, Re ≈ 100.       |
| `inputs.tg_amr`  | 32³ base + 1 refinement level (2× ratio), vorticity tagging.    |

Both decay monotonically from the analytical TG initial condition.
Expected behaviour at convergence:

- `|div u|_∞ ~ 10⁻¹⁴` per step (machine precision at MLMG default
  tolerance).
- MLMG converges in ~8 iterations per solve.
- `|u|_∞` decreases smoothly with the viscous dissipation rate.

## What's _not_ implemented yet

- **Immersed boundary**.  The matrix structure changes to a saddle-point
  system; see the `AGENTS.md` notes for the planned Trilinos /
  matrix-free path.  Trilinos has been temporarily removed from
  CMakeLists.txt and will return when this phase begins.
- Wall / inflow / outflow BCs (all-periodic only at this stage).
- Temporal subcycling between AMR levels.
- Sync-solve refluxing of `∇φ` (the projection is "approximate" — the
  per-level local gradient plus a post-projection `average_down`).
