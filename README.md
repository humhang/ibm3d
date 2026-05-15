# ibm3d — incompressible Navier–Stokes solver on AMReX (AMR-capable)

Staggered-grid (MAC) solver for the incompressible Navier–Stokes
equations using the **Perot 1997 exact-factorisation fractional step**
method ([J. Comput. Phys. 132(1)](https://doi.org/10.1006/jcph.1996.5587)),
built on AMReX with multi-level support (`AmrCore`).  The intended end
goal is the **Taira–Colonius immersed-boundary projection method**
([JCP 2007](https://doi.org/10.1016/j.jcp.2007.03.005)) — Perot 1997
is its underlying fractional-step framework.  The current code
implements the NS+AMR substrate; the IB block is not added yet.

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

Notation: `ε = ν dt / 2`, `B^N = Σ_{k=0}^{N} (εL)^k ≈ (I − εL)^{-1}`,
`G` cell→face gradient, `D` face→cell divergence, `L` face Laplacian.

```
A^n  = -(u^n · ∇) u^n                                            (explicit, centred)
r1   = (I + εL) u^n + dt A^n                                     (no pressure — Perot)
u*   = B^N r1                                                    (predictor)
solve (D B^N G) p = (1/dt) D u*                                  (modified Poisson)
u^{n+1} = u* − dt B^N G p                                        (projection)
```

Why the modified Poisson `D B^N G` and not the standard `∇²` of a
Chorin projection?  Perot 1997 shows that the exact block-LU
factorisation of the discrete `[A, G; D, 0]` system yields the
Schur complement `−D A^{-1} G` on the pressure; approximating `A^{-1}`
by the same `B^N` you used in the predictor and the projection makes
the fractional step consistent to the truncation order.  At `N = 0`
this reduces to plain Chorin; at `N = 1` it matches CN's `O(dt²)`.

The pressure `p` is solved for directly each step.  No incremental
form — `m_pressure` is the current pressure, and the previous step's
value is used as the CG warm start.

The modified Poisson is solved matrix-free with a hand-rolled CG.
`D B^N G` is composed from existing pieces (`ComputePressureGradient`,
the face Laplacian, the cell divergence).  Its spectrum is `−k² · …`
so the code negates both the operator and the RHS before handing to
CG — that's the same sign story you'd hit with any code that called
Laplacian-as-operator without an internal sign flip (e.g. AMReX
`MLPoisson`).  No preconditioner currently — for the test grids
(32³–64³) unpreconditioned CG converges in ~40–80 iters.

## AMR

- Per-level Perot solves with fine→coarse sync.  Each level's
  modified Poisson is solved independently; `average_down_faces` of
  `u*` before the solves and of `u^{n+1}` after the projection
  keeps coarse face values consistent with averaged fine values at
  the C/F interface.
- Ghost cells at intra-level patch boundaries: `FillBoundary` (via
  `FillPatchSingleLevel`).  Ghost cells at C/F boundaries: interpolation
  from coarse (`face_linear_interp` for velocity, `lincc_interp` for
  pressure) via `FillPatchTwoLevels`.
- Refinement is driven by `|ω|` (cell-centred vorticity magnitude)
  exceeding `ins.refine_vort`.
- A **proper composite** `D B^N G` — with FillPatch of every
  intermediate face term and average_down between L applications — is
  a natural future refactor when AMR divergence tolerance matters.
  Current 2-level test sees `|div u|_∞ ~ 10⁻¹¹`, at the CG tolerance.

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

- `|div u|_∞ ~ 10⁻¹¹` per step (CG tolerance dominated; tighten
  `ins.poisson_tol` for stricter divergence).
- Unpreconditioned CG converges in ~40–80 iters per solve at 32³.
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
