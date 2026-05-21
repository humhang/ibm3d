# ibm3d — incompressible Navier–Stokes solver on AMReX (AMR-capable)

Staggered-grid (MAC) solver for the incompressible Navier–Stokes
equations using the **Perot 1997 exact-factorisation fractional step**
method ([J. Comput. Phys. 132(1)](https://doi.org/10.1006/jcph.1996.5587)),
built on AMReX with multi-level support (`AmrCore`).  The solver now has
an initial **Taira–Colonius immersed-boundary projection method**
([JCP 2007](https://doi.org/10.1016/j.jcp.2007.03.005)) — Perot 1997
is its underlying fractional-step framework.  The current IB path handles
prescribed IB velocity on loaded line/surface elements and applies the
coupled projection on the finest AMR level.

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

With `ib.geometry` set, the finest level instead solves the
Taira-Colonius Schur system

```
[-D] B^N [G H] [p] = [        -D u*/dt]
[ E]     [   ] [f]   [(E u* - U_B)/dt]
```

and projects with `u^{n+1} = u* − dt B^N (G p + H f)`.  `H` spreads the
IB force component-wise to the matching MAC face family, and `E`
interpolates face velocity back to the same Lagrangian markers with
owner masks so shared patch faces are not double-counted.  The current
implementation uses one marker per immersed element centroid, with
element length/area as its quadrature weight.

Why the modified Poisson `D B^N G` and not the standard `∇²` of a
Chorin projection?  Perot 1997 shows that the exact block-LU
factorisation of the discrete `[A, G; D, 0]` system yields the
Schur complement `−D A^{-1} G` on the pressure; approximating `A^{-1}`
by the same `B^N` you used in the predictor and the projection makes
the fractional step consistent to the truncation order.  At `N = 0`
this reduces to plain Chorin; at `N = 1` it matches CN's `O(dt²)`.

The pressure `p` is solved for directly each step.  No incremental
form — `m_pressure` is the current pressure, and the previous step's
value is used as the Krylov warm start.

The modified Poisson is solved matrix-free with BiCGStab.
`D B^N G` is composed from existing pieces (`ComputePressureGradient`,
the face Laplacian, the cell divergence).  Its spectrum is `−k² · …`
on a full periodic/Neumann level, so the code negates both the operator
and the RHS before the Krylov solve.  Full-domain cases are effectively
SPD after that sign flip; partial AMR levels with C/F Dirichlet ghosts
are nonsymmetric, which is why the implementation uses BiCGStab.  No
preconditioner currently.

`ComputeDt` uses an unsplit advective CFL bound,
`dt <= cfl / max(Σ_d |u_d|/dx_d)`, plus a diffusive cap required by the
truncated Neumann series for `(I − εL)^{-1}`.  Prescribed Dirichlet wall
and inflow velocities are included explicitly in the CFL estimate, so a
moving lid is visible even before tangential wall speeds enter valid
face data.  `ins.fixed_dt` bypasses the advective CFL but still aborts if
it violates the `B^N` stability cap.

## AMR

- Per-level Perot solves with fine→coarse sync.  Each level's
  modified Poisson is solved independently; `average_down_faces` of
  `u*` before the solves and of `u^{n+1}` after the projection
  keeps coarse face values consistent with averaged fine values at
  the C/F interface.
- Ghost cells at intra-level patch boundaries: `FillBoundary` (via
  `FillPatchSingleLevel`).  Ghost cells at C/F boundaries: interpolation
  from coarse (`face_linear_interp` for velocity, `pc_interp` for
  pressure) via `FillPatchTwoLevels`.
- Refinement is driven by `|ω|` (cell-centred vorticity magnitude)
  exceeding `ins.refine_vort`.
- A **proper composite** `D B^N G` — with FillPatch of every
  intermediate face term and average_down between L applications — is
  a natural future refactor when AMR divergence tolerance matters.
  Periodic 2-level Taylor–Green reaches the Krylov tolerance; nonperiodic
  AMR with physical-boundary refinement can retain a bounded C/F
  divergence defect until a composite projection is added.

## Boundary conditions

Per-domain-face, read from the input file:

| `ins.bc_<face>` | velocity                       | pressure        |
| --------------- | ------------------------------ | --------------- |
| `periodic`      | wrap (geometry periodicity)    | wrap            |
| `noslip`        | Dirichlet `u = ins.vel_<face>` | Neumann ∂p/∂n=0 |
| `inflow`        | Dirichlet `u = ins.vel_<face>` | Neumann ∂p/∂n=0 |
| `slip`          | `u·n = 0`, tangential ∂/∂n = 0 | Neumann ∂p/∂n=0 |
| `outflow`       | linear extrapolation ghost, ∂²u/∂n² = 0 | Dirichlet p = 0 |

`<face>` ∈ {`xlo`,`xhi`,`ylo`,`yhi`,`zlo`,`zhi`}.  `noslip` and
`inflow` are numerically identical (a prescribed velocity vector on
the boundary); the names just document intent.  A non-zero
`ins.vel_<face>` on a `noslip` face is a *moving wall* (e.g. the
lid-driven cavity lid).  A periodic direction must be periodic on
both faces and match `geometry.is_periodic` (validated at startup).

The Dirichlet-velocity ⇒ Neumann-pressure / outflow-velocity ⇒
Dirichlet-pressure pairing is the standard projection-method choice.
When no face is `outflow` the pressure system is singular (pure
Neumann/periodic) and its mean is pinned; an `outflow` face makes it
non-singular and the mean is left free.

Staggered-grid specifics live in `src/INSSolver_BC.cpp`: at
Dirichlet/slip walls, the normal component sits exactly on the boundary
face and tangential components are reflected about the wall value half a
cell away.  At outflow, velocity ghosts use linear extrapolation while
the normal boundary face is left for the projection.
The truncated-Neumann-series operator applications use *homogeneous*
wall data; the prescribed velocity is re-imposed on `u*` and
`u^{n+1}` by `EnforceVelDirichlet` after the predictor and the
projection (standard treatment for low truncation order).  Pressure
Dirichlet outflow data is retained in the raw `G p` term of `B^N G p`;
only the higher-order Neumann-series work terms use homogeneous
velocity ghosts.  Set `ins.check_pressure_pin = 1` on an outflow case
to run a startup regression check that verifies a constant pressure
field is not in the modified-Poisson nullspace.

## Immersed boundary

Enable the first IB projection path with:

```text
ib.geometry       = path/to/body.stl    # 3D build; 2D uses the ASCII curve format
ib.velocity       = 0.0 0.0 0.0         # prescribed marker velocity, default zero
ib.refine_radius  = 4.0                 # cell-width radius for AMR tagging
```

In 3D, STL may be ASCII or binary.  In 2D, the curve format is:

```text
n_points n_segments
x y
...
i j
...
```

Connectivity may be zero-based or one-based and is stored internally as
zero-based.  Every MPI rank holds the full Lagrangian geometry and force
vector.  The interpolation operator uses `ParallelFor` and AMReX owner
masks so faces shared by two face-centred boxes are not counted twice.

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
./build-debug/src/ins_solver tests/tg/inputs.tg

# Run 2-level AMR Taylor–Green vortex
./build-debug/src/ins_solver tests/tg_amr/inputs.tg_amr
```

Plotfiles are written every `ins.plot_int` steps to `plt#####` (or
`plt_amr#####` for the AMR input) and can be viewed with VisIt or yt.

## Test inputs

| Case | Configuration |
| ---- | ------------- |
| `tests/tg/inputs.tg` | 32³ single-level Taylor–Green vortex, periodic, Re ≈ 100. |
| `tests/tg_amr/inputs.tg_amr` | 32³ base + 1 refinement level (2× ratio), vorticity tagging. |
| `tests/tg2d/inputs.tg2d` | 2D Taylor–Green analytic/self-convergence case. |
| `tests/lid/inputs.lid` | Lid-driven cavity, Re = 100 (no-slip walls, moving lid, z-periodic). |
| `tests/lid_amr/inputs.lid_amr` | AMR lid-driven cavity. |
| `tests/channel/inputs.channel` | Plane channel, uniform inflow / outflow, Re_h = 50. |
| `tests/ib_plane/inputs.ib_plane` | Single-level coupled IB projection smoke test with `ib_plane.stl`. |
| `tests/ib_plane_amr/inputs.ib_plane_amr` | Finest-level coupled IB projection smoke test with AMR and `ib_plane.stl`. |
| `tests/ib_cylinder_channel/inputs.ib_cylinder_channel` | Channel past a stationary cylindrical IB surface; coarse `1.5dx` STL smoke case. |

The Taylor–Green cases decay monotonically from the analytical IC;
at convergence `|div u|_∞ ~ 10⁻¹¹` per step (Krylov tolerance
dominated), and `|u|_∞` decays with the viscous rate.

The lid-driven cavity should approach a steady recirculating vortex;
the channel relaxes the uniform inlet toward the parabolic Poiseuille
profile and reports a *non-singular* pressure system (outflow pins
the pressure, so the mean is not removed).

## What's _not_ implemented yet

- Moving/deforming-body kinematics beyond uniform prescribed
  `ib.velocity`.
- Tpetra/Belos wrapping and MLMG preconditioning for the IB Schur
  operator.  The current coupled solve is still the local matrix-free
  BiCGStab path.
- Composite AMR IB projection.  IB coupling is applied on the finest
  level only; coarser data is synchronized by average-down where covered.
- Temporal subcycling between AMR levels.
- Sync-solve refluxing of `∇φ` (the projection is "approximate" — the
  per-level local gradient plus a post-projection `average_down`).
- Inhomogeneous Dirichlet data inside the truncated Neumann series
  (handled approximately: homogeneous in the series, re-imposed on
  `u*`/`u^{n+1}` afterwards — fine for low truncation order N).
- A proper composite `D B^N G` for AMR + non-periodic BCs (the
  domain-boundary physical BC is applied per-level after FillPatch;
  refinement is kept interior in the supplied tests so C/F and
  domain boundaries don't coincide).
