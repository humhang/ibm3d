# AGENTS.md

Guidelines for any agent (human or AI) contributing to **ibm3d**.

## Project status (read this first)

`ibm3d` is in the first **immersed-boundary projection** phase.  The
pure incompressible Navier–Stokes substrate is working, and the initial
Taira–Colonius IB path now loads geometry, builds element-centroid
markers, and solves the coupled projection for prescribed IB velocity.
The executable IB path now uses a matrix-free marker-space force Schur solve
with in-repo BiCGStab and a fixed AMReX `MLMG`/modified-Poisson approximate
pressure inverse.  The longer-term linear-algebra direction remains Belos
FGMRES with a local/overlapping marker-block preconditioner.

Key decisions already settled, do not re-litigate:

- **MAC staggered grid**, second-order centred differencing.  The solver stores
  every component of the symmetric viscous Cauchy stress
  `tau_ij = nu * (d(u_i)/d(x_j) + d(u_j)/d(x_i))`: diagonal components are
  cell-centred and off-diagonal components are i/j-edge-centred.  Each
  off-diagonal pair is stored identically.  Pressure remains separate, so the
  total stress is `sigma = -p I + tau`.
- **Explicit advection + Crank–Nicolson diffusion**, with `(I − εL)^{-1}`
  approximated by the truncated Neumann series `B^N = Σ_{k=0}^{N} (εL)^k`.
- **Perot 1997 fractional step**: predictor
  `r1 = u^n + (dt/2) div(tau^n) + dt A^n = (I + εL) u^n + dt A^n`
  (NO pressure gradient), `u* = B^N r1`, modified-Poisson
  `(D B^N G) p^{n+1} = (1/dt) D u*`, projection
  `u^{n+1} = u* − dt B^N G p^{n+1}`.  `m_pressure` is solved for
  directly each step — there is no incremental form.
  The equality uses the compatible MAC identity
  `div(tau) = nu (L u + G D u)` and the saved velocity's discrete
  incompressibility.  At partial C/F interfaces, `face_linear_interp` ghost
  values are not exactly divergence-free, so the symmetric-stress predictor
  legitimately retains a small local `nu G D u` term.
- **Modified-Poisson and IB solves are matrix-free**.  Pressure-only solves
  use hand-rolled BiCGStab on the composite hierarchy operator.  IB solves
  eliminate pressure and apply `S_f = M - B A^-1 C` in marker space, with
  `A^-1` approximated by fixed MLMG V-cycles plus fixed modified-operator
  defect corrections.  The original `[-D; E] B^N [G H]` residual always
  controls acceptance.  A Tpetra/Belos wrapper remains the intended scalable
  interface.
- **Coupled scaling and Schur caveat**: the IB solve scales
  pressure diagnostics by each level's geometric-mean spacing `h_l` and uses
  `f_k = h_f^(d-1)/w_k * fhat_k` for a marker with quadrature weight `w_k`.
  Stored forces and block diagnostics remain physical.  Do not restore the
  removed alternating force/pressure subiterations or pressure-only cleanup.
  The one-time exact pressure recovery and conditional full coupled Krylov
  refinement are part of residual verification, not alternating cleanup.
  The approximate Schur path accelerates the 2D cylinders and small plane/
  square cases, but the dense 3D cylinder channel still fails the recovered
  true residual and falls back to the old 2657-iteration coupled behavior.
  Do not hide this by loosening acceptance or reporting only the Schur
  residual.  See `.agent-memory/project_ib_matrixfree_plan.md`.
- **IB discretisation (current first pass)**: one Lagrangian marker at
  each line-segment/triangle centroid, element length/area as the
  quadrature weight, Peskin 4-point delta, component-wise MAC spreading
  (`f_x` to x-faces, etc.), and marker-centred finite-support
  spread/interpolate kernels.  Interpolation uses owner masks so shared
  patch faces are not double-counted.  Geometry and marker-space vectors are
  replicated on every MPI rank; force and Krylov vectors use
  `amrex::Gpu::DeviceVector`.  Interpolation reductions pass device pointers
  directly when `ParallelDescriptor::UseGpuAwareMpi()` is true and otherwise
  use a reusable pinned-host staging buffer.  IB coupling is applied only on
  the finest level; `ErrorEst` tags a configurable neighborhood of the markers
  so AMR keeps the body on the finest mesh.
- **`D B^N G` is negative semidefinite** (eigenvalues `−k² · …`).
  Both the composite pressure operator and the RHS in `ProjectPerot` are
  negated so full-domain levels have the positive sign.  Composite C/F
  interpolation and active-cell masking are not assumed symmetric, so the
  solver uses BiCGStab.  The projection
  `u^{n+1} = u* − dt B^N G p` still uses the natural `+B^N G p`.
- **Physical BCs** (`src/INSSolver_BC.cpp`): per-face `periodic` /
  `noslip` / `inflow` / `slip` / `outflow` from `ins.bc_<face>` +
  `ins.vel_<face>`.  Dirichlet velocity ⇒ Neumann pressure; outflow
  velocity ⇒ Dirichlet p=0.  `m_pressure_singular` (no outflow ⇒
  pure Neumann/periodic) gates composite mean-pinning — do NOT subtract
  the mean when an outflow is present.  Velocity FillPatch uses
  `PhysBCFunctNoOp`, then explicitly
  overwrites staggered domain-boundary ghosts with `FillVelGhostPhys`.
  Pressure FillPatch supplies callbacks that call `FillPresGhostPhys` on
  AMReX's interpolation scratch as well as the destination; this is required
  by `cell_cons_interp` near physical boundaries.
  Inhomogeneous wall data is kept out of the Neumann series
  (homogeneous there) and re-imposed by `EnforceVelDirichlet` on
  `u*` and `u^{n+1}`.  Do not apply `FillVelGhostPhys` directly to a
  freshly computed pressure-gradient field before `ApplyBNFace`: the
  raw `G p` term must retain outflow `p=0` Dirichlet boundary faces.
  Don't move staggered velocity BC handling into the generic AMReX BC
  functor — the normal/tangential split is the reason it stays explicit.
  Outflow velocity ghost fill is **linear extrapolation**
  (`ghost = 2u(N) − u(N−1)`) so that the face Laplacian at the
  outflow boundary face is zero in the normal direction (`∂²u/∂n² = 0`
  = biased diffusion stencil).  The normal boundary face itself is
  left untouched by `FillVelGhostPhys`; the projection sets it.

## Coding style

- **Format**: LLVM (`.clang-format` is checked in with `BasedOnStyle:
  LLVM`, `Standard: c++20`).  Run `clang-format -style=file` on every
  edited C/C++ source.  Zed's format-on-save with `clangd` reads the
  config.
- **Language**: C++20.  Use modern features where they improve clarity
  (concepts, ranges, `std::span`, designated initialisers,
  `[[likely]]` / `[[unlikely]]`).  Do **not** reach for them for
  novelty.
- **Headers**: `.H`, sources: `.cpp` — match AMReX conventions.

## Algorithm anchors (don't refactor away)

- Per-level FillPatch of the inputs to every operator apply
  (`FillFacePatch` / `FillCellPatch` helpers).  Intra-level halo +
  C/F interpolation are tied together in those helpers; downstream
  kernels assume their inputs have valid ghosts.
- `average_down_faces(m_vstar)` *before* the projection solve so coarse
  face values at the C/F interface equal the averaged fine values.
- `average_down_faces(m_vel)` + `average_down(m_pressure)` at the end of
  `Advance` to keep coarse representation consistent with averaged
  fine values after the projection.
- Recompute `m_stress` from FillPatched `m_vel` after initial hierarchy
  creation, after every regrid, and after the final velocity average-down in
  `Advance`.  The next predictor must use stress matching its saved velocity.
- The `B^N` factor in the projection step is the **same** polynomial
  used in the predictor and in the modified-Poisson operator — this is
  Perot's exact-factorisation consistency.  Don't replace any of the
  three `B^N` applications with the identity "for speed."
- `ComputeDt` uses the unsplit advective bound
  `dt <= cfl / max(Σ_d |u_d|/dx_d)`, includes prescribed Dirichlet wall
  and inflow speeds, and always enforces the `B^N` Neumann-series
  stability cap.  `ins.fixed_dt` overrides the advective CFL only; it
  must not bypass the `B^N` cap.

## Agent memory

In-repo at `.agent-memory/` (committed; see `MEMORY.md` there for the
one-line index).  

## Build / run

Standard operations are wired up as Zed tasks in `.zed/tasks.json`.
From the command palette: **`task: spawn`** → pick:

| Task                                     | What it does                                       |
| ---------------------------------------- | -------------------------------------------------- |
| `CMake: Configure Debug` / Release       | Configure `build-(debug\|release)/`, refresh symlink |
| `Build: ins_solver (Debug)` / Release    | Build the executable                                |
| `Build: all (Debug)` / Release           | Build everything in the build tree                  |
| `Clean: Debug build` / Release           | `cmake --build … --target clean`                    |
| `Clean: Rebuild Debug` / Release         | Wipe build tree, reconfigure, rebuild               |
| `Clean: Wipe everything`                 | Remove both build trees and `compile_commands.json` |
| `Run: ins_solver …`                      | Serial run with a selected `tests/{2d,3d}/<case>/inputs.*` |
| `Run: ins_solver MPI (…, 4 ranks)`       | `mpirun -np 4` run with a selected test input       |

The configure tasks pass `-DAMReX_DIR=/Users/hang/opt/amrex-26.01/install/lib/cmake/AMReX`.
Trilinos is **not** currently a dependency; when implementing the planned
Tpetra/Belos force-Schur wrapper, restore
`find_package(Trilinos REQUIRED COMPONENTS Tpetra Belos Ifpack2
Teuchos)` in the top-level `CMakeLists.txt` and re-add
`-DTrilinos_DIR=/Users/hang/opt/trilinos-17.0.0/install/lib/cmake/Trilinos`
to the Zed configure tasks.

After any configure task, `compile_commands.json` at the repo root is a
symlink to the most-recently-configured build's compile DB — that is
what `clangd` reads.

## Debugging

`.zed/debug.json` defines two CodeLLDB configurations: launch
`build-debug/src/ins_solver tests/3d/tg/inputs.tg` (with a pre-launch build), and
attach-by-PID (useful for attaching to one rank of an `mpirun`-launched
run).  CodeLLDB is auto-installed on first use.

## Test inputs

| File                                      | What it exercises                                          |
| ----------------------------------------- | ---------------------------------------------------------- |
| `tests/2d/tg2d/inputs.tg2d`               | Native 2D analytic Taylor–Green verification.              |
| `tests/2d/tg2d_amr/inputs.tg2d_amr`       | Native 2D AMR Taylor–Green verification.                   |
| `tests/2d/tg2d_cf/inputs.tg2d_cf`         | Convecting Taylor–Green across a partial C/F interface.    |
| `tests/2d/ib_square/inputs.ib_square`     | Native 2D coupled IB projection smoke test.                |
| `tests/2d/ib_square_amr/inputs.ib_square_amr` | Native 2D coupled IB projection smoke test with AMR.   |
| `tests/2d/ib_cylinder_re100/inputs.ib_cylinder_re100` | 2D AMR flow past a stationary cylinder, Re_D=100, finest D/dx=80. |
| `tests/2d/ib_cylinder_re100_coarse/inputs.ib_cylinder_re100_coarse` | Coarse local-debug cylinder, one refinement level, finest D/dx=20. |
| `tests/3d/tg/inputs.tg`                   | Single-level smoke test (32³, periodic, Taylor–Green).     |
| `tests/3d/tg_amr/inputs.tg_amr`           | 2-level AMR hierarchy pressure path + regrid + FillPatch.  |
| `tests/3d/tg2d/inputs.tg2d`               | Thin-periodic-z 2D Taylor–Green verification for 3D builds.|
| `tests/3d/lid/inputs.lid`                 | Lid-driven cavity — all-Dirichlet BCs, singular pressure.  |
| `tests/3d/lid_amr/inputs.lid_amr`         | AMR lid-driven cavity.                                     |
| `tests/3d/channel/inputs.channel`         | Inflow/outflow — non-singular pressure (outflow Dirichlet).|
| `tests/3d/ib_plane/inputs.ib_plane`       | Single-level coupled IB projection smoke test.             |
| `tests/3d/ib_plane_amr/inputs.ib_plane_amr` | Composite IB projection with finest-level marker coupling. |
| `tests/3d/ib_cylinder_channel/inputs.ib_cylinder_channel` | Single-level channel flow past a stationary cylindrical IB surface. |
| `tests/3d/ib_sphere_re100/inputs.ib_sphere_re100` | 3D AMR flow past a stationary sphere, Re_D=100, finest D/dx=80. |

Taylor–Green expectation: `|div u|_∞ ~ 10⁻¹¹` per step (Krylov-tolerance
dominated), monotonic energy decay.  Any drift is a regression.
`tests/3d/lid/inputs.lid` should develop a
single primary vortex and approach steady state; `tests/3d/channel/inputs.channel`
should report "non-singular (outflow Dirichlet)" and relax the inlet
toward Poiseuille.

## Conventions for changes

- Keep new code in the existing file structure unless adding a genuinely
  separate concern.  Extend `INSSolver_IB.cpp` for geometry, spread/interp,
  and single-level IB kernels; hierarchy projection and coupled Krylov work
  belongs in `INSSolver_Project.cpp`.  A wall BC extends the existing BC and
  diffusion/projection files.
- Do **not** comment what the code does — only why, and only when
  non-obvious.  AMReX domain knowledge is the bar: anything explainable
  by reading AMReX docs is too obvious to comment.
- Don't introduce new third-party deps without discussion.  AMReX + MPI
  is the current dependency surface; Trilinos returns when the
  Tpetra/Belos wrapper replaces the transitional in-repo Schur BiCGStab path.
- Match the AMReX style of the surrounding code (`amrex::Real`,
  `amrex::Box`, `MFIter`, `ParallelFor`, etc.) rather than mixing in
  raw STL/MPI primitives.
