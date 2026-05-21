# AGENTS.md

Guidelines for any agent (human or AI) contributing to **ibm3d**.

## Project status (read this first)

`ibm3d` is in the first **immersed-boundary projection** phase.  The
pure incompressible Navier–Stokes substrate is working, and the initial
Taira–Colonius IB path now loads geometry, builds element-centroid
markers, and solves the coupled finest-level projection for prescribed
IB velocity.  The longer-term linear-algebra direction is still
matrix-free `Tpetra::Operator` with AMReX `MLMG` as the preconditioner;
the current executable coupled path uses the local BiCGStab machinery.

Key decisions already settled, do not re-litigate:

- **MAC staggered grid**, second-order centred differencing.
- **Explicit advection + Crank–Nicolson diffusion**, with `(I − εL)^{-1}`
  approximated by the truncated Neumann series `B^N = Σ_{k=0}^{N} (εL)^k`.
- **Perot 1997 fractional step**: predictor `r1 = (I + εL) u^n + dt A^n`
  (NO pressure gradient), `u* = B^N r1`, modified-Poisson
  `(D B^N G) p^{n+1} = (1/dt) D u*`, projection
  `u^{n+1} = u* − dt B^N G p^{n+1}`.  `m_pressure` is solved for
  directly each step — there is no incremental form.
- **Modified-Poisson solve is matrix-free**, hand-rolled BiCGStab.  Operator
  composed from existing `ComputePressureGradient`, face Laplacian,
  and cell divergence pieces.  The current IB rows attach to this same
  operator as `[-D; E] B^N [G H]`; a Tpetra::Operator wrapper remains
  the intended scalable linear-algebra interface.
- **IB discretisation (current first pass)**: one Lagrangian marker at
  each line-segment/triangle centroid, element length/area as the
  quadrature weight, Peskin 4-point delta, component-wise MAC spreading
  (`f_x` to x-faces, etc.), and `ParallelFor` owner-mask interpolation
  so shared patch faces are not double-counted.  Geometry and force vectors are
  replicated on every MPI rank.  IB coupling is applied only on the
  finest level; `ErrorEst` tags a configurable neighborhood of the
  markers so AMR keeps the body on the finest mesh.
- **`D B^N G` is negative semidefinite** (eigenvalues `−k² · …`).
  Both the operator (`ApplyModifiedPoissonOp`) and the RHS in
  `ProjectPerot` are negated so full-domain levels have the positive
  sign.  Partial AMR levels with C/F Dirichlet ghosts are nonsymmetric,
  so the solver uses BiCGStab.  The projection
  `u^{n+1} = u* − dt B^N G p` still uses the natural `+B^N G p`.
- **Physical BCs** (`src/INSSolver_BC.cpp`): per-face `periodic` /
  `noslip` / `inflow` / `slip` / `outflow` from `ins.bc_<face>` +
  `ins.vel_<face>`.  Dirichlet velocity ⇒ Neumann pressure; outflow
  velocity ⇒ Dirichlet p=0.  `m_pressure_singular` (no outflow ⇒
  pure Neumann/periodic) gates the mean-pinning in
  `SolveModifiedPoisson` — do NOT subtract the mean when an outflow
  is present.  AMReX FillPatch uses `PhysBCFunctNoOp` (good for
  interior + C/F only); the staggered domain-boundary ghosts are
  overwritten afterwards by `FillVelGhostPhys` / `FillPresGhostPhys`.
  Inhomogeneous wall data is kept out of the Neumann series
  (homogeneous there) and re-imposed by `EnforceVelDirichlet` on
  `u*` and `u^{n+1}`.  Do not apply `FillVelGhostPhys` directly to a
  freshly computed pressure-gradient field before `ApplyBNFace`: the
  raw `G p` term must retain outflow `p=0` Dirichlet boundary faces.
  Don't move BC handling into the generic AMReX BC functor — the
  staggered normal/tangential split is the reason it's explicit.
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
- `average_down_faces(m_vstar)` *before* the per-level modified-Poisson
  solves so coarse face values at the C/F interface equal the
  averaged fine values.
- `average_down_faces(m_vel)` + `average_down(m_pressure)` at the end of
  `Advance` to keep coarse representation consistent with averaged
  fine values after the per-level projection.
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
| `Run: ins_solver …`                      | Serial run with a selected `tests/<case>/inputs.*`  |
| `Run: ins_solver MPI (…, 4 ranks)`       | `mpirun -np 4` run with a selected test input       |

The configure tasks pass `-DAMReX_DIR=/Users/hang/opt/amrex-26.01/install/lib/cmake/AMReX`.
Trilinos is **not** currently a dependency; when replacing the local
coupled BiCGStab with the planned Tpetra/Belos operator path, restore
`find_package(Trilinos REQUIRED COMPONENTS Tpetra Belos Ifpack2
Teuchos)` in the top-level `CMakeLists.txt` and re-add
`-DTrilinos_DIR=/Users/hang/opt/trilinos-17.0.0/install/lib/cmake/Trilinos`
to the Zed configure tasks.

After any configure task, `compile_commands.json` at the repo root is a
symlink to the most-recently-configured build's compile DB — that is
what `clangd` reads.

## Debugging

`.zed/debug.json` defines two CodeLLDB configurations: launch
`build-debug/src/ins_solver tests/tg/inputs.tg` (with a pre-launch build), and
attach-by-PID (useful for attaching to one rank of an `mpirun`-launched
run).  CodeLLDB is auto-installed on first use.

## Test inputs

| File                                      | What it exercises                                          |
| ----------------------------------------- | ---------------------------------------------------------- |
| `tests/tg/inputs.tg`                      | Single-level smoke test (32³, periodic, Taylor–Green).     |
| `tests/tg_amr/inputs.tg_amr`              | 2-level AMR per-level Perot path + regrid + FillPatch.     |
| `tests/tg2d/inputs.tg2d`                  | 2D analytic Taylor–Green verification.                     |
| `tests/lid/inputs.lid`                    | Lid-driven cavity — all-Dirichlet BCs, singular pressure.  |
| `tests/lid_amr/inputs.lid_amr`            | AMR lid-driven cavity.                                     |
| `tests/channel/inputs.channel`            | Inflow/outflow — non-singular pressure (outflow Dirichlet).|
| `tests/ib_plane/inputs.ib_plane`          | Single-level coupled IB projection smoke test.             |
| `tests/ib_plane_amr/inputs.ib_plane_amr`  | Finest-level coupled IB projection with AMR tagging.       |
| `tests/ib_cylinder_channel/inputs.ib_cylinder_channel` | Single-level channel flow past a stationary cylindrical IB surface. |

Taylor–Green expectation: `|div u|_∞ ~ 10⁻¹¹` per step (Krylov-tolerance
dominated), monotonic energy decay.  Any drift is a regression.
`tests/lid/inputs.lid` should develop a
single primary vortex and approach steady state; `tests/channel/inputs.channel`
should report "non-singular (outflow Dirichlet)" and relax the inlet
toward Poiseuille.

## Conventions for changes

- Keep new code in the existing file structure unless adding a
  genuinely separate concern.  Extend `INSSolver_IB.cpp` for IB
  spread/interp/coupled-Schur work; a wall BC would extend the existing
  diffusion/projection files.
- Do **not** comment what the code does — only why, and only when
  non-obvious.  AMReX domain knowledge is the bar: anything explainable
  by reading AMReX docs is too obvious to comment.
- Don't introduce new third-party deps without discussion.  AMReX + MPI
  is the current dependency surface; Trilinos returns when the
  Tpetra/Belos wrapper replaces the local coupled BiCGStab path.
- Match the AMReX style of the surrounding code (`amrex::Real`,
  `amrex::Box`, `MFIter`, `ParallelFor`, etc.) rather than mixing in
  raw STL/MPI primitives.
