# AGENTS.md

Guidelines for any agent (human or AI) contributing to **ibm3d**.

## Project status (read this first)

`ibm3d` is in **phase 1**: pure incompressible Navier–Stokes on an AMR
hierarchy.  The end goal is the **Taira–Colonius immersed-boundary
projection method** (JCP 2007); that machinery has *not* been added
yet, and the linear-algebra direction for it is settled (matrix-free
`Tpetra::Operator` with AMReX `MLMG` as the preconditioner — see
below).  Do not add IB code without explicit user direction.

Key decisions already settled, do not re-litigate:

- **MAC staggered grid**, second-order centred differencing.
- **Explicit advection + Crank–Nicolson diffusion**, with `(I − εL)^{-1}`
  approximated by the truncated Neumann series `B^N = Σ_{k=0}^{N} (εL)^k`.
- **Perot 1997 fractional step**: predictor `r1 = (I + εL) u^n + dt A^n`
  (NO pressure gradient), `u* = B^N r1`, modified-Poisson
  `(D B^N G) p^{n+1} = (1/dt) D u*`, projection
  `u^{n+1} = u* − dt B^N G p^{n+1}`.  `m_pressure` is solved for
  directly each step — there is no incremental form.
- **Modified-Poisson solve is matrix-free**, hand-rolled CG.  Operator
  composed from existing `ComputePressureGradient`, face Laplacian,
  and cell divergence pieces.  When the IB phase lands, the IB rows
  attach to this same operator (Tpetra::Operator wrapper) — keeping
  the modified-Poisson direction was the *purpose* of this design.
- **`D B^N G` is negative semidefinite** (eigenvalues `−k² · …`).
  Both the operator (`ApplyModifiedPoissonOp`) and the RHS in
  `ProjectPerot` are negated so CG sees a PSD system.  This is the
  same sign story as AMReX `MLPoisson` and bit us in this file's
  earlier revision; the projection `u^{n+1} = u* − dt B^N G p` still
  uses the natural `+B^N G p`.
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
  `u*` and `u^{n+1}`.  Don't move BC handling into the generic
  AMReX BC functor — the staggered normal/tangential split is the
  reason it's explicit.

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

## Setting up a fresh clone

This repo ships its Claude Code per-project memory inside the working
tree at `.agent-memory/`.  Claude Code itself reads/writes a path under
`~/.claude/projects/<encoded-abs-path>/memory/` — the standard bootstrap
makes that path a symlink into the repo:

```sh
./scripts/setup-agent-memory.sh
```

Run it once after cloning.  It's idempotent.  After that, every AI
session has access to the project's accumulated context (architectural
decisions, the matrix-free plan for the IB phase, install paths,
sign-convention gotchas, etc.) — see `.agent-memory/MEMORY.md` for the
index.

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
| `Run: ins_solver …`                      | Serial run with `inputs`                            |
| `Run: ins_solver MPI (…, 4 ranks)`       | `mpirun -np 4` run with `inputs`                    |

The configure tasks pass `-DAMReX_DIR=/Users/hang/opt/amrex-26.01/install/lib/cmake/AMReX`.
Trilinos is **not** currently a dependency; when the IB phase starts,
restore `find_package(Trilinos REQUIRED COMPONENTS Tpetra Belos Ifpack2
Teuchos)` in the top-level `CMakeLists.txt` and re-add
`-DTrilinos_DIR=/Users/hang/opt/trilinos-17.0.0/install/lib/cmake/Trilinos`
to the Zed configure tasks.

After any configure task, `compile_commands.json` at the repo root is a
symlink to the most-recently-configured build's compile DB — that is
what `clangd` reads.

## Debugging

`.zed/debug.json` defines two CodeLLDB configurations: launch
`build-debug/src/ins_solver inputs` (with a pre-launch build), and
attach-by-PID (useful for attaching to one rank of an `mpirun`-launched
run).  CodeLLDB is auto-installed on first use.

## Test inputs

| File             | What it exercises                                          |
| ---------------- | ---------------------------------------------------------- |
| `inputs`         | Single-level smoke test (32³, periodic, Taylor–Green).     |
| `inputs.tg_amr`  | 2-level AMR per-level Perot path + regrid + FillPatch.     |
| `inputs.lid`     | Lid-driven cavity — all-Dirichlet BCs, singular pressure.  |
| `inputs.channel` | Inflow/outflow — non-singular pressure (outflow Dirichlet).|

Taylor–Green expectation: `|div u|_∞ ~ 10⁻¹¹` per step (CG-tolerance
dominated), unpreconditioned CG ~40–80 iters at 32³, monotonic energy
decay.  Any drift is a regression.  `inputs.lid` should develop a
single primary vortex and approach steady state; `inputs.channel`
should report "non-singular (outflow Dirichlet)" and relax the inlet
toward Poiseuille.

## Conventions for changes

- Keep new code in the existing file structure unless adding a
  genuinely separate concern.  IB spread/interp would warrant a new
  `INSSolver_IB.cpp`; a wall BC would extend the existing
  diffusion/projection files.
- Do **not** comment what the code does — only why, and only when
  non-obvious.  AMReX domain knowledge is the bar: anything explainable
  by reading AMReX docs is too obvious to comment.
- Don't introduce new third-party deps without discussion.  AMReX + MPI
  is the current dependency surface; Trilinos returns when IB lands.
- Match the AMReX style of the surrounding code (`amrex::Real`,
  `amrex::Box`, `MFIter`, `ParallelFor`, etc.) rather than mixing in
  raw STL/MPI primitives.
