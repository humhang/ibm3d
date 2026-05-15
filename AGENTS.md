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
  approximated by the truncated Neumann series `Σ_{k=0}^{N} (εL)^k`.
- **Pressure Poisson via AMReX `MLMG`** while there's no IB.  When the IB
  phase begins, the operator becomes the Taira–Colonius modified Schur
  complement and is implemented **matrix-free** via a
  `Tpetra::Operator` subclass — *not* by assembling a `CrsMatrix`.
- **MLPoisson sign**: `Fapply` computes the *positive* discrete
  Laplacian, so `mlmg.solve(phi, rhs)` solves `∇²φ = rhs`.  In
  `ProjectComposite()` the RHS is therefore `+(1/dt) ∇·u*`, not
  negative.  This was a bug once; do not re-introduce it.
- **All-periodic BCs only** at this stage.  Wall/inflow/outflow is
  deferred until after the algorithm validates.

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
- `average_down_faces(m_vstar)` *before* forming the Poisson RHS so the
  composite divergence is level-consistent.
- `average_down_faces(m_vel)` + `average_down(m_pressure)` at the end of
  `Advance` to remove the small divergence residual the local
  projection leaves at C/F interfaces (the standard "approximate
  projection" pattern).

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
| `inputs`         | Single-level smoke test (32³, periodic, Taylor–Green).    |
| `inputs.tg_amr`  | 2-level AMR composite Poisson path + regrid + FillPatch.   |

Expected behaviour after fix-up: `|div u|_∞ ~ 10⁻¹⁴` per step; MLMG in
~8 iterations; monotonic energy decay.  Any drift from those numbers is
a regression.

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
