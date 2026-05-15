---
name: Project — code layout and current file roles
description: One-line purpose for every source file, so future agents don't have to grep to orient themselves.
type: project
originSessionId: 12fb2afb-57e7-4a3b-acaf-2f8c91188f9d
---
`src/` layout (as of 2026-05-15):

| File                     | Role                                                                                |
|--------------------------|-------------------------------------------------------------------------------------|
| `main.cpp`               | Trivial entry: `amrex::Initialize` → `INSSolver{}` → `InitData` → `Run`.            |
| `INSSolver.H`            | Class declaration, `AmrCore` overrides, FillPatch/AverageDown helper signatures.     |
| `INSSolver.cpp`          | Constructor, ParmParse, time loop, AmrCore hooks (MakeNewLevelFromScratch, MakeNewLevelFromCoarse, RemakeLevel, ClearLevel, ErrorEst), FillPatch helpers, average-down, plotfile writer, IC (3D Taylor–Green). |
| `INSSolver_Advect.cpp`   | `ComputeAdvection(lev, adv, vel_in)` — face-by-face `-(u·∇)u` on the MAC grid; takes already-FillPatched `vel_in`. |
| `INSSolver_Diffuse.cpp`  | `ApplyFaceLaplacian`, `ComputePressureGradient`, `ApplyCNDiffusion` (builds `u*` via Neumann series). |
| `INSSolver_Project.cpp`  | `ProjectComposite()` — average-down `u*`, build RHS, MLMG composite Poisson, per-level gradient + projection, `p ← p + φ`. |
| `CMakeLists.txt`         | Standalone executable target `ins_solver`, links MPI + AMReX (Trilinos not currently needed). |

Top-level files:

| File                   | Role                                                                       |
|------------------------|----------------------------------------------------------------------------|
| `CMakeLists.txt`       | `find_package(MPI)`, `find_package(AMReX)`, sets C++20.                    |
| `inputs`               | Single-level Taylor–Green test (`max_level=0`, 32³).                       |
| `inputs.tg_amr`        | 2-level AMR Taylor–Green test (`max_level=1`, vorticity tagging).          |
| `AGENTS.md`            | Coding-style + Zed-task documentation.                                     |
| `.clang-format`        | `BasedOnStyle: LLVM`, `Standard: c++20`.                                   |
| `.zed/tasks.json`      | Configure/build/clean/run/debug tasks pinning AMReX_DIR.                   |
| `.zed/debug.json`      | CodeLLDB launch + attach configs.                                          |

**How to apply**: when adding new functionality, prefer extending an
existing file in this list to creating a new file unless the new piece
is genuinely a separate concern.  E.g. immersed-boundary spread/interp
would belong in a new `INSSolver_IB.cpp`, but a new wall BC would live
inside the existing `INSSolver_Diffuse.cpp` next to the face Laplacian.

**Do not** restore `INSSolver_Stress.cpp` or `TrilinosPoissonSolver.H`
(deleted on 2026-05-14) — they were single-level IB-stress and assembled-
Trilinos placeholders, both superseded by the current direction.
