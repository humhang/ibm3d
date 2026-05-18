---
name: Project — code layout and current file roles
description: One-line purpose for every source file, so future agents don't have to grep to orient themselves.
type: project
originSessionId: 12fb2afb-57e7-4a3b-acaf-2f8c91188f9d
---
`src/` layout (as of 2026-05-15, third iteration — Perot + physical BCs):

| File                     | Role                                                                                |
|--------------------------|-------------------------------------------------------------------------------------|
| `main.cpp`               | Trivial entry: `amrex::Initialize` → `INSSolver{}` → `InitData` → `Run`.            |
| `INSSolver.H`            | Class declaration, `BCKind` enum, per-face BC storage, helper signatures.            |
| `INSSolver.cpp`          | Constructor, ParmParse, time loop, AmrCore hooks, FillPatch helpers (call physical BC after AMReX FillPatch), average-down, plotfile, IC (Taylor–Green *or* quiescent for BC-driven flows). |
| `INSSolver_BC.cpp`       | `ParseBCs`, `BuildBCRecs`, `FillVelGhostPhys` (Dirichlet/slip/outflow, normal vs tangential staggered handling, homogeneous flag), `FillPresGhostPhys` (Neumann walls / Dirichlet-0 outflow), `EnforceVelDirichlet`. |
| `INSSolver_Advect.cpp`   | `ComputeAdvection(lev, adv, vel_in)` — face-by-face `-(u·∇)u`; `vel_in` already FillPatched + physical-BC filled. |
| `INSSolver_Diffuse.cpp`  | `ApplyFaceLaplacian`, `ComputePressureGradient`, `ApplyBNFace` (B^N, homogeneous wall data), `ApplyCNDiffusion` (Perot predictor, then `EnforceVelDirichlet`). |
| `INSSolver_Project.cpp`  | `ApplyModifiedPoissonOp` (−D B^N G), `SolveModifiedPoisson` (matrix-free CG, mean-pin gated by `m_pressure_singular`), `ProjectPerot` (per-level solve+project, `EnforceVelDirichlet`). |
| `CMakeLists.txt`         | Executable `ins_solver`, links MPI + AMReX (Trilinos not currently needed). |

Top-level files:

| File                   | Role                                                                       |
|------------------------|----------------------------------------------------------------------------|
| `CMakeLists.txt`       | `find_package(MPI)`, `find_package(AMReX)`, sets C++20.                    |
| `inputs`               | Single-level Taylor–Green test (`max_level=0`, 32³).                       |
| `inputs.tg_amr`        | 2-level AMR Taylor–Green test (`max_level=1`, vorticity tagging).          |
| `inputs.lid`           | Lid-driven cavity (all-Dirichlet, singular pressure, `ic=quiescent`).      |
| `inputs.channel`       | Inflow/outflow channel (non-singular pressure).                           |
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
