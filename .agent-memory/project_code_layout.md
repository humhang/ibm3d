---
name: Project — code layout and current file roles
description: One-line purpose for every source file, so future agents don't have to grep to orient themselves.
type: project
originSessionId: 12fb2afb-57e7-4a3b-acaf-2f8c91188f9d
---
`src/` layout (as of 2026-05-20 — Perot + physical BCs + first IB projection):

| File                     | Role                                                                                |
|--------------------------|-------------------------------------------------------------------------------------|
| `main.cpp`               | Trivial entry: `amrex::Initialize` → `INSSolver{}` → `InitData` → `Run`.            |
| `INSSolver.H`            | Class declaration, `BCKind` enum, per-face BC storage, helper signatures.            |
| `INSSolver.cpp`          | Constructor, ParmParse, time loop, AmrCore hooks, FillPatch helpers (call physical BC after AMReX FillPatch), average-down, plotfile, IC (Taylor–Green *or* quiescent for BC-driven flows). |
| `INSSolver_BC.cpp`       | `ParseBCs`, `BuildBCRecs`, `FillVelGhostPhys` (Dirichlet/slip/outflow, normal vs tangential staggered handling, homogeneous flag), `FillPresGhostPhys` (Neumann walls / Dirichlet-0 outflow), `EnforceVelDirichlet`. |
| `INSSolver_Advect.cpp`   | `ComputeAdvection(lev, adv, vel_in)` — face-by-face `-(u·∇)u`; `vel_in` already FillPatched + physical-BC filled. |
| `INSSolver_Diffuse.cpp`  | `ApplyFaceLaplacian`, `ComputePressureGradient`, `ApplyBNFace` (B^N, raw valid k=0 term plus homogeneous k>=1 work-term ghosts), `ApplyCNDiffusion` (Perot predictor, then `EnforceVelDirichlet`). |
| `INSSolver_Project.cpp`  | `ApplyModifiedPoissonOp` (−D B^N G), `SolveModifiedPoisson` (pressure-only BiCGStab), `ProjectPerot` (pressure-only projection or finest-level coupled IB projection). |
| `INSSolver_IB.cpp`       | IB geometry initialization, Peskin 4-point `H/E`, finest-level IB tagging, coupled BiCGStab on `[-D;E]B^N[G H]`. |
| `IBGeometry.H/.cpp`      | Dimension-selected host loaders plus marker construction and device copies for IB geometry: 2D ASCII line-segment curves, 3D ASCII/binary STL triangle surfaces with exact coordinate de-duplication into indexed connectivity. |
| `CMakeLists.txt`         | Executables `ins_solver` and `ins_solver_2d`, link MPI + AMReX (Trilinos not currently needed). |

Top-level files:

| File                   | Role                                                                       |
|------------------------|----------------------------------------------------------------------------|
| `CMakeLists.txt`       | `find_package(MPI)`, `find_package(AMReX)`, sets C++20.                    |
| `tests/2d/tg2d/inputs.tg2d`            | Native 2D Taylor–Green analytic/self-convergence case.                     |
| `tests/2d/tg2d_amr/inputs.tg2d_amr`    | Native 2D Taylor–Green AMR case.                                           |
| `tests/2d/ib_square/inputs.ib_square`  | Native 2D coupled IB projection smoke case with local curve.               |
| `tests/2d/ib_square_amr/inputs.ib_square_amr` | Native 2D coupled IB projection AMR smoke case.                     |
| `tests/2d/ib_cylinder_re100/inputs.ib_cylinder_re100` | 2D flow past a stationary cylinder, Re_D=100, finest D/dx=20.    |
| `tests/3d/tg/inputs.tg`                | Single-level Taylor–Green test (`max_level=0`, 32³).                       |
| `tests/3d/tg_amr/inputs.tg_amr`        | 2-level AMR Taylor–Green test (`max_level=1`, vorticity tagging).          |
| `tests/3d/tg2d/inputs.tg2d`            | Thin-periodic-z Taylor–Green analytic/self-convergence case.               |
| `tests/3d/lid/inputs.lid`              | Lid-driven cavity (all-Dirichlet, singular pressure, `ic=quiescent`).      |
| `tests/3d/lid_amr/inputs.lid_amr`      | AMR lid-driven cavity.                                                     |
| `tests/3d/channel/inputs.channel`      | Inflow/outflow channel (non-singular pressure).                            |
| `tests/3d/ib_plane/inputs.ib_plane`    | Single-level coupled IB projection smoke case with local STL.              |
| `tests/3d/ib_plane_amr/inputs.ib_plane_amr` | Finest-level coupled IB projection AMR smoke case with local STL.      |
| `tests/3d/ib_cylinder_channel/inputs.ib_cylinder_channel` | Single-level channel past a stationary cylindrical IB surface; STL generated locally. |
| `tests/3d/ib_sphere_re100/inputs.ib_sphere_re100` | 3D flow past a stationary sphere, Re_D=100, finest D/dx=20; STL generated locally. |
| `AGENTS.md`            | Coding-style + Zed-task documentation.                                     |
| `.clang-format`        | `BasedOnStyle: LLVM`, `Standard: c++20`.                                   |
| `.zed/tasks.json`      | Configure/build/clean/run/debug tasks pinning AMReX_DIR.                   |
| `.zed/debug.json`      | CodeLLDB launch + attach configs.                                          |

**How to apply**: when adding new functionality, prefer extending an
existing file in this list to creating a new file unless the new piece
is genuinely a separate concern.  IB spread/interp and coupled-Schur
logic now live in `INSSolver_IB.cpp`; wall BCs still belong in the
existing BC/diffusion files.

**Do not** restore `INSSolver_Stress.cpp` or `TrilinosPoissonSolver.H`
(deleted on 2026-05-14) — they were single-level IB-stress and assembled-
Trilinos placeholders, both superseded by the current direction.
