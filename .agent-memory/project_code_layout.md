---
name: Project — code layout and current file roles
description: One-line purpose for every source file, so future agents don't have to grep to orient themselves.
type: project
originSessionId: 12fb2afb-57e7-4a3b-acaf-2f8c91188f9d
---
`src/` layout (updated 2026-07-17 — stored staggered stress):

| File                     | Role                                                                                |
|--------------------------|-------------------------------------------------------------------------------------|
| `main.cpp`               | Trivial entry: `amrex::Initialize` → `INSSolver{}` → `InitData` → `Run`.            |
| `INSSolver.H`            | Class declaration, `BCKind` enum, per-face BC storage, staggered velocity/stress hierarchy storage, helper signatures. |
| `INSSolver.cpp`          | Constructor, ParmParse, time loop, AmrCore hooks, velocity/stress allocation and lifecycle, FillPatch helpers (staggered velocity post-fill and pressure scratch-BC callbacks), average-down, plotfile, IC (Taylor–Green *or* quiescent for BC-driven flows). |
| `INSSolver_BC.cpp`       | `ParseBCs`, `BuildBCRecs`, `FillVelGhostPhys` (Dirichlet/slip/outflow, normal vs tangential staggered handling, homogeneous flag), `FillPresGhostPhys` (Neumann walls / Dirichlet-0 outflow), `EnforceVelDirichlet`. |
| `INSSolver_Advect.cpp`   | `ComputeAdvection(lev, adv, vel_in)` — face-by-face `-(u·∇)u`; `vel_in` already FillPatched + physical-BC filled. |
| `INSSolver_Diffuse.cpp`  | Staggered viscous-stress gradient/divergence and refresh kernels, face Laplacian/gradient kernels, per-level `ApplyBNFace`, hierarchy `ApplyCompositeBNFaces`, and predictor-RHS construction. |
| `INSSolver_Project.cpp`  | Composite modified-Poisson/IB blocks, pressure-only and marker force-Schur BiCGStab, MLMG pressure inverse, true coupled-residual fallback, `ProjectPerot`. |
| `INSSolver_IB.cpp`       | IB geometry initialization, Peskin 4-point spread/interpolate (`H/E`), finest-level IB tagging, and level-local kernels used by the composite solve. |
| `IBGeometry.H/.cpp`      | Dimension-selected host loaders plus marker construction and device copies for IB geometry: 2D ASCII line-segment curves, 3D ASCII/binary STL triangle surfaces with exact coordinate de-duplication into indexed connectivity. |
| `CMakeLists.txt`         | Executables `ins_solver` and `ins_solver_2d`, link MPI + AMReX (Trilinos not currently needed). |

Top-level files:

| File                   | Role                                                                       |
|------------------------|----------------------------------------------------------------------------|
| `CMakeLists.txt`       | `find_package(MPI)`, `find_package(AMReX)`, sets C++20.                    |
| `tests/2d/tg2d/inputs.tg2d`            | Native 2D Taylor–Green analytic/self-convergence case.                     |
| `tests/2d/tg2d_amr/inputs.tg2d_amr`    | Native 2D Taylor–Green AMR case.                                           |
| `tests/2d/tg2d_cf/inputs.tg2d_cf`      | Convecting Taylor–Green with a partial C/F interface.                       |
| `tests/2d/ib_square/inputs.ib_square`  | Native 2D coupled IB projection smoke case with local curve.               |
| `tests/2d/ib_square_amr/inputs.ib_square_amr` | Native 2D coupled IB projection AMR smoke case.                     |
| `tests/2d/ib_cylinder_re100/inputs.ib_cylinder_re100` | 2D flow past a stationary cylinder, Re_D=100, finest D/dx=80.    |
| `tests/2d/ib_cylinder_re100_coarse/inputs.ib_cylinder_re100_coarse` | Coarse local-debug cylinder, one refinement level, finest D/dx=20, 24 markers. |
| `tests/3d/tg/inputs.tg`                | Single-level Taylor–Green test (`max_level=0`, 32³).                       |
| `tests/3d/tg_amr/inputs.tg_amr`        | 2-level AMR Taylor–Green test (`max_level=1`, vorticity tagging).          |
| `tests/3d/tg2d/inputs.tg2d`            | Thin-periodic-z Taylor–Green analytic/self-convergence case.               |
| `tests/3d/lid/inputs.lid`              | Lid-driven cavity (all-Dirichlet, singular pressure, `ic=quiescent`).      |
| `tests/3d/lid_amr/inputs.lid_amr`      | AMR lid-driven cavity.                                                     |
| `tests/3d/channel/inputs.channel`      | Inflow/outflow channel (non-singular pressure).                            |
| `tests/3d/ib_plane/inputs.ib_plane`    | Single-level coupled IB projection smoke case with local STL.              |
| `tests/3d/ib_plane_amr/inputs.ib_plane_amr` | Composite AMR IB projection smoke case with finest-level marker coupling. |
| `tests/3d/ib_cylinder_channel/inputs.ib_cylinder_channel` | Single-level channel past a stationary cylindrical IB surface; STL generated locally. |
| `tests/3d/ib_sphere_re100/inputs.ib_sphere_re100` | 3D flow past a stationary sphere, Re_D=100, finest D/dx=80; STL generated locally. |
| `AGENTS.md`            | Coding-style + Zed-task documentation.                                     |
| `.clang-format`        | `BasedOnStyle: LLVM`, `Standard: c++20`.                                   |
| `.zed/tasks.json`      | Configure/build/clean/run/debug tasks pinning AMReX_DIR.                   |
| `.zed/debug.json`      | CodeLLDB launch + attach configs.                                          |

**How to apply**: when adding new functionality, prefer extending an
existing file in this list to creating a new file unless the new piece
is genuinely a separate concern.  IB geometry and spread/interp kernels
live in `INSSolver_IB.cpp`; composite projection operators and Krylov
orchestration live in `INSSolver_Project.cpp`.  Wall BCs belong in the
existing BC and diffusion/projection files.

**Do not** restore the historical `INSSolver_Stress.cpp` or
`TrilinosPoissonSolver.H` placeholders.  Staggered diffusive stress belongs in
`INSSolver_Diffuse.cpp`; the old file represented a different single-level IB
stress path.  The assembled-Trilinos placeholder is likewise superseded by
the current matrix-free direction.
