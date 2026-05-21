---
name: Reference — external dependency install paths on this machine
description: Filesystem locations of the AMReX and Trilinos installs; used by CMake `find_package` via the Zed configure tasks.
type: reference
originSessionId: 12fb2afb-57e7-4a3b-acaf-2f8c91188f9d
---
AMReX 26.01 (currently used):

- Install prefix: `/Users/hang/opt/amrex-26.01/install`
- CMake config dir: `/Users/hang/opt/amrex-26.01/install/lib/cmake/AMReX`
- `AMReXConfig.cmake`: confirmed present
- Built with MPI on (MPICH 4.1 from MacPorts: `/opt/local/lib/mpich-mp/`)
- HDF5 support enabled (1.14.6 from MacPorts)
- No OpenMP, no SIMD, no GPU backend

Trilinos 17.0.0 (not active in the current build; needed when replacing
the local IB BiCGStab path with the planned Tpetra/Belos solve):

- Install prefix: `/Users/hang/opt/trilinos-17.0.0/install`
- CMake config dir: `/Users/hang/opt/trilinos-17.0.0/install/lib/cmake/Trilinos`
- `TrilinosConfig.cmake`: confirmed present
- Components needed for IBPM: `Tpetra`, `Belos`, `Ifpack2`, `Teuchos`
  (and `Kokkos` is pulled in transitively)

These paths are baked into `.zed/tasks.json` as `-DAMReX_DIR=…` and
`-DTrilinos_DIR=…` on the configure commands.  If the installs move,
update those task strings in addition to whatever CMakeLists changes.

`.zed/` is gitignored because of these absolute paths — do not commit
the tasks file as-is.
