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
the hand-rolled IB BiCGStab path with the planned Tpetra/Belos solve):

- Install prefix: `/Users/hang/opt/trilinos-17.0.0/install`
- CMake config dir: `/Users/hang/opt/trilinos-17.0.0/install/lib/cmake/Trilinos`
- `TrilinosConfig.cmake`: confirmed present
- Components needed for IBPM: `Tpetra`, `Belos`, `Ifpack2`, `Teuchos`
  (and `Kokkos` is pulled in transitively)

The AMReX path is used by the configure tasks.  Re-add the Trilinos path
to those tasks only when Trilinos returns to `CMakeLists.txt`.  If the
installs move, update the task strings in addition to the CMake changes.

`.zed/` is gitignored because of these absolute paths — do not commit
the tasks file as-is.
