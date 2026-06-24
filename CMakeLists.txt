cmake_minimum_required(VERSION 3.18)
project(ibm3d C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR} CACHE PATH "Installing prefix of the project" FORCE)
endif()

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

find_package(MPI REQUIRED)
find_package(AMReX REQUIRED)

if(NOT AMReX_MPI)
  message(FATAL_ERROR "AMReX must be built with MPI support")
endif()
message(STATUS "AMReX_SIMD:          ${AMReX_SIMD}")
message(STATUS "AMReX_OMP:           ${AMReX_OMP}")
message(STATUS "AMReX_GPU_BACKEND:   ${AMReX_GPU_BACKEND}")

# LTO not compatible with CUDA pipeline, not sure about HIP/SYCL
# You can use LTO flags for pure host translation units with host compilers.
# But not for translation units with device codes, you can't forward LTO flags to host compilers.
# For this project: all sources are device codes if GPU enabled.
if("${CMAKE_BUILD_TYPE}" STREQUAL "Release"
   AND AMReX_GPU_BACKEND STREQUAL "NONE")
  include(CheckIPOSupported)
  check_ipo_supported(RESULT ipo_result OUTPUT ipo_output)
  if(ipo_result)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    message(STATUS "IPO enabled")
  else()
    message(STATUS "IPO not supported: ${ipo_output}")
  endif()
endif()

# you could accidently inherit flto flags from TPL, e.g., MPI from RHEL repo
# Need to disable it
# There should be better solution
if(NOT AMReX_GPU_BACKEND STREQUAL "NONE")
  add_compile_options($<$<CXX_COMPILER_ID:GNU>:-fno-lto>)
  add_link_options($<$<CXX_COMPILER_ID:GNU>:-fno-lto>)
endif()

# If you really want LTO for both host/device, you should do the 4 steps
# 1. g++ -flto for .cpp
# 2. nvcc -dlto for .cu.
#     Note: -Xcompiler -flto NOT ALLOWED. Host code inside .cu can't do LTO
# 3. nvcc -dlto linking device
# 4. g++ -flto for final linkage

# Trilinos will be re-introduced when the immersed-boundary projection is added
# (it is needed for the Taira–Colonius saddle-point operator).  The pure
# NS+AMR phase currently uses AMReX data structures plus a matrix-free
# modified-Poisson Krylov solve.

if(AMReX_GPU_BACKEND STREQUAL "CUDA")
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
elseif(AMReX_GPU_BACKEND STREQUAL "HIP")
  enable_language(HIP)
  set(CMAKE_HIP_STANDARD 20)
  set(CMAKE_HIP_STANDARD_REQUIRED ON)
  set(CMAKE_HIP_EXTENSIONS OFF)
  # These package below provide targets which are interfacial dependencies of amrex target,
  # I don't use it explicitly amrex config should call find_dependency, instead of me calling find_package here!
  # I submitted a ticket, once they fix it remove these
  find_package(hip REQUIRED CONFIG)
  find_package(hiprand REQUIRED CONFIG)
  find_package(rocrand REQUIRED CONFIG)
  find_package(rocprim REQUIRED CONFIG)
  find_package(rocsparse REQUIRED CONFIG)
elseif(AMReX_GPU_BACKEND STREQUAL "SYCL")
  # make sure the cxx compiler is a sycl compiler...
  # enumerate and match, although there may be better solution?
  set(_sycl_compiler_ids
      Clang IntelClang IntelDPCPP IntelLLVM)
  if (NOT CMAKE_CXX_COMPILER_ID IN_LIST _sycl_compiler_ids)
    message(FATAL_ERROR
              "AMReX was built with SYCL, but CMAKE_CXX_COMPILER is "
              "'${CMAKE_CXX_COMPILER}' (${CMAKE_CXX_COMPILER_ID}). "
              "Are you sure this is a SYCL compiler? "
              "Perhaps you need to reconfigure with the Intel oneAPI compiler, typically icpx.")
  endif()
  unset(_sycl_compiler_ids)
elseif(NOT AMReX_GPU_BACKEND STREQUAL "NONE")
  message(FATAL_ERROR "Unsupported AMReX GPU backend: "
        "'${AMReX_GPU_BACKEND}'")
else()
endif()

# if the MPI is a compiler wrapper, target MPI::MPI_CXX is empty, per CMake behavior,
# thus no additional MPI flags will be added if you link against MPI::MPI_CXX'.
# Not an issue for pure CXX code since the compiler is the wrapper.
# Won't work for CUDA/HIP, unless CUDA/HIP compilers are also MPI wrapper (for example Cray-CC in Frontier).
# For safe: don't use MPI::MPI_CXX, we create out own interface target.
add_library(ibm3d_mpi INTERFACE)
target_include_directories(ibm3d_mpi SYSTEM INTERFACE ${MPI_CXX_INCLUDE_DIRS})
target_compile_definitions(ibm3d_mpi INTERFACE ${MPI_CXX_COMPILE_DEFINITIONS})
target_compile_options(ibm3d_mpi INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:${MPI_CXX_COMPILE_OPTIONS}>")
target_link_libraries(ibm3d_mpi INTERFACE ${MPI_CXX_LIBRARIES})
if(MPI_CXX_LINK_FLAGS)
  separate_arguments(_mpi_cxx_link_options NATIVE_COMMAND "${MPI_CXX_LINK_FLAGS}")
  target_link_options(ibm3d_mpi INTERFACE ${_mpi_cxx_link_options})
endif()

add_subdirectory(src)
