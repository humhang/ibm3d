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

if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_result OUTPUT ipo_output)
    if(ipo_result)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
        message(STATUS "IPO enabled")
    else()
        message(STATUS "IPO not supported: ${ipo_output}")
    endif()
endif()

find_package(MPI REQUIRED)
find_package(AMReX REQUIRED)

if(NOT AMReX_MPI)
    message(FATAL_ERROR "AMReX must be built with MPI support")
endif()
message(STATUS "AMReX_SIMD:          ${AMReX_SIMD}")
message(STATUS "AMReX_OMP:           ${AMReX_OMP}")
message(STATUS "AMReX_GPU_BACKEND:   ${AMReX_GPU_BACKEND}")

# Trilinos will be re-introduced when the immersed-boundary projection is added
# (it is needed for the Taira–Colonius saddle-point operator).  The pure
# NS+AMR phase currently uses AMReX data structures plus a matrix-free
# modified-Poisson Krylov solve.

if(AMReX_GPU_BACKEND STREQUAL "CUDA")
    enable_language(CUDA)
elseif(AMReX_GPU_BACKEND STREQUAL "HIP")
    enable_language(HIP)
endif()

add_subdirectory(src)
