set(SOURCES
        main.cpp
        INSSolver.cpp
        INSSolver_BC.cpp
        INSSolver_Advect.cpp
        INSSolver_Diffuse.cpp
        INSSolver_Project.cpp
        INSSolver_IB.cpp
        IBGeometry.cpp
)

function(add_ins_solver_target target amrex_target)
  add_executable(${target} ${SOURCES})

  if(AMReX_GPU_BACKEND STREQUAL "CUDA")
    setup_target_for_cuda_compilation(${target}) # make it cuda source
  elseif(AMReX_GPU_BACKEND STREQUAL "HIP")
    # AMReX does not provide setup_target_for_hip_compilation; do it manually.
    set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE HIP)
  elseif(AMReX_GPU_BACKEND STREQUAL "SYCL")
    # don't need to do anything
  endif()

  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${target} PRIVATE
          MPI::MPI_CXX
          ${amrex_target})
endfunction()

if(AMReX_GPU_BACKEND STREQUAL "CUDA")
  include(AMReXTargetHelpers)
endif()

add_ins_solver_target(ins_solver AMReX::amrex_3d)
add_ins_solver_target(ins_solver_2d AMReX::amrex_2d)

install(TARGETS ins_solver ins_solver_2d
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
