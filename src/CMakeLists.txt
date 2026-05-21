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

add_executable(ins_solver ${SOURCES})

if(AMReX_GPU_BACKEND STREQUAL "CUDA")
  include(AMReXTargetHelpers)
  setup_target_for_cuda_compilation(ins_solver) #make it cuda source
elseif(AMReX_GPU_BACKEND STREQUAL "HIP")
  # AMReX does not provide like setup_target_for_hip_compilation, manually do it
  set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE HIP)
elseif(AMReX_GPU_BACKEND STREQUAL "SYCL")
  # don't need to do anything
endif()

target_include_directories(ins_solver PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ins_solver PRIVATE
        MPI::MPI_CXX
        AMReX::amrex)

install(TARGETS ins_solver RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
