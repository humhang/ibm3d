set(SOURCES
        main.cpp
        INSSolver.cpp
        INSSolver_BC.cpp
        INSSolver_Advect.cpp
        INSSolver_Diffuse.cpp
        INSSolver_Project.cpp
)

add_executable(ins_solver ${SOURCES})

target_include_directories(ins_solver PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ins_solver PRIVATE
        MPI::MPI_CXX
        AMReX::amrex)
