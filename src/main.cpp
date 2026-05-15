#include <AMReX.H>
#include "INSSolver.H"

int main(int argc, char *argv[]) {
  amrex::Initialize(argc, argv);
  {
    INSSolver solver;
    solver.InitData();
    solver.Run();
  }
  amrex::Finalize();
  return 0;
}
