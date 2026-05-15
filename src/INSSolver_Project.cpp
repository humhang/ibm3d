/**
 * INSSolver_Project.cpp
 *
 * Composite pressure Poisson and projection across the AMR hierarchy.
 *
 * Steps performed by ProjectComposite():
 *
 *  1.  Average u* fine→coarse so that the coarse face values at every C/F
 *      interface match the average of the fine faces beneath them.  The
 *      cell-local divergence of u* on the coarse side then equals the
 *      composite divergence.
 *
 *  2.  RHS_lev =  +(1/dt) ∇·u*_lev          (AMReX MLPoisson's Fapply
 *      computes the *positive* discrete Laplacian, so the equation it
 *      solves is ∇²φ = RHS directly.)
 *
 *  3.  MLMG solves the composite cell-centred Poisson with periodic BCs.
 *      Coarse–fine flux matching at the C/F interfaces is handled
 *      internally by MLPoisson's composite stencil.
 *
 *  4.  Per level: FillPatch φ (so the gradient stencil sees coarse-
 *      interpolated values across the C/F interface), then form
 *      ∇φ on faces and apply u^{n+1} = u* − dt ∇φ.  p^{n+1} = p^n + φ.
 *
 *  5.  The caller (Advance) then averages u^{n+1} and p^{n+1} fine→coarse
 *      to remove the divergence residual the local projection leaves on
 *      coarse cells adjacent to the C/F interface — the standard
 *      "approximate projection" pattern.
 */

#include "INSSolver.H"

#include <AMReX_MLMG.H>
#include <AMReX_MLPoisson.H>
#include <AMReX_Print.H>

using namespace amrex;

void INSSolver::ProjectComposite() {
  const int nlev = finest_level + 1;

  // ---- 1) sync u* fine→coarse ----
  AverageDownVelocity(m_vstar);

  // ---- 2) RHS = -(1/dt) ∇·u* on each level ----
  Vector<MultiFab> rhs(nlev);
  for (int lev = 0; lev < nlev; ++lev) {
    rhs[lev].define(grids[lev], dmap[lev], 1, 0);
    rhs[lev].setVal(0.0);

    const Real *dx = geom[lev].CellSize();
    const Real cx = 1.0_rt / (m_dt * dx[0]);
    const Real cy = 1.0_rt / (m_dt * dx[1]);
#if AMREX_SPACEDIM == 3
    const Real cz = 1.0_rt / (m_dt * dx[2]);
#endif

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(rhs[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const Box &bx = mfi.tilebox();
      auto const &u = m_vstar[lev][0]->const_array(mfi);
      auto const &v = m_vstar[lev][1]->const_array(mfi);
#if AMREX_SPACEDIM == 3
      auto const &w = m_vstar[lev][2]->const_array(mfi);
#endif
      auto const &r = rhs[lev].array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        Real d = cx * (u(i + 1, j, k) - u(i, j, k)) +
                 cy * (v(i, j + 1, k) - v(i, j, k));
#if AMREX_SPACEDIM == 3
        d += cz * (w(i, j, k + 1) - w(i, j, k));
#endif
        r(i, j, k) = d;
      });
    }
  }

  // ---- 3) MLMG composite Poisson ----
  Vector<Geometry> mlgeom(geom.begin(), geom.begin() + nlev);
  Vector<BoxArray> mlba(grids.begin(), grids.begin() + nlev);
  Vector<DistributionMapping> mldm(dmap.begin(), dmap.begin() + nlev);

  LPInfo lpinfo;
  MLPoisson mlpoisson(mlgeom, mlba, mldm, lpinfo);

  std::array<LinOpBCType, AMREX_SPACEDIM> bc_lo, bc_hi;
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    bc_lo[d] = LinOpBCType::Periodic;
    bc_hi[d] = LinOpBCType::Periodic;
  }
  mlpoisson.setDomainBC(bc_lo, bc_hi);

  Vector<MultiFab *> phi_ptrs(nlev);
  Vector<const MultiFab *> rhs_ptrs(nlev);
  for (int lev = 0; lev < nlev; ++lev) {
    mlpoisson.setLevelBC(lev, nullptr); // no inhomogeneous BC data
    m_phi[lev]->setVal(0.0);            // zero initial guess
    phi_ptrs[lev] = m_phi[lev].get();
    rhs_ptrs[lev] = &rhs[lev];
  }

  MLMG mlmg(mlpoisson);
  mlmg.setMaxIter(200);
  mlmg.setMaxFmgIter(0);
  mlmg.setVerbose(std::max(0, m_verbose - 1));
  mlmg.setBottomTolerance(1.0e-6);

  const Real reltol = 1.0e-10;
  const Real abstol = 0.0;
  mlmg.solve(phi_ptrs, rhs_ptrs, reltol, abstol);

  // ---- 4) gradient + projection per level ----
  for (int lev = 0; lev < nlev; ++lev) {
    // φ with 1 ghost (intra-level + C/F interpolation from coarse)
    MultiFab phi_g(grids[lev], dmap[lev], 1, 1);
    FillCellPatch(lev, m_phi, phi_g, m_cur_time, m_bc_pres);

    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
      BoxArray fba = amrex::convert(grids[lev], IntVect::TheDimensionVector(d));
      MultiFab gphi(fba, dmap[lev], 1, 0);
      ComputePressureGradient(lev, d, phi_g, gphi);

      // u^{n+1} = u* - dt ∇φ
      MultiFab::Copy(*m_vel[lev][d], *m_vstar[lev][d], 0, 0, 1, 0);
      MultiFab::Saxpy(*m_vel[lev][d], -m_dt, gphi, 0, 0, 1, 0);
    }

    // p^{n+1} = p^n + φ
    MultiFab::Add(*m_pressure[lev], *m_phi[lev], 0, 0, 1, 0);
  }
}
