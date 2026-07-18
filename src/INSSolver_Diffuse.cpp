/**
 * INSSolver_Diffuse.cpp
 *
 * Crank–Nicolson predictor (Perot 1997 form, no pressure):
 *
 *   tau^n = ν (grad(u^n) + grad(u^n)^T)
 *   r1 = u^n + (dt/2) div(tau^n) + dt A^n      ε = ν dt / 2
 *   u* = B^N r1                                B^N = Σ_{k=0}^N (εL)^k
 *
 * Convergence of the series requires ‖εL‖ < 1, i.e. ν dt / h² < 1/d.
 * The pressure gradient is intentionally NOT included here — under
 * Perot's block-LU factorisation the predictor has no pressure term;
 * the pressure enters only through the modified-Poisson projection.
 *
 * The face Laplacian and gradient helpers in this file are reused
 * by INSSolver_Project.cpp for applying B^N to G p and forming the
 * modified-Poisson operator D B^N G.
 */

#include "INSSolver.H"

#include <AMReX_Array4.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_MFIter.H>

using namespace amrex;

// ============================================================
//  ComputeViscousStress — tau = nu * (grad(u) + grad(u)^T)
// ============================================================
void INSSolver::ComputeViscousStress(
    int lev, const std::array<const MultiFab *, AMREX_SPACEDIM> &vel,
    StressMFArray &stress) {
  BL_PROFILE("INSSolver::ComputeViscousStress()");

  const auto inv_dx = geom[lev].InvCellSizeArray();

  for (int velocity_component = 0; velocity_component < AMREX_SPACEDIM;
       ++velocity_component) {
    for (int gradient_direction = velocity_component;
         gradient_direction < AMREX_SPACEDIM; ++gradient_direction) {
      MultiFab &tau = *stress[velocity_component][gradient_direction];
      const bool diagonal = velocity_component == gradient_direction;
      const Real gradient_coefficient = m_nu * inv_dx[gradient_direction];
      const Real transpose_coefficient = m_nu * inv_dx[velocity_component];
      const int gradient_i = gradient_direction == 0;
      const int gradient_j = gradient_direction == 1;
      const int gradient_k = gradient_direction == 2;
      const int transpose_i = velocity_component == 0;
      const int transpose_j = velocity_component == 1;
      const int transpose_k = velocity_component == 2;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
      for (MFIter mfi(tau); mfi.isValid(); ++mfi) {
        const Box &bx = mfi.fabbox();
        auto const &u = vel[velocity_component]->const_array(mfi);
        auto const &u_transpose = vel[gradient_direction]->const_array(mfi);
        auto const &t = tau.array(mfi);
        auto const &t_transpose =
            stress[gradient_direction][velocity_component]->array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j,
                                                    int k) noexcept {
          Real value;
          if (diagonal) {
            value = 2.0_rt * gradient_coefficient *
                    (u(i + gradient_i, j + gradient_j, k + gradient_k) -
                     u(i, j, k));
          } else {
            value = gradient_coefficient *
                        (u(i, j, k) -
                         u(i - gradient_i, j - gradient_j, k - gradient_k)) +
                    transpose_coefficient *
                        (u_transpose(i, j, k) - u_transpose(i - transpose_i,
                                                            j - transpose_j,
                                                            k - transpose_k));
          }
          t(i, j, k) = value;
          if (!diagonal)
            t_transpose(i, j, k) = value;
        });
      }
    }
  }
}

// ============================================================
//  ComputeStressDivergence — div(tau) on velocity-component faces
// ============================================================
void INSSolver::ComputeStressDivergence(int lev, int velocity_component,
                                        const StressMFArray &stress,
                                        MultiFab &div_tau) {
  BL_PROFILE("INSSolver::ComputeStressDivergence()");

  const auto inv_dx = geom[lev].InvCellSizeArray();
  const bool diagonal_x = velocity_component == 0;
  const bool diagonal_y = velocity_component == 1;
#if AMREX_SPACEDIM == 3
  const bool diagonal_z = velocity_component == 2;
#endif
  const IntVect nod = IntVect::TheDimensionVector(velocity_component);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
  for (MFIter mfi(div_tau, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    const Box &bx = mfi.tilebox(nod);
    auto const &tx = stress[velocity_component][0]->const_array(mfi);
    auto const &ty = stress[velocity_component][1]->const_array(mfi);
#if AMREX_SPACEDIM == 3
    auto const &tz = stress[velocity_component][2]->const_array(mfi);
#endif
    auto const &d = div_tau.array(mfi);

    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      Real value = inv_dx[0] * (diagonal_x ? tx(i, j, k) - tx(i - 1, j, k)
                                           : tx(i + 1, j, k) - tx(i, j, k)) +
                   inv_dx[1] * (diagonal_y ? ty(i, j, k) - ty(i, j - 1, k)
                                           : ty(i, j + 1, k) - ty(i, j, k));
#if AMREX_SPACEDIM == 3
      value += inv_dx[2] * (diagonal_z ? tz(i, j, k) - tz(i, j, k - 1)
                                       : tz(i, j, k + 1) - tz(i, j, k));
#endif
      d(i, j, k) = value;
    });
  }
}

void INSSolver::UpdateStoredStress(Real time) {
  BL_PROFILE("INSSolver::UpdateStoredStress()");

  for (int lev = 0; lev <= finest_level; ++lev) {
    const BoxArray &ba = grids[lev];
    const DistributionMapping &dm = dmap[lev];
    std::array<MultiFab, AMREX_SPACEDIM> velocity;
    std::array<const MultiFab *, AMREX_SPACEDIM> velocity_ptr;

    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
      BoxArray fba = amrex::convert(ba, IntVect::TheDimensionVector(d));
      velocity[d].define(fba, dm, 1, 2);
      FillFacePatch(lev, d, m_vel, velocity[d], time);
      velocity_ptr[d] = &velocity[d];
    }
    ComputeViscousStress(lev, velocity_ptr, m_stress[lev]);
  }
}

// ============================================================
//  ApplyFaceLaplacian — 2nd-order centred face Laplacian
// ============================================================
void INSSolver::ApplyFaceLaplacian(int lev, int dir, const MultiFab &fin,
                                   MultiFab &Lf) {
  BL_PROFILE("INSSolver::ApplyFaceLaplacian()");

  const Real *dx = geom[lev].CellSize();
  const Real dxi2 = 1.0_rt / (dx[0] * dx[0]);
  const Real dyi2 = 1.0_rt / (dx[1] * dx[1]);
#if AMREX_SPACEDIM == 3
  const Real dzi2 = 1.0_rt / (dx[2] * dx[2]);
#endif

  const IntVect nod = IntVect::TheDimensionVector(dir);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
  for (MFIter mfi(Lf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    const Box &bx = mfi.tilebox(nod);
    auto const &f = fin.const_array(mfi);
    auto const &Lfa = Lf.array(mfi);
    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      Real lap =
          dxi2 * (f(i + 1, j, k) - 2.0_rt * f(i, j, k) + f(i - 1, j, k)) +
          dyi2 * (f(i, j + 1, k) - 2.0_rt * f(i, j, k) + f(i, j - 1, k));
#if AMREX_SPACEDIM == 3
      lap += dzi2 * (f(i, j, k + 1) - 2.0_rt * f(i, j, k) + f(i, j, k - 1));
#endif
      Lfa(i, j, k) = lap;
    });
  }
}

// ============================================================
//  ComputePressureGradient — (∂p/∂x_dir) on dir-aligned faces
// ============================================================
void INSSolver::ComputePressureGradient(int lev, int dir, const MultiFab &p,
                                        MultiFab &gp) {
  BL_PROFILE("INSSolver::ComputePressureGradient()");

  const Real idx = 1.0_rt / geom[lev].CellSize(dir);
  const IntVect nod = IntVect::TheDimensionVector(dir);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
  for (MFIter mfi(gp, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    const Box &bx = mfi.tilebox(nod);
    auto const &pa = p.const_array(mfi);
    auto const &gpa = gp.array(mfi);
    if (dir == 0) {
      amrex::ParallelFor(bx,
                         [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                           gpa(i, j, k) = idx * (pa(i, j, k) - pa(i - 1, j, k));
                         });
    } else if (dir == 1) {
      amrex::ParallelFor(bx,
                         [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                           gpa(i, j, k) = idx * (pa(i, j, k) - pa(i, j - 1, k));
                         });
#if AMREX_SPACEDIM == 3
    } else {
      amrex::ParallelFor(bx,
                         [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                           gpa(i, j, k) = idx * (pa(i, j, k) - pa(i, j, k - 1));
                         });
#endif
    }
  }
}

// ============================================================
//  ApplyBNFace — B^N = Σ_{k=0}^N (εL)^k applied to a face MF
//
//  Iterative form:
//    term ← src                        (k = 0)
//    dst  ← src
//    for k = 1 .. N:
//      term ← ε L term
//      dst  ← dst + term
//
//  Only valid values are copied from `src`.  The iterated k>=1 work
//  term fills homogeneous physical/intra-level ghosts before each L.
//  This preserves pressure-gradient boundary faces in the k=0 term
//  while keeping the Neumann-series terms tied to the homogeneous
//  velocity operator.
// ============================================================
void INSSolver::ApplyBNFace(int lev, int dir, const MultiFab &src,
                            MultiFab &dst) {
  BL_PROFILE("INSSolver::ApplyBNFace()");

  const Real eps = 0.5_rt * m_nu * m_dt;
  const int N = m_cn_order;

  const BoxArray fba = src.boxArray();
  const DistributionMapping dm = src.DistributionMap();

  // dst ← src   (k=0 term)
  MultiFab::Copy(dst, src, 0, 0, 1, 0);
  if (N == 0)
    return;

  // term holds (εL)^k src as we iterate.  The iterated terms use
  // *homogeneous* physical BCs — for the truncated Neumann series the
  // inhomogeneous wall data is re-imposed on u* afterwards by
  // EnforceVelDirichlet (standard treatment for low truncation order).
  MultiFab term(fba, dm, 1, 2);
  term.setVal(0.0); // C/F ghosts deterministically 0 (per-level approx)
  MultiFab::Copy(term, src, 0, 0, 1, 0);
  FillVelGhostPhys(lev, dir, term, /*homogeneous=*/true);

  MultiFab Lterm(fba, dm, 1, 0);
  for (int k = 1; k <= N; ++k) {
    ApplyFaceLaplacian(lev, dir, term, Lterm);
    // term ← ε * Lterm
    MultiFab::LinComb(term, 0.0_rt, term, 0, eps, Lterm, 0, 0, 1, 0);
    FillVelGhostPhys(lev, dir, term, /*homogeneous=*/true);
    MultiFab::Add(dst, term, 0, 0, 1, 0);
  }
}

// ============================================================
//  ApplyCompositeBNFaces — hierarchy B^N with C/F-filled work terms
// ============================================================
void INSSolver::ApplyCompositeBNFaces(Vector<FaceMFArray> &src,
                                      Vector<FaceMFArray> &dst) {
  BL_PROFILE("INSSolver::ApplyCompositeBNFaces()");

  const int nlev = finest_level + 1;
  const Real eps = 0.5_rt * m_nu * m_dt;

  Vector<FaceMFArray> term(nlev);
  Vector<FaceMFArray> lap_term(nlev);
  for (int lev = 0; lev < nlev; ++lev) {
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
      const BoxArray &fba = src[lev][d]->boxArray();
      const DistributionMapping &dm = src[lev][d]->DistributionMap();
      term[lev][d] = std::make_unique<MultiFab>(fba, dm, 1, 1);
      lap_term[lev][d] = std::make_unique<MultiFab>(fba, dm, 1, 0);
      term[lev][d]->setVal(0.0);
      MultiFab::Copy(*term[lev][d], *src[lev][d], 0, 0, 1, 0);
      MultiFab::Copy(*dst[lev][d], *src[lev][d], 0, 0, 1, 0);
    }
  }

  AverageDownVelocity(term);
  AverageDownVelocity(dst);

  for (int k = 1; k <= m_cn_order; ++k) {
    for (int lev = 0; lev < nlev; ++lev) {
      for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        const BoxArray &fba = term[lev][d]->boxArray();
        const DistributionMapping &dm = term[lev][d]->DistributionMap();
        MultiFab term_g(fba, dm, 1, 1);
        FillFacePatch(lev, d, term, term_g, m_cur_time,
                      /*homogeneous=*/true);
        ApplyFaceLaplacian(lev, d, term_g, *lap_term[lev][d]);
      }
    }

    for (int lev = 0; lev < nlev; ++lev) {
      for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        MultiFab::LinComb(*term[lev][d], 0.0_rt, *term[lev][d], 0, eps,
                          *lap_term[lev][d], 0, 0, 1, 0);
        MultiFab::Add(*dst[lev][d], *term[lev][d], 0, 0, 1, 0);
      }
    }
    AverageDownVelocity(term);
  }

  AverageDownVelocity(dst);
}

// ============================================================
//  BuildCNPredictorRHS — u^n + (dt/2) div(tau^n) + dt A^n
// ============================================================
void INSSolver::BuildCNPredictorRHS(
    int lev, const std::array<MultiFab *, AMREX_SPACEDIM> &rhs,
    const std::array<const MultiFab *, AMREX_SPACEDIM> &u_n,
    const std::array<const MultiFab *, AMREX_SPACEDIM> &adv,
    const StressMFArray &stress) {
  BL_PROFILE("INSSolver::BuildCNPredictorRHS()");

  const BoxArray &ba = grids[lev];
  const DistributionMapping &dm = dmap[lev];

  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    BoxArray fba = amrex::convert(ba, IntVect::TheDimensionVector(d));

    // F = u^n + (dt/2) div(tau^n) + dt A^n
    MultiFab &F = *rhs[d];
    MultiFab::Copy(F, *u_n[d], 0, 0, 1, 2); // u_n carries 2 ghosts

    MultiFab div_tau(fba, dm, 1, 0);
    ComputeStressDivergence(lev, d, stress, div_tau);
    MultiFab::Saxpy(F, 0.5_rt * m_dt, div_tau, 0, 0, 1, 0);

    MultiFab::Saxpy(F, m_dt, *adv[d], 0, 0, 1, 0); // F += dt A^n

    FillVelGhostPhys(lev, d, F, /*homogeneous=*/false);
  }
}
