/**
 * INSSolver_Diffuse.cpp
 *
 * Crank–Nicolson predictor (Perot 1997 form, no pressure):
 *
 *   r1 = (I + εL) u^n  +  dt A^n               ε = ν dt / 2
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
#include <AMReX_MFIter.H>

using namespace amrex;

// ============================================================
//  ApplyFaceLaplacian — 2nd-order centred face Laplacian
// ============================================================
void INSSolver::ApplyFaceLaplacian(int lev, int dir, const MultiFab &fin,
                                   MultiFab &Lf) {
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
      Real lap = dxi2 * (f(i + 1, j, k) - 2.0_rt * f(i, j, k) + f(i - 1, j, k)) +
                 dyi2 * (f(i, j + 1, k) - 2.0_rt * f(i, j, k) + f(i, j - 1, k));
#if AMREX_SPACEDIM == 3
      lap +=
          dzi2 * (f(i, j, k + 1) - 2.0_rt * f(i, j, k) + f(i, j, k - 1));
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
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        gpa(i, j, k) = idx * (pa(i, j, k) - pa(i - 1, j, k));
      });
    } else if (dir == 1) {
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        gpa(i, j, k) = idx * (pa(i, j, k) - pa(i, j - 1, k));
      });
#if AMREX_SPACEDIM == 3
    } else {
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
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
//  Caller must ensure `src` has its 2-ghost layer filled.  After each
//  L application we FillBoundary(periodicity) so the next L sees valid
//  surroundings — single-level only; multi-level intermediate terms
//  use FillBoundary as an approximation (see header).
// ============================================================
void INSSolver::ApplyBNFace(int lev, int dir, const MultiFab &src,
                            MultiFab &dst) {
  const Real eps = 0.5_rt * m_nu * m_dt;
  const int N = m_cn_order;

  const BoxArray fba = src.boxArray();
  const DistributionMapping dm = src.DistributionMap();

  // dst ← src   (k=0 term)
  MultiFab::Copy(dst, src, 0, 0, 1, 0);
  if (N == 0)
    return;

  // term holds (εL)^k src as we iterate
  MultiFab term(fba, dm, 1, 2);
  MultiFab::Copy(term, src, 0, 0, 1, std::min(src.nGrow(), 2));
  // (src ghosts are caller's responsibility; copy then refresh just in case)
  term.FillBoundary(geom[lev].periodicity());

  MultiFab Lterm(fba, dm, 1, 0);
  for (int k = 1; k <= N; ++k) {
    ApplyFaceLaplacian(lev, dir, term, Lterm);
    // term ← ε * Lterm
    MultiFab::LinComb(term, 0.0_rt, term, 0, eps, Lterm, 0, 0, 1, 0);
    term.FillBoundary(geom[lev].periodicity());
    MultiFab::Add(dst, term, 0, 0, 1, 0);
  }
}

// ============================================================
//  ApplyCNDiffusion — Perot predictor:  u* = B^N [(I + εL) u^n + dt A^n]
// ============================================================
void INSSolver::ApplyCNDiffusion(
    int lev, const std::array<MultiFab *, AMREX_SPACEDIM> &vstar,
    const std::array<const MultiFab *, AMREX_SPACEDIM> &u_n,
    const std::array<const MultiFab *, AMREX_SPACEDIM> &adv) {
  const Real eps = 0.5_rt * m_nu * m_dt;

  const BoxArray &ba = grids[lev];
  const DistributionMapping &dm = dmap[lev];

  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    BoxArray fba = amrex::convert(ba, IntVect::TheDimensionVector(d));

    // F = (I + εL) u^n + dt A^n   (no pressure gradient — Perot)
    MultiFab F(fba, dm, 1, 2);
    MultiFab::Copy(F, *u_n[d], 0, 0, 1, 2); // u_n carries 2 ghosts

    MultiFab LuN(fba, dm, 1, 0);
    ApplyFaceLaplacian(lev, d, F, LuN);
    MultiFab::Saxpy(F, eps, LuN, 0, 0, 1, 0); // F += εL u^n

    MultiFab::Saxpy(F, m_dt, *adv[d], 0, 0, 1, 0); // F += dt A^n

    F.FillBoundary(geom[lev].periodicity());

    // u* = B^N F
    ApplyBNFace(lev, d, F, *vstar[d]);
  }
}
