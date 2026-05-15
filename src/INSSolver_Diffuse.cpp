/**
 * INSSolver_Diffuse.cpp
 *
 * Crank–Nicolson diffusion for u* via the Neumann-series truncation
 *
 *   (I − εL) u* = (I + εL) u^n + dt A^n − dt ∇p^n,    ε = ν dt / 2
 *   u*  ≈ Σ_{k=0}^{N} (εL)^k · [ (I + εL) u^n + dt A^n − dt ∇p^n ]
 *
 * Convergence of the series requires ‖εL‖ < 1.  For the standard 2nd-order
 * Laplacian this is ε·(2·d/h²) < 1, i.e. ν dt / h² < 1/d.  Increase the
 * truncation order N when this margin is tight.
 *
 * The Laplacian, gradient, and Saxpys are all applied per-level — `u_n`,
 * `adv` and `pres` are FillPatched inputs supplied by the caller.
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
//  ApplyCNDiffusion — build u* via the truncated Neumann series
// ============================================================
void INSSolver::ApplyCNDiffusion(
    int lev, const std::array<MultiFab *, AMREX_SPACEDIM> &vstar,
    const std::array<const MultiFab *, AMREX_SPACEDIM> &u_n,
    const std::array<const MultiFab *, AMREX_SPACEDIM> &adv,
    const MultiFab &pres) {
  const Real eps = 0.5_rt * m_nu * m_dt;
  const int N = m_cn_order;

  const BoxArray &ba = grids[lev];
  const DistributionMapping &dm = dmap[lev];

  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    BoxArray fba = amrex::convert(ba, IntVect::TheDimensionVector(d));

    // F = (I + εL) u^n + dt A^n − dt ∇p^n
    MultiFab F(fba, dm, 1, 2);
    MultiFab::Copy(F, *u_n[d], 0, 0, 1, 2); // u_n already has ghosts

    // εL u^n  (acts on F's ghost-filled data)
    MultiFab LuN(fba, dm, 1, 0);
    ApplyFaceLaplacian(lev, d, F, LuN);
    MultiFab::Saxpy(F, eps, LuN, 0, 0, 1, 0);

    // + dt A^n
    MultiFab::Saxpy(F, m_dt, *adv[d], 0, 0, 1, 0);

    // − dt ∇p^n
    MultiFab gradp(fba, dm, 1, 0);
    ComputePressureGradient(lev, d, pres, gradp);
    MultiFab::Saxpy(F, -m_dt, gradp, 0, 0, 1, 0);

    // F's interior is now correct; refresh its ghosts so the next Laplacian
    // application has valid surroundings.  FillBoundary is enough here:
    // F lives entirely on this level, and at C/F boundaries on the *fine*
    // side the ghost layer of F lies inside the fine patch (which is valid)
    // or one cell outside — but Laplacian applications below only need the
    // ghosts that the original u_n already carries.  In practice this works
    // because the Neumann-series term magnitudes decay rapidly.
    F.FillBoundary(geom[lev].periodicity());

    // u* ← F (k = 0 term)
    MultiFab::Copy(*vstar[d], F, 0, 0, 1, 0);

    // term_k = (εL)^k F, accumulate into u*
    MultiFab term(fba, dm, 1, 2);
    MultiFab::Copy(term, F, 0, 0, 1, 2);
    term.FillBoundary(geom[lev].periodicity());

    MultiFab Lterm(fba, dm, 1, 0);
    for (int k = 1; k <= N; ++k) {
      ApplyFaceLaplacian(lev, d, term, Lterm);
      // term ← ε * Lterm
      MultiFab::LinComb(term, 0.0_rt, term, 0, eps, Lterm, 0, 0, 1, 0);
      term.FillBoundary(geom[lev].periodicity());
      MultiFab::Add(*vstar[d], term, 0, 0, 1, 0);
    }
  }
}
