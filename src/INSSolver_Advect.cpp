/**
 * INSSolver_Advect.cpp
 *
 * Explicit centred-difference advection on the staggered (MAC) grid:
 *
 *   A_u = -(u ∂u/∂x + v ∂u/∂y + w ∂u/∂z)  on x-faces
 *   A_v = -(u ∂v/∂x + v ∂v/∂y + w ∂v/∂z)  on y-faces
 *   A_w = -(u ∂w/∂x + v ∂w/∂y + w ∂w/∂z)  on z-faces
 *
 * Cross-velocity components are averaged from the four surrounding faces
 * to the current face location.  All derivatives are second-order centred.
 *
 * `vel_in` is the input velocity with 2 ghost cells already FillPatched
 * (intra-level halos + C/F interpolation from coarse).
 */

#include "INSSolver.H"

#include <AMReX_Array4.H>
#include <AMReX_MFIter.H>

using namespace amrex;

void INSSolver::ComputeAdvection(
    int lev, const std::array<MultiFab *, AMREX_SPACEDIM> &adv,
    const std::array<const MultiFab *, AMREX_SPACEDIM> &vel_in) {
  const Real *dx = geom[lev].CellSize();
  const Real dxi = 1.0_rt / dx[0];
  const Real dyi = 1.0_rt / dx[1];
#if AMREX_SPACEDIM == 3
  const Real dzi = 1.0_rt / dx[2];
#endif

  const MultiFab &U = *vel_in[0];
  const MultiFab &V = *vel_in[1];
#if AMREX_SPACEDIM == 3
  const MultiFab &W = *vel_in[2];
#endif

  MultiFab &Au = *adv[0];
  MultiFab &Av = *adv[1];
#if AMREX_SPACEDIM == 3
  MultiFab &Aw = *adv[2];
#endif

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
  for (MFIter mfi(Au, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    auto const &u = U.const_array(mfi);
    auto const &v = V.const_array(mfi);
#if AMREX_SPACEDIM == 3
    auto const &w = W.const_array(mfi);
#endif

    // --- x-face advection ---
    {
      const Box &bx = mfi.tilebox(IntVect::TheDimensionVector(0));
      auto const &au = Au.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        Real dudx = 0.5_rt * dxi * (u(i + 1, j, k) - u(i - 1, j, k));
        Real v_at_u = 0.25_rt * (v(i - 1, j, k) + v(i, j, k) +
                                 v(i - 1, j + 1, k) + v(i, j + 1, k));
        Real dudy = 0.5_rt * dyi * (u(i, j + 1, k) - u(i, j - 1, k));
#if AMREX_SPACEDIM == 3
        Real w_at_u = 0.25_rt * (w(i - 1, j, k) + w(i, j, k) +
                                 w(i - 1, j, k + 1) + w(i, j, k + 1));
        Real dudz = 0.5_rt * dzi * (u(i, j, k + 1) - u(i, j, k - 1));
        au(i, j, k) =
            -(u(i, j, k) * dudx + v_at_u * dudy + w_at_u * dudz);
#else
        au(i, j, k) = -(u(i, j, k) * dudx + v_at_u * dudy);
#endif
      });
    }

    // --- y-face advection ---
    {
      const Box &bx = mfi.tilebox(IntVect::TheDimensionVector(1));
      auto const &av = Av.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        Real u_at_v = 0.25_rt * (u(i, j - 1, k) + u(i + 1, j - 1, k) +
                                 u(i, j, k) + u(i + 1, j, k));
        Real dvdx = 0.5_rt * dxi * (v(i + 1, j, k) - v(i - 1, j, k));
        Real dvdy = 0.5_rt * dyi * (v(i, j + 1, k) - v(i, j - 1, k));
#if AMREX_SPACEDIM == 3
        Real w_at_v = 0.25_rt * (w(i, j - 1, k) + w(i, j, k) +
                                 w(i, j - 1, k + 1) + w(i, j, k + 1));
        Real dvdz = 0.5_rt * dzi * (v(i, j, k + 1) - v(i, j, k - 1));
        av(i, j, k) =
            -(u_at_v * dvdx + v(i, j, k) * dvdy + w_at_v * dvdz);
#else
        av(i, j, k) = -(u_at_v * dvdx + v(i, j, k) * dvdy);
#endif
      });
    }

#if AMREX_SPACEDIM == 3
    // --- z-face advection ---
    {
      const Box &bx = mfi.tilebox(IntVect::TheDimensionVector(2));
      auto const &aw = Aw.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        Real u_at_w = 0.25_rt * (u(i, j, k - 1) + u(i + 1, j, k - 1) +
                                 u(i, j, k) + u(i + 1, j, k));
        Real dwdx = 0.5_rt * dxi * (w(i + 1, j, k) - w(i - 1, j, k));
        Real v_at_w = 0.25_rt * (v(i, j, k - 1) + v(i, j + 1, k - 1) +
                                 v(i, j, k) + v(i, j + 1, k));
        Real dwdy = 0.5_rt * dyi * (w(i, j + 1, k) - w(i, j - 1, k));
        Real dwdz = 0.5_rt * dzi * (w(i, j, k + 1) - w(i, j, k - 1));
        aw(i, j, k) =
            -(u_at_w * dwdx + v_at_w * dwdy + w(i, j, k) * dwdz);
      });
    }
#endif
  }
}
