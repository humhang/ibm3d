/**
 * INSSolver_IB.cpp
 *
 * Immersed-boundary pieces for the Taira-Colonius projection method.
 *
 * Current scope: prescribed rigid/stationary IB velocity, one Lagrangian
 * marker at each geometry element centroid, and finest-level-only coupling.
 */

#include "INSSolver.H"

#include <AMReX.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_Math.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>

using namespace amrex;

namespace {

struct Peskin4Kernel {
  static constexpr int support_width = 4;
  static constexpr Real support_radius = Real(0.5) * support_width;

  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static Real phi(Real r) noexcept {
    r = std::abs(r);
    if (r < 1.0_rt) {
      return 0.125_rt * (3.0_rt - 2.0_rt * r +
                         std::sqrt(1.0_rt + 4.0_rt * r - 4.0_rt * r * r));
    }
    if (r < support_radius) {
      return 0.125_rt * (5.0_rt - 2.0_rt * r -
                         std::sqrt(-7.0_rt + 12.0_rt * r - 4.0_rt * r * r));
    }
    return 0.0_rt;
  }

  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static Real
  periodic_displacement(Real x, Real x0, Real length, int periodic) noexcept {
    Real r = x - x0;
    if (periodic) {
      if (r > 0.5_rt * length)
        r -= length;
      if (r < -0.5_rt * length)
        r += length;
    }
    return r;
  }

  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static Real
  face_offset(int dir, int d) noexcept {
    return dir == d ? 0.0_rt : 0.5_rt;
  }

  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static Real
  face_coordinate(int dir, int d, int index,
                  const GpuArray<Real, AMREX_SPACEDIM> &plo,
                  const GpuArray<Real, AMREX_SPACEDIM> &dx) noexcept {
    return plo[d] + (index + face_offset(dir, d)) * dx[d];
  }

  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void
  support_bounds(Real marker_coord, int dir, int d,
                 const GpuArray<Real, AMREX_SPACEDIM> &plo,
                 const GpuArray<Real, AMREX_SPACEDIM> &dx, int &lo,
                 int &hi) noexcept {
    const Real center = (marker_coord - plo[d]) / dx[d] - face_offset(dir, d);
    lo = static_cast<int>(amrex::Math::floor(center - support_radius));
    hi = static_cast<int>(amrex::Math::floor(center + support_radius));
  }

  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static Real
  delta(int dir, int i, int j, int k,
        const GpuArray<Real, AMREX_SPACEDIM> &marker_x,
        const GpuArray<Real, AMREX_SPACEDIM> &plo,
        const GpuArray<Real, AMREX_SPACEDIM> &dx) noexcept {
    const int index[AMREX_SPACEDIM] = {AMREX_D_DECL(i, j, k)};
    Real value = 1.0_rt;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
      const Real x = face_coordinate(dir, d, index[d], plo, dx);
      const Real ph = phi((x - marker_x[d]) / dx[d]);
      if (ph == 0.0_rt)
        return 0.0_rt;
      value *= ph / dx[d];
    }
    return value;
  }

  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static GpuArray<Real, AMREX_SPACEDIM>
  face_position(int dir, int i, int j, int k,
                const GpuArray<Real, AMREX_SPACEDIM> &plo,
                const GpuArray<Real, AMREX_SPACEDIM> &dx) noexcept {
    GpuArray<Real, AMREX_SPACEDIM> x{};
    x[0] = face_coordinate(dir, 0, i, plo, dx);
    x[1] = face_coordinate(dir, 1, j, plo, dx);
#if AMREX_SPACEDIM == 3
    x[2] = face_coordinate(dir, 2, k, plo, dx);
#endif
    return x;
  }
};

Real vector_dot(const std::vector<Real> &a, const std::vector<Real> &b) {
  return std::inner_product(a.begin(), a.end(), b.begin(), Real(0.0));
}

void vector_lincomb(std::vector<Real> &dst, Real a, const std::vector<Real> &x,
                    Real b, const std::vector<Real> &y) {
  for (std::size_t i = 0; i < dst.size(); ++i)
    dst[i] = a * x[i] + b * y[i];
}

void vector_saxpy(std::vector<Real> &dst, Real a, const std::vector<Real> &x) {
  for (std::size_t i = 0; i < dst.size(); ++i)
    dst[i] += a * x[i];
}

Real coupled_dot(const MultiFab &a_p, const std::vector<Real> &a_ib,
                 const MultiFab &b_p, const std::vector<Real> &b_ib) {
  return MultiFab::Dot(a_p, 0, b_p, 0, 1, 0) + vector_dot(a_ib, b_ib);
}

} // namespace

void INSSolver::InitializeIBGeometry() {
  BL_PROFILE("INSSolver::InitializeIBGeometry()");

  if (!m_ib_enabled)
    return;

  if (m_ib_geometry_file.empty())
    amrex::Abort("ib.enabled=1 requires ib.geometry or ib.geom_file");

  m_ib_geometry = ibm3d::LoadIBGeometry(m_ib_geometry_file);
  m_ib_geometry.UploadToDevice();

  m_ib_force.assign(m_ib_geometry.markers.size() * AMREX_SPACEDIM, 0.0);

  if (m_verbose > 0) {
    Print() << "Immersed boundary enabled\n"
            << "  geometry    = " << m_ib_geometry_file << "\n"
            << "  points      = " << m_ib_geometry.points.size() << "\n"
            << "  elements    = " << m_ib_geometry.elements.size() << "\n"
            << "  markers     = " << m_ib_geometry.markers.size() << "\n";
  }
}

void INSSolver::SpreadIBForce(int lev, int dir, const std::vector<Real> &force,
                              MultiFab &hforce) {
  BL_PROFILE("INSSolver::SpreadIBForce()");

  hforce.setVal(0.0);
  if (!m_ib_enabled || m_ib_geometry.markers.empty())
    return;

  Gpu::DeviceVector<Real> force_device(force.size());
  Gpu::copy(Gpu::hostToDevice, force.begin(), force.end(),
            force_device.begin());

  const auto *markers = m_ib_geometry.device_markers.data();
  const auto *f = force_device.data();
  const int nmarkers = static_cast<int>(m_ib_geometry.markers.size());

  const Geometry &gm = geom[lev];
  const auto dx = gm.CellSizeArray();
  const auto plo = gm.ProbLoArray();
  const auto phi = gm.ProbHiArray();
  GpuArray<Real, AMREX_SPACEDIM> prob_len{};
  GpuArray<int, AMREX_SPACEDIM> periodic{};
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    prob_len[d] = phi[d] - plo[d];
    periodic[d] = gm.isPeriodic(d);
  }

  const IntVect nod = IntVect::TheDimensionVector(dir);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
  for (MFIter mfi(hforce, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    const Box &bx = mfi.tilebox(nod);
    auto const &h = hforce.array(mfi);
    const int bxlo0 = bx.smallEnd(0);
    const int bxhi0 = bx.bigEnd(0);
    const int bxlo1 = bx.smallEnd(1);
    const int bxhi1 = bx.bigEnd(1);
#if AMREX_SPACEDIM == 3
    const int bxlo2 = bx.smallEnd(2);
    const int bxhi2 = bx.bigEnd(2);
#endif
    amrex::ParallelFor(nmarkers, [=] AMREX_GPU_DEVICE(int marker) noexcept {
      const auto marker_data = markers[marker];
      const Real marker_force =
          f[marker * AMREX_SPACEDIM + dir] * marker_data.weight;
      if (marker_force == 0.0_rt)
        return;

      const int sx_begin = periodic[0] ? -1 : 0;
      const int sx_end = periodic[0] ? 1 : 0;
      const int sy_begin = periodic[1] ? -1 : 0;
      const int sy_end = periodic[1] ? 1 : 0;
#if AMREX_SPACEDIM == 3
      const int sz_begin = periodic[2] ? -1 : 0;
      const int sz_end = periodic[2] ? 1 : 0;
#endif

      for (int sx = sx_begin; sx <= sx_end; ++sx) {
        for (int sy = sy_begin; sy <= sy_end; ++sy) {
#if AMREX_SPACEDIM == 3
          for (int sz = sz_begin; sz <= sz_end; ++sz) {
#endif
            GpuArray<Real, AMREX_SPACEDIM> marker_x = marker_data.x;
            marker_x[0] += static_cast<Real>(sx) * prob_len[0];
            marker_x[1] += static_cast<Real>(sy) * prob_len[1];
#if AMREX_SPACEDIM == 3
            marker_x[2] += static_cast<Real>(sz) * prob_len[2];
#endif

            int ilo, ihi, jlo, jhi;
            Peskin4Kernel::support_bounds(marker_x[0], dir, 0, plo, dx, ilo,
                                          ihi);
            Peskin4Kernel::support_bounds(marker_x[1], dir, 1, plo, dx, jlo,
                                          jhi);
            const int ii0 = (ilo > bxlo0) ? ilo : bxlo0;
            const int ii1 = (ihi < bxhi0) ? ihi : bxhi0;
            const int jj0 = (jlo > bxlo1) ? jlo : bxlo1;
            const int jj1 = (jhi < bxhi1) ? jhi : bxhi1;
            if (ii0 > ii1 || jj0 > jj1)
              continue;

            int kk0 = 0;
            int kk1 = 0;
#if AMREX_SPACEDIM == 3
            int klo, khi;
            Peskin4Kernel::support_bounds(marker_x[2], dir, 2, plo, dx, klo,
                                          khi);
            kk0 = (klo > bxlo2) ? klo : bxlo2;
            kk1 = (khi < bxhi2) ? khi : bxhi2;
            if (kk0 > kk1)
              continue;
#endif
            for (int k = kk0; k <= kk1; ++k) {
              for (int j = jj0; j <= jj1; ++j) {
                for (int i = ii0; i <= ii1; ++i) {
                  const Real delta =
                      Peskin4Kernel::delta(dir, i, j, k, marker_x, plo, dx);
                  if (delta != 0.0_rt)
                    HostDevice::Atomic::Add(&h(i, j, k), marker_force * delta);
                }
              }
            }
#if AMREX_SPACEDIM == 3
          }
#endif
        }
      }
    });
  }
  Gpu::streamSynchronize();
}

void INSSolver::InterpolateIBVelocity(
    int lev, const std::array<const MultiFab *, AMREX_SPACEDIM> &vel,
    std::vector<Real> &marker_vel) const {
  BL_PROFILE("INSSolver::InterpolateIBVelocity()");

  const auto &markers = m_ib_geometry.markers;
  marker_vel.assign(markers.size() * AMREX_SPACEDIM, 0.0);
  if (!m_ib_enabled || markers.empty())
    return;

  const auto *markers_p = m_ib_geometry.device_markers.data();
  const int nmarkers = static_cast<int>(markers.size());
  Gpu::DeviceVector<Real> marker_vel_device(marker_vel.size());
  Gpu::copy(Gpu::hostToDevice, marker_vel.begin(), marker_vel.end(),
            marker_vel_device.begin());
  Real *marker_vel_p = marker_vel_device.data();

  const Geometry &gm = geom[lev];
  const auto dx = gm.CellSizeArray();
  const auto plo = gm.ProbLoArray();
  const auto phi = gm.ProbHiArray();
  GpuArray<Real, AMREX_SPACEDIM> prob_len{};
  GpuArray<int, AMREX_SPACEDIM> periodic{};
  Real dv = 1.0_rt;
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    prob_len[d] = phi[d] - plo[d];
    periodic[d] = gm.isPeriodic(d);
    dv *= dx[d];
  }

  for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
    const MultiFab &mf = *vel[dir];
    const auto owner = mf.OwnerMask(gm.periodicity());

    for (MFIter mfi(mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const Box &bx = mfi.tilebox();
      auto const &u = mf.const_array(mfi);
      auto const &mask = owner->const_array(mfi);
      const int bxlo0 = bx.smallEnd(0);
      const int bxhi0 = bx.bigEnd(0);
      const int bxlo1 = bx.smallEnd(1);
      const int bxhi1 = bx.bigEnd(1);
#if AMREX_SPACEDIM == 3
      const int bxlo2 = bx.smallEnd(2);
      const int bxhi2 = bx.bigEnd(2);
#endif
      amrex::ParallelFor(nmarkers, [=] AMREX_GPU_DEVICE(int marker) noexcept {
        const auto marker_data = markers_p[marker];
        Real marker_sum = 0.0_rt;

        const int sx_begin = periodic[0] ? -1 : 0;
        const int sx_end = periodic[0] ? 1 : 0;
        const int sy_begin = periodic[1] ? -1 : 0;
        const int sy_end = periodic[1] ? 1 : 0;
#if AMREX_SPACEDIM == 3
        const int sz_begin = periodic[2] ? -1 : 0;
        const int sz_end = periodic[2] ? 1 : 0;
#endif

        for (int sx = sx_begin; sx <= sx_end; ++sx) {
          for (int sy = sy_begin; sy <= sy_end; ++sy) {
#if AMREX_SPACEDIM == 3
            for (int sz = sz_begin; sz <= sz_end; ++sz) {
#endif
              GpuArray<Real, AMREX_SPACEDIM> marker_x = marker_data.x;
              marker_x[0] += static_cast<Real>(sx) * prob_len[0];
              marker_x[1] += static_cast<Real>(sy) * prob_len[1];
#if AMREX_SPACEDIM == 3
              marker_x[2] += static_cast<Real>(sz) * prob_len[2];
#endif

              int ilo, ihi, jlo, jhi;
              Peskin4Kernel::support_bounds(marker_x[0], dir, 0, plo, dx, ilo,
                                            ihi);
              Peskin4Kernel::support_bounds(marker_x[1], dir, 1, plo, dx, jlo,
                                            jhi);
              const int ii0 = (ilo > bxlo0) ? ilo : bxlo0;
              const int ii1 = (ihi < bxhi0) ? ihi : bxhi0;
              const int jj0 = (jlo > bxlo1) ? jlo : bxlo1;
              const int jj1 = (jhi < bxhi1) ? jhi : bxhi1;
              if (ii0 > ii1 || jj0 > jj1)
                continue;

              int kk0 = 0;
              int kk1 = 0;
#if AMREX_SPACEDIM == 3
              int klo, khi;
              Peskin4Kernel::support_bounds(marker_x[2], dir, 2, plo, dx, klo,
                                            khi);
              kk0 = (klo > bxlo2) ? klo : bxlo2;
              kk1 = (khi < bxhi2) ? khi : bxhi2;
              if (kk0 > kk1)
                continue;
#endif
              for (int k = kk0; k <= kk1; ++k) {
                for (int j = jj0; j <= jj1; ++j) {
                  for (int i = ii0; i <= ii1; ++i) {
                    if (mask(i, j, k) == 0)
                      continue;
                    const Real delta =
                        Peskin4Kernel::delta(dir, i, j, k, marker_x, plo, dx);
                    marker_sum += u(i, j, k) * delta * dv;
                  }
                }
              }
#if AMREX_SPACEDIM == 3
            }
#endif
          }
        }
        if (marker_sum != 0.0_rt) {
          HostDevice::Atomic::Add(&marker_vel_p[marker * AMREX_SPACEDIM + dir],
                                  marker_sum);
        }
      });
    }
    Gpu::streamSynchronize();
  }

  Gpu::copy(Gpu::deviceToHost, marker_vel_device.begin(),
            marker_vel_device.end(), marker_vel.begin());
  Gpu::streamSynchronize();

  if (!marker_vel.empty()) {
    ParallelDescriptor::ReduceRealSum(marker_vel.data(),
                                      static_cast<int>(marker_vel.size()));
  }
}

void INSSolver::ApplyIBSchurOp(int lev, const MultiFab &phi,
                               const std::vector<Real> &force,
                               MultiFab &result_p,
                               std::vector<Real> &result_ib) {
  BL_PROFILE("INSSolver::ApplyIBSchurOp()");

  const BoxArray &ba = grids[lev];
  const DistributionMapping &dm = dmap[lev];

  std::array<MultiFab, AMREX_SPACEDIM> bq;
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    BoxArray fba = amrex::convert(ba, IntVect::TheDimensionVector(d));
    MultiFab q(fba, dm, 1, 2);
    MultiFab h(fba, dm, 1, 0);
    bq[d].define(fba, dm, 1, 0);

    q.setVal(0.0);
    ComputePressureGradient(lev, d, phi, q);
    SpreadIBForce(lev, d, force, h);
    MultiFab::Add(q, h, 0, 0, 1, 0);

    ApplyBNFace(lev, d, q, bq[d]);
  }

  const Real *dx = geom[lev].CellSize();
  const Real cx = 1.0_rt / dx[0];
  const Real cy = 1.0_rt / dx[1];
#if AMREX_SPACEDIM == 3
  const Real cz = 1.0_rt / dx[2];
#endif

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
  for (MFIter mfi(result_p, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    const Box &bx = mfi.tilebox();
    auto const &u = bq[0].const_array(mfi);
    auto const &v = bq[1].const_array(mfi);
#if AMREX_SPACEDIM == 3
    auto const &w = bq[2].const_array(mfi);
#endif
    auto const &r = result_p.array(mfi);
    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      Real d = cx * (u(i + 1, j, k) - u(i, j, k)) +
               cy * (v(i, j + 1, k) - v(i, j, k));
#if AMREX_SPACEDIM == 3
      d += cz * (w(i, j, k + 1) - w(i, j, k));
#endif
      r(i, j, k) = -d;
    });
  }

  InterpolateIBVelocity(lev, {AMREX_D_DECL(&bq[0], &bq[1], &bq[2])}, result_ib);
}

int INSSolver::SolveIBProjection(int lev, MultiFab &p, const MultiFab &rhs_p_in,
                                 const std::vector<Real> &rhs_ib) {
  BL_PROFILE("INSSolver::SolveIBProjection()");

  const BoxArray &ba = grids[lev];
  const DistributionMapping &dm = dmap[lev];

  const bool level_singular =
      m_pressure_singular &&
      (grids[lev].numPts() == geom[lev].Domain().numPts());

  MultiFab rhs_p(ba, dm, 1, 0);
  MultiFab::Copy(rhs_p, rhs_p_in, 0, 0, 1, 0);
  if (level_singular) {
    SubtractMean(lev, rhs_p);
    SubtractMean(lev, p);
  }

  MultiFab p_g(ba, dm, 1, 1);
  FillCellPatch(lev, m_pressure, p_g, m_cur_time, m_bc_pres);

  MultiFab rr_p(ba, dm, 1, 0), rhat_p(ba, dm, 1, 0);
  MultiFab pv_p(ba, dm, 1, 1), v_p(ba, dm, 1, 0);
  MultiFab s_p(ba, dm, 1, 1), t_p(ba, dm, 1, 0);
  MultiFab Ax_p(ba, dm, 1, 0);
  pv_p.setVal(0.0);
  v_p.setVal(0.0);
  s_p.setVal(0.0);

  std::vector<Real> Ax_ib(rhs_ib.size(), 0.0);
  std::vector<Real> rr_ib(rhs_ib.size(), 0.0);
  std::vector<Real> rhat_ib(rhs_ib.size(), 0.0);
  std::vector<Real> pv_ib(rhs_ib.size(), 0.0);
  std::vector<Real> v_ib(rhs_ib.size(), 0.0);
  std::vector<Real> s_ib(rhs_ib.size(), 0.0);
  std::vector<Real> t_ib(rhs_ib.size(), 0.0);

  ApplyIBSchurOp(lev, p_g, m_ib_force, Ax_p, Ax_ib);
  MultiFab::LinComb(rr_p, 1.0_rt, rhs_p, 0, -1.0_rt, Ax_p, 0, 0, 1, 0);
  for (std::size_t i = 0; i < rr_ib.size(); ++i)
    rr_ib[i] = rhs_ib[i] - Ax_ib[i];
  MultiFab::Copy(rhat_p, rr_p, 0, 0, 1, 0);
  rhat_ib = rr_ib;

  const Real rhs_norm2 =
      MultiFab::Dot(rhs_p, 0, rhs_p, 0, 1, 0) + vector_dot(rhs_ib, rhs_ib);
  const Real tol2 =
      m_poisson_tol * m_poisson_tol * std::max(rhs_norm2, Real(1.0e-300));

  Real rho = 1.0_rt, alpha = 1.0_rt, omega = 1.0_rt;
  Real rsold = coupled_dot(rr_p, rr_ib, rr_p, rr_ib);

  const Real tiny = 1.0e-300;
  const char *breakdown = nullptr;
  int iter = 0;
  for (; iter < m_poisson_max_iter; ++iter) {
    if (rsold < tol2)
      break;
    if (!std::isfinite(rsold)) {
      breakdown = "non-finite residual";
      break;
    }

    const Real rho_new = coupled_dot(rhat_p, rhat_ib, rr_p, rr_ib);
    if (std::abs(rho_new) < tiny) {
      breakdown = "rho breakdown";
      break;
    }
    const Real beta = (rho_new / rho) * (alpha / omega);

    MultiFab::LinComb(pv_p, 1.0_rt, pv_p, 0, -omega, v_p, 0, 0, 1, 0);
    MultiFab::LinComb(pv_p, beta, pv_p, 0, 1.0_rt, rr_p, 0, 0, 1, 0);
    for (std::size_t i = 0; i < pv_ib.size(); ++i)
      pv_ib[i] = rr_ib[i] + beta * (pv_ib[i] - omega * v_ib[i]);

    FillPresGhostPhys(lev, pv_p);
    ApplyIBSchurOp(lev, pv_p, pv_ib, v_p, v_ib);

    const Real rhatv = coupled_dot(rhat_p, rhat_ib, v_p, v_ib);
    if (std::abs(rhatv) < tiny) {
      breakdown = "rhat-v breakdown";
      break;
    }
    alpha = rho_new / rhatv;

    MultiFab::LinComb(s_p, 1.0_rt, rr_p, 0, -alpha, v_p, 0, 0, 1, 0);
    vector_lincomb(s_ib, 1.0_rt, rr_ib, -alpha, v_ib);

    const Real s2 = coupled_dot(s_p, s_ib, s_p, s_ib);
    if (s2 < tol2) {
      MultiFab::Saxpy(p, alpha, pv_p, 0, 0, 1, 0);
      vector_saxpy(m_ib_force, alpha, pv_ib);
      rsold = s2;
      ++iter;
      break;
    }

    FillPresGhostPhys(lev, s_p);
    ApplyIBSchurOp(lev, s_p, s_ib, t_p, t_ib);

    const Real tt = coupled_dot(t_p, t_ib, t_p, t_ib);
    omega = (tt > tiny) ? coupled_dot(t_p, t_ib, s_p, s_ib) / tt : 0.0_rt;

    MultiFab::Saxpy(p, alpha, pv_p, 0, 0, 1, 0);
    MultiFab::Saxpy(p, omega, s_p, 0, 0, 1, 0);
    vector_saxpy(m_ib_force, alpha, pv_ib);
    vector_saxpy(m_ib_force, omega, s_ib);

    MultiFab::LinComb(rr_p, 1.0_rt, s_p, 0, -omega, t_p, 0, 0, 1, 0);
    vector_lincomb(rr_ib, 1.0_rt, s_ib, -omega, t_ib);

    rho = rho_new;
    rsold = coupled_dot(rr_p, rr_ib, rr_p, rr_ib);
    if (std::abs(omega) < tiny) {
      breakdown = "omega breakdown";
      break;
    }
  }

  if (level_singular)
    SubtractMean(lev, p);

  const bool converged = std::isfinite(rsold) && rsold < tol2;
  if (!converged) {
    const Real relres = std::sqrt(std::max(rsold, Real(0.0)) /
                                  std::max(rhs_norm2, Real(1.0e-300)));
    std::ostringstream msg;
    msg << "IB coupled projection on level " << lev;
    if (breakdown != nullptr) {
      msg << " stopped by " << breakdown;
    } else if (iter >= m_poisson_max_iter) {
      msg << " reached max_iter";
    } else {
      msg << " failed to converge";
    }
    msg << "; the coupled IB system may be singular or rank deficient"
        << " (markers = " << m_ib_geometry.markers.size()
        << ", constraints = " << rhs_ib.size()
        << ", cells = " << grids[lev].numPts() << ", relres = " << relres
        << ").";
    amrex::Warning(msg.str());
  }

  if (m_verbose > 1) {
    const Real relres = std::sqrt(std::max(rsold, Real(0.0)) /
                                  std::max(rhs_norm2, Real(1.0e-300)));
    Print() << "  IB BiCGStab lev " << lev << ": " << iter
            << " iters, |r|/|rhs| = " << relres << "\n";
  }
  return iter;
}

void INSSolver::AddIBTags(int lev, TagBoxArray &tags) const {
  BL_PROFILE("INSSolver::AddIBTags()");

  if (!m_ib_enabled || m_ib_geometry.markers.empty() || lev >= max_level)
    return;

  const Geometry &gm = geom[lev];
  const auto dx = gm.CellSizeArray();
  const auto plo = gm.ProbLoArray();
  const auto phi = gm.ProbHiArray();
  GpuArray<int, AMREX_SPACEDIM> periodic{};
  GpuArray<Real, AMREX_SPACEDIM> prob_len{};
  Real radius = 0.0_rt;
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    radius = std::max(radius, dx[d]);
    prob_len[d] = phi[d] - plo[d];
    periodic[d] = gm.isPeriodic(d);
  }
  radius *= m_ib_refine_radius;
  const Real radius2 = radius * radius;
  const auto *markers = m_ib_geometry.device_markers.data();
  const int nmarkers = static_cast<int>(m_ib_geometry.markers.size());

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
  for (MFIter mfi(tags, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    const Box &bx = mfi.tilebox();
    auto const &tag = tags.array(mfi);
    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      GpuArray<Real, AMREX_SPACEDIM> x{};
      x[0] = plo[0] + (i + 0.5_rt) * dx[0];
      x[1] = plo[1] + (j + 0.5_rt) * dx[1];
#if AMREX_SPACEDIM == 3
      x[2] = plo[2] + (k + 0.5_rt) * dx[2];
#endif
      for (int marker = 0; marker < nmarkers; ++marker) {
        Real dist2 = 0.0_rt;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
          const Real r = Peskin4Kernel::periodic_displacement(
              x[d], markers[marker].x[d], prob_len[d], periodic[d]);
          dist2 += r * r;
        }
        if (dist2 <= radius2) {
          tag(i, j, k) = TagBox::SET;
          break;
        }
      }
    });
  }
}
