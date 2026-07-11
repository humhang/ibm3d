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
