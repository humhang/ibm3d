/**
 * INSSolver_IB.cpp
 *
 * Immersed-boundary pieces for the Taira-Colonius projection method.
 *
 * Current scope: prescribed rigid/stationary IB velocity, one Lagrangian
 * marker at each geometry element centroid, and finest-level-only coupling.
 */

#include "INSSolver.H"

#include <AMReX_Gpu.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <cmath>
#include <numeric>

using namespace amrex;

namespace {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real ib_phi4(Real r) noexcept {
  r = std::abs(r);
  if (r < 1.0_rt) {
    return 0.125_rt * (3.0_rt - 2.0_rt * r +
                       std::sqrt(1.0_rt + 4.0_rt * r - 4.0_rt * r * r));
  }
  if (r < 2.0_rt) {
    return 0.125_rt * (5.0_rt - 2.0_rt * r -
                       std::sqrt(-7.0_rt + 12.0_rt * r - 4.0_rt * r * r));
  }
  return 0.0_rt;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
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

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
ib_delta(const GpuArray<Real, AMREX_SPACEDIM> &x,
         const ibm3d::IBMarker &marker,
         const GpuArray<Real, AMREX_SPACEDIM> &dx,
         const GpuArray<Real, AMREX_SPACEDIM> &prob_len,
         const GpuArray<int, AMREX_SPACEDIM> &periodic) noexcept {
  Real delta = 1.0_rt;
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    const Real r =
        periodic_displacement(x[d], marker.x[d], prob_len[d], periodic[d]);
    const Real ph = ib_phi4(r / dx[d]);
    if (ph == 0.0_rt)
      return 0.0_rt;
    delta *= ph / dx[d];
  }
  return delta;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE GpuArray<Real, AMREX_SPACEDIM>
face_position(int dir, int i, int j, int k,
              const GpuArray<Real, AMREX_SPACEDIM> &plo,
              const GpuArray<Real, AMREX_SPACEDIM> &dx) noexcept {
  GpuArray<Real, AMREX_SPACEDIM> x{};
  x[0] = plo[0] + (i + (dir == 0 ? 0.0_rt : 0.5_rt)) * dx[0];
  x[1] = plo[1] + (j + (dir == 1 ? 0.0_rt : 0.5_rt)) * dx[1];
#if AMREX_SPACEDIM == 3
  x[2] = plo[2] + (k + (dir == 2 ? 0.0_rt : 0.5_rt)) * dx[2];
#endif
  return x;
}

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
    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      const auto x = face_position(dir, i, j, k, plo, dx);
      Real sum = 0.0_rt;
      for (int marker = 0; marker < nmarkers; ++marker) {
        const Real delta =
            ib_delta(x, markers[marker], dx, prob_len, periodic);
        sum +=
            f[marker * AMREX_SPACEDIM + dir] * markers[marker].weight * delta;
      }
      h(i, j, k) = sum;
    });
  }
  Gpu::streamSynchronize();
}

void INSSolver::InterpolateIBVelocity(
    int lev, const std::array<const MultiFab *, AMREX_SPACEDIM> &vel,
    std::vector<Real> &marker_vel) const {
  const auto &markers = m_ib_geometry.markers;
  marker_vel.assign(markers.size() * AMREX_SPACEDIM, 0.0);
  if (!m_ib_enabled || markers.empty())
    return;

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

    for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
      const Box &bx = mfi.validbox();
      auto const &u = mf.const_array(mfi);
      auto const &mask = owner->const_array(mfi);
      amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
        if (mask(i, j, k) == 0)
          return;
        const auto x = face_position(dir, i, j, k, plo, dx);
        for (std::size_t marker = 0; marker < markers.size(); ++marker) {
          const Real delta =
              ib_delta(x, markers[marker], dx, prob_len, periodic);
          if (delta != 0.0_rt) {
            marker_vel[marker * AMREX_SPACEDIM + dir] +=
                u(i, j, k) * delta * dv;
          }
        }
      });
    }
  }

  if (!marker_vel.empty()) {
    ParallelDescriptor::ReduceRealSum(marker_vel.data(),
                                      static_cast<int>(marker_vel.size()));
  }
}

void INSSolver::ApplyIBSchurOp(int lev, const MultiFab &phi,
                               const std::vector<Real> &force,
                               MultiFab &result_p,
                               std::vector<Real> &result_ib) {
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
  int iter = 0;
  for (; iter < m_poisson_max_iter; ++iter) {
    if (rsold < tol2 || !std::isfinite(rsold))
      break;

    const Real rho_new = coupled_dot(rhat_p, rhat_ib, rr_p, rr_ib);
    if (std::abs(rho_new) < tiny)
      break;
    const Real beta = (rho_new / rho) * (alpha / omega);

    MultiFab::LinComb(pv_p, 1.0_rt, pv_p, 0, -omega, v_p, 0, 0, 1, 0);
    MultiFab::LinComb(pv_p, beta, pv_p, 0, 1.0_rt, rr_p, 0, 0, 1, 0);
    for (std::size_t i = 0; i < pv_ib.size(); ++i)
      pv_ib[i] = rr_ib[i] + beta * (pv_ib[i] - omega * v_ib[i]);

    FillPresGhostPhys(lev, pv_p);
    ApplyIBSchurOp(lev, pv_p, pv_ib, v_p, v_ib);

    const Real rhatv = coupled_dot(rhat_p, rhat_ib, v_p, v_ib);
    if (std::abs(rhatv) < tiny)
      break;
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
    if (std::abs(omega) < tiny)
      break;
  }

  if (level_singular)
    SubtractMean(lev, p);

  if (m_verbose > 1) {
    const Real relres = std::sqrt(std::max(rsold, Real(0.0)) /
                                  std::max(rhs_norm2, Real(1.0e-300)));
    Print() << "  IB BiCGStab lev " << lev << ": " << iter
            << " iters, |r|/|rhs| = " << relres << "\n";
  }
  return iter;
}

void INSSolver::AddIBTags(int lev, TagBoxArray &tags) const {
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
  const char tagval = TagBox::SET;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
  for (MFIter mfi(tags, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    const Box &bx = mfi.tilebox();
    auto const &tag = tags.array(mfi);
    amrex::ParallelFor(
        bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          GpuArray<Real, AMREX_SPACEDIM> x{};
          x[0] = plo[0] + (i + 0.5_rt) * dx[0];
          x[1] = plo[1] + (j + 0.5_rt) * dx[1];
#if AMREX_SPACEDIM == 3
          x[2] = plo[2] + (k + 0.5_rt) * dx[2];
#endif
          for (int marker = 0; marker < nmarkers; ++marker) {
            Real dist2 = 0.0_rt;
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
              const Real r = periodic_displacement(
                  x[d], markers[marker].x[d], prob_len[d], periodic[d]);
              dist2 += r * r;
            }
            if (dist2 <= radius2) {
              tag(i, j, k) = tagval;
              break;
            }
          }
        });
  }
}
