/**
 * INSSolver_Project.cpp
 *
 * Perot 1997 / Taira–Colonius projection step.
 *
 *   Modified Poisson:   (D B^N G) p^{n+1} = (1/dt) D u*
 *   Projection:         u^{n+1} = u* − dt B^N G p^{n+1}
 *
 *   B^N = Σ_{k=0}^N (εL)^k,   ε = ν dt / 2
 *
 * The operator D B^N G is applied matrix-free (compose existing
 * gradient, face-Laplacian, divergence pieces).  Pressure-only solves use
 * hierarchy BiCGStab.  IB solves eliminate pressure with a marker-space
 * force Schur operator whose fixed pressure inverse combines MLMG V-cycles
 * with modified-operator corrections; final acceptance always uses the
 * original coupled hierarchy operator.
 *
 * Multi-level projection uses a hierarchy-wide Krylov solve whose fine
 * pressure ghosts are filled from the current coarse iterate.  With IB
 * enabled, pressure spans all levels while the Schur Krylov vector contains
 * finest-level Lagrangian forces only.  Applying the same IB rows on coarser
 * levels makes the marker constraints over-dense and singular.
 */

#include "INSSolver.H"

#include <AMReX.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLPoisson.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_Print.H>
#include <AMReX_Reduce.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

using namespace amrex;

namespace {

using CellHierarchy = Vector<std::unique_ptr<MultiFab>>;
using MaskHierarchy = Vector<std::unique_ptr<iMultiFab>>;
using MarkerVector = INSSolver::MarkerVector;

void mask_covered_cells(MultiFab &mf, const iMultiFab &mask) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
  for (MFIter mfi(mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    const Box &bx = mfi.tilebox();
    auto const &a = mf.array(mfi);
    auto const &m = mask.const_array(mfi);
    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      if (m(i, j, k) == 0)
        a(i, j, k) = 0.0_rt;
    });
  }
}

void mask_composite(CellHierarchy &mf, const MaskHierarchy &mask) {
  for (int lev = 0; lev < static_cast<int>(mf.size()); ++lev)
    mask_covered_cells(*mf[lev], *mask[lev]);
}

void define_composite_like(CellHierarchy &dst, const CellHierarchy &src,
                           int ngrow) {
  dst.resize(src.size());
  for (int lev = 0; lev < static_cast<int>(src.size()); ++lev) {
    dst[lev] = std::make_unique<MultiFab>(
        src[lev]->boxArray(), src[lev]->DistributionMap(), 1, ngrow);
    dst[lev]->setVal(0.0);
  }
}

void copy_composite(CellHierarchy &dst, const CellHierarchy &src,
                    const MaskHierarchy &mask) {
  for (int lev = 0; lev < static_cast<int>(dst.size()); ++lev)
    MultiFab::Copy(*dst[lev], *src[lev], 0, 0, 1, 0);
  mask_composite(dst, mask);
}

void copy_composite_unmasked(CellHierarchy &dst, const CellHierarchy &src) {
  for (int lev = 0; lev < static_cast<int>(dst.size()); ++lev)
    MultiFab::Copy(*dst[lev], *src[lev], 0, 0, 1, 0);
}

void lincomb_composite(CellHierarchy &dst, Real a, const CellHierarchy &x,
                       Real b, const CellHierarchy &y,
                       const MaskHierarchy &mask) {
  for (int lev = 0; lev < static_cast<int>(dst.size()); ++lev) {
    MultiFab::LinComb(*dst[lev], a, *x[lev], 0, b, *y[lev], 0, 0, 1, 0);
  }
  mask_composite(dst, mask);
}

void saxpy_composite(CellHierarchy &dst, Real a, const CellHierarchy &x,
                     const MaskHierarchy &mask) {
  for (int lev = 0; lev < static_cast<int>(dst.size()); ++lev)
    MultiFab::Saxpy(*dst[lev], a, *x[lev], 0, 0, 1, 0);
  mask_composite(dst, mask);
}

Real dot_composite(const CellHierarchy &a, const CellHierarchy &b) {
  Real value = 0.0_rt;
  for (int lev = 0; lev < static_cast<int>(a.size()); ++lev)
    value += MultiFab::Dot(*a[lev], 0, *b[lev], 0, 1, 0);
  return value;
}

void set_vector(MarkerVector &values, Real value) {
  Real *values_p = values.data();
  const auto size = static_cast<Long>(values.size());
  amrex::ParallelFor(
      size, [=] AMREX_GPU_DEVICE(Long i) noexcept { values_p[i] = value; });
}

void copy_vector(MarkerVector &dst, const MarkerVector &src) {
  dst.resize(src.size());
  Gpu::copyAsync(Gpu::deviceToDevice, src.begin(), src.end(), dst.begin());
}

Real dot_vector(const MarkerVector &a, const MarkerVector &b) {
  AMREX_ALWAYS_ASSERT(a.size() == b.size());
  const Real *a_p = a.data();
  const Real *b_p = b.data();
  return Reduce::Sum<Real>(
      static_cast<Long>(a.size()),
      [=] AMREX_GPU_DEVICE(Long i) noexcept { return a_p[i] * b_p[i]; });
}

Real dot_coupled(const CellHierarchy &a_p, const MarkerVector &a_ib,
                 const CellHierarchy &b_p, const MarkerVector &b_ib) {
  return dot_composite(a_p, b_p) + dot_vector(a_ib, b_ib);
}

void saxpy_vector(MarkerVector &dst, Real a, const MarkerVector &x) {
  AMREX_ALWAYS_ASSERT(dst.size() == x.size());
  Real *dst_p = dst.data();
  const Real *x_p = x.data();
  const auto size = static_cast<Long>(dst.size());
  amrex::ParallelFor(
      size, [=] AMREX_GPU_DEVICE(Long i) noexcept { dst_p[i] += a * x_p[i]; });
}

void lincomb_vector(MarkerVector &dst, Real a, const MarkerVector &x, Real b,
                    const MarkerVector &y) {
  AMREX_ALWAYS_ASSERT(x.size() == y.size());
  dst.resize(x.size());
  Real *dst_p = dst.data();
  const Real *x_p = x.data();
  const Real *y_p = y.data();
  const auto size = static_cast<Long>(dst.size());
  amrex::ParallelFor(size, [=] AMREX_GPU_DEVICE(Long i) noexcept {
    dst_p[i] = a * x_p[i] + b * y_p[i];
  });
}

void saxpy_coupled(CellHierarchy &dst_p, MarkerVector &dst_ib, Real a,
                   const CellHierarchy &x_p, const MarkerVector &x_ib,
                   const MaskHierarchy &mask) {
  saxpy_composite(dst_p, a, x_p, mask);
  saxpy_vector(dst_ib, a, x_ib);
}

void lincomb_coupled(CellHierarchy &dst_p, MarkerVector &dst_ib, Real a,
                     const CellHierarchy &x_p, const MarkerVector &x_ib, Real b,
                     const CellHierarchy &y_p, const MarkerVector &y_ib,
                     const MaskHierarchy &mask) {
  lincomb_composite(dst_p, a, x_p, b, y_p, mask);
  lincomb_vector(dst_ib, a, x_ib, b, y_ib);
}

Real norm2_vector(const MarkerVector &a) { return dot_vector(a, a); }

Real marker_slip_norm_inf(const MarkerVector &marker_velocity,
                          GpuArray<Real, AMREX_SPACEDIM> target_velocity) {
  const Real *velocity_p = marker_velocity.data();
  return Reduce::Max<Real>(
      static_cast<Long>(marker_velocity.size()),
      [=] AMREX_GPU_DEVICE(Long i) noexcept {
        const Real difference =
            velocity_p[i] - target_velocity[i % AMREX_SPACEDIM];
        return difference < 0.0_rt ? -difference : difference;
      },
      0.0_rt);
}

struct CoupledResidual {
  Real pressure_norm2;
  Real ib_norm2;
};

// H already contains marker quadrature.  This scaling equilibrates all four
// uniform-grid blocks at O(1/h) and recovers their adjoint scaling.
struct CoupledScaling {
  static void
  initialize_force_scaling(MarkerVector &force_columns,
                           const Gpu::DeviceVector<ibm3d::IBMarker> &markers,
                           Real marker_measure) {
    const auto *markers_p = markers.data();
    Real *force_columns_p = force_columns.data();
    const auto size = static_cast<Long>(force_columns.size());
    amrex::ParallelFor(size, [=] AMREX_GPU_DEVICE(Long i) noexcept {
      force_columns_p[i] =
          marker_measure / markers_p[i / AMREX_SPACEDIM].weight;
    });
  }
  CoupledScaling(const Vector<Geometry> &geometry, int finest_level,
                 const Gpu::DeviceVector<ibm3d::IBMarker> &markers)
      : pressure_rows(finest_level + 1),
        force_columns(markers.size() * AMREX_SPACEDIM) {
    Real finest_cell_volume = 1.0_rt;
    for (int lev = 0; lev <= finest_level; ++lev) {
      const auto dx = geometry[lev].CellSizeArray();
      Real cell_volume = 1.0_rt;
      for (int d = 0; d < AMREX_SPACEDIM; ++d)
        cell_volume *= dx[d];
      pressure_rows[lev] =
          std::pow(cell_volume, 1.0_rt / static_cast<Real>(AMREX_SPACEDIM));
      if (lev == finest_level)
        finest_cell_volume = cell_volume;
    }

    const Real marker_measure =
        finest_cell_volume / pressure_rows[finest_level];
    initialize_force_scaling(force_columns, markers, marker_measure);
  }

  void scale_pressure(CellHierarchy &values) const {
    AMREX_ALWAYS_ASSERT(values.size() == pressure_rows.size());
    for (int lev = 0; lev < static_cast<int>(values.size()); ++lev)
      values[lev]->mult(pressure_rows[lev], 0, 1, 0);
  }

  void to_scaled_force(const MarkerVector &physical,
                       MarkerVector &scaled) const {
    AMREX_ALWAYS_ASSERT(physical.size() == force_columns.size());
    scaled.resize(physical.size());
    const Real *physical_p = physical.data();
    const Real *force_columns_p = force_columns.data();
    Real *scaled_p = scaled.data();
    const auto size = static_cast<Long>(scaled.size());
    amrex::ParallelFor(size, [=] AMREX_GPU_DEVICE(Long i) noexcept {
      scaled_p[i] = physical_p[i] / force_columns_p[i];
    });
  }

  void to_physical_force(const MarkerVector &scaled,
                         MarkerVector &physical) const {
    AMREX_ALWAYS_ASSERT(scaled.size() == force_columns.size());
    physical.resize(scaled.size());
    const Real *scaled_p = scaled.data();
    const Real *force_columns_p = force_columns.data();
    Real *physical_p = physical.data();
    const auto size = static_cast<Long>(scaled.size());
    amrex::ParallelFor(size, [=] AMREX_GPU_DEVICE(Long i) noexcept {
      physical_p[i] = force_columns_p[i] * scaled_p[i];
    });
  }

  Vector<Real> pressure_rows;
  MarkerVector force_columns;
};

Real sum_composite(const CellHierarchy &a) {
  Real value = 0.0_rt;
  for (const auto &mf : a)
    value += mf->sum(0, true);
  ParallelDescriptor::ReduceRealSum(value);
  return value;
}

Real active_cell_count(const MaskHierarchy &mask) {
  Long value = 0;
  for (const auto &m : mask)
    value += m->sum(0, 0, true);
  ParallelDescriptor::ReduceLongSum(value);
  return static_cast<Real>(value);
}

void subtract_composite_mean(CellHierarchy &mf, const MaskHierarchy &mask) {
  const Real n = active_cell_count(mask);
  if (n <= 0.0_rt)
    return;
  const Real mean = sum_composite(mf) / n;
  for (auto &level_mf : mf)
    level_mf->plus(-mean, 0, 1, 0);
  mask_composite(mf, mask);
}

void negative_divergence_from_faces(
    const Geometry &gm,
    const std::array<std::unique_ptr<MultiFab>, AMREX_SPACEDIM> &face,
    MultiFab &result) {
  const Real *dx = gm.CellSize();
  const Real cx = 1.0_rt / dx[0];
  const Real cy = 1.0_rt / dx[1];
#if AMREX_SPACEDIM == 3
  const Real cz = 1.0_rt / dx[2];
#endif

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
  for (MFIter mfi(result, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    const Box &bx = mfi.tilebox();
    auto const &u = face[0]->const_array(mfi);
    auto const &v = face[1]->const_array(mfi);
#if AMREX_SPACEDIM == 3
    auto const &w = face[2]->const_array(mfi);
#endif
    auto const &r = result.array(mfi);
    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      Real d = cx * (u(i + 1, j, k) - u(i, j, k)) +
               cy * (v(i, j + 1, k) - v(i, j, k));
#if AMREX_SPACEDIM == 3
      d += cz * (w(i, j, k + 1) - w(i, j, k));
#endif
      r(i, j, k) = -d;
    });
  }
}

} // namespace

MaskHierarchy INSSolver::BuildCompositeActiveMasks() const {
  BL_PROFILE("INSSolver::BuildCompositeActiveMasks()");

  const int nlev = finest_level + 1;
  MaskHierarchy active_masks(nlev);
  for (int lev = 0; lev < finest_level; ++lev) {
    active_masks[lev] = std::make_unique<iMultiFab>(amrex::makeFineMask(
        grids[lev], dmap[lev], grids[lev + 1], ref_ratio[lev], 1, 0));
  }
  active_masks[finest_level] = std::make_unique<iMultiFab>(
      grids[finest_level], dmap[finest_level], 1, 0);
  active_masks[finest_level]->setVal(1);
  return active_masks;
}

void INSSolver::ApplyCompositeModifiedPoissonOp(
    CellHierarchy &phi, CellHierarchy &result,
    const MaskHierarchy &active_masks) {
  BL_PROFILE("INSSolver::ApplyCompositeModifiedPoissonOp()");

  ApplyCompositeProjectionBlocks(phi, nullptr, result, nullptr, active_masks);
}

void INSSolver::ApplyCompositeProjectionBlocks(
    CellHierarchy &phi, const MarkerVector *force, CellHierarchy &result_p,
    MarkerVector *result_ib, const MaskHierarchy &active_masks) {
  BL_PROFILE("INSSolver::ApplyCompositeProjectionBlocks()");

  CellHierarchy phi_sync;
  define_composite_like(phi_sync, phi, 1);
  copy_composite_unmasked(phi_sync, phi);
  for (int lev = finest_level - 1; lev >= 0; --lev)
    amrex::average_down(*phi_sync[lev + 1], *phi_sync[lev], 0, 1,
                        ref_ratio[lev]);

  if (result_ib != nullptr) {
    result_ib->resize(m_ib_geometry.markers.size() * AMREX_SPACEDIM);
    set_vector(*result_ib, 0.0_rt);
  }

  Vector<FaceMFArray> q_hierarchy(finest_level + 1);
  Vector<FaceMFArray> bq(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    MultiFab phi_g(grids[lev], dmap[lev], 1, 1);
    FillCellPatch(lev, phi_sync, phi_g, m_cur_time, m_bc_pres);

    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
      BoxArray fba = amrex::convert(grids[lev], IntVect::TheDimensionVector(d));
      q_hierarchy[lev][d] = std::make_unique<MultiFab>(fba, dmap[lev], 1, 1);
      bq[lev][d] = std::make_unique<MultiFab>(fba, dmap[lev], 1, 0);

      q_hierarchy[lev][d]->setVal(0.0);
      ComputePressureGradient(lev, d, phi_g, *q_hierarchy[lev][d]);
      if (force != nullptr && lev == finest_level) {
        MultiFab h(fba, dmap[lev], 1, 0);
        SpreadIBForce(lev, d, *force, h);
        MultiFab::Add(*q_hierarchy[lev][d], h, 0, 0, 1, 0);
      }
    }
  }

  ApplyCompositeBNFaces(q_hierarchy, bq);

  for (int lev = 0; lev <= finest_level; ++lev) {
    negative_divergence_from_faces(geom[lev], bq[lev], *result_p[lev]);
    mask_covered_cells(*result_p[lev], *active_masks[lev]);
  }
  if (m_pressure_singular)
    subtract_composite_mean(result_p, active_masks);

  if (result_ib != nullptr) {
    InterpolateIBVelocity(
        finest_level,
        {AMREX_D_DECL(bq[finest_level][0].get(), bq[finest_level][1].get(),
                      bq[finest_level][2].get())},
        *result_ib);
  }
}

void INSSolver::ApplyCompositeIBProjectionOp(
    CellHierarchy &phi, const MarkerVector &force, CellHierarchy &result_p,
    MarkerVector &result_ib, const MaskHierarchy &active_masks) {
  BL_PROFILE("INSSolver::ApplyCompositeIBProjectionOp()");

  ApplyCompositeProjectionBlocks(phi, &force, result_p, &result_ib,
                                 active_masks);
}

void INSSolver::BuildCompositePoissonInitialGuess(
    CellHierarchy &p, const CellHierarchy &rhs,
    const MaskHierarchy &active_masks) {
  BL_PROFILE("INSSolver::BuildCompositePoissonInitialGuess()");

  const int nlev = finest_level + 1;
  Vector<Geometry> mg_geom(geom.begin(), geom.begin() + nlev);
  Vector<BoxArray> mg_grids(grids.begin(), grids.begin() + nlev);
  Vector<DistributionMapping> mg_dmap(dmap.begin(), dmap.begin() + nlev);

  MLPoisson poisson(mg_geom, mg_grids, mg_dmap);
  Array<LinOpBCType, AMREX_SPACEDIM> lobc;
  Array<LinOpBCType, AMREX_SPACEDIM> hibc;
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    if (geom[0].isPeriodic(d)) {
      lobc[d] = LinOpBCType::Periodic;
      hibc[d] = LinOpBCType::Periodic;
    } else {
      lobc[d] = (m_bc_lo[d] == BCKind::outflow) ? LinOpBCType::Dirichlet
                                                : LinOpBCType::Neumann;
      hibc[d] = (m_bc_hi[d] == BCKind::outflow) ? LinOpBCType::Dirichlet
                                                : LinOpBCType::Neumann;
    }
  }
  poisson.setDomainBC(lobc, hibc);
  poisson.setMaxOrder(2);
  for (int lev = 0; lev < nlev; ++lev)
    poisson.setLevelBC(lev, nullptr);

  CellHierarchy neg_rhs;
  define_composite_like(neg_rhs, rhs, 0);
  Vector<MultiFab *> sol(nlev);
  Vector<const MultiFab *> rhs_ptr(nlev);
  for (int lev = 0; lev < nlev; ++lev) {
    p[lev]->setVal(0.0);
    MultiFab::Copy(*neg_rhs[lev], *rhs[lev], 0, 0, 1, 0);
    neg_rhs[lev]->mult(-1.0_rt, 0, 1, 0);
    sol[lev] = p[lev].get();
    rhs_ptr[lev] = neg_rhs[lev].get();
  }

  MLMG mlmg(poisson);
  mlmg.setVerbose(0);
  mlmg.setBottomVerbose(0);
  mlmg.setMaxIter(100);
  mlmg.solve(sol, rhs_ptr, 1.0e-10_rt, 0.0_rt);

  mask_composite(p, active_masks);
  if (m_pressure_singular)
    subtract_composite_mean(p, active_masks);
}

int INSSolver::SolveCompositeModifiedPoisson(CellHierarchy &p,
                                             const CellHierarchy &rhs_in,
                                             const MaskHierarchy &active_masks,
                                             Real relative_tolerance) {
  BL_PROFILE("INSSolver::SolveCompositeModifiedPoisson()");

  CellHierarchy rhs;
  define_composite_like(rhs, rhs_in, 0);
  copy_composite(rhs, rhs_in, active_masks);
  mask_composite(p, active_masks);

  if (m_pressure_singular) {
    subtract_composite_mean(rhs, active_masks);
    subtract_composite_mean(p, active_masks);
  }

  CellHierarchy rr, rhat, pv, v, s, t, Ax;
  define_composite_like(rr, p, 0);
  define_composite_like(rhat, p, 0);
  define_composite_like(pv, p, 1);
  define_composite_like(v, p, 0);
  define_composite_like(s, p, 1);
  define_composite_like(t, p, 0);
  define_composite_like(Ax, p, 0);

  const Real rhs_norm2 = dot_composite(rhs, rhs);
  const Real solve_tol =
      (relative_tolerance > 0.0_rt) ? relative_tolerance : m_poisson_tol;
  const Real tol2 = solve_tol * solve_tol * std::max(rhs_norm2, Real(1.0e-300));

  if (!m_pressure_singular) {
    CellHierarchy saved_p;
    define_composite_like(saved_p, p, 1);
    copy_composite_unmasked(saved_p, p);

    ApplyCompositeModifiedPoissonOp(p, Ax, active_masks);
    lincomb_composite(rr, 1.0_rt, rhs, -1.0_rt, Ax, active_masks);
    const Real cold_resid2 = dot_composite(rr, rr);

    BuildCompositePoissonInitialGuess(p, rhs, active_masks);
    ApplyCompositeModifiedPoissonOp(p, Ax, active_masks);
    lincomb_composite(rr, 1.0_rt, rhs, -1.0_rt, Ax, active_masks);
    const Real warm_resid2 = dot_composite(rr, rr);
    if (!std::isfinite(warm_resid2) || warm_resid2 >= cold_resid2)
      copy_composite_unmasked(p, saved_p);
  }

  const Real tiny = 1.0e-300;
  const char *breakdown = nullptr;
  Real rsold = std::numeric_limits<Real>::infinity();
  int iter = 0;
  constexpr int residual_refresh = 100;
  while (iter < m_poisson_max_iter) {
    ApplyCompositeModifiedPoissonOp(p, Ax, active_masks);
    lincomb_composite(rr, 1.0_rt, rhs, -1.0_rt, Ax, active_masks);
    rsold = dot_composite(rr, rr);
    if (!std::isfinite(rsold)) {
      breakdown = "non-finite residual";
      break;
    }
    if (rsold < tol2)
      break;

    copy_composite(rhat, rr, active_masks);
    for (auto &level : pv)
      level->setVal(0.0);
    for (auto &level : v)
      level->setVal(0.0);

    Real rho = 1.0_rt;
    Real alpha = 1.0_rt;
    Real omega = 1.0_rt;
    const int cycle_end = std::min(iter + residual_refresh, m_poisson_max_iter);
    bool cycle_breakdown = false;

    while (iter < cycle_end) {
      const Real rho_new = dot_composite(rhat, rr);
      if (std::abs(rho_new) < tiny) {
        breakdown = "rho breakdown";
        cycle_breakdown = true;
        break;
      }
      const Real beta = (rho_new / rho) * (alpha / omega);

      lincomb_composite(pv, 1.0_rt, pv, -omega, v, active_masks);
      lincomb_composite(pv, beta, pv, 1.0_rt, rr, active_masks);

      ApplyCompositeModifiedPoissonOp(pv, v, active_masks);

      const Real rhatv = dot_composite(rhat, v);
      if (std::abs(rhatv) < tiny) {
        breakdown = "rhat-v breakdown";
        cycle_breakdown = true;
        break;
      }
      alpha = rho_new / rhatv;

      lincomb_composite(s, 1.0_rt, rr, -alpha, v, active_masks);

      const Real s2 = dot_composite(s, s);
      if (s2 < tol2) {
        saxpy_composite(p, alpha, pv, active_masks);
        ++iter;
        break;
      }

      ApplyCompositeModifiedPoissonOp(s, t, active_masks);

      const Real tt = dot_composite(t, t);
      omega = (tt > tiny) ? dot_composite(t, s) / tt : 0.0_rt;

      saxpy_composite(p, alpha, pv, active_masks);
      saxpy_composite(p, omega, s, active_masks);

      lincomb_composite(rr, 1.0_rt, s, -omega, t, active_masks);

      rho = rho_new;
      rsold = dot_composite(rr, rr);
      ++iter;
      if (std::abs(omega) < tiny) {
        breakdown = "omega breakdown";
        cycle_breakdown = true;
        break;
      }
      if (!std::isfinite(rsold)) {
        breakdown = "non-finite residual";
        cycle_breakdown = true;
        break;
      }
      if (rsold < tol2)
        break;
    }

    if (cycle_breakdown)
      break;
  }

  if (m_pressure_singular)
    subtract_composite_mean(p, active_masks);

  ApplyCompositeModifiedPoissonOp(p, Ax, active_masks);
  lincomb_composite(rr, 1.0_rt, rhs, -1.0_rt, Ax, active_masks);
  rsold = dot_composite(rr, rr);

  const bool converged = std::isfinite(rsold) && rsold < tol2;
  if (!converged) {
    const Real relres = std::sqrt(std::max(rsold, Real(0.0)) /
                                  std::max(rhs_norm2, Real(1.0e-300)));
    std::ostringstream msg;
    msg << "Composite modified Poisson solve";
    if (breakdown != nullptr) {
      msg << " stopped by " << breakdown;
    } else if (iter >= m_poisson_max_iter) {
      msg << " reached max_iter";
    } else {
      msg << " failed to converge";
    }
    msg << "; the pressure system may be singular, inconsistent, or "
           "ill-conditioned"
        << " (levels = " << finest_level + 1
        << ", pressure_singular = " << m_pressure_singular
        << ", iter = " << iter << ", relres = " << relres << ").";
    amrex::Warning(msg.str());
  }

  if (m_verbose > 1) {
    const Real relres = std::sqrt(std::max(rsold, Real(0.0)) /
                                  std::max(rhs_norm2, Real(1.0e-300)));
    Print() << "  Composite Perot BiCGStab: " << iter
            << " iters, |r|/|rhs| = " << relres << "\n";
  }
  return iter;
}

int INSSolver::SolveCompositeIBProjection(CellHierarchy &p,
                                          const CellHierarchy &rhs_p_in,
                                          const MarkerVector &rhs_ib,
                                          const MaskHierarchy &active_masks) {
  BL_PROFILE("INSSolver::SolveCompositeIBProjection()");

  CellHierarchy rhs_p;
  define_composite_like(rhs_p, rhs_p_in, 0);
  copy_composite(rhs_p, rhs_p_in, active_masks);
  mask_composite(p, active_masks);

  if (m_pressure_singular) {
    subtract_composite_mean(rhs_p, active_masks);
    subtract_composite_mean(p, active_masks);
  }

  AMREX_ALWAYS_ASSERT(rhs_ib.size() ==
                      m_ib_geometry.markers.size() * AMREX_SPACEDIM);
  AMREX_ALWAYS_ASSERT(m_ib_force.size() == rhs_ib.size());
  const CoupledScaling scaling(geom, finest_level,
                               m_ib_geometry.device_markers);
  MarkerVector scaled_force;
  MarkerVector physical_force;
  scaling.to_scaled_force(m_ib_force, scaled_force);
  CellHierarchy initial_p;
  define_composite_like(initial_p, p, 1);
  copy_composite_unmasked(initial_p, p);
  MarkerVector initial_scaled_force;
  copy_vector(initial_scaled_force, scaled_force);

  CellHierarchy scaled_rhs_p;
  define_composite_like(scaled_rhs_p, rhs_p, 0);
  copy_composite(scaled_rhs_p, rhs_p, active_masks);
  scaling.scale_pressure(scaled_rhs_p);

  const Real rhs_p_norm2 = dot_composite(rhs_p, rhs_p);
  const Real rhs_ib_norm2 = dot_vector(rhs_ib, rhs_ib);
  const Real rhs_norm2 =
      dot_composite(scaled_rhs_p, scaled_rhs_p) + rhs_ib_norm2;
  const Real rhs_norm = std::sqrt(std::max(rhs_norm2, Real(1.0e-300)));
  const Real coupled_tol = m_poisson_tol * rhs_norm;
  const Real tiny = 1.0e-300;
  const int nlev = finest_level + 1;
  Vector<Geometry> mg_geom(geom.begin(), geom.begin() + nlev);
  Vector<BoxArray> mg_grids(grids.begin(), grids.begin() + nlev);
  Vector<DistributionMapping> mg_dmap(dmap.begin(), dmap.begin() + nlev);

  MLPoisson poisson(mg_geom, mg_grids, mg_dmap);
  Array<LinOpBCType, AMREX_SPACEDIM> lobc;
  Array<LinOpBCType, AMREX_SPACEDIM> hibc;
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    if (geom[0].isPeriodic(d)) {
      lobc[d] = LinOpBCType::Periodic;
      hibc[d] = LinOpBCType::Periodic;
    } else {
      lobc[d] = (m_bc_lo[d] == BCKind::outflow) ? LinOpBCType::Dirichlet
                                                : LinOpBCType::Neumann;
      hibc[d] = (m_bc_hi[d] == BCKind::outflow) ? LinOpBCType::Dirichlet
                                                : LinOpBCType::Neumann;
    }
  }
  poisson.setDomainBC(lobc, hibc);
  poisson.setMaxOrder(2);
  poisson.setEnforceSingularSolvable(m_pressure_singular);
  for (int lev = 0; lev < nlev; ++lev)
    poisson.setLevelBC(lev, nullptr);

  MLMG pressure_mlmg(poisson);
  pressure_mlmg.setVerbose(0);
  pressure_mlmg.setBottomVerbose(0);
  pressure_mlmg.setPrecondIter(m_ib_schur_mg_iters);
  pressure_mlmg.setMaxFmgIter(0);
  pressure_mlmg.setBottomSolver(BottomSolver::smoother);

  CellHierarchy poisson_rhs, inverse_residual, inverse_image, inverse_delta;
  define_composite_like(poisson_rhs, rhs_p, 0);
  define_composite_like(inverse_residual, rhs_p, 0);
  define_composite_like(inverse_image, rhs_p, 0);
  define_composite_like(inverse_delta, p, 1);
  Vector<MultiFab *> mlmg_solution(nlev);
  Vector<const MultiFab *> mlmg_rhs(nlev);

  int pressure_inverse_applications = 0;
  int standard_poisson_applications = 0;
  int modified_poisson_applications = 0;

  const auto apply_standard_poisson_inverse = [&](const CellHierarchy &input,
                                                  CellHierarchy &solution) {
    ++standard_poisson_applications;
    for (int lev = 0; lev < nlev; ++lev) {
      solution[lev]->setVal(0.0_rt);
      MultiFab::Copy(*poisson_rhs[lev], *input[lev], 0, 0, 1, 0);
      poisson_rhs[lev]->mult(-1.0_rt, 0, 1, 0);
      mlmg_solution[lev] = solution[lev].get();
      mlmg_rhs[lev] = poisson_rhs[lev].get();
    }
    mask_composite(poisson_rhs, active_masks);
    if (m_pressure_singular)
      subtract_composite_mean(poisson_rhs, active_masks);

    pressure_mlmg.precond(mlmg_solution, mlmg_rhs, 0.0_rt, 0.0_rt);

    mask_composite(solution, active_masks);
    if (m_pressure_singular)
      subtract_composite_mean(solution, active_masks);
  };

  const auto apply_pressure_inverse = [&](const CellHierarchy &input,
                                          CellHierarchy &solution) {
    BL_PROFILE("INSSolver::ApplyIBPressureInverse()");

    ++pressure_inverse_applications;
    apply_standard_poisson_inverse(input, solution);
    // A fixed correction polynomial keeps this map reproducible for BiCGStab.
    for (int correction = 0; correction < m_ib_schur_pressure_corrections;
         ++correction) {
      ApplyCompositeModifiedPoissonOp(solution, inverse_image, active_masks);
      ++modified_poisson_applications;
      lincomb_composite(inverse_residual, 1.0_rt, input, -1.0_rt, inverse_image,
                        active_masks);
      if (m_pressure_singular)
        subtract_composite_mean(inverse_residual, active_masks);
      apply_standard_poisson_inverse(inverse_residual, inverse_delta);
      saxpy_composite(solution, 1.0_rt, inverse_delta, active_masks);
      if (m_pressure_singular)
        subtract_composite_mean(solution, active_masks);
    }
  };

  CellHierarchy zero_p, force_p, pressure_rhs, pressure_rhs_response,
      pressure_response, pressure_image_p;
  define_composite_like(zero_p, p, 1);
  define_composite_like(force_p, rhs_p, 0);
  define_composite_like(pressure_rhs, rhs_p, 0);
  define_composite_like(pressure_rhs_response, p, 1);
  define_composite_like(pressure_response, p, 1);
  define_composite_like(pressure_image_p, rhs_p, 0);
  for (auto &level : zero_p)
    level->setVal(0.0_rt);

  MarkerVector force_ib(rhs_ib.size());
  MarkerVector pressure_ib(rhs_ib.size());
  MarkerVector base_pressure_ib(rhs_ib.size());
  set_vector(force_ib, 0.0_rt);
  set_vector(pressure_ib, 0.0_rt);
  set_vector(base_pressure_ib, 0.0_rt);
  int schur_applications = 0;

  const auto apply_force_blocks = [&](const MarkerVector &input_force) {
    scaling.to_physical_force(input_force, physical_force);
    ApplyCompositeProjectionBlocks(zero_p, &physical_force, force_p, &force_ib,
                                   active_masks);
  };

  const auto apply_pressure_blocks = [&](CellHierarchy &input_pressure) {
    ApplyCompositeProjectionBlocks(input_pressure, nullptr, pressure_image_p,
                                   &pressure_ib, active_masks);
  };

  const auto apply_schur = [&](const MarkerVector &input_force,
                               MarkerVector &result) {
    BL_PROFILE("INSSolver::ApplyIBForceSchurOp()");

    ++schur_applications;
    apply_force_blocks(input_force);
    // This affine form is algebraically M f - B P C f for linear P, but
    // avoids subtracting two large pressure fields before applying B.
    lincomb_composite(pressure_rhs, 1.0_rt, rhs_p, -1.0_rt, force_p,
                      active_masks);
    if (m_pressure_singular)
      subtract_composite_mean(pressure_rhs, active_masks);
    apply_pressure_inverse(pressure_rhs, pressure_response);
    apply_pressure_blocks(pressure_response);
    lincomb_vector(result, 1.0_rt, force_ib, 1.0_rt, pressure_ib);
    saxpy_vector(result, -1.0_rt, base_pressure_ib);
  };

  apply_pressure_inverse(rhs_p, pressure_rhs_response);
  apply_pressure_blocks(pressure_rhs_response);
  copy_vector(base_pressure_ib, pressure_ib);
  MarkerVector schur_rhs(rhs_ib.size());
  lincomb_vector(schur_rhs, 1.0_rt, rhs_ib, -1.0_rt, pressure_ib);

  const Real schur_rhs_norm2 = norm2_vector(schur_rhs);
  const Real schur_rhs_norm =
      std::sqrt(std::max(schur_rhs_norm2, Real(1.0e-300)));
  const Real schur_tol2 =
      m_poisson_tol * m_poisson_tol * std::max(schur_rhs_norm2, Real(1.0e-300));

  MarkerVector residual(rhs_ib.size());
  MarkerVector rhat(rhs_ib.size());
  MarkerVector search(rhs_ib.size());
  MarkerVector image(rhs_ib.size());
  MarkerVector intermediate(rhs_ib.size());
  MarkerVector image_intermediate(rhs_ib.size());
  set_vector(residual, 0.0_rt);
  set_vector(rhat, 0.0_rt);
  set_vector(search, 0.0_rt);
  set_vector(image, 0.0_rt);
  set_vector(intermediate, 0.0_rt);
  set_vector(image_intermediate, 0.0_rt);

  const auto evaluate_schur_residual = [&]() {
    apply_schur(scaled_force, image);
    lincomb_vector(residual, 1.0_rt, schur_rhs, -1.0_rt, image);
    return norm2_vector(residual);
  };

  Real schur_resid2 = evaluate_schur_residual();
  const Real initial_schur_resid = std::sqrt(std::max(schur_resid2, Real(0.0)));
  MarkerVector best_force;
  copy_vector(best_force, scaled_force);
  Real best_schur_resid2 = schur_resid2;

  constexpr int residual_refresh = 50;
  const char *breakdown = nullptr;
  int iter = 0;
  while (iter < m_poisson_max_iter) {
    if (!std::isfinite(schur_resid2)) {
      breakdown = "non-finite Schur residual";
      break;
    }
    if (schur_resid2 <= schur_tol2)
      break;

    copy_vector(rhat, residual);
    set_vector(search, 0.0_rt);
    set_vector(image, 0.0_rt);

    Real rho = 1.0_rt;
    Real alpha = 1.0_rt;
    Real omega = 1.0_rt;
    const int cycle_end = std::min(iter + residual_refresh, m_poisson_max_iter);
    bool cycle_breakdown = false;

    while (iter < cycle_end) {
      const Real rho_new = dot_vector(rhat, residual);
      if (!std::isfinite(rho_new) || std::abs(rho_new) < tiny) {
        breakdown = std::isfinite(rho_new) ? "rho breakdown" : "non-finite rho";
        cycle_breakdown = true;
        break;
      }
      const Real beta = (rho_new / rho) * (alpha / omega);
      if (!std::isfinite(beta)) {
        breakdown = "non-finite beta";
        cycle_breakdown = true;
        break;
      }

      lincomb_vector(search, 1.0_rt, search, -omega, image);
      lincomb_vector(search, beta, search, 1.0_rt, residual);
      apply_schur(search, image);

      const Real rhat_image = dot_vector(rhat, image);
      if (!std::isfinite(rhat_image) || std::abs(rhat_image) < tiny) {
        breakdown = std::isfinite(rhat_image) ? "rhat-v breakdown"
                                              : "non-finite rhat-v";
        cycle_breakdown = true;
        break;
      }
      alpha = rho_new / rhat_image;
      if (!std::isfinite(alpha)) {
        breakdown = "non-finite alpha";
        cycle_breakdown = true;
        break;
      }

      lincomb_vector(intermediate, 1.0_rt, residual, -alpha, image);
      const Real intermediate_resid2 = norm2_vector(intermediate);
      if (!std::isfinite(intermediate_resid2)) {
        breakdown = "non-finite Schur residual";
        cycle_breakdown = true;
        break;
      }
      if (intermediate_resid2 <= schur_tol2) {
        saxpy_vector(scaled_force, alpha, search);
        ++iter;
        break;
      }

      apply_schur(intermediate, image_intermediate);
      const Real image_norm2 = norm2_vector(image_intermediate);
      omega = (image_norm2 > tiny)
                  ? dot_vector(image_intermediate, intermediate) / image_norm2
                  : 0.0_rt;
      if (!std::isfinite(omega)) {
        breakdown = "non-finite omega";
        cycle_breakdown = true;
        break;
      }

      saxpy_vector(scaled_force, alpha, search);
      saxpy_vector(scaled_force, omega, intermediate);
      lincomb_vector(residual, 1.0_rt, intermediate, -omega,
                     image_intermediate);

      rho = rho_new;
      schur_resid2 = norm2_vector(residual);
      ++iter;
      if (std::abs(omega) < tiny) {
        breakdown = "omega breakdown";
        cycle_breakdown = true;
        break;
      }
      if (!std::isfinite(schur_resid2)) {
        breakdown = "non-finite Schur residual";
        cycle_breakdown = true;
        break;
      }
      if (schur_resid2 <= schur_tol2)
        break;
    }

    schur_resid2 = evaluate_schur_residual();
    if (std::isfinite(schur_resid2) && schur_resid2 < best_schur_resid2) {
      best_schur_resid2 = schur_resid2;
      copy_vector(best_force, scaled_force);
    }
    if (cycle_breakdown)
      break;
  }

  if (std::isfinite(best_schur_resid2) &&
      (!std::isfinite(schur_resid2) || best_schur_resid2 < schur_resid2)) {
    copy_vector(scaled_force, best_force);
    schur_resid2 = evaluate_schur_residual();
  }

  apply_force_blocks(scaled_force);
  lincomb_composite(pressure_rhs, 1.0_rt, rhs_p, -1.0_rt, force_p,
                    active_masks);
  if (m_pressure_singular)
    subtract_composite_mean(pressure_rhs, active_masks);
  apply_pressure_inverse(pressure_rhs, p);
  const Real pressure_rhs_norm = std::sqrt(
      std::max(dot_composite(pressure_rhs, pressure_rhs), Real(1.0e-300)));
  const Real max_pressure_row = *std::max_element(scaling.pressure_rows.begin(),
                                                  scaling.pressure_rows.end());
  const Real pressure_recovery_tol =
      std::min(m_poisson_tol,
               0.25_rt * coupled_tol / (max_pressure_row * pressure_rhs_norm));
  if (m_verbose > 2) {
    ApplyCompositeModifiedPoissonOp(p, inverse_image, active_masks);
    lincomb_composite(inverse_residual, 1.0_rt, pressure_rhs, -1.0_rt,
                      inverse_image, active_masks);
    Print() << "    approximate pressure residual = "
            << std::sqrt(dot_composite(inverse_residual, inverse_residual) /
                         std::max(dot_composite(pressure_rhs, pressure_rhs),
                                  Real(1.0e-300)))
            << "\n";
  }
  const int pressure_recovery_iters = SolveCompositeModifiedPoisson(
      p, pressure_rhs, active_masks, pressure_recovery_tol);
  if (m_verbose > 2) {
    ApplyCompositeModifiedPoissonOp(p, inverse_image, active_masks);
    lincomb_composite(inverse_residual, 1.0_rt, pressure_rhs, -1.0_rt,
                      inverse_image, active_masks);
    Print() << "    recovered pressure residual = "
            << std::sqrt(dot_composite(inverse_residual, inverse_residual) /
                         std::max(dot_composite(pressure_rhs, pressure_rhs),
                                  Real(1.0e-300)))
            << "\n";
  }
  scaling.to_physical_force(scaled_force, m_ib_force);

  CellHierarchy coupled_image_p, coupled_residual_p, scaled_residual_p;
  define_composite_like(coupled_image_p, rhs_p, 0);
  define_composite_like(coupled_residual_p, rhs_p, 0);
  define_composite_like(scaled_residual_p, rhs_p, 0);
  MarkerVector coupled_image_ib(rhs_ib.size());
  MarkerVector coupled_residual_ib(rhs_ib.size());
  set_vector(coupled_image_ib, 0.0_rt);
  set_vector(coupled_residual_ib, 0.0_rt);

  Real coupled_resid = std::numeric_limits<Real>::infinity();
  const auto evaluate_physical_coupled_residual = [&]() {
    scaling.to_physical_force(scaled_force, m_ib_force);
    ApplyCompositeIBProjectionOp(p, m_ib_force, coupled_image_p,
                                 coupled_image_ib, active_masks);
    lincomb_composite(coupled_residual_p, 1.0_rt, rhs_p, -1.0_rt,
                      coupled_image_p, active_masks);
    lincomb_vector(coupled_residual_ib, 1.0_rt, rhs_ib, -1.0_rt,
                   coupled_image_ib);

    copy_composite(scaled_residual_p, coupled_residual_p, active_masks);
    scaling.scale_pressure(scaled_residual_p);
    CoupledResidual result{
        dot_composite(coupled_residual_p, coupled_residual_p),
        norm2_vector(coupled_residual_ib)};
    coupled_resid = std::sqrt(std::max(
        dot_composite(scaled_residual_p, scaled_residual_p) + result.ib_norm2,
        Real(0.0)));
    return result;
  };

  CoupledResidual final_residual = evaluate_physical_coupled_residual();
  const Real post_schur_coupled_resid = coupled_resid;
  if (coupled_resid > 10.0_rt * coupled_tol) {
    copy_composite_unmasked(p, initial_p);
    copy_vector(scaled_force, initial_scaled_force);
    final_residual = evaluate_physical_coupled_residual();
    if (!m_pressure_singular) {
      CellHierarchy saved_initial_p;
      define_composite_like(saved_initial_p, p, 1);
      copy_composite_unmasked(saved_initial_p, p);
      const Real initial_resid = coupled_resid;

      apply_force_blocks(scaled_force);
      lincomb_composite(pressure_rhs, 1.0_rt, rhs_p, -1.0_rt, force_p,
                        active_masks);
      BuildCompositePoissonInitialGuess(p, pressure_rhs, active_masks);
      final_residual = evaluate_physical_coupled_residual();
      if (!std::isfinite(coupled_resid) || coupled_resid >= initial_resid) {
        copy_composite_unmasked(p, saved_initial_p);
        final_residual = evaluate_physical_coupled_residual();
      }
    }
  }
  const Real refinement_initial_resid = coupled_resid;
  Real refinement_resid = coupled_resid;
  const char *refinement_breakdown = nullptr;
  int refinement_iters = 0;

  if (!std::isfinite(coupled_resid) || coupled_resid > coupled_tol) {
    CellHierarchy refinement_rhat, refinement_search, refinement_image,
        refinement_intermediate, refinement_intermediate_image;
    define_composite_like(refinement_rhat, p, 0);
    define_composite_like(refinement_search, p, 1);
    define_composite_like(refinement_image, p, 0);
    define_composite_like(refinement_intermediate, p, 1);
    define_composite_like(refinement_intermediate_image, p, 0);
    MarkerVector refinement_rhat_ib(rhs_ib.size());
    MarkerVector refinement_search_ib(rhs_ib.size());
    MarkerVector refinement_image_ib(rhs_ib.size());
    MarkerVector refinement_intermediate_ib(rhs_ib.size());
    MarkerVector refinement_intermediate_image_ib(rhs_ib.size());
    set_vector(refinement_rhat_ib, 0.0_rt);
    set_vector(refinement_search_ib, 0.0_rt);
    set_vector(refinement_image_ib, 0.0_rt);
    set_vector(refinement_intermediate_ib, 0.0_rt);
    set_vector(refinement_intermediate_image_ib, 0.0_rt);

    const auto apply_scaled_coupled =
        [&](CellHierarchy &input_p, const MarkerVector &input_force,
            CellHierarchy &result_p, MarkerVector &result_ib) {
          scaling.to_physical_force(input_force, physical_force);
          ApplyCompositeIBProjectionOp(input_p, physical_force, result_p,
                                       result_ib, active_masks);
          scaling.scale_pressure(result_p);
        };

    const auto evaluate_scaled_coupled_residual = [&]() {
      apply_scaled_coupled(p, scaled_force, coupled_image_p, coupled_image_ib);
      lincomb_composite(scaled_residual_p, 1.0_rt, scaled_rhs_p, -1.0_rt,
                        coupled_image_p, active_masks);
      lincomb_vector(coupled_residual_ib, 1.0_rt, rhs_ib, -1.0_rt,
                     coupled_image_ib);
      return std::sqrt(
          std::max(dot_composite(scaled_residual_p, scaled_residual_p) +
                       norm2_vector(coupled_residual_ib),
                   Real(0.0)));
    };

    refinement_resid = evaluate_scaled_coupled_residual();
    CellHierarchy best_refinement_p;
    define_composite_like(best_refinement_p, p, 1);
    copy_composite_unmasked(best_refinement_p, p);
    MarkerVector best_refinement_force;
    copy_vector(best_refinement_force, scaled_force);
    Real best_refinement_resid = refinement_resid;

    constexpr int coupled_residual_refresh = 100;
    while (refinement_iters < m_poisson_max_iter) {
      if (!std::isfinite(refinement_resid)) {
        refinement_breakdown = "non-finite coupled residual";
        break;
      }
      if (refinement_resid <= coupled_tol)
        break;

      copy_composite(refinement_rhat, scaled_residual_p, active_masks);
      copy_vector(refinement_rhat_ib, coupled_residual_ib);
      for (auto &level : refinement_search)
        level->setVal(0.0_rt);
      for (auto &level : refinement_image)
        level->setVal(0.0_rt);
      set_vector(refinement_search_ib, 0.0_rt);
      set_vector(refinement_image_ib, 0.0_rt);

      Real rho = 1.0_rt;
      Real alpha = 1.0_rt;
      Real omega = 1.0_rt;
      const int cycle_end = std::min(
          refinement_iters + coupled_residual_refresh, m_poisson_max_iter);
      bool cycle_breakdown = false;

      while (refinement_iters < cycle_end) {
        const Real rho_new =
            dot_coupled(refinement_rhat, refinement_rhat_ib, scaled_residual_p,
                        coupled_residual_ib);
        if (!std::isfinite(rho_new) || std::abs(rho_new) < tiny) {
          refinement_breakdown =
              std::isfinite(rho_new) ? "rho breakdown" : "non-finite rho";
          cycle_breakdown = true;
          break;
        }
        const Real beta = (rho_new / rho) * (alpha / omega);
        if (!std::isfinite(beta)) {
          refinement_breakdown = "non-finite beta";
          cycle_breakdown = true;
          break;
        }

        lincomb_coupled(refinement_search, refinement_search_ib, 1.0_rt,
                        refinement_search, refinement_search_ib, -omega,
                        refinement_image, refinement_image_ib, active_masks);
        lincomb_coupled(refinement_search, refinement_search_ib, beta,
                        refinement_search, refinement_search_ib, 1.0_rt,
                        scaled_residual_p, coupled_residual_ib, active_masks);

        apply_scaled_coupled(refinement_search, refinement_search_ib,
                             refinement_image, refinement_image_ib);
        const Real rhat_image =
            dot_coupled(refinement_rhat, refinement_rhat_ib, refinement_image,
                        refinement_image_ib);
        if (!std::isfinite(rhat_image) || std::abs(rhat_image) < tiny) {
          refinement_breakdown = std::isfinite(rhat_image)
                                     ? "rhat-v breakdown"
                                     : "non-finite rhat-v";
          cycle_breakdown = true;
          break;
        }
        alpha = rho_new / rhat_image;

        lincomb_coupled(refinement_intermediate, refinement_intermediate_ib,
                        1.0_rt, scaled_residual_p, coupled_residual_ib, -alpha,
                        refinement_image, refinement_image_ib, active_masks);
        const Real intermediate_resid2 =
            dot_coupled(refinement_intermediate, refinement_intermediate_ib,
                        refinement_intermediate, refinement_intermediate_ib);
        if (intermediate_resid2 <= coupled_tol * coupled_tol) {
          saxpy_coupled(p, scaled_force, alpha, refinement_search,
                        refinement_search_ib, active_masks);
          ++refinement_iters;
          break;
        }

        apply_scaled_coupled(
            refinement_intermediate, refinement_intermediate_ib,
            refinement_intermediate_image, refinement_intermediate_image_ib);
        const Real image_norm2 = dot_coupled(
            refinement_intermediate_image, refinement_intermediate_image_ib,
            refinement_intermediate_image, refinement_intermediate_image_ib);
        omega = (image_norm2 > tiny)
                    ? dot_coupled(refinement_intermediate_image,
                                  refinement_intermediate_image_ib,
                                  refinement_intermediate,
                                  refinement_intermediate_ib) /
                          image_norm2
                    : 0.0_rt;
        if (!std::isfinite(omega)) {
          refinement_breakdown = "non-finite omega";
          cycle_breakdown = true;
          break;
        }

        saxpy_coupled(p, scaled_force, alpha, refinement_search,
                      refinement_search_ib, active_masks);
        saxpy_coupled(p, scaled_force, omega, refinement_intermediate,
                      refinement_intermediate_ib, active_masks);
        lincomb_coupled(scaled_residual_p, coupled_residual_ib, 1.0_rt,
                        refinement_intermediate, refinement_intermediate_ib,
                        -omega, refinement_intermediate_image,
                        refinement_intermediate_image_ib, active_masks);

        rho = rho_new;
        ++refinement_iters;
        if (std::abs(omega) < tiny) {
          refinement_breakdown = "omega breakdown";
          cycle_breakdown = true;
          break;
        }
        const Real recursive_resid2 =
            dot_coupled(scaled_residual_p, coupled_residual_ib,
                        scaled_residual_p, coupled_residual_ib);
        if (!std::isfinite(recursive_resid2)) {
          refinement_breakdown = "non-finite coupled residual";
          cycle_breakdown = true;
          break;
        }
        if (recursive_resid2 <= coupled_tol * coupled_tol)
          break;
      }

      refinement_resid = evaluate_scaled_coupled_residual();
      if (std::isfinite(refinement_resid) &&
          refinement_resid < best_refinement_resid) {
        best_refinement_resid = refinement_resid;
        copy_composite_unmasked(best_refinement_p, p);
        copy_vector(best_refinement_force, scaled_force);
      }
      if (cycle_breakdown)
        break;
    }

    if (std::isfinite(best_refinement_resid) &&
        (!std::isfinite(refinement_resid) ||
         best_refinement_resid < refinement_resid)) {
      copy_composite_unmasked(p, best_refinement_p);
      copy_vector(scaled_force, best_refinement_force);
      refinement_resid = evaluate_scaled_coupled_residual();
    }
    if (m_pressure_singular)
      subtract_composite_mean(p, active_masks);
    final_residual = evaluate_physical_coupled_residual();
  }

  const bool converged =
      std::isfinite(coupled_resid) && coupled_resid <= coupled_tol;

  const Real schur_resid = std::sqrt(std::max(schur_resid2, Real(0.0)));
  const Real schur_relres = schur_resid / schur_rhs_norm;
  const Real average_rate =
      (iter > 0 && initial_schur_resid > 0.0_rt && std::isfinite(schur_resid))
          ? std::pow(schur_resid / initial_schur_resid,
                     1.0_rt / static_cast<Real>(iter))
          : 0.0_rt;
  const Real refinement_rate =
      (refinement_iters > 0 && refinement_initial_resid > 0.0_rt &&
       std::isfinite(coupled_resid))
          ? std::pow(coupled_resid / refinement_initial_resid,
                     1.0_rt / static_cast<Real>(refinement_iters))
          : 0.0_rt;
  const Real scaled_relres = coupled_resid / rhs_norm;
  const Real post_schur_scaled_relres = post_schur_coupled_resid / rhs_norm;

  if (!converged) {
    const Real abs_p =
        std::sqrt(std::max(final_residual.pressure_norm2, Real(0.0)));
    const Real abs_ib = std::sqrt(std::max(final_residual.ib_norm2, Real(0.0)));
    const Real rhs_p_norm = std::sqrt(std::max(rhs_p_norm2, Real(0.0)));
    const Real rhs_ib_norm = std::sqrt(std::max(rhs_ib_norm2, Real(0.0)));
    const Real relres_p = abs_p / std::max(rhs_p_norm, Real(1.0e-300));
    const Real relres_ib =
        (rhs_ib_norm > Real(1.0e-300))
            ? abs_ib / rhs_ib_norm
            : (abs_ib == Real(0.0) ? Real(0.0)
                                   : std::numeric_limits<Real>::infinity());
    std::ostringstream msg;
    msg << "Composite IB force-Schur solve";
    if (refinement_breakdown != nullptr) {
      msg << " coupled refinement stopped by " << refinement_breakdown;
    } else if (refinement_iters >= m_poisson_max_iter) {
      msg << " coupled refinement reached max_iter";
    } else if (breakdown != nullptr) {
      msg << " stopped by " << breakdown;
    } else if (iter >= m_poisson_max_iter) {
      msg << " reached max_iter";
    } else {
      msg << " failed the true coupled-residual check";
    }
    msg << " (levels = " << finest_level + 1
        << ", finest_level = " << finest_level
        << ", markers = " << m_ib_geometry.markers.size()
        << ", constraints = " << rhs_ib.size()
        << ", pressure_singular = " << m_pressure_singular
        << ", Schur_relres = " << schur_relres
        << ", Schur_average_rate = " << average_rate
        << ", refinement_iters = " << refinement_iters
        << ", refinement_average_rate = " << refinement_rate
        << ", scaled_relres = " << scaled_relres
        << ", pressure_relres = " << relres_p << ", ib_relres = " << relres_ib
        << ", |r_p| = " << abs_p << ", |rhs_p| = " << rhs_p_norm
        << ", |r_ib| = " << abs_ib << ", |rhs_ib| = " << rhs_ib_norm
        << ", pressure_MLMG_iters = " << m_ib_schur_mg_iters
        << ", pressure_corrections = " << m_ib_schur_pressure_corrections
        << ").";
    amrex::Warning(msg.str());
  }

  if (m_verbose > 1) {
    Print() << "  Composite IB force-Schur BiCGStab: " << iter
            << " iters, Schur |r|/|rhs| = " << schur_relres
            << ", avg factor/iter = " << average_rate
            << ", post-Schur true scaled |r|/|rhs| = "
            << post_schur_scaled_relres << "\n"
            << "    pressure inverse: " << pressure_inverse_applications
            << " applies, " << standard_poisson_applications
            << " standard-Poisson applications, "
            << modified_poisson_applications
            << " modified-op corrections; Schur applies = "
            << schur_applications
            << ", exact pressure recovery iters = " << pressure_recovery_iters
            << " at reltol " << pressure_recovery_tol << "\n";
    if (refinement_iters > 0) {
      Print() << "    exact coupled refinement: " << refinement_iters
              << " iters, avg factor/iter = " << refinement_rate
              << ", scaled |r|/|rhs| = " << scaled_relres << "\n";
    }
  }
  return iter;
}

// ============================================================
//  CheckOutflowPressurePin — constant p must not be a null mode
//                            when outflow p=0 is active.
// ============================================================
void INSSolver::CheckOutflowPressurePin() {
  BL_PROFILE("INSSolver::CheckOutflowPressurePin()");

  if (m_pressure_singular) {
    if (m_verbose > 0)
      Print() << "Pressure-pin check skipped: no outflow Dirichlet face\n";
    return;
  }

  const int nlev = finest_level + 1;
  auto active_masks = BuildCompositeActiveMasks();
  CellHierarchy phi(nlev);
  CellHierarchy Aphi(nlev);
  for (int lev = 0; lev < nlev; ++lev) {
    phi[lev] = std::make_unique<MultiFab>(grids[lev], dmap[lev], 1, 1);
    Aphi[lev] = std::make_unique<MultiFab>(grids[lev], dmap[lev], 1, 0);
    phi[lev]->setVal(1.0);
    Aphi[lev]->setVal(0.0);
  }
  mask_composite(phi, active_masks);
  ApplyCompositeModifiedPoissonOp(phi, Aphi, active_masks);

  Real norm = 0.0_rt;
  for (int lev = 0; lev < nlev; ++lev)
    norm = std::max(norm, Aphi[lev]->norminf(0, 0));
  ParallelDescriptor::ReduceRealMax(norm);
  if (m_verbose > 0)
    Print() << "Pressure-pin check: composite |-D B^N G 1|_inf = " << norm
            << "\n";

  const Real threshold = 1024.0_rt * std::numeric_limits<Real>::epsilon();
  if (!std::isfinite(norm) || norm <= threshold) {
    amrex::Abort("Pressure-pin check failed: outflow Dirichlet pressure "
                 "left the constant mode in the operator nullspace.");
  }
}

// ============================================================
//  ProjectPerot — orchestrate the hierarchy Perot projection.
// ============================================================
void INSSolver::ProjectPerot() {
  BL_PROFILE("INSSolver::ProjectPerot()");

  const int nlev = finest_level + 1;
  const bool use_ib =
      m_ib_enabled && !m_ib_geometry.markers.empty() && finest_level >= 0;

  // Sync u* fine→coarse so coarse face values at C/F equal the
  // averaged fine flux before the hierarchy projection builds its RHS.
  AverageDownVelocity(m_vstar);

  auto active_masks = BuildCompositeActiveMasks();

  CellHierarchy rhs(nlev);
  for (int lev = 0; lev < nlev; ++lev) {
    const BoxArray &ba = grids[lev];
    const DistributionMapping &dm = dmap[lev];

    // RHS = −(1/dt) D u*   (negated to pair with the negated operator)
    rhs[lev] = std::make_unique<MultiFab>(ba, dm, 1, 0);
    rhs[lev]->setVal(0.0);

    const Real *dx = geom[lev].CellSize();
    const Real cx = -1.0_rt / (m_dt * dx[0]);
    const Real cy = -1.0_rt / (m_dt * dx[1]);
#if AMREX_SPACEDIM == 3
    const Real cz = -1.0_rt / (m_dt * dx[2]);
#endif

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*rhs[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const Box &bx = mfi.tilebox();
      auto const &u = m_vstar[lev][0]->const_array(mfi);
      auto const &v = m_vstar[lev][1]->const_array(mfi);
#if AMREX_SPACEDIM == 3
      auto const &w = m_vstar[lev][2]->const_array(mfi);
#endif
      auto const &r = rhs[lev]->array(mfi);
      amrex::ParallelFor(bx,
                         [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                           Real d = cx * (u(i + 1, j, k) - u(i, j, k)) +
                                    cy * (v(i, j + 1, k) - v(i, j, k));
#if AMREX_SPACEDIM == 3
                           d += cz * (w(i, j, k + 1) - w(i, j, k));
#endif
                           r(i, j, k) = d;
                         });
    }
    mask_covered_cells(*rhs[lev], *active_masks[lev]);
  }

  if (use_ib) {
    MarkerVector rhs_ib;
    InterpolateIBVelocity(finest_level,
                          {AMREX_D_DECL(m_vstar[finest_level][0].get(),
                                        m_vstar[finest_level][1].get(),
                                        m_vstar[finest_level][2].get())},
                          rhs_ib);
    GpuArray<Real, AMREX_SPACEDIM> ib_velocity{};
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
      ib_velocity[d] = m_ib_velocity[d];
    Real *rhs_ib_p = rhs_ib.data();
    const Real dt = m_dt;
    amrex::ParallelFor(static_cast<Long>(rhs_ib.size()), [=] AMREX_GPU_DEVICE(
                                                             Long i) noexcept {
      rhs_ib_p[i] = (rhs_ib_p[i] - ib_velocity[i % AMREX_SPACEDIM]) / dt;
    });

    SolveCompositeIBProjection(m_pressure, rhs, rhs_ib, active_masks);
  } else {
    // Solve (D B^N G) p^{n+1} = rhs across the pressure hierarchy.
    SolveCompositeModifiedPoisson(m_pressure, rhs, active_masks);
  }

  AverageDownPressure();

  Vector<FaceMFArray> projection_source(nlev);
  Vector<FaceMFArray> projection_correction(nlev);
  for (int lev = 0; lev < nlev; ++lev) {
    const BoxArray &ba = grids[lev];
    const DistributionMapping &dm = dmap[lev];
    const bool use_ib_on_level = use_ib && lev == finest_level;

    // Projection:  u^{n+1} = u* − dt B^N G p^{n+1}
    //  (same B^N as in the predictor and the operator — this is the
    //   Perot consistency that makes the projection exact.)
    // Same C/F (Dirichlet-from-coarse) + domain BC fill as the hierarchy
    // operator apply; IB force is applied only on the finest level.
    MultiFab phi_g(ba, dm, 1, 1);
    FillCellPatch(lev, m_pressure, phi_g, m_cur_time, m_bc_pres);

    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
      BoxArray fba = amrex::convert(ba, IntVect::TheDimensionVector(d));
      projection_source[lev][d] = std::make_unique<MultiFab>(fba, dm, 1, 1);
      projection_correction[lev][d] = std::make_unique<MultiFab>(fba, dm, 1, 0);

      projection_source[lev][d]->setVal(0.0);
      ComputePressureGradient(lev, d, phi_g, *projection_source[lev][d]);
      if (use_ib_on_level) {
        MultiFab hf(fba, dm, 1, 0);
        SpreadIBForce(lev, d, m_ib_force, hf);
        MultiFab::Add(*projection_source[lev][d], hf, 0, 0, 1, 0);
      }
    }
  }

  ApplyCompositeBNFaces(projection_source, projection_correction);

  for (int lev = 0; lev < nlev; ++lev) {
    const bool use_ib_on_level = use_ib && lev == finest_level;
    std::array<MultiFab *, AMREX_SPACEDIM> vel_out;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
      MultiFab::Copy(*m_vel[lev][d], *m_vstar[lev][d], 0, 0, 1, 0);
      MultiFab::Saxpy(*m_vel[lev][d], -m_dt, *projection_correction[lev][d], 0,
                      0, 1, 0);
      vel_out[d] = m_vel[lev][d].get();
    }

    // Re-impose the prescribed wall velocity (the projection
    // does not preserve inhomogeneous Dirichlet data exactly).
    EnforceVelDirichlet(lev, vel_out);

    if (use_ib_on_level && m_verbose > 1) {
      MarkerVector marker_vel;
      InterpolateIBVelocity(
          lev,
          {AMREX_D_DECL(m_vel[lev][0].get(), m_vel[lev][1].get(),
                        m_vel[lev][2].get())},
          marker_vel);
      GpuArray<Real, AMREX_SPACEDIM> ib_velocity{};
      for (int d = 0; d < AMREX_SPACEDIM; ++d)
        ib_velocity[d] = m_ib_velocity[d];
      const Real slip = marker_slip_norm_inf(marker_vel, ib_velocity);
      Print() << "  |E u - U_ib|_inf = " << slip << "\n";
    }
  }
}
