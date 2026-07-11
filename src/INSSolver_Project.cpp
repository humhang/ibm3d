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
 * gradient, face-Laplacian, divergence pieces).  The pressure-only
 * hierarchy solve and the coupled AMR IB block use BiCGStab.  AMReX's
 * composite Poisson solve supplies a checked
 * pressure-block initial guess for non-singular modified/IB systems.
 *
 * Multi-level projection uses a hierarchy-wide Krylov solve whose fine
 * pressure ghosts are filled from the current coarse iterate.  With IB
 * enabled, the Krylov vector spans all pressure levels plus the
 * Lagrangian force unknowns, while the IB rows/force columns are applied
 * only on the finest level; applying the same IB rows on coarser levels
 * makes the marker constraints over-dense and singular.
 */

#include "INSSolver.H"

#include <AMReX.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLPoisson.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <vector>

using namespace amrex;

namespace {

using CellHierarchy = Vector<std::unique_ptr<MultiFab>>;
using MaskHierarchy = Vector<std::unique_ptr<iMultiFab>>;

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

Real dot_vector(const std::vector<Real> &a, const std::vector<Real> &b) {
  return std::inner_product(a.begin(), a.end(), b.begin(), Real(0.0));
}

void saxpy_vector(std::vector<Real> &dst, Real a, const std::vector<Real> &x) {
  for (std::size_t i = 0; i < dst.size(); ++i)
    dst[i] += a * x[i];
}

void lincomb_vector(std::vector<Real> &dst, Real a, const std::vector<Real> &x,
                    Real b, const std::vector<Real> &y) {
  for (std::size_t i = 0; i < dst.size(); ++i)
    dst[i] = a * x[i] + b * y[i];
}

Real dot_coupled(const CellHierarchy &a_p, const std::vector<Real> &a_ib,
                 const CellHierarchy &b_p, const std::vector<Real> &b_ib) {
  return dot_composite(a_p, b_p) + dot_vector(a_ib, b_ib);
}

Real norm2_vector(const std::vector<Real> &a) { return dot_vector(a, a); }

struct CoupledResidual {
  Real pressure_norm2;
  Real ib_norm2;

  [[nodiscard]] Real norm2() const { return pressure_norm2 + ib_norm2; }

  [[nodiscard]] Real norm() const {
    return std::sqrt(std::max(norm2(), Real(0.0)));
  }
};

// H already contains marker quadrature.  This scaling equilibrates all four
// uniform-grid blocks at O(1/h) and recovers their adjoint scaling.
struct CoupledScaling {
  CoupledScaling(const Vector<Geometry> &geometry, int finest_level,
                 const std::vector<ibm3d::IBMarker> &markers)
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
    for (std::size_t marker = 0; marker < markers.size(); ++marker) {
      const Real scale = marker_measure / markers[marker].weight;
      for (int d = 0; d < AMREX_SPACEDIM; ++d)
        force_columns[marker * AMREX_SPACEDIM + d] = scale;
    }
  }

  void scale_pressure(CellHierarchy &values) const {
    AMREX_ALWAYS_ASSERT(values.size() == pressure_rows.size());
    for (int lev = 0; lev < static_cast<int>(values.size()); ++lev)
      values[lev]->mult(pressure_rows[lev], 0, 1, 0);
  }

  void unscale_pressure(CellHierarchy &values) const {
    AMREX_ALWAYS_ASSERT(values.size() == pressure_rows.size());
    for (int lev = 0; lev < static_cast<int>(values.size()); ++lev)
      values[lev]->mult(1.0_rt / pressure_rows[lev], 0, 1, 0);
  }

  void to_scaled_force(const std::vector<Real> &physical,
                       std::vector<Real> &scaled) const {
    AMREX_ALWAYS_ASSERT(physical.size() == force_columns.size());
    scaled.resize(physical.size());
    for (std::size_t i = 0; i < physical.size(); ++i)
      scaled[i] = physical[i] / force_columns[i];
  }

  void to_physical_force(const std::vector<Real> &scaled,
                         std::vector<Real> &physical) const {
    AMREX_ALWAYS_ASSERT(scaled.size() == force_columns.size());
    physical.resize(scaled.size());
    for (std::size_t i = 0; i < scaled.size(); ++i)
      physical[i] = force_columns[i] * scaled[i];
  }

  Vector<Real> pressure_rows;
  std::vector<Real> force_columns;
};

void saxpy_coupled(CellHierarchy &dst_p, std::vector<Real> &dst_ib, Real a,
                   const CellHierarchy &x_p, const std::vector<Real> &x_ib,
                   const MaskHierarchy &mask) {
  saxpy_composite(dst_p, a, x_p, mask);
  saxpy_vector(dst_ib, a, x_ib);
}

void lincomb_coupled(CellHierarchy &dst_p, std::vector<Real> &dst_ib, Real a,
                     const CellHierarchy &x_p, const std::vector<Real> &x_ib,
                     Real b, const CellHierarchy &y_p,
                     const std::vector<Real> &y_ib, const MaskHierarchy &mask) {
  lincomb_composite(dst_p, a, x_p, b, y_p, mask);
  lincomb_vector(dst_ib, a, x_ib, b, y_ib);
}

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
    CellHierarchy &phi, const std::vector<Real> *force, CellHierarchy &result_p,
    std::vector<Real> *result_ib, const MaskHierarchy &active_masks) {
  AMREX_ASSERT(result_ib == nullptr || force != nullptr);

  CellHierarchy phi_sync;
  define_composite_like(phi_sync, phi, 1);
  copy_composite_unmasked(phi_sync, phi);
  for (int lev = finest_level - 1; lev >= 0; --lev)
    amrex::average_down(*phi_sync[lev + 1], *phi_sync[lev], 0, 1,
                        ref_ratio[lev]);

  if (result_ib != nullptr)
    result_ib->assign(force->size(), 0.0_rt);

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
    CellHierarchy &phi, const std::vector<Real> &force, CellHierarchy &result_p,
    std::vector<Real> &result_ib, const MaskHierarchy &active_masks) {
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

int INSSolver::SolveCompositeModifiedPoisson(
    CellHierarchy &p, const CellHierarchy &rhs_in,
    const MaskHierarchy &active_masks) {
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
  const Real tol2 =
      m_poisson_tol * m_poisson_tol * std::max(rhs_norm2, Real(1.0e-300));

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
                                          const std::vector<Real> &rhs_ib,
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
  const CoupledScaling scaling(geom, finest_level, m_ib_geometry.markers);
  std::vector<Real> scaled_force;
  std::vector<Real> physical_force;
  scaling.to_scaled_force(m_ib_force, scaled_force);

  CellHierarchy scaled_rhs_p;
  define_composite_like(scaled_rhs_p, rhs_p, 0);
  copy_composite(scaled_rhs_p, rhs_p, active_masks);
  scaling.scale_pressure(scaled_rhs_p);

  const Real rhs_p_norm2 = dot_composite(rhs_p, rhs_p);
  const Real rhs_ib_norm2 = dot_vector(rhs_ib, rhs_ib);
  const Real rhs_norm2 =
      dot_composite(scaled_rhs_p, scaled_rhs_p) + rhs_ib_norm2;
  const Real rhs_norm = std::sqrt(std::max(rhs_norm2, Real(1.0e-300)));
  const Real tol = m_poisson_tol * rhs_norm;
  const Real tiny = 1.0e-300;
  const Real tol2 =
      m_poisson_tol * m_poisson_tol * std::max(rhs_norm2, Real(1.0e-300));

  CellHierarchy Ax_p, rr_p;
  define_composite_like(Ax_p, p, 0);
  define_composite_like(rr_p, p, 1);
  std::vector<Real> Ax_ib(rhs_ib.size(), 0.0_rt);
  std::vector<Real> rr_ib(rhs_ib.size(), 0.0_rt);

  const auto apply_scaled_operator =
      [&](CellHierarchy &input_p, const std::vector<Real> &input_force,
          CellHierarchy &result_p, std::vector<Real> &result_ib) {
        scaling.to_physical_force(input_force, physical_force);
        ApplyCompositeIBProjectionOp(input_p, physical_force, result_p,
                                     result_ib, active_masks);
        scaling.scale_pressure(result_p);
      };

  const auto evaluate_residual = [&]() {
    apply_scaled_operator(p, scaled_force, Ax_p, Ax_ib);
    lincomb_composite(rr_p, 1.0_rt, scaled_rhs_p, -1.0_rt, Ax_p, active_masks);
    for (std::size_t i = 0; i < rr_ib.size(); ++i)
      rr_ib[i] = rhs_ib[i] - Ax_ib[i];
    return CoupledResidual{dot_composite(rr_p, rr_p), norm2_vector(rr_ib)};
  };

  Real resid = evaluate_residual().norm();

  CellHierarchy best_p;
  define_composite_like(best_p, p, 1);
  copy_composite_unmasked(best_p, p);
  std::vector<Real> best_ib = scaled_force;
  Real best_resid = resid;

  const auto remember_current_if_better = [&](Real candidate_resid) {
    if (!std::isfinite(candidate_resid) || candidate_resid >= best_resid)
      return false;
    copy_composite_unmasked(best_p, p);
    best_ib = scaled_force;
    best_resid = candidate_resid;
    return true;
  };

  if (!m_pressure_singular) {
    CellHierarchy saved_p, zero_p, force_p, guess_rhs;
    define_composite_like(saved_p, p, 1);
    define_composite_like(zero_p, p, 1);
    define_composite_like(force_p, p, 0);
    define_composite_like(guess_rhs, rhs_p, 0);
    copy_composite_unmasked(saved_p, p);

    scaling.to_physical_force(scaled_force, physical_force);
    ApplyCompositeProjectionBlocks(zero_p, &physical_force, force_p, nullptr,
                                   active_masks);
    lincomb_composite(guess_rhs, 1.0_rt, rhs_p, -1.0_rt, force_p, active_masks);
    BuildCompositePoissonInitialGuess(p, guess_rhs, active_masks);

    const Real warm_resid = evaluate_residual().norm();
    if (!remember_current_if_better(warm_resid))
      copy_composite_unmasked(p, saved_p);
  }

  CellHierarchy rhat_p, search_p, image_p, intermediate_p, image_intermediate_p;
  define_composite_like(rhat_p, p, 0);
  define_composite_like(search_p, p, 1);
  define_composite_like(image_p, p, 0);
  define_composite_like(intermediate_p, p, 1);
  define_composite_like(image_intermediate_p, p, 0);
  std::vector<Real> rhat_ib(rhs_ib.size(), 0.0_rt);
  std::vector<Real> search_ib(rhs_ib.size(), 0.0_rt);
  std::vector<Real> image_ib(rhs_ib.size(), 0.0_rt);
  std::vector<Real> intermediate_ib(rhs_ib.size(), 0.0_rt);
  std::vector<Real> image_intermediate_ib(rhs_ib.size(), 0.0_rt);

  constexpr int residual_refresh = 100;
  const char *breakdown = nullptr;
  int iter = 0;
  while (iter < m_poisson_max_iter) {
    const CoupledResidual exact_residual = evaluate_residual();
    const Real exact_resid2 = exact_residual.norm2();
    resid = exact_residual.norm();
    remember_current_if_better(resid);
    if (!std::isfinite(exact_resid2)) {
      breakdown = "non-finite residual";
      break;
    }
    if (exact_resid2 <= tol2)
      break;

    copy_composite(rhat_p, rr_p, active_masks);
    rhat_ib = rr_ib;
    for (auto &level : search_p)
      level->setVal(0.0);
    for (auto &level : image_p)
      level->setVal(0.0);
    std::fill(search_ib.begin(), search_ib.end(), 0.0_rt);
    std::fill(image_ib.begin(), image_ib.end(), 0.0_rt);

    Real rho = 1.0_rt;
    Real alpha = 1.0_rt;
    Real omega = 1.0_rt;
    const int cycle_end = std::min(iter + residual_refresh, m_poisson_max_iter);
    bool cycle_breakdown = false;

    while (iter < cycle_end) {
      const Real rho_new = dot_coupled(rhat_p, rhat_ib, rr_p, rr_ib);
      if (!std::isfinite(rho_new)) {
        breakdown = "non-finite rho";
        cycle_breakdown = true;
        break;
      }
      if (std::abs(rho_new) < tiny) {
        breakdown = "rho breakdown";
        cycle_breakdown = true;
        break;
      }
      const Real beta = (rho_new / rho) * (alpha / omega);
      if (!std::isfinite(beta)) {
        breakdown = "non-finite beta";
        cycle_breakdown = true;
        break;
      }

      lincomb_coupled(search_p, search_ib, 1.0_rt, search_p, search_ib, -omega,
                      image_p, image_ib, active_masks);
      lincomb_coupled(search_p, search_ib, beta, search_p, search_ib, 1.0_rt,
                      rr_p, rr_ib, active_masks);

      apply_scaled_operator(search_p, search_ib, image_p, image_ib);
      const Real rhat_image = dot_coupled(rhat_p, rhat_ib, image_p, image_ib);
      if (!std::isfinite(rhat_image)) {
        breakdown = "non-finite rhat-v";
        cycle_breakdown = true;
        break;
      }
      if (std::abs(rhat_image) < tiny) {
        breakdown = "rhat-v breakdown";
        cycle_breakdown = true;
        break;
      }
      alpha = rho_new / rhat_image;
      if (!std::isfinite(alpha)) {
        breakdown = "non-finite alpha";
        cycle_breakdown = true;
        break;
      }

      lincomb_coupled(intermediate_p, intermediate_ib, 1.0_rt, rr_p, rr_ib,
                      -alpha, image_p, image_ib, active_masks);
      const Real intermediate_resid2 = dot_coupled(
          intermediate_p, intermediate_ib, intermediate_p, intermediate_ib);
      if (!std::isfinite(intermediate_resid2)) {
        breakdown = "non-finite residual";
        cycle_breakdown = true;
        break;
      }
      if (intermediate_resid2 <= tol2) {
        saxpy_coupled(p, scaled_force, alpha, search_p, search_ib,
                      active_masks);
        ++iter;
        break;
      }

      apply_scaled_operator(intermediate_p, intermediate_ib,
                            image_intermediate_p, image_intermediate_ib);
      const Real image_norm2 =
          dot_coupled(image_intermediate_p, image_intermediate_ib,
                      image_intermediate_p, image_intermediate_ib);
      if (!std::isfinite(image_norm2)) {
        breakdown = "non-finite t-t";
        cycle_breakdown = true;
        break;
      }
      omega = (image_norm2 > tiny)
                  ? dot_coupled(image_intermediate_p, image_intermediate_ib,
                                intermediate_p, intermediate_ib) /
                        image_norm2
                  : 0.0_rt;
      if (!std::isfinite(omega)) {
        breakdown = "non-finite omega";
        cycle_breakdown = true;
        break;
      }

      saxpy_coupled(p, scaled_force, alpha, search_p, search_ib, active_masks);
      saxpy_coupled(p, scaled_force, omega, intermediate_p, intermediate_ib,
                    active_masks);
      lincomb_coupled(rr_p, rr_ib, 1.0_rt, intermediate_p, intermediate_ib,
                      -omega, image_intermediate_p, image_intermediate_ib,
                      active_masks);

      rho = rho_new;
      const Real recursive_resid2 = dot_coupled(rr_p, rr_ib, rr_p, rr_ib);
      ++iter;
      if (std::abs(omega) < tiny) {
        breakdown = "omega breakdown";
        cycle_breakdown = true;
        break;
      }
      if (!std::isfinite(recursive_resid2)) {
        breakdown = "non-finite residual";
        cycle_breakdown = true;
        break;
      }
      if (recursive_resid2 <= tol2)
        break;
    }

    if (cycle_breakdown)
      break;
  }

  if (m_pressure_singular)
    subtract_composite_mean(p, active_masks);

  resid = evaluate_residual().norm();
  remember_current_if_better(resid);

  if (std::isfinite(best_resid) &&
      (!std::isfinite(resid) || best_resid < resid)) {
    copy_composite_unmasked(p, best_p);
    scaled_force = best_ib;
    if (m_pressure_singular)
      subtract_composite_mean(p, active_masks);
    resid = evaluate_residual().norm();
  }

  const bool converged = std::isfinite(resid) && resid <= tol;

  scaling.to_physical_force(scaled_force, m_ib_force);
  scaling.unscale_pressure(rr_p);
  const CoupledResidual final_residual{dot_composite(rr_p, rr_p),
                                       norm2_vector(rr_ib)};

  if (!converged) {
    const Real scaled_relres = resid / rhs_norm;
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
    msg << "Composite IB projection solve";
    if (breakdown != nullptr) {
      msg << " stopped by " << breakdown;
    } else if (iter >= m_poisson_max_iter) {
      msg << " reached max_iter";
    } else {
      msg << " failed to converge";
    }
    msg << "; the coupled IB system may be singular, rank deficient, or "
           "ill-conditioned"
        << " (levels = " << finest_level + 1
        << ", finest_level = " << finest_level
        << ", markers = " << m_ib_geometry.markers.size()
        << ", constraints = " << rhs_ib.size()
        << ", pressure_singular = " << m_pressure_singular
        << ", scaled_relres = " << scaled_relres
        << ", pressure_relres = " << relres_p << ", ib_relres = " << relres_ib
        << ", |r_p| = " << abs_p << ", |rhs_p| = " << rhs_p_norm
        << ", |r_ib| = " << abs_ib << ", |rhs_ib| = " << rhs_ib_norm << ").";
    amrex::Warning(msg.str());
  }

  if (m_verbose > 1) {
    const Real scaled_relres = resid / rhs_norm;
    Print() << "  Composite IB BiCGStab: " << iter
            << " iters, scaled |r|/|rhs| = " << scaled_relres << "\n";
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
    std::vector<Real> rhs_ib;
    InterpolateIBVelocity(finest_level,
                          {AMREX_D_DECL(m_vstar[finest_level][0].get(),
                                        m_vstar[finest_level][1].get(),
                                        m_vstar[finest_level][2].get())},
                          rhs_ib);
    for (std::size_t marker = 0; marker < m_ib_geometry.markers.size();
         ++marker) {
      for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        const auto n = marker * AMREX_SPACEDIM + d;
        rhs_ib[n] = (rhs_ib[n] - m_ib_velocity[d]) / m_dt;
      }
    }

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
      std::vector<Real> marker_vel;
      InterpolateIBVelocity(
          lev,
          {AMREX_D_DECL(m_vel[lev][0].get(), m_vel[lev][1].get(),
                        m_vel[lev][2].get())},
          marker_vel);
      Real slip = 0.0_rt;
      for (std::size_t marker = 0; marker < m_ib_geometry.markers.size();
           ++marker) {
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
          const auto n = marker * AMREX_SPACEDIM + d;
          slip = std::max(slip, std::abs(marker_vel[n] - m_ib_velocity[d]));
        }
      }
      Print() << "  |E u - U_ib|_inf = " << slip << "\n";
    }
  }
}
