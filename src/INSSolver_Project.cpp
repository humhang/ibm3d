/**
 * INSSolver_Project.cpp
 *
 * Perot 1997 / Taira–Colonius (without IB) projection step.
 *
 *   Modified Poisson:   (D B^N G) p^{n+1} = (1/dt) D u*
 *   Projection:         u^{n+1} = u* − dt B^N G p^{n+1}
 *
 *   B^N = Σ_{k=0}^N (εL)^k,   ε = ν dt / 2
 *
 * The operator D B^N G is applied matrix-free (compose existing
 * gradient, face-Laplacian, divergence pieces) and the system is
 * solved with a hand-rolled CG.  No preconditioner — for the test
 * grids (32³–64³) unpreconditioned CG converges in a few-dozen
 * iterations; MLMG-as-preconditioner can be added later if needed.
 *
 * Multi-level handling is approximate: each level's modified Poisson
 * is solved independently, and `average_down_faces` / `average_down`
 * sync coarse with fine after the projection.  A proper composite
 * D B^N G that handles C/F coupling inside the operator apply is
 * deferred.
 */

#include "INSSolver.H"

#include <AMReX_MultiFabUtil.H>
#include <AMReX_Print.H>

#include <cmath>

using namespace amrex;

// ============================================================
//  ApplyModifiedPoissonOp — result = −D B^N G φ   (cell → cell)
//
//  We return the negated operator so CG sees a positive
//  semidefinite system: the natural D B^N G has eigenvalues −k²
//  on every Fourier mode (same sign convention as ∇²).  The
//  caller passes a negated RHS to compensate.  The projection
//  step still uses B^N G p with its natural sign.
//
//  φ must have ≥ 1 ghost cell filled by the caller.
// ============================================================
void INSSolver::ApplyModifiedPoissonOp(int lev, const MultiFab &phi,
                                       MultiFab &result) {
  const BoxArray &ba = grids[lev];
  const DistributionMapping &dm = dmap[lev];

  // Compute G φ on faces, then apply B^N face-by-face.
  std::array<MultiFab, AMREX_SPACEDIM> bgphi;
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    BoxArray fba = amrex::convert(ba, IntVect::TheDimensionVector(d));
    MultiFab gphi(fba, dm, 1, 2); // 2 ghosts for B^N's L applications
    bgphi[d].define(fba, dm, 1, 0);

    // Zero everything first so that coarse–fine ghost cells (which the
    // per-level operator never fills) are a deterministic 0 — the
    // per-level "Dirichlet-0 at C/F" approximation.  Without this they
    // are uninitialised and the operator blows up whenever a fine
    // patch has an interior C/F interface (e.g. the lid cavity).
    gphi.setVal(0.0);
    ComputePressureGradient(lev, d, phi, gphi);
    FillVelGhostPhys(lev, d, gphi, /*homogeneous=*/true);

    ApplyBNFace(lev, d, gphi, bgphi[d]);
  }

  // result = D (B^N G φ)
  const Real *dx = geom[lev].CellSize();
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
    auto const &u = bgphi[0].const_array(mfi);
    auto const &v = bgphi[1].const_array(mfi);
#if AMREX_SPACEDIM == 3
    auto const &w = bgphi[2].const_array(mfi);
#endif
    auto const &r = result.array(mfi);
    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      Real d = cx * (u(i + 1, j, k) - u(i, j, k)) +
               cy * (v(i, j + 1, k) - v(i, j, k));
#if AMREX_SPACEDIM == 3
      d += cz * (w(i, j, k + 1) - w(i, j, k));
#endif
      r(i, j, k) = -d; // negate so the operator is positive semidefinite
    });
  }
}

// ============================================================
//  SubtractMean — for the singular periodic system: pin Σ φ ≡ 0
// ============================================================
void INSSolver::SubtractMean(int lev, MultiFab &mf) {
  const Real n_cells = static_cast<Real>(grids[lev].numPts());
  Real s = mf.sum(0);
  s /= n_cells;
  mf.plus(-s, 0, 1, 0);
}

// ============================================================
//  SolveModifiedPoisson — matrix-free CG on (D B^N G) p = rhs
//
//  p:   in = warm start, out = solution (cell-centred, ≥1 ghost)
//  rhs: read-only cell-centred MF (0 ghosts is fine)
// ============================================================
int INSSolver::SolveModifiedPoisson(int lev, MultiFab &p,
                                    const MultiFab &rhs_in) {
  const BoxArray &ba = grids[lev];
  const DistributionMapping &dm = dmap[lev];

  // A level's pressure problem is singular (solution defined up to a
  // constant) only when it is pure Neumann/periodic — i.e. no outflow
  // (m_pressure_singular) AND the level tiles the whole (refined)
  // domain so there is no coarse–fine interface supplying Dirichlet
  // data.  A partial fine patch gets Dirichlet-from-coarse at its C/F
  // boundary and is non-singular: pinning its mean would corrupt it.
  const bool level_singular =
      m_pressure_singular &&
      (grids[lev].numPts() == geom[lev].Domain().numPts());

  MultiFab rhs(ba, dm, 1, 0);
  MultiFab::Copy(rhs, rhs_in, 0, 0, 1, 0);
  if (level_singular) {
    SubtractMean(lev, rhs);
    SubtractMean(lev, p);
  }

  // Boundary data for p: domain physical BCs AND, at the C/F
  // interface, the already-solved coarser pressure interpolated in
  // (Dirichlet-from-coarse level solve — IAMR style).  Separate ghost
  // buffer (FillCellPatch's FillPatchTwoLevels must not alias the fine
  // source).  p_g.valid = this level's p, p_g C/F = coarse-interp.
  MultiFab p_g(ba, dm, 1, 1);
  FillCellPatch(lev, m_pressure, p_g, m_cur_time, m_bc_pres);

  // ---- Matrix-free BiCGStab ----
  // The per-level operator −D B^N G is NOT symmetric once a fine patch
  // has a coarse–fine interface (the staggered B^N + ad-hoc C/F ghost
  // breaks the D/G adjointness).  CG diverges on it; BiCGStab tolerates
  // the asymmetry.  All search vectors are *corrections*: homogeneous
  // at C/F (ghost = 0 via setVal, never refilled there) — only
  // FillPresGhostPhys touches their domain ghosts before each apply.
  MultiFab rr(ba, dm, 1, 0), rhat(ba, dm, 1, 0);
  MultiFab pv(ba, dm, 1, 1), v(ba, dm, 1, 0);
  MultiFab s(ba, dm, 1, 1), t(ba, dm, 1, 0);
  MultiFab Ax(ba, dm, 1, 0);
  pv.setVal(0.0);
  s.setVal(0.0);

  // r0 = rhs − A p_g
  ApplyModifiedPoissonOp(lev, p_g, Ax);
  MultiFab::LinComb(rr, 1.0_rt, rhs, 0, -1.0_rt, Ax, 0, 0, 1, 0);
  MultiFab::Copy(rhat, rr, 0, 0, 1, 0);

  const Real rhs_norm2 = MultiFab::Dot(rhs, 0, rhs, 0, 1, 0);
  const Real tol2 =
      m_poisson_tol * m_poisson_tol * std::max(rhs_norm2, Real(1.0e-300));

  Real rho = 1.0, alpha = 1.0, omega = 1.0;
  v.setVal(0.0);
  Real rsold = MultiFab::Dot(rr, 0, rr, 0, 1, 0);

  const Real tiny = 1.0e-300;
  int iter = 0;
  for (; iter < m_poisson_max_iter; ++iter) {
    if (rsold < tol2 || !std::isfinite(rsold))
      break;

    const Real rho_new = MultiFab::Dot(rhat, 0, rr, 0, 1, 0);
    if (std::abs(rho_new) < tiny)
      break; // breakdown
    const Real beta = (rho_new / rho) * (alpha / omega);

    // pv ← rr + beta (pv − omega v)
    MultiFab::LinComb(pv, 1.0_rt, pv, 0, -omega, v, 0, 0, 1, 0);
    MultiFab::LinComb(pv, beta, pv, 0, 1.0_rt, rr, 0, 0, 1, 0);

    FillPresGhostPhys(lev, pv);
    ApplyModifiedPoissonOp(lev, pv, v);

    const Real rhatv = MultiFab::Dot(rhat, 0, v, 0, 1, 0);
    if (std::abs(rhatv) < tiny)
      break;
    alpha = rho_new / rhatv;

    // s ← rr − alpha v
    MultiFab::LinComb(s, 1.0_rt, rr, 0, -alpha, v, 0, 0, 1, 0);

    const Real s2 = MultiFab::Dot(s, 0, s, 0, 1, 0);
    if (s2 < tol2) {
      MultiFab::Saxpy(p, alpha, pv, 0, 0, 1, 0);
      rsold = s2;
      ++iter;
      break;
    }

    FillPresGhostPhys(lev, s);
    ApplyModifiedPoissonOp(lev, s, t);

    const Real tt = MultiFab::Dot(t, 0, t, 0, 1, 0);
    omega = (tt > tiny) ? MultiFab::Dot(t, 0, s, 0, 1, 0) / tt : 0.0;

    // p ← p + alpha pv + omega s
    MultiFab::Saxpy(p, alpha, pv, 0, 0, 1, 0);
    MultiFab::Saxpy(p, omega, s, 0, 0, 1, 0);

    // rr ← s − omega t
    MultiFab::LinComb(rr, 1.0_rt, s, 0, -omega, t, 0, 0, 1, 0);

    rho = rho_new;
    rsold = MultiFab::Dot(rr, 0, rr, 0, 1, 0);
    if (std::abs(omega) < tiny)
      break; // breakdown
  }

  // Re-pin the mean of the solution (truly-singular level only).
  if (level_singular)
    SubtractMean(lev, p);

  if (m_verbose > 1) {
    const Real relres =
        std::sqrt(std::max(rsold, Real(0.0)) /
                  std::max(rhs_norm2, Real(1.0e-300)));
    Print() << "  Perot BiCGStab lev " << lev << ": " << iter
            << " iters, |r|/|rhs| = " << relres << "\n";
  }
  return iter;
}

// ============================================================
//  ProjectPerot — orchestrate the per-level Perot projection.
// ============================================================
void INSSolver::ProjectPerot() {
  const int nlev = finest_level + 1;

  // Sync u* fine→coarse so coarse face values at C/F equal the
  // averaged fine flux — gives a composite-consistent divergence
  // even though the per-level solves themselves are not composite.
  AverageDownVelocity(m_vstar);

  for (int lev = 0; lev < nlev; ++lev) {
    const BoxArray &ba = grids[lev];
    const DistributionMapping &dm = dmap[lev];

    // RHS = −(1/dt) D u*   (negated to pair with the negated operator)
    MultiFab rhs(ba, dm, 1, 0);
    rhs.setVal(0.0);

    const Real *dx = geom[lev].CellSize();
    const Real cx = -1.0_rt / (m_dt * dx[0]);
    const Real cy = -1.0_rt / (m_dt * dx[1]);
#if AMREX_SPACEDIM == 3
    const Real cz = -1.0_rt / (m_dt * dx[2]);
#endif

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const Box &bx = mfi.tilebox();
      auto const &u = m_vstar[lev][0]->const_array(mfi);
      auto const &v = m_vstar[lev][1]->const_array(mfi);
#if AMREX_SPACEDIM == 3
      auto const &w = m_vstar[lev][2]->const_array(mfi);
#endif
      auto const &r = rhs.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        Real d = cx * (u(i + 1, j, k) - u(i, j, k)) +
                 cy * (v(i, j + 1, k) - v(i, j, k));
#if AMREX_SPACEDIM == 3
        d += cz * (w(i, j, k + 1) - w(i, j, k));
#endif
        r(i, j, k) = d;
      });
    }

    // Solve  (D B^N G) p^{n+1} = rhs.   Warm-start with the previous pressure.
    SolveModifiedPoisson(lev, *m_pressure[lev], rhs);

    // Projection:  u^{n+1} = u* − dt B^N G p^{n+1}
    //  (same B^N as in the predictor and the operator — this is the
    //   Perot consistency that makes the projection exact.)
    // Same C/F (Dirichlet-from-coarse) + domain BC fill the operator
    // used inside the solve — the projection's ∇p must be applied to
    // the *identical* ghost-filled pressure or D u^{n+1} ≠ 0.
    MultiFab phi_g(ba, dm, 1, 1);
    FillCellPatch(lev, m_pressure, phi_g, m_cur_time, m_bc_pres);

    std::array<MultiFab *, AMREX_SPACEDIM> vel_out;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
      BoxArray fba = amrex::convert(ba, IntVect::TheDimensionVector(d));
      MultiFab gp(fba, dm, 1, 2);
      MultiFab bgp(fba, dm, 1, 0);

      ComputePressureGradient(lev, d, phi_g, gp);
      FillVelGhostPhys(lev, d, gp, /*homogeneous=*/true);

      ApplyBNFace(lev, d, gp, bgp);

      MultiFab::Copy(*m_vel[lev][d], *m_vstar[lev][d], 0, 0, 1, 0);
      MultiFab::Saxpy(*m_vel[lev][d], -m_dt, bgp, 0, 0, 1, 0);
      vel_out[d] = m_vel[lev][d].get();
    }

    // Re-impose the prescribed wall velocity (the local projection
    // does not preserve inhomogeneous Dirichlet data exactly).
    EnforceVelDirichlet(lev, vel_out);
  }
}
