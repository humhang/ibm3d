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

  // For a pure Neumann/periodic pressure system the solution is
  // defined only up to a constant — pin it by subtracting the mean
  // of the RHS and the warm-start.  With an outflow (Dirichlet p=0)
  // the system is non-singular and the mean must NOT be removed.
  MultiFab rhs(ba, dm, 1, 0);
  MultiFab::Copy(rhs, rhs_in, 0, 0, 1, 0);
  if (m_pressure_singular) {
    SubtractMean(lev, rhs);
    SubtractMean(lev, p);
  }

  // Working MFs
  MultiFab r(ba, dm, 1, 0);
  MultiFab d_(ba, dm, 1, 1); // search direction; 1 ghost for op apply
  MultiFab Ad(ba, dm, 1, 0);

  // r0 = rhs − A p
  FillPresGhostPhys(lev, p);
  ApplyModifiedPoissonOp(lev, p, Ad); // Ad temporarily = A p
  MultiFab::LinComb(r, 1.0_rt, rhs, 0, -1.0_rt, Ad, 0, 0, 1, 0);

  MultiFab::Copy(d_, r, 0, 0, 1, 0);

  Real rsold = MultiFab::Dot(r, 0, r, 0, 1, 0);
  const Real rhs_norm2 = MultiFab::Dot(rhs, 0, rhs, 0, 1, 0);
  const Real tol2 =
      m_poisson_tol * m_poisson_tol * std::max(rhs_norm2, Real(1.0e-300));

  int iter = 0;
  for (; iter < m_poisson_max_iter; ++iter) {
    if (rsold < tol2)
      break;

    FillPresGhostPhys(lev, d_);
    ApplyModifiedPoissonOp(lev, d_, Ad);

    Real dAd = MultiFab::Dot(d_, 0, Ad, 0, 1, 0);
    if (dAd <= 0.0) {
      Print() << "  WARNING: Perot CG dAd <= 0 (" << dAd
              << ") — operator non-positive on this iterate.\n";
      break;
    }
    Real alpha = rsold / dAd;

    MultiFab::Saxpy(p, alpha, d_, 0, 0, 1, 0);
    MultiFab::Saxpy(r, -alpha, Ad, 0, 0, 1, 0);

    Real rsnew = MultiFab::Dot(r, 0, r, 0, 1, 0);
    Real beta = rsnew / rsold;
    // d ← r + β d
    MultiFab::LinComb(d_, beta, d_, 0, 1.0_rt, r, 0, 0, 1, 0);
    rsold = rsnew;
  }

  // Re-pin the mean of the solution (singular system only).
  if (m_pressure_singular)
    SubtractMean(lev, p);

  if (m_verbose > 1) {
    const Real relres =
        std::sqrt(rsold / std::max(rhs_norm2, Real(1.0e-300)));
    Print() << "  Perot CG lev " << lev << ": " << iter
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
    MultiFab phi_g(ba, dm, 1, 1);
    MultiFab::Copy(phi_g, *m_pressure[lev], 0, 0, 1, 0);
    FillPresGhostPhys(lev, phi_g);

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
