/**
 * INSSolver_BC.cpp
 *
 * Physical boundary conditions on the staggered (MAC) grid.
 *
 * Supported per-domain-face kinds (see BCKind in INSSolver.H):
 *
 *   periodic    geometry periodicity (nothing to do here)
 *   dirichlet   prescribed velocity vector g  (no-slip wall, moving
 *               lid, inflow — same numerics).  Pressure: Neumann.
 *   slip        u·n = 0, tangential zero-gradient.  Pressure: Neumann.
 *   outflow     velocity linear extrapolation: ghost = 2u(N)−u(N−1), giving
 *               ∂²u/∂n² = 0 at the boundary face (biased-stencil diffusion).
 *               Normal boundary face is left as-is (set by projection).
 *               Pressure: Dirichlet p = 0.
 *
 * Staggered-grid conventions used below
 * -------------------------------------
 *   * The velocity component whose face is normal to the boundary
 *     (comp == bdir) sits exactly ON the boundary at the lo face index
 *     = domain.smallEnd(bdir) and the hi face index
 *     = domain.bigEnd(bdir)+1.
 *   * The tangential components (comp != bdir) are cell-centred in
 *     `bdir`; the wall is half a cell outside the first interior face.
 *
 * The pressure-Poisson BC pairing (Dirichlet velocity ⇒ Neumann
 * pressure, outflow velocity ⇒ Dirichlet pressure) is the standard
 * one for projection methods and is what the user specified.
 */

#include "INSSolver.H"

#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <string>

using namespace amrex;

namespace {

BCKind kind_from_string(const std::string &s) {
  if (s == "periodic")
    return BCKind::periodic;
  if (s == "noslip" || s == "inflow" || s == "dirichlet" || s == "wall")
    return BCKind::dirichlet;
  if (s == "slip")
    return BCKind::slip;
  if (s == "outflow")
    return BCKind::outflow;
  amrex::Abort("INSSolver: unknown BC kind '" + s + "'");
  return BCKind::periodic; // unreachable
}

const char *kind_name(BCKind k) {
  switch (k) {
  case BCKind::periodic:
    return "periodic";
  case BCKind::dirichlet:
    return "dirichlet";
  case BCKind::slip:
    return "slip";
  case BCKind::outflow:
    return "outflow";
  }
  return "?";
}

} // namespace

// ============================================================
//  ParseBCs — read ins.bc_* / ins.vel_* and validate
// ============================================================
void INSSolver::ParseBCs() {
  ParmParse pp("ins");

  const char *lo_key[3] = {"bc_xlo", "bc_ylo", "bc_zlo"};
  const char *hi_key[3] = {"bc_xhi", "bc_yhi", "bc_zhi"};
  const char *vlo_key[3] = {"vel_xlo", "vel_ylo", "vel_zlo"};
  const char *vhi_key[3] = {"vel_xhi", "vel_yhi", "vel_zhi"};

  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    std::string lo = "periodic", hi = "periodic";
    pp.query(lo_key[d], lo);
    pp.query(hi_key[d], hi);
    m_bc_lo[d] = kind_from_string(lo);
    m_bc_hi[d] = kind_from_string(hi);

    m_bcvel_lo[d] = RealVect::TheZeroVector();
    m_bcvel_hi[d] = RealVect::TheZeroVector();
    Vector<Real> v;
    if (pp.queryarr(vlo_key[d], v) && v.size() >= AMREX_SPACEDIM)
      for (int c = 0; c < AMREX_SPACEDIM; ++c)
        m_bcvel_lo[d][c] = v[c];
    v.clear();
    if (pp.queryarr(vhi_key[d], v) && v.size() >= AMREX_SPACEDIM)
      for (int c = 0; c < AMREX_SPACEDIM; ++c)
        m_bcvel_hi[d][c] = v[c];
  }

  // Consistency with geometry.is_periodic.
  Vector<int> isper(AMREX_SPACEDIM, 0);
  {
    ParmParse pg("geometry");
    pg.queryarr("is_periodic", isper);
  }
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    const bool per_bc =
        (m_bc_lo[d] == BCKind::periodic) && (m_bc_hi[d] == BCKind::periodic);
    const bool per_geom = isper[d] != 0;
    if (per_bc != per_geom) {
      amrex::Abort("INSSolver: BC/periodicity mismatch in direction " +
                   std::to_string(d) +
                   " — geometry.is_periodic and ins.bc_* disagree.");
    }
    if ((m_bc_lo[d] == BCKind::periodic) != (m_bc_hi[d] == BCKind::periodic))
      amrex::Abort("INSSolver: a direction must be periodic on both faces "
                   "or neither.");
  }

  // The pressure system is singular (pure Neumann/periodic) unless at
  // least one face is an outflow (which pins the pressure via p=0).
  m_pressure_singular = true;
  for (int d = 0; d < AMREX_SPACEDIM; ++d)
    if (m_bc_lo[d] == BCKind::outflow || m_bc_hi[d] == BCKind::outflow)
      m_pressure_singular = false;

  if (m_verbose > 0) {
    Print() << "Boundary conditions:\n";
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
      Print() << "  dir " << d << ":  lo=" << kind_name(m_bc_lo[d])
              << "  hi=" << kind_name(m_bc_hi[d]) << "\n";
    Print() << "  pressure system: "
            << (m_pressure_singular ? "singular (mean pinned)"
                                    : "non-singular (outflow Dirichlet)")
            << "\n";
  }
}

// ============================================================
//  BuildBCRecs — BCKind → AMReX BCRec (used by FillPatch C/F interp)
//
//  These only matter at coarse–fine boundaries that coincide with a
//  domain boundary; our tests keep refinement interior so the exact
//  choice is not exercised, but we set sane values anyway:
//    velocity: ext_dir for dirichlet, foextrap for slip/outflow
//    pressure: foextrap for dirichlet/slip, ext_dir for outflow
// ============================================================
void INSSolver::BuildBCRecs() {
  auto vel_type = [](BCKind k) -> int {
    switch (k) {
    case BCKind::periodic:
      return BCType::int_dir;
    case BCKind::dirichlet:
      return BCType::ext_dir;
    case BCKind::slip:
    case BCKind::outflow:
      return BCType::foextrap;
    }
    return BCType::foextrap;
  };
  auto pres_type = [](BCKind k) -> int {
    switch (k) {
    case BCKind::periodic:
      return BCType::int_dir;
    case BCKind::dirichlet:
    case BCKind::slip:
      return BCType::foextrap;
    case BCKind::outflow:
      return BCType::ext_dir;
    }
    return BCType::foextrap;
  };

  m_bc_pres.resize(1);
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    m_bc_pres[0].setLo(d, pres_type(m_bc_lo[d]));
    m_bc_pres[0].setHi(d, pres_type(m_bc_hi[d]));
  }
  for (int comp = 0; comp < AMREX_SPACEDIM; ++comp) {
    m_bc_vel[comp].resize(1);
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
      m_bc_vel[comp][0].setLo(d, vel_type(m_bc_lo[d]));
      m_bc_vel[comp][0].setHi(d, vel_type(m_bc_hi[d]));
    }
  }
}

// ============================================================
//  FillVelGhostPhys — fill domain-boundary ghosts of velocity
//                     component `comp`.
// ============================================================
void INSSolver::FillVelGhostPhys(int lev, int comp, MultiFab &mf,
                                 bool homogeneous) {
  mf.FillBoundary(geom[lev].periodicity()); // periodic + intra-level

  const Box &domain = geom[lev].Domain();

  for (int bdir = 0; bdir < AMREX_SPACEDIM; ++bdir) {
    if (geom[lev].isPeriodic(bdir))
      continue;

    const bool normal = (comp == bdir);
    const int dlo = domain.smallEnd(bdir);
    const int dhi = domain.bigEnd(bdir);
    // Boundary face index of the *normal* component:
    const int floidx = dlo;       // lo boundary face
    const int fhiidx = dhi + 1;   // hi boundary face

    for (int side = 0; side < 2; ++side) {
      const BCKind kind = (side == 0) ? m_bc_lo[bdir] : m_bc_hi[bdir];
      if (kind == BCKind::periodic)
        continue;
      const Real g =
          homogeneous ? 0.0
                      : ((side == 0) ? m_bcvel_lo[bdir][comp]
                                     : m_bcvel_hi[bdir][comp]);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
      for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
        const Box fb = mfi.fabbox(); // grown (ghost) box
        auto a = mf.array(mfi);

        // Build the slab of indices to write for this boundary.
        Box slab = fb;
        if (side == 0) {
          if (normal)
            slab.setRange(bdir, fb.smallEnd(bdir),
                          floidx - fb.smallEnd(bdir) + 1); // ghosts + face
          else
            slab.setRange(bdir, fb.smallEnd(bdir), dlo - fb.smallEnd(bdir));
        } else {
          if (normal)
            slab.setRange(bdir, fhiidx, fb.bigEnd(bdir) - fhiidx + 1);
          else
            slab.setRange(bdir, dhi + 1, fb.bigEnd(bdir) - dhi);
        }
        if (slab.isEmpty())
          continue;

        const int s = side;
        const int bd = bdir;
        const BCKind kd = kind;

        amrex::ParallelFor(
            slab, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
              int ijk[3] = {i, j, k};
              const int idx = ijk[bd];

              auto interior_normal = [&](int off) {
                int t[3] = {i, j, k};
                t[bd] = (s == 0) ? (floidx + off) : (fhiidx - off);
                return a(t[0], t[1], t[2]);
              };
              auto interior_tang = [&](int off) {
                int t[3] = {i, j, k};
                t[bd] = (s == 0) ? (dlo + off) : (dhi - off);
                return a(t[0], t[1], t[2]);
              };

              if (normal) {
                const int bidx = (s == 0) ? floidx : fhiidx;
                if (kd == BCKind::dirichlet || kd == BCKind::slip) {
                  const Real wall = (kd == BCKind::slip) ? 0.0 : g;
                  if (idx == bidx) {
                    a(i, j, k) = wall;
                  } else {
                    const int off = (s == 0) ? (bidx - idx) : (idx - bidx);
                    a(i, j, k) = 2.0_rt * wall - interior_normal(off);
                  }
                } else {
                  // outflow: boundary face is set by the projection;
                  // ghost cells get the linear extrapolation beyond it
                  // so that L(u)[bidx] = 0 in the normal direction.
                  if (idx != bidx) {
                    const int off = (s == 0) ? (bidx - idx) : (idx - bidx);
                    a(i, j, k) = (1.0_rt + off) * interior_normal(0) -
                                 (Real)off * interior_normal(1);
                  }
                }
              } else {
                // Tangential face: wall is between ghost(-1) and
                // interior(0); value at the wall = average.
                const int m = (s == 0) ? (dlo - idx) : (idx - dhi);
                if (kd == BCKind::dirichlet) {
                  a(i, j, k) = 2.0_rt * g - interior_tang(m - 1);
                } else if (kd == BCKind::slip) {
                  a(i, j, k) = interior_tang(m - 1);
                } else {
                  // outflow: linear extrapolation gives ∂²u/∂n² = 0
                  a(i, j, k) = (1.0_rt + m) * interior_tang(0) -
                               (Real)m * interior_tang(1);
                }
              }
            });
      }
    }
  }
}

// ============================================================
//  FillPresGhostPhys — Neumann at dirichlet/slip, p=0 at outflow
// ============================================================
void INSSolver::FillPresGhostPhys(int lev, MultiFab &mf) {
  mf.FillBoundary(geom[lev].periodicity());

  const Box &domain = geom[lev].Domain();

  for (int bdir = 0; bdir < AMREX_SPACEDIM; ++bdir) {
    if (geom[lev].isPeriodic(bdir))
      continue;
    const int dlo = domain.smallEnd(bdir);
    const int dhi = domain.bigEnd(bdir);

    for (int side = 0; side < 2; ++side) {
      const BCKind kind = (side == 0) ? m_bc_lo[bdir] : m_bc_hi[bdir];
      if (kind == BCKind::periodic)
        continue;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
      for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
        const Box fb = mfi.fabbox();
        auto a = mf.array(mfi);

        Box slab = fb;
        if (side == 0)
          slab.setRange(bdir, fb.smallEnd(bdir), dlo - fb.smallEnd(bdir));
        else
          slab.setRange(bdir, dhi + 1, fb.bigEnd(bdir) - dhi);
        if (slab.isEmpty())
          continue;

        const int s = side;
        const int bd = bdir;
        const bool dirichlet0 = (kind == BCKind::outflow);

        amrex::ParallelFor(
            slab, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
              int ijk[3] = {i, j, k};
              const int idx = ijk[bd];
              const int m = (s == 0) ? (dlo - idx) : (idx - dhi);
              int t[3] = {i, j, k};
              t[bd] = (s == 0) ? (dlo + m - 1) : (dhi - m + 1);
              const Real pin = a(t[0], t[1], t[2]);
              // Neumann: ghost = mirror interior.  Dirichlet p=0:
              // odd reflection so the face value is 0.
              a(i, j, k) = dirichlet0 ? -pin : pin;
            });
      }
    }
  }
}

// ============================================================
//  EnforceVelDirichlet — re-impose prescribed wall velocity
// ============================================================
void INSSolver::EnforceVelDirichlet(
    int lev, const std::array<MultiFab *, AMREX_SPACEDIM> &vel) {
  for (int comp = 0; comp < AMREX_SPACEDIM; ++comp)
    FillVelGhostPhys(lev, comp, *vel[comp], /*homogeneous=*/false);
}
