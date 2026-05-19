/**
 * INSSolver.cpp
 *
 * Construction, parameter reading, time loop, AMR machinery
 * (regridding, FillPatch helpers, average-down, tagging), I/O.
 */

#include "INSSolver.H"

#include <AMReX_FillPatchUtil.H>
#include <AMReX_Interpolater.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <AMReX_TagBox.H>
#include <AMReX_VisMF.H>

#include <algorithm>
#include <cmath>
#include <limits>

using namespace amrex;

// ============================================================
// Construction
// ============================================================
INSSolver::INSSolver() {
  ReadParameters();
  ParseBCs();
  BuildBCRecs();

  // Build the initial hierarchy (drives MakeNewLevelFromScratch on each level
  // up to max_level).  AmrCore handles tagging through ErrorEst.
  InitFromScratch(m_cur_time);

  // After the initial hierarchy is built, sync fine→coarse so the coarsest
  // levels reflect the higher-resolution IC where they overlap.
  AverageDownVelocity(m_vel);
  AverageDownPressure();

  Print() << "INSSolver constructed\n"
          << "  nu          = " << m_nu << "\n"
          << "  cfl         = " << m_cfl << "\n"
          << "  t_stop      = " << m_t_stop << "\n"
          << "  max_step    = " << m_max_step << "\n"
          << "  cn_order    = " << m_cn_order << "\n"
          << "  regrid_int  = " << m_regrid_int << "\n"
          << "  max_level   = " << max_level << "\n"
          << "  finest_level (initial) = " << finest_level << "\n";
}

void INSSolver::ReadParameters() {
  ParmParse pp("ins");
  pp.query("nu", m_nu);
  pp.query("cfl", m_cfl);
  pp.query("fixed_dt", m_fixed_dt);
  pp.query("t_stop", m_t_stop);
  pp.query("max_step", m_max_step);
  pp.query("plot_int", m_plot_int);
  pp.query("regrid_int", m_regrid_int);
  pp.query("cn_order", m_cn_order);
  pp.query("refine_vort", m_refine_vort);
  pp.query("poisson_tol", m_poisson_tol);
  pp.query("poisson_max_iter", m_poisson_max_iter);
  pp.query("verbose", m_verbose);
  pp.query("plot_prefix", m_plot_prefix);
  pp.query("ic", m_ic);
  pp.query("tg_uc", m_tg_uc);
  pp.query("tg_vc", m_tg_vc);
  pp.query("tg2d_dump", m_tg2d_dump);
  pp.query("tg2d_cmp", m_tg2d_cmp);
}

// ============================================================
// InitData / Run
// ============================================================
void INSSolver::InitData() {
  if (m_plot_int > 0)
    WritePlotFile();
}

void INSSolver::Run() {
  for (m_step = 1; m_step <= m_max_step && m_cur_time < m_t_stop; ++m_step) {
    ComputeDt();
    if (m_cur_time + m_dt > m_t_stop)
      m_dt = m_t_stop - m_cur_time;

    if (m_verbose > 0) {
      Print() << "Step " << m_step << "  t = " << m_cur_time
              << "  dt = " << m_dt << "  finest = " << finest_level << "\n";
    }

    Advance();
    m_cur_time += m_dt;

    if (m_regrid_int > 0 && max_level > 0 && (m_step % m_regrid_int == 0)) {
      regrid(0, m_cur_time);
      m_ab2_valid = false; // grids changed → A^{n-1} no longer valid
    }

    if (m_plot_int > 0 && (m_step % m_plot_int == 0)) {
      WritePlotFile();
    }

    if (m_verbose > 1) {
      Print() << "  |u|_inf = " << ComputeMaxVelocity()
              << "   |div u|_inf = " << ComputeMaxDivergence() << "\n";
    }
  }
  Print() << "Done. t = " << m_cur_time << ", step = " << m_step - 1 << "\n";
  if (m_plot_int > 0 && ((m_step - 1) % m_plot_int != 0))
    WritePlotFile();

  if (m_ic == "tg2d")
    WriteTG2DDiagnostics();
}

// ============================================================
// ComputeDt — CFL on advective velocity across all levels
// ============================================================
void INSSolver::ComputeDt() {
  if (m_fixed_dt > 0.0) {
    m_dt = m_fixed_dt;
    return;
  }

  // ---- Advective CFL (∞ when the field is at rest) ----
  Real max_inv = 0.0;
  for (int lev = 0; lev <= finest_level; ++lev) {
    const Real *dx = geom[lev].CellSize();
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
      Real umax = m_vel[lev][d]->norminf(0, 0);
      max_inv = std::max(max_inv, umax / dx[d]);
    }
  }
  ParallelDescriptor::ReduceRealMax(max_inv);
  const Real dt_adv =
      (max_inv > 0.0) ? (m_cfl / max_inv) : std::numeric_limits<Real>::max();

  // ---- Diffusive limit set by Neumann-series stability ----
  // B^N = Σ (εL)^k approximates (I − εL)^{-1} only if the spectral
  // radius ε‖L‖ < 1.  ‖L‖ = 4·dim/h² (discrete Laplacian) and
  // ε = ν dt/2, so ε‖L‖ = 2·dim·ν·dt/h².  Cap dt so ε‖L‖ ≤ α with
  // α = 0.5 — beyond ε‖L‖ ≥ 1 the modified-Poisson operator D B^N G
  // turns indefinite and the CG converges to garbage (lid cavity
  // with the viscous-limited dt was hitting ε‖L‖ = 1.5).
  const Real alpha = 0.5;
  const Real *dxf = geom[finest_level].CellSize();
  Real h2 = dxf[0] * dxf[0];
  for (int d = 1; d < AMREX_SPACEDIM; ++d)
    h2 = std::min(h2, dxf[d] * dxf[d]);
  const Real dt_diff =
      alpha * h2 / (2.0 * AMREX_SPACEDIM * std::max(m_nu, 1.0e-12));

  m_dt = std::min(dt_adv, dt_diff);

  if (m_verbose > 1)
    amrex::Print() << "  dt: adv=" << dt_adv << " diff=" << dt_diff
                   << " -> " << m_dt << "\n";
}

// ============================================================
// AmrCore overrides
// ============================================================
void INSSolver::MakeNewLevelFromScratch(int lev, Real /*time*/,
                                        const BoxArray &ba,
                                        const DistributionMapping &dm) {
  SetBoxArray(lev, ba);
  SetDistributionMap(lev, dm);
  AllocateLevelStorage(lev, ba, dm);
  InitFlowField(lev);
}

void INSSolver::MakeNewLevelFromCoarse(int lev, Real time, const BoxArray &ba,
                                       const DistributionMapping &dm) {
  SetBoxArray(lev, ba);
  SetDistributionMap(lev, dm);
  AllocateLevelStorage(lev, ba, dm);

  PhysBCFunctNoOp bc_func;
  for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
    InterpFromCoarseLevel(*m_vel[lev][dir], time, *m_vel[lev - 1][dir], 0, 0, 1,
                          geom[lev - 1], geom[lev], bc_func, 0, bc_func, 0,
                          ref_ratio[lev - 1], &face_linear_interp,
                          m_bc_vel[dir], 0);
  }
  // pc_interp, not lincc_interp: conservative-linear's limited slopes
  // read the coarse scratch's domain ghosts, which InterpFromCoarseLevel
  // leaves uninitialised under PhysBCFunctNoOp → NaN pressure when a
  // newly-refined patch hugs a no-slip wall (lid cavity, on regrid).
  InterpFromCoarseLevel(*m_pressure[lev], time, *m_pressure[lev - 1], 0, 0, 1,
                        geom[lev - 1], geom[lev], bc_func, 0, bc_func, 0,
                        ref_ratio[lev - 1], &pc_interp, m_bc_pres, 0);
}

void INSSolver::RemakeLevel(int lev, Real time, const BoxArray &ba,
                            const DistributionMapping &dm) {
  // Stash current data; rebuild storage with the new layout; then FillPatch
  // from the old data (and the coarse level, for newly-exposed regions).
  FaceMFArray old_vel;
  for (int d = 0; d < AMREX_SPACEDIM; ++d)
    old_vel[d] = std::move(m_vel[lev][d]);
  std::unique_ptr<MultiFab> old_pres = std::move(m_pressure[lev]);

  SetBoxArray(lev, ba);
  SetDistributionMap(lev, dm);
  AllocateLevelStorage(lev, ba, dm);

  PhysBCFunctNoOp bc_func;
  for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
    Vector<MultiFab *> cmf{m_vel[lev - 1][dir].get()};
    Vector<MultiFab *> fmf{old_vel[dir].get()};
    Vector<Real> ct{time}, ft{time};
    FillPatchTwoLevels(*m_vel[lev][dir], time, cmf, ct, fmf, ft, 0, 0, 1,
                       geom[lev - 1], geom[lev], bc_func, 0, bc_func, 0,
                       ref_ratio[lev - 1], &face_linear_interp, m_bc_vel[dir],
                       0);
  }
  {
    Vector<MultiFab *> cmf{m_pressure[lev - 1].get()};
    Vector<MultiFab *> fmf{old_pres.get()};
    Vector<Real> ct{time}, ft{time};
    // pc_interp (see MakeNewLevelFromCoarse): lincc_interp NaNs the
    // pressure on regrid when a remade patch touches a no-slip wall.
    FillPatchTwoLevels(*m_pressure[lev], time, cmf, ct, fmf, ft, 0, 0, 1,
                       geom[lev - 1], geom[lev], bc_func, 0, bc_func, 0,
                       ref_ratio[lev - 1], &pc_interp, m_bc_pres, 0);
  }
}

void INSSolver::ClearLevel(int lev) {
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    m_vel[lev][d].reset();
    m_advect[lev][d].reset();
    m_advect_old[lev][d].reset();
    m_vstar[lev][d].reset();
  }
  m_pressure[lev].reset();
}

// ============================================================
// AllocateLevelStorage — face MFs for velocity / advection / u*,
//                       cell MFs for pressure / phi.
// ============================================================
void INSSolver::AllocateLevelStorage(int lev, const BoxArray &ba,
                                     const DistributionMapping &dm) {
  constexpr int nghost_vel = 2;
  constexpr int nghost_pre = 1;

  std::array<BoxArray, AMREX_SPACEDIM> face_ba;
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    face_ba[d] = amrex::convert(ba, IntVect::TheDimensionVector(d));
  }

  const int N = lev + 1;
  if (static_cast<int>(m_vel.size()) < N) {
    m_vel.resize(N);
    m_advect.resize(N);
    m_advect_old.resize(N);
    m_vstar.resize(N);
    m_pressure.resize(N);
  }

  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    m_vel[lev][d] = std::make_unique<MultiFab>(face_ba[d], dm, 1, nghost_vel);
    m_advect[lev][d] = std::make_unique<MultiFab>(face_ba[d], dm, 1, 0);
    m_advect_old[lev][d] = std::make_unique<MultiFab>(face_ba[d], dm, 1, 0);
    m_vstar[lev][d] = std::make_unique<MultiFab>(face_ba[d], dm, 1, nghost_vel);
    m_vel[lev][d]->setVal(0.0);
    m_advect[lev][d]->setVal(0.0);
    m_advect_old[lev][d]->setVal(0.0);
    m_vstar[lev][d]->setVal(0.0);
  }

  m_pressure[lev] = std::make_unique<MultiFab>(ba, dm, 1, nghost_pre);
  m_pressure[lev]->setVal(0.0);
}

void INSSolver::SyncFillICGhosts(int lev) {
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    m_vel[lev][d]->OverrideSync(geom[lev].periodicity());
    FillVelGhostPhys(lev, d, *m_vel[lev][d], /*homogeneous=*/false);
  }
  FillPresGhostPhys(lev, *m_pressure[lev]);
}

// ============================================================
// InitFlowField — 3D Taylor–Green vortex (or 2D fallback)
// ============================================================
void INSSolver::InitFlowField(int lev) {
  const Geometry &gm = geom[lev];
  const Real *dx = gm.CellSize();
  const Real *plo = gm.ProbLo();

  if (m_ic == "quiescent") {
    // Start from rest; the boundary conditions drive the flow
    // (lid-driven cavity, channel inflow/outflow, …).
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
      m_vel[lev][d]->setVal(0.0);
    m_pressure[lev]->setVal(0.0);
    SyncFillICGhosts(lev);
    return;
  }

  if (m_ic == "tg2d") {
    // 2D Taylor–Green decaying vortex, optionally translated by a
    // uniform mean flow (m_tg_uc, m_tg_vc) — an EXACT solution of
    // incompressible NS (Galilean boost; z-invariant, periodic):
    //   u = Uc − cos ξ sin η · e^{-2νt},  ξ = x − Uc t
    //   v = Vc + sin ξ cos η · e^{-2νt},  η = y − Vc t
    //   p = -¼(cos 2ξ + cos 2η) · e^{-4νt}
    // Uc=Vc=0 → stationary vortex (advection is a pure gradient →
    // projected out → probes only the diffusion time order = cn_order).
    // Uc≠0 → convecting vortex: advection is rotational, so this
    // probes the AB2 advection time order too.  (t=0 ⇒ ξ=x, η=y.)
    const Real Uc = m_tg_uc, Vc = m_tg_vc;
    for (MFIter mfi(*m_pressure[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      auto const &p = m_pressure[lev]->array(mfi);
      auto const &u = m_vel[lev][0]->array(mfi);
      auto const &v = m_vel[lev][1]->array(mfi);
#if AMREX_SPACEDIM == 3
      auto const &w = m_vel[lev][2]->array(mfi);
#endif
      const Box &bx_cc = mfi.tilebox();
      amrex::ParallelFor(bx_cc,
                         [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                           const Real x = plo[0] + (i + 0.5_rt) * dx[0];
                           const Real y = plo[1] + (j + 0.5_rt) * dx[1];
                           p(i, j, k) = -0.25_rt * (std::cos(2.0_rt * x) +
                                                    std::cos(2.0_rt * y));
                         });
      const Box &bx_u = mfi.tilebox(IntVect::TheDimensionVector(0));
      amrex::ParallelFor(bx_u,
                         [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                           const Real x = plo[0] + i * dx[0];
                           const Real y = plo[1] + (j + 0.5_rt) * dx[1];
                           u(i, j, k) = Uc - std::cos(x) * std::sin(y);
                         });
      const Box &bx_v = mfi.tilebox(IntVect::TheDimensionVector(1));
      amrex::ParallelFor(bx_v,
                         [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                           const Real x = plo[0] + (i + 0.5_rt) * dx[0];
                           const Real y = plo[1] + j * dx[1];
                           v(i, j, k) = Vc + std::sin(x) * std::cos(y);
                         });
#if AMREX_SPACEDIM == 3
      const Box &bx_w = mfi.tilebox(IntVect::TheDimensionVector(2));
      amrex::ParallelFor(
          bx_w, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            w(i, j, k) = 0.0_rt;
          });
#endif
    }
    SyncFillICGhosts(lev);
    return;
  }

  // Default: 3D Taylor–Green vortex (periodic verification case).
  for (MFIter mfi(*m_pressure[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    const Box &bx_cc = mfi.tilebox();
    auto const &p = m_pressure[lev]->array(mfi);
    auto const &u = m_vel[lev][0]->array(mfi);
    auto const &v = m_vel[lev][1]->array(mfi);
#if AMREX_SPACEDIM == 3
    auto const &w = m_vel[lev][2]->array(mfi);
#endif

    amrex::ParallelFor(bx_cc, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      const Real x = plo[0] + (i + 0.5_rt) * dx[0];
      const Real y = plo[1] + (j + 0.5_rt) * dx[1];
#if AMREX_SPACEDIM == 3
      const Real z = plo[2] + (k + 0.5_rt) * dx[2];
      p(i, j, k) = (1.0_rt / 16.0_rt) *
                   (std::cos(2.0_rt * x) + std::cos(2.0_rt * y)) *
                   (std::cos(2.0_rt * z) + 2.0_rt);
#else
      p(i, j, k) = -0.25_rt * (std::cos(2.0_rt * x) + std::cos(2.0_rt * y));
#endif
    });

    const Box &bx_u = mfi.tilebox(IntVect::TheDimensionVector(0));
    amrex::ParallelFor(bx_u, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      const Real x = plo[0] + i * dx[0];
      const Real y = plo[1] + (j + 0.5_rt) * dx[1];
#if AMREX_SPACEDIM == 3
      const Real z = plo[2] + (k + 0.5_rt) * dx[2];
      u(i, j, k) = std::sin(x) * std::cos(y) * std::cos(z);
#else
      u(i, j, k) = std::sin(x) * std::cos(y);
#endif
    });

    const Box &bx_v = mfi.tilebox(IntVect::TheDimensionVector(1));
    amrex::ParallelFor(bx_v, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      const Real x = plo[0] + (i + 0.5_rt) * dx[0];
      const Real y = plo[1] + j * dx[1];
#if AMREX_SPACEDIM == 3
      const Real z = plo[2] + (k + 0.5_rt) * dx[2];
      v(i, j, k) = -std::cos(x) * std::sin(y) * std::cos(z);
#else
      v(i, j, k) = -std::cos(x) * std::sin(y);
#endif
    });

#if AMREX_SPACEDIM == 3
    const Box &bx_w = mfi.tilebox(IntVect::TheDimensionVector(2));
    amrex::ParallelFor(bx_w, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      w(i, j, k) = 0.0_rt;
    });
#endif
  }

  SyncFillICGhosts(lev);
}

// ============================================================
// Advance — single time step across all levels (no subcycling)
// ============================================================
void INSSolver::Advance() {
  const Real time = m_cur_time;

  // ---- 1) Build u* on every level ----
  for (int lev = 0; lev <= finest_level; ++lev) {
    const BoxArray &ba = grids[lev];
    const DistributionMapping &dm = dmap[lev];

    // Filled u^n: 2 ghost cells so the advection + Laplacian stencils work
    // at patch and C/F boundaries.
    std::array<MultiFab, AMREX_SPACEDIM> u_n;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
      BoxArray fba = amrex::convert(ba, IntVect::TheDimensionVector(d));
      u_n[d].define(fba, dm, 1, 2);
      FillFacePatch(lev, d, m_vel, u_n[d], time);
    }

    std::array<MultiFab *, AMREX_SPACEDIM> adv_p;
    std::array<const MultiFab *, AMREX_SPACEDIM> u_n_p;
    std::array<MultiFab *, AMREX_SPACEDIM> vs_p;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
      adv_p[d] = m_advect[lev][d].get();
      u_n_p[d] = &u_n[d];
      vs_p[d] = m_vstar[lev][d].get();
    }

    ComputeAdvection(lev, adv_p, u_n_p); // A^n → m_advect[lev]

    // 2nd-order Adams–Bashforth advection 3/2 A^n − 1/2 A^{n-1},
    // blended in place into m_advect_old (no per-step allocation).
    // No valid history (step 1 / first step after a regrid) ⇒ Euler.
    std::array<const MultiFab *, AMREX_SPACEDIM> adv_eff;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
      if (m_ab2_valid) {
        MultiFab::LinComb(*m_advect_old[lev][d], 1.5_rt, *m_advect[lev][d], 0,
                          -0.5_rt, *m_advect_old[lev][d], 0, 0, 1, 0);
        adv_eff[d] = m_advect_old[lev][d].get();
      } else {
        adv_eff[d] = m_advect[lev][d].get();
      }
    }

    ApplyCNDiffusion(lev, vs_p, {AMREX_D_DECL(&u_n[0], &u_n[1], &u_n[2])},
                     {AMREX_D_DECL(adv_eff[0], adv_eff[1], adv_eff[2])});

    // Promote A^n to next step's A^{n-1} by swapping the buffers
    // (m_advect still holds A^n; m_advect_old holds the spent blend
    // or stale history, which the next ComputeAdvection overwrites).
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
      std::swap(m_advect[lev][d], m_advect_old[lev][d]);
  }
  m_ab2_valid = true;

  // ---- 2) Perot modified-Poisson solve + projection on every level ----
  ProjectPerot();

  // ---- 3) Sync coarse with averaged-down fine ----
  AverageDownVelocity(m_vel);
  AverageDownPressure();
}

// ============================================================
// FillPatch helpers.  AMReX FillPatch handles interior + C/F
// interpolation (PhysBCFunctNoOp — the generic functor is poor for
// staggered physical BCs); the domain-boundary ghosts are then
// overwritten by our explicit staggered BC routines.
// ============================================================
void INSSolver::FillFacePatch(int lev, int dir, Vector<FaceMFArray> &source,
                              MultiFab &dst, Real time) {
  PhysBCFunctNoOp bc_func;
  if (lev == 0) {
    Vector<MultiFab *> smf{source[lev][dir].get()};
    Vector<Real> st{time};
    FillPatchSingleLevel(dst, time, smf, st, 0, 0, 1, geom[lev], bc_func, 0);
  } else {
    // The coarse source must carry correct domain-boundary ghosts
    // BEFORE the two-level interpolation: PhysBCFunctNoOp does not
    // fill them, and fine patches that hug a wall (e.g. the lid)
    // interpolate from coarse cells that would otherwise be garbage.
    FillVelGhostPhys(lev - 1, dir, *source[lev - 1][dir],
                     /*homogeneous=*/false);
    Vector<MultiFab *> cmf{source[lev - 1][dir].get()};
    Vector<MultiFab *> fmf{source[lev][dir].get()};
    Vector<Real> ct{time}, ft{time};
    FillPatchTwoLevels(dst, time, cmf, ct, fmf, ft, 0, 0, 1, geom[lev - 1],
                       geom[lev], bc_func, 0, bc_func, 0, ref_ratio[lev - 1],
                       &face_linear_interp, m_bc_vel[dir], 0);
  }
  FillVelGhostPhys(lev, dir, dst, /*homogeneous=*/false);
}

void INSSolver::FillCellPatch(int lev,
                              Vector<std::unique_ptr<MultiFab>> &source,
                              MultiFab &dst, Real time,
                              const Vector<BCRec> &bcs) {
  PhysBCFunctNoOp bc_func;
  if (lev == 0) {
    Vector<MultiFab *> smf{source[lev].get()};
    Vector<Real> st{time};
    FillPatchSingleLevel(dst, time, smf, st, 0, 0, 1, geom[lev], bc_func, 0);
  } else {
    // Fill the coarse source's domain-boundary ghosts first (see the
    // FillFacePatch note) so the coarse→fine interpolation near a
    // wall does not read uninitialised coarse ghost cells.
    FillPresGhostPhys(lev - 1, *source[lev - 1]);
    Vector<MultiFab *> cmf{source[lev - 1].get()};
    Vector<MultiFab *> fmf{source[lev].get()};
    Vector<Real> ct{time}, ft{time};
    // Piecewise-constant C/F interpolation (not lincc): conservative-
    // linear computes slopes that read the coarse scratch's
    // domain-boundary ghosts, which FillPatchTwoLevels leaves
    // uninitialised under PhysBCFunctNoOp → NaN for fine patches that
    // hug a wall (the lid).  pc_interp uses no slopes, so it is
    // immune.  The per-level pressure C/F coupling is already
    // approximate, so the order reduction here is acceptable.
    FillPatchTwoLevels(dst, time, cmf, ct, fmf, ft, 0, 0, 1, geom[lev - 1],
                       geom[lev], bc_func, 0, bc_func, 0, ref_ratio[lev - 1],
                       &pc_interp, bcs, 0);
  }
  FillPresGhostPhys(lev, dst);
}

// ============================================================
// Average-down — fine→coarse sync of overlap
// ============================================================
void INSSolver::AverageDownVelocity(Vector<FaceMFArray> &v) {
  for (int lev = finest_level - 1; lev >= 0; --lev) {
    Array<const MultiFab *, AMREX_SPACEDIM> fine;
    Array<MultiFab *, AMREX_SPACEDIM> crse;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
      fine[d] = v[lev + 1][d].get();
      crse[d] = v[lev][d].get();
    }
    amrex::average_down_faces(fine, crse, ref_ratio[lev], geom[lev]);
  }
}

void INSSolver::AverageDownPressure() {
  for (int lev = finest_level - 1; lev >= 0; --lev) {
    amrex::average_down(*m_pressure[lev + 1], *m_pressure[lev], 0, 1,
                        ref_ratio[lev]);
  }
}

// ============================================================
// Tagging — refine where |ω| > threshold
// ============================================================
void INSSolver::ComputeCellCenteredVorticityMag(int lev, MultiFab &vortmag) {
  // Need 1 ghost on velocity to take centred differences at cell centres.
  const BoxArray &ba = grids[lev];
  const DistributionMapping &dm = dmap[lev];

  std::array<MultiFab, AMREX_SPACEDIM> u_g;
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    BoxArray fba = amrex::convert(ba, IntVect::TheDimensionVector(d));
    u_g[d].define(fba, dm, 1, 1);
    FillFacePatch(lev, d, m_vel, u_g[d], m_cur_time);
  }

  const Real *dx = geom[lev].CellSize();
  const Real idx = 1.0_rt / dx[0];
  const Real idy = 1.0_rt / dx[1];
#if AMREX_SPACEDIM == 3
  const Real idz = 1.0_rt / dx[2];
#endif

  for (MFIter mfi(vortmag, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    const Box &bx = mfi.tilebox();
    auto const &u = u_g[0].const_array(mfi);
    auto const &v = u_g[1].const_array(mfi);
#if AMREX_SPACEDIM == 3
    auto const &w = u_g[2].const_array(mfi);
#endif
    auto const &om = vortmag.array(mfi);

    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      // Cell-centred derivatives via averaged face values.
      // d v/dx at cell (i,j,k): average over the two y-faces (j, j+1)
      Real dvdx = 0.25_rt * idx *
                  (v(i + 1, j, k) + v(i + 1, j + 1, k) - v(i - 1, j, k) -
                   v(i - 1, j + 1, k));
      Real dudy = 0.25_rt * idy *
                  (u(i, j + 1, k) + u(i + 1, j + 1, k) - u(i, j - 1, k) -
                   u(i + 1, j - 1, k));
#if AMREX_SPACEDIM == 3
      Real dwdy = 0.25_rt * idy *
                  (w(i, j + 1, k) + w(i, j + 1, k + 1) - w(i, j - 1, k) -
                   w(i, j - 1, k + 1));
      Real dvdz = 0.25_rt * idz *
                  (v(i, j, k + 1) + v(i, j + 1, k + 1) - v(i, j, k - 1) -
                   v(i, j + 1, k - 1));
      Real dudz = 0.25_rt * idz *
                  (u(i, j, k + 1) + u(i + 1, j, k + 1) - u(i, j, k - 1) -
                   u(i + 1, j, k - 1));
      Real dwdx = 0.25_rt * idx *
                  (w(i + 1, j, k) + w(i + 1, j, k + 1) - w(i - 1, j, k) -
                   w(i - 1, j, k + 1));
      Real ox = dwdy - dvdz;
      Real oy = dudz - dwdx;
      Real oz = dvdx - dudy;
      om(i, j, k) = std::sqrt(ox * ox + oy * oy + oz * oz);
#else
      om(i, j, k) = std::abs(dvdx - dudy);
#endif
    });
  }
}

void INSSolver::ErrorEst(int lev, TagBoxArray &tags, Real /*time*/,
                         int /*ngrow*/) {
  if (m_refine_vort <= 0.0)
    return;

  MultiFab vortmag(grids[lev], dmap[lev], 1, 0);
  vortmag.setVal(0.0);
  ComputeCellCenteredVorticityMag(lev, vortmag);

  const Real thr = m_refine_vort;
  const char tagval = TagBox::SET;

  for (MFIter mfi(vortmag, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    const Box &bx = mfi.tilebox();
    auto const &om = vortmag.const_array(mfi);
    auto const &tag = tags.array(mfi);
    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      if (om(i, j, k) > thr)
        tag(i, j, k) = tagval;
    });
  }
}

// ============================================================
// Plotfile — multi-level, cell-centred derived fields
// ============================================================
void INSSolver::WritePlotFile() {
  const int nout = AMREX_SPACEDIM + 2; // velocities + pressure + |ω|

  Vector<MultiFab> mfout(finest_level + 1);
  for (int lev = 0; lev <= finest_level; ++lev) {
    mfout[lev].define(grids[lev], dmap[lev], nout, 0);

    // |ω| → component nout-1
    MultiFab vortmag(grids[lev], dmap[lev], 1, 0);
    vortmag.setVal(0.0);
    ComputeCellCenteredVorticityMag(lev, vortmag);

    for (MFIter mfi(mfout[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const Box &bx = mfi.tilebox();
      auto const &out = mfout[lev].array(mfi);
      auto const &u = m_vel[lev][0]->const_array(mfi);
      auto const &v = m_vel[lev][1]->const_array(mfi);
#if AMREX_SPACEDIM == 3
      auto const &w = m_vel[lev][2]->const_array(mfi);
#endif
      auto const &pres = m_pressure[lev]->const_array(mfi);
      auto const &om = vortmag.const_array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        out(i, j, k, 0) = 0.5_rt * (u(i, j, k) + u(i + 1, j, k));
        out(i, j, k, 1) = 0.5_rt * (v(i, j, k) + v(i, j + 1, k));
#if AMREX_SPACEDIM == 3
        out(i, j, k, 2) = 0.5_rt * (w(i, j, k) + w(i, j, k + 1));
        out(i, j, k, 3) = pres(i, j, k);
        out(i, j, k, 4) = om(i, j, k);
#else
        out(i, j, k, 2) = pres(i, j, k);
        out(i, j, k, 3) = om(i, j, k);
#endif
      });
    }
  }

  Vector<std::string> names = {"u", "v"
#if AMREX_SPACEDIM == 3
                               ,
                               "w"
#endif
                               ,
                               "pressure", "vort_mag"};

  Vector<const MultiFab *> mf_ptrs;
  Vector<Geometry> g;
  Vector<int> level_steps;
  Vector<IntVect> rratio;
  for (int lev = 0; lev <= finest_level; ++lev) {
    mf_ptrs.push_back(&mfout[lev]);
    g.push_back(geom[lev]);
    level_steps.push_back(m_step);
  }
  for (int lev = 0; lev < finest_level; ++lev) {
    rratio.push_back(ref_ratio[lev]);
  }

  const std::string name = amrex::Concatenate(m_plot_prefix, m_step, 6);
  WriteMultiLevelPlotfile(name, finest_level + 1, mf_ptrs, names, g,
                          m_cur_time, level_steps, rratio);
  Print() << "Wrote " << name << "\n";
}

// ============================================================
// Diagnostics
// ============================================================
Real INSSolver::ComputeMaxVelocity() const {
  Real m = 0.0;
  for (int lev = 0; lev <= finest_level; ++lev) {
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
      m = std::max(m, m_vel[lev][d]->norminf(0, 0));
    }
  }
  ParallelDescriptor::ReduceRealMax(m);
  return m;
}

Real INSSolver::ComputeMaxDivergence() const {
  Real m = 0.0;
  for (int lev = 0; lev <= finest_level; ++lev) {
    const Real *dx = geom[lev].CellSize();
    const Real idx = 1.0_rt / dx[0];
    const Real idy = 1.0_rt / dx[1];
#if AMREX_SPACEDIM == 3
    const Real idz = 1.0_rt / dx[2];
#endif

    MultiFab div(grids[lev], dmap[lev], 1, 0);
    div.setVal(0.0);

    for (MFIter mfi(div, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const Box &bx = mfi.tilebox();
      auto const &u = m_vel[lev][0]->const_array(mfi);
      auto const &v = m_vel[lev][1]->const_array(mfi);
#if AMREX_SPACEDIM == 3
      auto const &w = m_vel[lev][2]->const_array(mfi);
#endif
      auto const &dv = div.array(mfi);
      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        Real d = idx * (u(i + 1, j, k) - u(i, j, k)) +
                 idy * (v(i, j + 1, k) - v(i, j, k));
#if AMREX_SPACEDIM == 3
        d += idz * (w(i, j, k + 1) - w(i, j, k));
#endif
        dv(i, j, k) = d;
      });
    }
    m = std::max(m, div.norminf(0, 0));
  }
  ParallelDescriptor::ReduceRealMax(m);
  return m;
}

// ============================================================
// ComputeTG2DError — discrete error vs the exact 2D Taylor–Green
//   u = -cos x sin y · F,  v = sin x cos y · F,   F = e^{-2νt}
// (level 0; periodic verification case only)
// ============================================================
void INSSolver::ComputeTG2DError(Real time, Real &l2, Real &linf) const {
  const int lev = 0;
  const Real *dx = geom[lev].CellSize();
  const Real *plo = geom[lev].ProbLo();
  const Real F = std::exp(-2.0 * m_nu * time);
  const Real Uc = m_tg_uc, Vc = m_tg_vc;

  Real sum2 = 0.0, mx = 0.0;
  long npts = 0;

  for (int comp = 0; comp < 2; ++comp) {
    const MultiFab &vel = *m_vel[lev][comp];
    for (MFIter mfi(vel); mfi.isValid(); ++mfi) {
      const Box &bx = mfi.validbox();
      auto const &a = vel.const_array(mfi);
      // exact face value at this component's nodal location;
      // convecting frame ξ = x − Uc t, η = y − Vc t
      amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
        Real x, y, ex;
        if (comp == 0) { // u on x-faces
          x = plo[0] + i * dx[0];
          y = plo[1] + (j + 0.5) * dx[1];
          ex = Uc - std::cos(x - Uc * time) * std::sin(y - Vc * time) * F;
        } else { // v on y-faces
          x = plo[0] + (i + 0.5) * dx[0];
          y = plo[1] + j * dx[1];
          ex = Vc + std::sin(x - Uc * time) * std::cos(y - Vc * time) * F;
        }
        const Real e = a(i, j, k) - ex;
        sum2 += e * e;
        mx = std::max(mx, std::abs(e));
        ++npts;
      });
    }
  }
  ParallelDescriptor::ReduceRealSum(sum2);
  ParallelDescriptor::ReduceLongSum(npts);
  ParallelDescriptor::ReduceRealMax(mx);

  l2 = (npts > 0) ? std::sqrt(sum2 / static_cast<Real>(npts)) : 0.0;
  linf = mx;
}

// ============================================================
// WriteTG2DDiagnostics — analytic error + (optional) fixed-grid
//   self-convergence dump/compare.  Called once after the time loop.
// ============================================================
void INSSolver::WriteTG2DDiagnostics() const {
  Real l2 = 0.0, linf = 0.0;
  ComputeTG2DError(m_cur_time, l2, linf);
  Print() << "TG2D-ERROR  ncell=" << geom[0].Domain().length(0)
          << "  dx=" << geom[0].CellSize()[0] << "  dt=" << m_dt
          << "  t=" << m_cur_time << "  L2=" << l2 << "  Linf=" << linf
          << "\n";

  if (m_tg2d_dump.empty() && m_tg2d_cmp.empty())
    return;

  // Final level-0 cell-centred velocity; comparing two dt runs on the
  // same grid cancels the (identical) spatial error, isolating the
  // temporal order.
  constexpr int nc = AMREX_SPACEDIM;
  MultiFab cc(grids[0], dmap[0], nc, 0);
  amrex::average_face_to_cellcenter(
      cc, 0,
      Array<const MultiFab *, AMREX_SPACEDIM>{AMREX_D_DECL(
          m_vel[0][0].get(), m_vel[0][1].get(), m_vel[0][2].get())});

  if (!m_tg2d_cmp.empty()) {
    MultiFab ref(grids[0], dmap[0], nc, 0);
    VisMF::Read(ref, m_tg2d_cmp);
    MultiFab diff(grids[0], dmap[0], nc, 0);
    MultiFab::LinComb(diff, 1.0_rt, cc, 0, -1.0_rt, ref, 0, 0, nc, 0);
    const Real d2 = MultiFab::Dot(diff, 0, diff, 0, nc, 0);
    const Real n = static_cast<Real>(nc * grids[0].numPts());
    Print() << "TG2D-SELFCONV  dt=" << m_dt << "  vs " << m_tg2d_cmp
            << "  L2diff=" << std::sqrt(d2 / n) << "\n";
  }
  if (!m_tg2d_dump.empty())
    VisMF::Write(cc, m_tg2d_dump);
}
