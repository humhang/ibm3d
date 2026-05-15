#ifndef INS_SOLVER_H
#define INS_SOLVER_H

#include <AMReX_AmrCore.H>
#include <AMReX_BCRec.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <array>
#include <memory>
#include <string>

/// Incompressible Navier–Stokes solver on a staggered (MAC) AMR grid.
///
/// Layout
/// ------
///   u, v, w   face-centred (one component each)
///   p, phi    cell-centred (one component)
///
/// Algorithm (per time step, non-subcycled across AMR levels)
/// ----------------------------------------------------------
///   1.  A^n  = -(u^n · ∇) u^n               (explicit, centred, face-by-face)
///   2.  u*   = (I − εL)^{-1} [(I + εL) u^n + dt A^n − dt ∇p^n],   ε = ν dt / 2
///       with (I − εL)^{-1} replaced by the truncated Neumann series
///       Σ_{k=0}^{N} (εL)^k.
///   3.  ∇²φ  = (1/dt) ∇·u*                  (composite multi-level solve via AMReX MLMG)
///   4.  u^{n+1} = u* − dt ∇φ                (level-local projection)
///   5.  p^{n+1} = p^n + φ
///   6.  Average u^{n+1}, p^{n+1} fine→coarse (sync of overlapping coarse cells).
///
/// Coarse–fine treatment
/// ---------------------
///   * Per-level operator applications (advection, Laplacian, gradient) use
///     ghost cells filled by AMReX FillPatch infrastructure: intra-level halo
///     via `FillBoundary`, C/F ghosts via interpolation from the coarse level.
///   * The pressure-Poisson solve is delegated to MLMG, which handles flux
///     matching at the C/F interface inside its operator apply.
///   * Velocities are averaged-down (fine→coarse on overlapping faces) before
///     forming the Poisson RHS and after projection, so the composite
///     divergence operator and the projected velocity field are level-consistent.
class INSSolver : public amrex::AmrCore {
public:
  INSSolver();
  ~INSSolver() override = default;

  void InitData();
  void Run();

  // -------- AmrCore overrides --------
  void ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real time,
                int ngrow) override;
  void MakeNewLevelFromScratch(int lev, amrex::Real time,
                               const amrex::BoxArray &ba,
                               const amrex::DistributionMapping &dm) override;
  void MakeNewLevelFromCoarse(int lev, amrex::Real time,
                              const amrex::BoxArray &ba,
                              const amrex::DistributionMapping &dm) override;
  void RemakeLevel(int lev, amrex::Real time, const amrex::BoxArray &ba,
                   const amrex::DistributionMapping &dm) override;
  void ClearLevel(int lev) override;

private:
  // ---------------- Parameters (ParmParse) ----------------
  amrex::Real m_nu = 1.0e-3;     ///< kinematic viscosity
  amrex::Real m_cfl = 0.5;       ///< advective CFL
  amrex::Real m_fixed_dt = -1.0; ///< if > 0, override CFL-derived dt
  amrex::Real m_t_stop = 1.0;
  int m_max_step = 10000;
  int m_plot_int = 100;
  int m_regrid_int = 5;
  int m_cn_order = 2; ///< Neumann-series truncation order
  amrex::Real m_refine_vort = 5.0;
  int m_verbose = 1;
  std::string m_plot_prefix = "plt";

  // ---------------- State ----------------
  amrex::Real m_cur_time = 0.0;
  amrex::Real m_dt = 0.0;
  int m_step = 0;

  // ---------------- Level data ----------------
  using FaceMFArray =
      std::array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM>;

  amrex::Vector<FaceMFArray> m_vel;    ///< current velocity (u, v, w)
  amrex::Vector<FaceMFArray> m_advect; ///< explicit advection term A^n
  amrex::Vector<FaceMFArray> m_vstar;  ///< intermediate u*
  amrex::Vector<std::unique_ptr<amrex::MultiFab>> m_pressure;
  amrex::Vector<std::unique_ptr<amrex::MultiFab>> m_phi;

  // ---------------- BC records ----------------
  std::array<amrex::Vector<amrex::BCRec>, AMREX_SPACEDIM> m_bc_vel;
  amrex::Vector<amrex::BCRec> m_bc_pres;

  // ---------------- Helpers ----------------
  void ReadParameters();
  void AllocateLevelStorage(int lev, const amrex::BoxArray &ba,
                            const amrex::DistributionMapping &dm);
  void InitFlowField(int lev);

  void ComputeDt();
  void Advance();

  void ComputeAdvection(
      int lev, const std::array<amrex::MultiFab *, AMREX_SPACEDIM> &adv,
      const std::array<const amrex::MultiFab *, AMREX_SPACEDIM> &vel_in);

  void ApplyCNDiffusion(
      int lev, const std::array<amrex::MultiFab *, AMREX_SPACEDIM> &vstar,
      const std::array<const amrex::MultiFab *, AMREX_SPACEDIM> &vel_in,
      const std::array<const amrex::MultiFab *, AMREX_SPACEDIM> &adv,
      const amrex::MultiFab &pres);

  void ApplyFaceLaplacian(int lev, int dir, const amrex::MultiFab &fin,
                          amrex::MultiFab &Lf);

  void ComputePressureGradient(int lev, int dir, const amrex::MultiFab &p,
                               amrex::MultiFab &gp);

  void ProjectComposite();

  // FillPatch helpers
  void FillFacePatch(int lev, int dir,
                     amrex::Vector<FaceMFArray> &source,
                     amrex::MultiFab &dst, amrex::Real time);
  void FillCellPatch(int lev,
                     amrex::Vector<std::unique_ptr<amrex::MultiFab>> &source,
                     amrex::MultiFab &dst, amrex::Real time,
                     const amrex::Vector<amrex::BCRec> &bcs);

  // Average down across levels
  void AverageDownVelocity(amrex::Vector<FaceMFArray> &v);
  void AverageDownPressure();

  // Tagging
  void ComputeCellCenteredVorticityMag(int lev, amrex::MultiFab &vortmag);

  // Diagnostics / I/O
  void WritePlotFile();
  amrex::Real ComputeMaxVelocity() const;
  amrex::Real ComputeMaxDivergence() const;
};

#endif
