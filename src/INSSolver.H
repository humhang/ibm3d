#ifndef INS_SOLVER_H
#define INS_SOLVER_H

#include <AMReX_AmrCore.H>
#include <AMReX_BCRec.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_RealVect.H>

#include <array>
#include <memory>
#include <string>

/// Incompressible Navier–Stokes solver on a staggered (MAC) AMR grid,
/// using the Perot 1997 exact-factorisation fractional-step method
/// (= Taira–Colonius 2007 minus the immersed-boundary block).
///
/// Layout
/// ------
///   u, v, w   face-centred (one component each)
///   p         cell-centred (one component)
///
/// Algorithm (per time step, single-level proper; multi-level
/// approximated by per-level Perot solves + average-down sync)
/// ------------------------------------------------------------
///   ε = ν dt / 2,  B^N = Σ_{k=0}^N (εL)^k   (truncated Neumann series for A^{−1})
///
///   1.  A^n  = -(u^n · ∇) u^n                          (explicit, centred)
///   2.  r1   = (I + εL) u^n + dt A^n                   (NO pressure gradient — Perot)
///   3.  u*   = B^N r1                                   (predictor)
///   4.  Solve  (D B^N G) p^{n+1} = (1/dt) D u*          (modified Poisson, matrix-free CG)
///   5.  u^{n+1} = u* − dt B^N G p^{n+1}                  (Perot projection)
///   6.  Average u^{n+1} and p^{n+1} fine→coarse on overlapping faces / cells.
///
/// Why this differs from a textbook Chorin / incremental projection
/// ----------------------------------------------------------------
///   * The predictor does NOT contain ∇p^n.  Perot derives the
///     fractional-step *as* the LU factorisation of the block system,
///     and the LU has no pressure in the predictor block.
///   * The Poisson operator is D B^N G (a polynomial in L) rather than
///     the standard ∇² = D G.  At N=0 they coincide (Chorin); at N=1
///     the operator matches the CN diffusion to O(dt²).
///   * The projection multiplies by B^N as well — it is `u* − dt B^N G p`,
///     not `u* − dt G p`.
///   * `m_pressure` is solved for directly, not accumulated.  Use the
///     previous step's value as the CG warm start.
///
/// Coarse–fine treatment (current, approximate)
/// --------------------------------------------
/// Each level's modified Poisson is solved independently; u* and the
/// projected u are averaged-down fine→coarse to keep coarse face values
/// consistent at the C/F interface.  A proper composite D B^N G — with
/// fine ghosts of every intermediate face term FillPatched from coarse
/// and average_down between L applications — is feasible but deferred.
/// Per-domain-face boundary condition kind.
///
///   periodic   handled by AMReX geometry periodicity
///   dirichlet  prescribed velocity vector (no-slip wall, moving lid,
///              or inflow — all the same numerically: u = g on the
///              boundary).  Pressure: homogeneous Neumann ∂p/∂n = 0.
///   slip       free-slip wall: normal velocity = 0, tangential
///              zero-gradient.  Pressure: homogeneous Neumann.
///   outflow    velocity zero-gradient (Neumann).  Pressure:
///              homogeneous Dirichlet p = 0.
enum class BCKind { periodic, dirichlet, slip, outflow };

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
  int m_cn_order = 1; ///< Neumann-series truncation order N (B^N = I+εL+…+(εL)^N)
  amrex::Real m_refine_vort = 5.0;
  amrex::Real m_poisson_tol = 1.0e-10; ///< CG relative tolerance
  int m_poisson_max_iter = 500;
  int m_verbose = 1;
  std::string m_plot_prefix = "plt";
  std::string m_ic = "taylor_green"; ///< "taylor_green" | "quiescent"

  // ---------------- Boundary conditions ----------------
  // Indexed by direction (0=x, 1=y, 2=z); lo and hi domain faces.
  std::array<BCKind, AMREX_SPACEDIM> m_bc_lo{};
  std::array<BCKind, AMREX_SPACEDIM> m_bc_hi{};
  std::array<amrex::RealVect, AMREX_SPACEDIM> m_bcvel_lo{};
  std::array<amrex::RealVect, AMREX_SPACEDIM> m_bcvel_hi{};
  /// True when the pressure system is singular (no outflow ⇒ pure
  /// Neumann/periodic ⇒ solution defined up to a constant).
  bool m_pressure_singular = true;

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

  // ---------------- BC records ----------------
  std::array<amrex::Vector<amrex::BCRec>, AMREX_SPACEDIM> m_bc_vel;
  amrex::Vector<amrex::BCRec> m_bc_pres;

  // ---------------- Helpers ----------------
  void ReadParameters();
  void ParseBCs();      ///< read ins.bc_* / ins.vel_* and validate
  void BuildBCRecs();   ///< translate BCKind → AMReX BCRec for FillPatch
  void AllocateLevelStorage(int lev, const amrex::BoxArray &ba,
                            const amrex::DistributionMapping &dm);
  void InitFlowField(int lev);

  // ---- Physical boundary conditions (staggered grid) ----
  /// Fill the domain-boundary ghost layer (and, for the normal
  /// component, the boundary face itself) of velocity component
  /// `comp`.  `homogeneous` uses zero wall data (for CG operator
  /// applications); otherwise the prescribed `ins.vel_*` values.
  /// Also calls FillBoundary for periodic / intra-level halos.
  void FillVelGhostPhys(int lev, int comp, amrex::MultiFab &mf,
                        bool homogeneous);
  /// Fill pressure ghosts: Neumann at dirichlet/slip walls,
  /// Dirichlet p=0 at outflow.  (Always homogeneous — pressure has
  /// no inhomogeneous wall data in this formulation.)
  void FillPresGhostPhys(int lev, amrex::MultiFab &mf);
  /// Re-impose the prescribed wall velocity on a velocity field
  /// after the predictor / projection (the linear operators do not
  /// preserve inhomogeneous Dirichlet data exactly).
  void EnforceVelDirichlet(
      int lev, const std::array<amrex::MultiFab *, AMREX_SPACEDIM> &vel);

  void ComputeDt();
  void Advance();

  void ComputeAdvection(
      int lev, const std::array<amrex::MultiFab *, AMREX_SPACEDIM> &adv,
      const std::array<const amrex::MultiFab *, AMREX_SPACEDIM> &vel_in);

  /// CN predictor: u* = B^N [ (I + εL) u^n + dt A^n ].   NO pressure term.
  void ApplyCNDiffusion(
      int lev, const std::array<amrex::MultiFab *, AMREX_SPACEDIM> &vstar,
      const std::array<const amrex::MultiFab *, AMREX_SPACEDIM> &vel_in,
      const std::array<const amrex::MultiFab *, AMREX_SPACEDIM> &adv);

  void ApplyFaceLaplacian(int lev, int dir, const amrex::MultiFab &fin,
                          amrex::MultiFab &Lf);

  void ComputePressureGradient(int lev, int dir, const amrex::MultiFab &p,
                               amrex::MultiFab &gp);

  /// Apply B^N = Σ_{k=0}^N (εL)^k to a face-centred MultiFab.
  /// `src` must have ≥2 ghost cells already filled (FillBoundary/FillPatch).
  void ApplyBNFace(int lev, int dir, const amrex::MultiFab &src,
                   amrex::MultiFab &dst);

  /// Apply the modified-Poisson operator (D B^N G) to a cell-centred φ.
  /// `phi` must have its 1-cell ghost layer filled.
  void ApplyModifiedPoissonOp(int lev, const amrex::MultiFab &phi,
                              amrex::MultiFab &result);

  /// Matrix-free CG on (D B^N G) p = rhs.
  /// `p` is in/out — pass the previous-step pressure as the warm start.
  /// `rhs` is read-only.  Returns the iteration count.
  int SolveModifiedPoisson(int lev, amrex::MultiFab &p,
                           const amrex::MultiFab &rhs);

  /// Perot projection across all levels: per-level modified-Poisson solve,
  /// per-level u^{n+1} = u* − dt B^N G p, then fine→coarse sync.
  void ProjectPerot();

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

  /// Discrete L2 and L∞ error of the face velocities against the exact
  /// 2D Taylor–Green decaying vortex at `time` (level 0 only — used
  /// for the analytic convergence study; no-op unless ic == tg2d).
  void ComputeTG2DError(amrex::Real time, amrex::Real &l2, amrex::Real &linf)
      const;

  // Compatibility helper: subtract the (volume-weighted) mean of a
  // cell-centred MultiFab in place — needed for the singular periodic
  // pressure system.
  void SubtractMean(int lev, amrex::MultiFab &mf);
};

#endif
