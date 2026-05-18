# ============================================================
# inputs.lid — lid-driven cavity (classic validation case)
#
# Square cavity, thin & periodic in z to recover the canonical
# 2D problem.  All walls no-slip; the top wall (y_hi) slides in
# +x with unit speed.  Re = U L / nu = 1 * 1 / 0.01 = 100.
#
# Expected: a primary recirculating vortex fills the cavity, a
# steady state is approached, |div u| stays at the CG tolerance.
#
#   ./build-release/src/ins_solver inputs.lid
# ============================================================

# ---- Domain ----
geometry.prob_lo     =  0.0  0.0  0.0
geometry.prob_hi     =  1.0  1.0  0.125
geometry.is_periodic =  0  0  1

# ---- Grid ----
amr.n_cell           =  64  64  8
amr.max_level        =  0
amr.blocking_factor  =  8
amr.max_grid_size    =  32
amr.v                =  1

# ---- Boundary conditions ----
#  noslip  : Dirichlet velocity (stationary wall or moving lid)
#  vel_*   : prescribed wall velocity vector (only for noslip/inflow)
ins.bc_xlo = noslip
ins.bc_xhi = noslip
ins.bc_ylo = noslip
ins.bc_yhi = noslip
ins.bc_zlo = periodic
ins.bc_zhi = periodic

ins.vel_yhi = 1.0 0.0 0.0     # moving lid (tangential)

# ---- Solver ----
ins.ic               =  quiescent
ins.nu               =  1.0e-2     # Re = 100
ins.cfl              =  0.4
ins.t_stop           =  40.0
ins.max_step         =  400
ins.plot_int         =  50
ins.cn_order         =  1
ins.poisson_tol      =  1.0e-10
ins.poisson_max_iter =  1000
ins.verbose          =  1
ins.plot_prefix      =  plt_lid
