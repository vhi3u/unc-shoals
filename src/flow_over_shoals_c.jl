using Oceananigans
using Oceananigans.Grids: Periodic, Bounded
using Oceananigans.Units
using Oceananigans.BoundaryConditions: OpenBoundaryCondition, FieldBoundaryConditions
using Oceananigans.TurbulenceClosures
using Oceananigans.Solvers: ConjugateGradientPoissonSolver, FFTBasedPoissonSolver
using Oceananigans.Models: buoyancy_operation
using Oceananigans.OutputWriters
using Oceananigans.Forcings
using Statistics: mean
using Oceanostics: RossbyNumber, ErtelPotentialVorticity,
    KineticEnergy, KineticEnergyDissipationRate, TurbulentKineticEnergy,
    XShearProductionRate, YShearProductionRate, ZShearProductionRate
using Oceanostics.ProgressMessengers: TimedMessenger
using SeawaterPolynomials.TEOS10
using Printf: @sprintf
using NCDatasets
using DataFrames
using CUDA: has_cuda_gpu, allowscalar
using Base.Threads

if has_cuda_gpu()
    arch = GPU()
    allowscalar(false)
else
    arch = CPU()
end
@info "architecture = $(arch)"
include("dshoal_vn_param.jl")

@info "Julia threads = $(nthreads())"


run_number = 441 # <-- change this for each new run
sim_runtime = 10days
callback_interval = 86400seconds
run_tag = "TEST_clean$(run_number)"  # e.g. "periodic_run1"

params = (; Lx=100e3, Ly=300e3, Lz=50, Nx=30, Ny=30, Nz=10)
params = (; params..., Nx=200, Ny=600, Nz=50)

x, y, z = (0, params.Lx), (0, params.Ly), (-params.Lz, 0)
grid = RectilinearGrid(arch; size=(params.Nx, params.Ny, params.Nz), halo=(4, 4, 4), x, y, z, topology=(Bounded, Periodic, Bounded))

# @inline slope_bottom(x, y) = ifelse(x < 10e3, -10.0 - 20.0 * (x / 10e3), -30.0 - 20.0 * ((x - 10e3) / 90e3))

const _shelf_length = 30e3
const _shelf_depth = -20.0
const _shoal_length = 30e3
const _shoal_crest_depth = -5.0
const _deep_ocean_depth = -50.0
const _sigma_shoal = 8e3
const _Hs_shoal = 15.0
const _Ly_shoal = 300e3
const _y0_shoal = params.Ly / 2.0
const _half_extent_shoal = _Ly_shoal / 2.0

@inline bottom(x, y) = _param_shoal_bottom(x, y, _y0_shoal, _sigma_shoal, _Hs_shoal,
    _half_extent_shoal, _shelf_length, _shelf_depth,
    _shoal_length, _shoal_crest_depth, _deep_ocean_depth)

# @inline slope_bottom(x, y) = -params.Lz * (x / params.Lx)
# @inline slope_bottom(x, y) = ifelse(x < 10e3, -30.0 * (x / 10e3), -30.0 - 20.0 * ((x - 10e3) / 90e3)) # piecewise slope 



@inline function slope_bottom(x, y)
    # Along-shore compact window
    window = param_shoal_window(y, _y0_shoal, _half_extent_shoal)

    # Background depth (3-piece piecewise linear)
    hw_slope = param_background_depth(x, _shelf_length, _shelf_depth, _deep_ocean_depth)
    hw = _deep_ocean_depth + (hw_slope - _deep_ocean_depth) * window

    # Shoal centerline x-profile
    hs_center = param_shoal_xprofile(x, _shoal_length, _shoal_crest_depth)

    # Bump above background (only positive)
    bump = max(0.0, hs_center - hw_slope)

    # Offshore taper kills the bump past the shoal
    taper = param_shoal_taper(x, _shoal_length)

    # # Gaussian cross-section in y
    # gauss_norm = exp(-((y - _y0_shoal)^2) / (2.0 * _sigma_shoal^2))

    # raw = hw + bump * taper * gauss_norm * window
    raw = hw + bump * taper * window

    # Shift entire bathymetry down by 5 m, clamp at deep_ocean_depth
    return max(raw - 5.0, _deep_ocean_depth)
end

# GFB = GridFittedBottom(slope_bottom)
GFB = GridFittedBottom(slope_bottom)
ib_grid = ImmersedBoundaryGrid(grid, GFB)

@info ib_grid

v₀ = 0.10

params = (; params..., v₀=v₀)

# @inline function T_north_pwl(z)
#     # Linear from 20.5389 at z=-5 to 14.3323 at z=-35
#     t = clamp((z - (-5.0)) / (-35.0 - (-5.0)), 0.0, 1.0)
#     return 20.5389 + t * (14.3323 - 20.5389)
# end

# @inline function T_south_pwl(z)
#     # Linear from 24.5378 at z=-5 to 23.4116 at z=-30
#     t = clamp((z - (-5.0)) / (-30.0 - (-5.0)), 0.0, 1.0)
#     return 24.5378 + t * (23.4116 - 24.5378)
# end

# @inline function S_north_pwl(z)
#     # Linear from 32.6264 at z=-5 to 33.2648 at z=-35
#     t = clamp((z - (-5.0)) / (-35.0 - (-5.0)), 0.0, 1.0)
#     return 32.6264 + t * (33.2648 - 32.6264)
# end

# @inline function S_south_pwl(z)
#     # Linear from 35.5830 at z=-5 to 36.1776 at z=-30
#     t = clamp((z - (-5.0)) / (-30.0 - (-5.0)), 0.0, 1.0)
#     return 35.5830 + t * (36.1776 - 35.5830)
# end

# const δ_smooth = 2.5  # smoothing length scale in meters

# @inline smooth_step(z, z0) = 0.5 * (1.0 - tanh((z - z0) / δ_smooth))

# @inline function T_north_pwl(z)
#     z1, z2, z3 = -5.0, -15.0, -35.0
#     v1, v2, v3 = 20.5389, 17.8875, 14.3323
#     m12 = (v2 - v1) / (z2 - z1)
#     m23 = (v3 - v2) / (z3 - z2)
#     val1 = v1
#     val2 = v1 + m12 * (z - z1)
#     val3 = v2 + m23 * (z - z2)
#     val4 = v3
#     w1 = smooth_step(z, z1)
#     w2 = smooth_step(z, z2)
#     w3 = smooth_step(z, z3)
#     return val1 * (1 - w1) + val2 * (w1 - w2) + val3 * (w2 - w3) + val4 * w3
# end

# @inline function T_south_pwl(z)
#     z1, z2, z3 = -5.0, -15.0, -30.0
#     v1, v2, v3 = 24.5378, 24.3073, 23.4116
#     m12 = (v2 - v1) / (z2 - z1)
#     m23 = (v3 - v2) / (z3 - z2)
#     val1 = v1
#     val2 = v1 + m12 * (z - z1)
#     val3 = v2 + m23 * (z - z2)
#     val4 = v3
#     w1 = smooth_step(z, z1)
#     w2 = smooth_step(z, z2)
#     w3 = smooth_step(z, z3)
#     return val1 * (1 - w1) + val2 * (w1 - w2) + val3 * (w2 - w3) + val4 * w3
# end

# @inline function S_north_pwl(z)
#     z1, z2, z3 = -5.0, -15.0, -35.0
#     v1, v2, v3 = 32.6264, 33.7062, 33.2648
#     m12 = (v2 - v1) / (z2 - z1)
#     m23 = (v3 - v2) / (z3 - z2)
#     val1 = v1
#     val2 = v1 + m12 * (z - z1)
#     val3 = v2 + m23 * (z - z2)
#     val4 = v3
#     w1 = smooth_step(z, z1)
#     w2 = smooth_step(z, z2)
#     w3 = smooth_step(z, z3)
#     return val1 * (1 - w1) + val2 * (w1 - w2) + val3 * (w2 - w3) + val4 * w3
# end

# @inline function S_south_pwl(z)
#     z1, z2, z3 = -5.0, -15.0, -30.0
#     v1, v2, v3 = 35.5830, 35.9986, 36.1776
#     m12 = (v2 - v1) / (z2 - z1)
#     m23 = (v3 - v2) / (z3 - z2)
#     val1 = v1
#     val2 = v1 + m12 * (z - z1)
#     val3 = v2 + m23 * (z - z2)
#     val4 = v3
#     w1 = smooth_step(z, z1)
#     w2 = smooth_step(z, z2)
#     w3 = smooth_step(z, z3)
#     return val1 * (1 - w1) + val2 * (w1 - w2) + val3 * (w2 - w3) + val4 * w3
# end

# const Tₑ_val = 23.11
# const Sₑ_val = 35.5

# @inline drag_u(x, y, t, u, v, p) = -p.cᴰ * √(u^2 + v^2) * u
# @inline drag_v(x, y, t, u, v, p) = -p.cᴰ * √(u^2 + v^2) * v
# drag_bc_u = FluxBoundaryCondition(drag_u, field_dependencies=(:u, :v), parameters=(; cᴰ=cᴰ,))
# drag_bc_v = FluxBoundaryCondition(drag_v, field_dependencies=(:u, :v), parameters=(; cᴰ=cᴰ,))

# @inline T_south_pwl(z) = 24.5378 #+ (23.4116 - 24.5378) / (-50.0) * z
# @inline T_north_pwl(z) = 20.5389 #+ (14.3323 - 20.5389) / (-50.0) * z
# @inline S_south_pwl(z) = 35.5830 #+ (36.1776 - 35.5830) / (-50.0) * z
# @inline S_north_pwl(z) = 32.6264 #+ (33.2648 - 32.6264) / (-50.0) * z

# @inline tsbc(x, z, t) = T_south_pwl(z)
# @inline tnbc(x, z, t) = T_north_pwl(z)
# @inline ssbc(x, z, t) = S_south_pwl(z)
# @inline snbc(x, z, t) = S_north_pwl(z)

reltol = sqrt(eps(ib_grid))
abstol = sqrt(eps(ib_grid))
@info "reltol = $reltol, abstol = $abstol"

model = NonhydrostaticModel(ib_grid;
    timestepper=:RungeKutta3,
    advection=WENO(order=5),
    #closure=AnisotropicMinimumDissipation(),
    #hydrostatic_pressure_anomaly=CenterField(ib_grid),
    pressure_solver=ConjugateGradientPoissonSolver(ib_grid, reltol=reltol, abstol=abstol), #; preconditioner=FFTBasedPoissonSolver(grid)),
    tracers=(:T, :S),
    #buoyancy=SeawaterBuoyancy(),
    #coriolis=FPlane(latitude=35.2480),
    #boundary_conditions=bcs
)



@info "" model

@info "creating output fields"

@info "u storage = $(typeof(parent(model.velocities.u)))"
@info "v storage = $(typeof(parent(model.velocities.v)))"
@info "w storage = $(typeof(parent(model.velocities.w)))"

cfl_values = Float64[]       # Stores CFL at each step
cfl_times = Float64[]       # Stores model time

simulation = Simulation(model, Δt=15minutes, stop_time=sim_runtime)

conjure_time_step_wizard!(simulation, cfl=0.9)

progress = TimedMessenger()

simulation.callbacks[:progress] = Callback(progress, TimeInterval(callback_interval))

function print_solver_iterations(sim)
    solver = sim.model.pressure_solver
    if hasproperty(solver, :conjugate_gradient_solver)
        cg = solver.conjugate_gradient_solver
        @info @sprintf("Pressure solver: %d CG iterations (t = %.2f days)",
            cg.iteration, time(sim) / 86400)
    end
end
simulation.callbacks[:solver_iters] = Callback(print_solver_iterations, TimeInterval(callback_interval))

u, v, w = model.velocities
# b = buoyancy_operation(model)

u_c = @at (Center, Center, Center) u
v_c = @at (Center, Center, Center) v
w_c = @at (Center, Center, Center) w
T = model.tracers.T
S = model.tracers.S
# Ro = @at (Center, Center, Center) RossbyNumber(model)

slice_fields = (; u_c, v_c, w_c, T, S)

# Surface XY slice (top layer)
simulation.output_writers[:surface_slice] =
    NetCDFWriter(model, slice_fields,
        filename="top_$(run_tag).nc",
        schedule=TimeInterval(callback_interval),
        indices=(:, :, params.Nz),
        overwrite_existing=true)

# Mid-y XZ slice (cross-shore transect at domain center)
simulation.output_writers[:midy_slice] =
    NetCDFWriter(model, slice_fields,
        filename="midy_$(run_tag).nc",
        schedule=TimeInterval(callback_interval),
        indices=(:, round(Int, params.Ny / 2), :),
        overwrite_existing=true)


# initial conditions
# uᵢ = 0.005 * rand(size(u)...)
# vᵢ = 0.005 * rand(size(v)...)
# wᵢ = 0.005 * rand(size(w)...)
# uᵢ .-= mean(uᵢ)
# vᵢ .-= mean(vᵢ)
# wᵢ .-= mean(wᵢ)
# uᵢ .+= 0
# vᵢ .+= v₀
vᵢ = v₀

# @inline Tᵢ(x, y, z) = T_south_pwl(z)
# @inline Sᵢ(x, y, z) = S_south_pwl(z)

# set!(model, v=vᵢ, T=Tᵢ, S=Sᵢ)
set!(model, v=vᵢ)

run!(simulation)