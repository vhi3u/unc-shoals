# using Pkg
# Pkg.instantiate() # Only need to do this once when you started the repo in another machine
# Pkg.resolve()
# import Pkg;
# Pkg.add("Oceananigans");
# Pkg.add("NCDatasets");
# Pkg.add("DataFrames");
# Pkg.add("Interpolations");
# Pkg.add("CUDA");
# Pkg.add("Oceanostics");
# Pkg.add("CSV");
# Pkg.add("Statistics");
# Pkg.add("SeawaterPolynomials");
# import Pkg;
# # Pkg.add("Rasters");
# Pkg.instantiate() # Only need to do this once when you started the repo in another machine
# Pkg.resolve()

using Oceananigans
using Oceananigans.Grids: Periodic, Bounded, minimum_zspacing
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

# build
@info "building domain"

# switches
LES = true
mass_flux = true
periodic_y = true
gradient_IC = true
sigmoid_v_bc = true
sigmoid_ic = true
is_coriolis = true
checkpointing = false
shoal_bath = true
if has_cuda_gpu()
    arch = GPU()
else
    arch = CPU()
end
@info "architecture = $(arch)"
include("dshoal_vn_param.jl")

# simulation knobs
run_number = 1 # <-- change this for each new run
sim_runtime = 30days
callback_interval = 86400seconds
run_tag = (periodic_y ? "periodic" : "bounded") * "_shoals$(run_number)"  # e.g. "periodic_run1"

if LES
    params = (; Lx=200e3, Ly=200e3, Lz=50)
else
    params = (; Lx=100000, Ly=200000, Lz=50)
end
if arch == CPU()
    params = (; params..., Nx=60, Ny=60, Nz=10)
else
    params = (; params..., Nx=200, Ny=200, Nz=50)
end

x, y, z = (0, params.Lx), (0, params.Ly), (-params.Lz, 0)

# grid  

if periodic_y
    grid = RectilinearGrid(arch; size=(params.Nx, params.Ny, params.Nz), halo=(4, 4, 4), x, y, z, topology=(Bounded, Periodic, Bounded))
else
    grid = RectilinearGrid(arch; size=(params.Nx, params.Ny, params.Nz), halo=(4, 4, 4), x, y, z, topology=(Bounded, Bounded, Bounded))
end

# model parameters

# # quadratic drag (log-law)
# const κᵛᵏ = 0.4    # von Kármán constant
# const Rz = 2.5e-4 # roughness fraction of domain depth
# z_0 = Rz * params.Lz
# z₁ = minimum_zspacing(grid, Center(), Center(), Center()) / 2
# c_dz = (κᵛᵏ / log(z₁ / z_0))^2

# @info "Using z₁ = $z₁"
# @info "Quadratic drag coefficient c_dz = $c_dz"

# drag = BulkDrag(coefficient=c_dz)

if shoal_bath
    Hs = 15.0
    shoal_length = 20000.0
    sigma = 8000.0
    shelf_depth = -25.0
    shelf_break_end = 12000.0

    slope_bottom = dshoal_param_bottom(params.Ly;
        Hs=Hs,
        shoal_length=shoal_length,
        sigma=sigma,
        shelf_depth=shelf_depth,
        shelf_break_end=shelf_break_end)
    GFB = GridFittedBottom(slope_bottom)
    ib_grid = ImmersedBoundaryGrid(grid, GFB)
else
    ib_grid = grid
end

@info ib_grid

if mass_flux
    v₀ = 0.10
else
    v₀ = 0.0
end

# defaults
T_north_v1, S_north_v1 = 20.5389, 32.6264
T_south_v1, S_south_v1 = 24.5378, 35.5830

params = (; params...,
    v₀=v₀,
    Ls=20e3,
    Le=100e3,
    Lw=10e3,
    τ=24hours,
    T_north_v1=T_north_v1,
    T_south_v1=T_south_v1,
    S_north_v1=S_north_v1,
    S_south_v1=S_south_v1,
    wind_stress=0.0)

# GPU-compatible SMOOTH piecewise linear T/S profiles (from CTD data)
# B1 = North, B2 = South
# Uses tanh blending for smooth transitions (equivalent to MATLAB smoothdata)
# δ = smoothing length scale (meters), set to ~2.5m for 5m effective smoothing

const δ_smooth = 2.5  # smoothing length scale in meters

# Smooth transition function: 0 when z >> z0, 1 when z << z0
@inline smooth_step(z, z0) = 0.5 * (1.0 - tanh((z - z0) / δ_smooth))

# Temperature at North boundary (B1) - SMOOTHED
@inline function T_north_pwl(z, v1=20.5389)
    z1, z2, z3 = -5.0, -15.0, -35.0
    v2, v3 = 17.8875, 14.3323
    m12 = (v2 - v1) / (z2 - z1)
    m23 = (v3 - v2) / (z3 - z2)
    val1 = v1
    val2 = v1 + m12 * (z - z1)
    val3 = v2 + m23 * (z - z2)
    val4 = v3
    w1 = smooth_step(z, z1)
    w2 = smooth_step(z, z2)
    w3 = smooth_step(z, z3)
    return val1 * (1 - w1) + val2 * (w1 - w2) + val3 * (w2 - w3) + val4 * w3
end

@inline function T_south_pwl(z, v1=24.5378)
    z1, z2, z3 = -5.0, -15.0, -30.0
    v2, v3 = 24.3073, 23.4116
    m12 = (v2 - v1) / (z2 - z1)
    m23 = (v3 - v2) / (z3 - z2)
    val1 = v1
    val2 = v1 + m12 * (z - z1)
    val3 = v2 + m23 * (z - z2)
    val4 = v3
    w1 = smooth_step(z, z1)
    w2 = smooth_step(z, z2)
    w3 = smooth_step(z, z3)
    return val1 * (1 - w1) + val2 * (w1 - w2) + val3 * (w2 - w3) + val4 * w3
end

@inline function S_north_pwl(z, v1=32.6264)
    z1, z2, z3 = -5.0, -15.0, -35.0
    v2, v3 = 33.7062, 33.2648
    m12 = (v2 - v1) / (z2 - z1)
    m23 = (v3 - v2) / (z3 - z2)
    val1 = v1
    val2 = v1 + m12 * (z - z1)
    val3 = v2 + m23 * (z - z2)
    val4 = v3
    w1 = smooth_step(z, z1)
    w2 = smooth_step(z, z2)
    w3 = smooth_step(z, z3)
    return val1 * (1 - w1) + val2 * (w1 - w2) + val3 * (w2 - w3) + val4 * w3
end

@inline function S_south_pwl(z, v1=35.5830)
    z1, z2, z3 = -5.0, -15.0, -30.0
    v2, v3 = 35.9986, 36.1776
    m12 = (v2 - v1) / (z2 - z1)
    m23 = (v3 - v2) / (z3 - z2)
    val1 = v1
    val2 = v1 + m12 * (z - z1)
    val3 = v2 + m23 * (z - z2)
    val4 = v3
    w1 = smooth_step(z, z1)
    w2 = smooth_step(z, z2)
    w3 = smooth_step(z, z3)
    return val1 * (1 - w1) + val2 * (w1 - w2) + val3 * (w2 - w3) + val4 * w3
end

# Temperature at East boundary (Offshore) - STRATIFIED
@inline function T_east_pwl(z)
    z1, z2, z3 = -5.0, -25.0, -45.0
    v1, v2, v3 = 25.0, 23.0, 21.0
    m12 = (v2 - v1) / (z2 - z1)
    m23 = (v3 - v2) / (z3 - z2)
    val1 = v1
    val2 = v1 + m12 * (z - z1)
    val3 = v2 + m23 * (z - z2)
    val4 = v3
    w1 = smooth_step(z, z1)
    w2 = smooth_step(z, z2)
    w3 = smooth_step(z, z3)
    return val1 * (1 - w1) + val2 * (w1 - w2) + val3 * (w2 - w3) + val4 * w3
end

# Salinity at East boundary (Offshore) - STABLE (Saltier at depth)
@inline function S_east_pwl(z)
    z1, z2, z3 = -5.0, -25.0, -45.0
    v1, v2, v3 = 35.8, 36.0, 36.2      # Flipped: 35.8 at surface, 36.2 at bottom
    m12 = (v2 - v1) / (z2 - z1)
    m23 = (v3 - v2) / (z3 - z2)
    val1 = v1
    val2 = v1 + m12 * (z - z1)
    val3 = v2 + m23 * (z - z2)
    val4 = v3
    w1 = smooth_step(z, z1)
    w2 = smooth_step(z, z2)
    w3 = smooth_step(z, z3)
    return val1 * (1 - w1) + val2 * (w1 - w2) + val3 * (w2 - w3) + val4 * w3
end

# Eastern boundary targets are now functions of z
params = (; params...)

# T/S boundary condition helpers
cᴰ = 2.5e-3
# bottom drag (z-boundary): signature (x, y, t, field_deps..., params)
@inline drag_u(x, y, t, u, v, cᴰ) = -cᴰ * u * sqrt(u^2 + v^2)
@inline drag_v(x, y, t, u, v, cᴰ) = -cᴰ * v * sqrt(u^2 + v^2)
# immersed drag (immersed boundary): signature (x, y, z, t, field_deps..., params)
@inline immersed_drag_u(x, y, z, t, u, v, cᴰ) = -cᴰ * u * sqrt(u^2 + v^2)
@inline immersed_drag_v(x, y, z, t, u, v, cᴰ) = -cᴰ * v * sqrt(u^2 + v^2)
drag_bc_u = FluxBoundaryCondition(drag_u, field_dependencies=(:u, :v), parameters=cᴰ)
drag_bc_v = FluxBoundaryCondition(drag_v, field_dependencies=(:u, :v), parameters=cᴰ)
immersed_drag_bc_u = FluxBoundaryCondition(immersed_drag_u, field_dependencies=(:u, :v), parameters=cᴰ)
immersed_drag_bc_v = FluxBoundaryCondition(immersed_drag_v, field_dependencies=(:u, :v), parameters=cᴰ)
if LES
    @inline tsbc(x, z, t) = T_south_pwl(z, T_south_v1)
    @inline tnbc(x, z, t) = T_north_pwl(z, T_north_v1)
    @inline ssbc(x, z, t) = S_south_pwl(z, S_south_v1)
    @inline snbc(x, z, t) = S_north_pwl(z, S_north_v1)
end

# wind stress BCs from wind speed (bulk formula, cf. kencode.jl)
# Ramp up over τ_ramp to suppress near-inertial oscillations from impulsive start
# ρₐ = 1.225   # kg m⁻³, average density of air at sea-level
# ρₒ = 1028.0  # kg m⁻³, average density of seawater
# u_w = 0.0    # m s⁻¹, 10-m wind speed (cross-shore)
# v_w = 10.0   # m s⁻¹, 10-m wind speed (along-shore)
# const Qu_wind_full = -ρₐ / ρₒ * cᴰ * u_w * abs(u_w)  # m² s⁻²
# const Qv_wind_full = -ρₐ / ρₒ * cᴰ * v_w * abs(v_w)  # m² s⁻²
# const τ_ramp = 2 * 86400.0  # ramp-up time in seconds (~2 days ≈ 2 inertial periods)
# @inline wind_ramp(t) = tanh(t / τ_ramp)
# @inline wind_flux_u(x, y, t) = Qu_wind_full * wind_ramp(t)
# @inline wind_flux_v(x, y, t) = Qv_wind_full * wind_ramp(t)
# wind_bc_u = FluxBoundaryCondition(wind_flux_u)
# wind_bc_v = FluxBoundaryCondition(wind_flux_v)

# new sponge masks using built-in functions from Oceananigans

const north_mask = PiecewiseLinearMask{:y}(center=params.Ly, width=params.Ls)
const south_mask = PiecewiseLinearMask{:y}(center=0, width=params.Ls)
const east_mask = PiecewiseLinearMask{:x}(center=params.Lx, width=params.Le)
@inline offshore_mask_uvw(x, y, z) = 0.5 * (1.0 + tanh((x - 65e3) / 10e3))
const global_params = params

if periodic_y
    @inline sponge_mask(x, y, z) = north_mask(x, y, z)
    @inline sponge_mask_uvw(x, y, z) = min(north_mask(x, y, z) + offshore_mask_uvw(x, y, z), 1.0)

    @inline function T_target(x, y, z, t)
        return T_south_pwl(z)
    end

    @inline function S_target(x, y, z, t)
        return S_south_pwl(z)
    end

    @inline function v_target(x, y, z, t)
        return v∞(x, z, t, global_params)
    end
else
    @inline sponge_mask(x, y, z) = min(north_mask(x, y, z) + south_mask(x, y, z), 1.0)
    @inline sponge_mask_uvw(x, y, z) = min(north_mask(x, y, z) + south_mask(x, y, z) + offshore_mask_uvw(x, y, z), 1.0)

    @inline function T_target(x, y, z, t)
        n = north_mask(x, y, z)
        s = south_mask(x, y, z)
        tot = n + s
        return tot > 0 ? (n * T_north_pwl(z) + s * T_south_pwl(z)) / tot : T_south_pwl(z)
    end

    @inline function S_target(x, y, z, t)
        n = north_mask(x, y, z)
        s = south_mask(x, y, z)
        tot = n + s
        return tot > 0 ? (n * S_north_pwl(z) + s * S_south_pwl(z)) / tot : S_south_pwl(z)
    end

    @inline function v_target(x, y, z, t)
        return v∞(x, z, t, global_params)
    end
end

# velocity function
if sigmoid_v_bc
    @inline function v∞(x, z, t, p)
        xC = 3e3
        xS = 65e3
        Lw = p.Lx
        k1 = 80 / Lw
        k2 = 20 / Lw

        s1 = 1 / (1 + exp(-k1 * (x - xC)))
        s2 = 1 / (1 + exp(k2 * (x - xS)))
        s = (s1 - 1) + s2
        sc = clamp(s, 0.0, 1.0)
        return p.v₀ * sc
    end
else
    @inline function v∞(x, z, t, p)
        return p.v₀
    end
end

u_nudging = Relaxation(; rate=1 / global_params.τ, mask=sponge_mask_uvw, target=0.0)
v_nudging = Relaxation(; rate=1 / global_params.τ, mask=sponge_mask_uvw, target=v_target)
w_nudging = Relaxation(; rate=1 / global_params.τ, mask=sponge_mask_uvw, target=0.0)
T_nudging = Relaxation(; rate=1 / global_params.τ, mask=sponge_mask, target=T_target)
S_nudging = Relaxation(; rate=1 / global_params.τ, mask=sponge_mask, target=S_target)

# forcing functions
if mass_flux
    forcings = (u=u_nudging, v=v_nudging, w=w_nudging, T=T_nudging, S=S_nudging)
else
    forcings = (T=T_nudging, S=S_nudging)
end

if periodic_y
    T_bcs = FieldBoundaryConditions()
    S_bcs = FieldBoundaryConditions()
    u_bcs = FieldBoundaryConditions(immersed=immersed_drag_bc_u) # top=wind_bc_u
    v_bcs = FieldBoundaryConditions(immersed=immersed_drag_bc_v) # top=wind_bc_v
    w_bcs = FieldBoundaryConditions()
else
    open_bc = OpenBoundaryCondition(v∞; parameters=params, scheme=PerturbationAdvection())
    open_zero = OpenBoundaryCondition(0.0)
    T_bcs = FieldBoundaryConditions(south=ValueBoundaryCondition(tsbc), north=ValueBoundaryCondition(tnbc))
    S_bcs = FieldBoundaryConditions(south=ValueBoundaryCondition(ssbc), north=ValueBoundaryCondition(snbc))
    u_bcs = FieldBoundaryConditions(immersed=immersed_drag_bc_u)
    v_bcs = FieldBoundaryConditions(immersed=immersed_drag_bc_v, north=open_bc, south=open_bc)
    w_bcs = FieldBoundaryConditions()
end

bcs = (u=u_bcs, v=v_bcs, w=w_bcs, T=T_bcs, S=S_bcs)
if is_coriolis
    coriolis = FPlane(latitude=35.2480)
else
    coriolis = nothing
end

# reltol = 1e-5
# maxiter = 500  # prevent CG solver from grinding millions of iters if convergence stalls

if periodic_y
    model = NonhydrostaticModel(ib_grid;
        timestepper=:RungeKutta3,
        advection=WENO(order=5),
        closure=AnisotropicMinimumDissipation(),
        hydrostatic_pressure_anomaly=CenterField(ib_grid),
        pressure_solver=ConjugateGradientPoissonSolver(ib_grid),
        tracers=(:T, :S),
        buoyancy=SeawaterBuoyancy(),
        coriolis=coriolis,
        boundary_conditions=bcs,
        forcing=forcings
    )
else
    model = NonhydrostaticModel(ib_grid;
        timestepper=:RungeKutta3,
        advection=WENO(order=5),
        closure=AnisotropicMinimumDissipation(),
        pressure_solver=ConjugateGradientPoissonSolver(ib_grid; reltol=reltol, maxiter=maxiter),
        tracers=(:T, :S),
        buoyancy=SeawaterBuoyancy(),
        coriolis=coriolis,
        boundary_conditions=bcs,
        forcing=forcings
    )
end

@info "" model

# Check for existing checkpoint to determine if we should pickup or start fresh
if checkpointing
    checkpoint_prefix = periodic_y ? "checkpoint_$(run_tag)" : "checkpoint_$(run_tag)"
    checkpoint_files = filter(f -> startswith(f, checkpoint_prefix) && endswith(f, ".jld2"), readdir("."))
    if !isempty(checkpoint_files)
        @info "Found checkpoint file(s) - will restore when simulation runs"
        pickup = true
    else
        pickup = false
    end
else
    pickup = false
end

# output

@info "creating output fields"

# Don't overwrite the NetCDF file when picking up from a checkpoint
overwrite_existing = !pickup

cfl_values = Float64[]       # Stores CFL at each step
cfl_times = Float64[]       # Stores model time

simulation = Simulation(model, Δt=15minutes, stop_time=sim_runtime)

conjure_time_step_wizard!(simulation, cfl=0.7, diffusive_cfl=0.7)

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
T = model.tracers.T
S = model.tracers.S
Ro = @at (Center, Center, Center) RossbyNumber(model)
KE = @at (Center, Center, Center) KineticEnergy(model)

# Centered velocities for consistency
u_c = @at (Center, Center, Center) u
v_c = @at (Center, Center, Center) v
w_c = @at (Center, Center, Center) w

# Cross-correlations for EKE and Fluxes
# EKE = 0.5 * (⟨uu⟩ - ⟨u⟩² + ⟨vv⟩ - ⟨v⟩² + ⟨ww⟩ - ⟨w⟩²)  — computed in post-processing from tavg_fields
uu = Field(u_c * u_c)
vv = Field(v_c * v_c)
ww = Field(w_c * w_c)
uT = Field(u_c * T)
uS = Field(u_c * S)
vT = Field(v_c * T)
vS = Field(v_c * S)
wT = Field(w_c * T)
wS = Field(w_c * S)

slice_fields = (; u_c, v_c, w_c, T, S, Ro, KE)
tavg_fields = (; u_c, v_c, w_c, uu, vv, ww, T, S, uT, uS, vT, vS, wT, wS)

# (1) 2D snapshots (every 1 day)
# Surface XY slice (top layer)
simulation.output_writers[:surface_slice] = NetCDFWriter(model, slice_fields,
    filename="top_$(run_tag).nc",
    schedule=TimeInterval(callback_interval),
    indices=(:, :, params.Nz),
    overwrite_existing=overwrite_existing)

# Mid-y XZ slice (cross-shore transect at domain center)
simulation.output_writers[:midy_slice] = NetCDFWriter(model, slice_fields,
    filename="midy_$(run_tag).nc",
    schedule=TimeInterval(callback_interval),
    indices=(:, round(Int, params.Ny / 2), :),
    overwrite_existing=overwrite_existing)

# # Mid-x YZ slice (along-shore transect at domain center)
# simulation.output_writers[:midx_slice] = NetCDFWriter(model, slice_fields,
#     filename="midx_$(run_tag).nc",
#     schedule=TimeInterval(callback_interval),
#     indices=(round(Int, params.Nx / 2), :, :),
#     overwrite_existing=overwrite_existing)

# # (2) 3D snapshots (every 20 days)
# simulation.output_writers[:snapshots_3d] = NetCDFWriter(model, slice_fields,
#     filename="snapshots_3d_$(run_tag).nc",
#     schedule=TimeInterval(20days),
#     overwrite_existing=overwrite_existing)

# # (3) 3D Time Averages (10 day window)
# simulation.output_writers[:time_avg_3d] = NetCDFWriter(model, tavg_fields,
#     filename="time_avg_3d_$(run_tag).nc",
#     schedule=AveragedTimeInterval(10days, window=10days),
#     overwrite_existing=overwrite_existing)

# # Domain-integrated KE time series
# ∫KE = Integral(KE)
# simulation.output_writers[:ke] = NetCDFWriter(model, (; ∫KE),
#     schedule=TimeInterval(callback_interval),
#     filename="KE_$(run_tag).nc",
#     overwrite_existing=overwrite_existing)

if checkpointing
    checkpoint_prefix = periodic_y ? "checkpoint_$(run_tag)" : "checkpoint_$(run_tag)"
    simulation.output_writers[:checkpointer] = Checkpointer(model,
        schedule=TimeInterval(5days),
        prefix=checkpoint_prefix,
        overwrite_existing=true,
        cleanup=true)
end

if !pickup
    @info "No checkpoint found, setting initial conditions"
    # initial conditions
    if sigmoid_ic
        v_init = (x, y, z) -> v∞(x, z, 0, params)
    else
        v_init = v₀
    end

    if gradient_IC
        @inline α_lin(y) = clamp(y / params.Ly, 0.0, 1.0)
        @inline blend(a, b, α) = (1 - α) * a + α * b
        @inline Tᵢ(x, y, z) = blend(T_south_pwl(z, T_south_v1), T_north_pwl(z, T_north_v1), α_lin(y))
        @inline Sᵢ(x, y, z) = blend(S_south_pwl(z, S_south_v1), S_north_pwl(z, S_north_v1), α_lin(y))
    else
        @inline Tᵢ(x, y, z) = T_south_pwl(z, T_south_v1)
        @inline Sᵢ(x, y, z) = S_south_pwl(z, S_south_v1)
    end

    set!(model, u=0.0, v=v_init, w=0.0, T=Tᵢ, S=Sᵢ)
end

# run simulation
@info """
════════════════════════════════════════════════════════
 SIMULATION CONFIGURATION: $(run_tag)
════════════════════════════════════════════════════════
 Run number:      $(run_number)
 Runtime:         $(sim_runtime)
 Architecture:    $(arch)

 ── Switches ──
 LES:             $(LES)
 mass_flux:       $(mass_flux)
 periodic_y:      $(periodic_y)
 gradient_IC:     $(gradient_IC)
 sigmoid_v_bc:    $(sigmoid_v_bc)
 sigmoid_ic:      $(sigmoid_ic)
 is_coriolis:     $(is_coriolis)
 checkpointing:   $(checkpointing)
 shoal_bath:      $(shoal_bath)
 ════════════════════════════════════════════════════════
"""
run!(simulation, pickup=pickup)
