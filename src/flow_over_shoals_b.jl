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
using Oceananigans.Grids: Bounded
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
gradient_IC = false
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
run_number = 20 # <-- change this for each new run
sim_runtime = 10days
callback_interval = 86400seconds
run_tag = "bdd_shoals$(run_number)"

if LES
    params = (; Lx=100e3, Ly=300e3, Lz=50, Nx=50, Ny=150, Nz=10)
else
    params = (; Lx=100000, Ly=300000, Lz=50, Nx=50, Ny=150, Nz=10)
end
if arch == CPU()
    params = (; params..., Nx=50, Ny=150, Nz=10) # keep the same for now
else
    params = (; params..., Nx=200, Ny=600, Nz=50)
end

x, y, z = (0, params.Lx), (0, params.Ly), (-params.Lz, 0)

# grid (bounded in all directions)
grid = RectilinearGrid(arch; size=(params.Nx, params.Ny, params.Nz), halo=(4, 4, 4), x, y, z, topology=(Bounded, Bounded, Bounded))

# model parameters

if shoal_bath
    # Define shoal parameters (align with the new sigmoidal setup)
    Hs = 15.0         # Height of shoal above -25m shelf
    sigma = 8e3       # Gaussian width of shoal (half crossover)
    shoal_length = 20e3 # Horizontal span of the shoal ridge

    slope_bottom = dshoal_param_bottom(params.Ly; Hs=Hs, sigma=sigma, shoal_length=shoal_length)
    GFB = GridFittedBottom(slope_bottom)
    ib_grid = ImmersedBoundaryGrid(grid, GFB)
else
    ib_grid = grid
end

@info ib_grid

if mass_flux
    v₀ = 0.10 # [m/s] mean flow velocity
else
    v₀ = 0.0
end

# store parameters for sponge setup
params = (; params...,
    v₀=v₀, # add v₀ to params for GPU compatibility
    Ls=50e3, # sponge layer size (north and south)
    Le=10e3, # sponge layer size (east)
    Lw=10e3, # sponge layer size (west)
    τₙ=24hours, # relaxation timescale for north sponge
    τₛ=24hours, # relaxation timescale for south sponge
    τₑ=5days, # relaxation timescale for east sponge
    τ_w=24hours, # relaxation timescale for west sponge
    τ_ts=24hours) # relaxation timescale for temperature and salinity at the north and south boundaries

# GPU-compatible SMOOTH piecewise linear T/S profiles (from CTD data)
# B1 = North, B2 = South
# Uses tanh blending for smooth transitions (equivalent to MATLAB smoothdata)
# δ = smoothing length scale (meters), set to ~2.5m for 5m effective smoothing

const δ_smooth = 2.5  # smoothing length scale in meters

# Smooth transition function: 0 when z >> z0, 1 when z << z0
@inline smooth_step(z, z0) = 0.5 * (1.0 - tanh((z - z0) / δ_smooth))

# Temperature at North boundary (B1) - SMOOTHED
@inline function T_north_pwl(z)
    z1, z2, z3 = -5.0, -15.0, -35.0
    v1, v2, v3 = 20.5389, 17.8875, 14.3323
    m12 = (v2 - v1) / (z2 - z1)
    m23 = (v3 - v2) / (z3 - z2)
    # Piecewise linear values
    val1 = v1
    val2 = v1 + m12 * (z - z1)
    val3 = v2 + m23 * (z - z2)
    val4 = v3
    # Smooth blending between segments
    w1 = smooth_step(z, z1)  # 0 above z1, 1 below z1
    w2 = smooth_step(z, z2)  # 0 above z2, 1 below z2
    w3 = smooth_step(z, z3)  # 0 above z3, 1 below z3
    # Blend: start with val1, transition to val2 at z1, to val3 at z2, to val4 at z3
    return val1 * (1 - w1) + val2 * (w1 - w2) + val3 * (w2 - w3) + val4 * w3
end

# Temperature at South boundary (B2) - SMOOTHED
@inline function T_south_pwl(z)
    z1, z2, z3 = -5.0, -15.0, -30.0
    v1, v2, v3 = 24.5378, 24.3073, 23.4116
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

# Salinity at North boundary (B1) - SMOOTHED
@inline function S_north_pwl(z)
    z1, z2, z3 = -5.0, -15.0, -35.0
    v1, v2, v3 = 32.6264, 33.7062, 33.2648
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

# Salinity at South boundary (B2) - SMOOTHED
@inline function S_south_pwl(z)
    z1, z2, z3 = -5.0, -15.0, -30.0
    v1, v2, v3 = 35.5830, 35.9986, 36.1776
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

# Eastern boundary T/S constants
Tₑ_val = 23.11
Sₑ_val = 35.5
params = (; params..., Tₑ=Tₑ_val, Sₑ=Sₑ_val)

# bottom drag parameters
# cᴰ = 2.5e-3 # dimensionless drag coefficient

# quadratic drag:
const κᵛᵏ = 0.4 # von Karman constant
const Rz = 2.5e-4
z_0 = Rz * params.Lz
z₁ = minimum_zspacing(grid, Center(), Center(), Center()) / 2
c_dz = (κᵛᵏ / log(z₁ / z_0))^2

@info "Using z₁ = $z₁"
@info "Quadratic drag coefficient c_dz = $c_dz"

drag = BulkDrag(coefficient=c_dz)
# # wind stress parameters
# # τ_wind = 0.14 N/m² southward (from observational data)
# # Convert to kinematic stress: τ_kinematic = τ_wind / ρ₀  [m²/s²]
# ρ₀ = 1020.0            # reference seawater density [kg/m³]
# τ_wind = -0.14          # wind stress magnitude [N/m²]
# τ_kinematic_v = τ_wind / ρ₀   # negative = southward (−y direction)

if LES
    # @inline drag_u(x, y, t, u, v, p) = -p.cᴰ * √(u^2 + v^2) * u
    # @inline drag_v(x, y, t, u, v, p) = -p.cᴰ * √(u^2 + v^2) * v
    # drag_bc_u = FluxBoundaryCondition(drag_u, field_dependencies=(:u, :v), parameters=(; cᴰ=cᴰ,))
    # drag_bc_v = FluxBoundaryCondition(drag_v, field_dependencies=(:u, :v), parameters=(; cᴰ=cᴰ,))
    # wind_bc_v = FluxBoundaryCondition(τ_kinematic_v)  # uniform wind stress on v at surface
    @inline tsbc(x, z, t) = T_south_pwl(z)
    @inline tnbc(x, z, t) = T_north_pwl(z)
    @inline ssbc(x, z, t) = S_south_pwl(z)
    @inline snbc(x, z, t) = S_north_pwl(z)
end

# mask functions
@inline function south_mask(x, y, z, p)
    y0 = 0
    y1 = p.Ls
    if y0 <= y <= y1
        return 1 - y / y1
    else
        return 0.0
    end
end

@inline function north_mask(x, y, z, p)
    y0 = p.Ly - p.Ls
    y1 = p.Ly
    if y0 <= y <= y1
        return (y - y0) / (y1 - y0)
    else
        return 0.0
    end
end

@inline function east_mask(x, y, z, p)
    x0 = p.Lx - p.Le
    x1 = p.Lx
    if x0 <= x <= x1
        return (x - x0) / (x1 - x0)
    else
        return 0.0
    end
end

@inline function west_mask(x, y, z, p)
    x1 = p.Lw
    if x <= x1
        return 1.0 - x / x1
    else
        return 0.0
    end
end

# velocity function
if sigmoid_v_bc
    @inline function v∞(x, z, t, p)
        xC = 3e3
        xS = 60e3
        Lw = p.Lx
        k1 = 80 / Lw
        k2 = 40 / Lw

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

# sponge functions (bounded setup — additive sponge layers)
if mass_flux
    @inline sponge_u(x, y, z, t, u, p) = -(
        south_mask(x, y, z, p) * u / p.τₛ +
        north_mask(x, y, z, p) * u / p.τₙ +
        east_mask(x, y, z, p) * u / p.τₑ
    )

    @inline sponge_v(x, y, z, t, v, p) = -(
        south_mask(x, y, z, p) * (v - v∞(x, z, t, p)) / p.τₛ +
        north_mask(x, y, z, p) * (v - v∞(x, z, t, p)) / p.τₙ +
        east_mask(x, y, z, p) * v / p.τₑ
    )
    @inline sponge_w(x, y, z, t, w, p) = -(
        south_mask(x, y, z, p) * w / p.τₛ +
        north_mask(x, y, z, p) * w / p.τₙ +
        east_mask(x, y, z, p) * w / p.τₑ
    )

    @inline sponge_T(x, y, z, t, T, p) = -(
        south_mask(x, y, z, p) * (T - T_south_pwl(z)) / p.τ_ts +
        north_mask(x, y, z, p) * (T - T_south_pwl(z)) / p.τ_ts
    )

    @inline sponge_S(x, y, z, t, S, p) = -(
        south_mask(x, y, z, p) * (S - S_south_pwl(z)) / p.τ_ts +
        north_mask(x, y, z, p) * (S - S_south_pwl(z)) / p.τ_ts
    )
end

# forcing functions
FT = Forcing(sponge_T, field_dependencies=:T, parameters=params)
FS = Forcing(sponge_S, field_dependencies=:S, parameters=params)
Fᵤ = Forcing(sponge_u, field_dependencies=:u, parameters=params)
Fᵥ = Forcing(sponge_v, field_dependencies=:v, parameters=params)
F_w = Forcing(sponge_w, field_dependencies=:w, parameters=params)
forcings = (u=Fᵤ, v=Fᵥ, w=F_w, T=FT, S=FS)

# boundary conditions (bounded setup)
v_north = OpenBoundaryCondition(v∞; parameters=params, scheme=PerturbationAdvection(inflow_timescale=Inf, outflow_timescale=0.0))
v_south = OpenBoundaryCondition(v∞; parameters=params, scheme=PerturbationAdvection(inflow_timescale=0.0, outflow_timescale=Inf))
v_east = OpenBoundaryCondition(0.0; scheme=PerturbationAdvection(inflow_timescale=1day, outflow_timescale=15minutes))

T_bcs = FieldBoundaryConditions(south=ValueBoundaryCondition(tsbc), north=ValueBoundaryCondition(tsbc), east=GradientBoundaryCondition(0.0))
S_bcs = FieldBoundaryConditions(south=ValueBoundaryCondition(ssbc), north=ValueBoundaryCondition(ssbc), east=GradientBoundaryCondition(0.0))
u_bcs = FieldBoundaryConditions(immersed=drag, east=v_east)
v_bcs = FieldBoundaryConditions(north=v_north, south=v_south, immersed=drag, east=v_east)
w_bcs = FieldBoundaryConditions(immersed=drag)

bcs = (u=u_bcs, v=v_bcs, w=w_bcs, T=T_bcs, S=S_bcs)
if is_coriolis
    coriolis = FPlane(latitude=35.2480)
else
    coriolis = nothing
end

# reltol = sqrt(eps(ib_grid))
# abstol = sqrt(eps(ib_grid))
# @info "reltol = $reltol, abstol = $abstol"

model = NonhydrostaticModel(ib_grid;
    timestepper=:RungeKutta3,
    advection=WENO(order=5, minimum_buffer_upwind_order=1),
    closure=DynamicSmagorinsky(),
    pressure_solver=ConjugateGradientPoissonSolver(ib_grid),
    tracers=(:T, :S),
    buoyancy=SeawaterBuoyancy(),
    coriolis=coriolis,
    boundary_conditions=bcs,
    #forcing=forcings
)

@info "" model

pickup = false

# output

@info "creating output fields"

# Don't overwrite the NetCDF file when picking up from a checkpoint
overwrite_existing = !pickup

cfl_values = Float64[]       # Stores CFL at each step
cfl_times = Float64[]       # Stores model time

simulation = Simulation(model, Δt=15minutes, stop_time=sim_runtime)

conjure_time_step_wizard!(simulation, cfl=0.9, diffusive_cfl=0.8)

progress = TimedMessenger()

function print_solver_iterations(sim)
    solver = sim.model.pressure_solver
    if hasproperty(solver, :conjugate_gradient_solver)
        cg = solver.conjugate_gradient_solver
        @info @sprintf("Pressure solver: %d CG iterations (t = %.2f days)",
            cg.iteration, time(sim) / 86400)
    end
end

simulation.callbacks[:progress] = Callback(progress, TimeInterval(callback_interval))

u, v, w = model.velocities
T = model.tracers.T
S = model.tracers.S
b = buoyancy_operation(model)

u_c = @at (Center, Center, Center) u
v_c = @at (Center, Center, Center) v
w_c = @at (Center, Center, Center) w

# # vorticity, potential vorticity, rossby number calculations
# ζ = @at (Center, Center, Center) Field(∂x(v_c) - ∂y(u_c))

# PV = @at (Center, Center, Center) ErtelPotentialVorticity(model, u, v, w, b, model.coriolis)
Ro = @at (Center, Center, Center) RossbyNumber(model)

slice_fields = (; u_c, v_c, w_c, T, S, Ro)

# Surface XY slice (top layer)
simulation.output_writers[:surface_slice] =
    NetCDFWriter(model, slice_fields,
        filename="top_$(run_tag).nc",
        schedule=TimeInterval(callback_interval),
        indices=(:, :, params.Nz),
        overwrite_existing=overwrite_existing)

# Mid-y XZ slice (cross-shore transect at domain center)
simulation.output_writers[:midy_slice] =
    NetCDFWriter(model, slice_fields,
        filename="midy_$(run_tag).nc",
        schedule=TimeInterval(callback_interval),
        indices=(:, round(Int, params.Ny / 2), :),
        overwrite_existing=overwrite_existing)

# # Cross-correlation fields (Reynolds stresses & tracer fluxes)
# uu = Field((@at (Center, Center, Center) u * u))
# vv = Field((@at (Center, Center, Center) v * v))
# ww = Field((@at (Center, Center, Center) w * w))
# uv = Field((@at (Center, Center, Center) u * v))
# uw = Field((@at (Center, Center, Center) u * w))
# vw = Field((@at (Center, Center, Center) v * w))
# uT = Field((@at (Center, Center, Center) u * T))
# vT = Field((@at (Center, Center, Center) v * T))
# wT = Field((@at (Center, Center, Center) w * T))
# uS = Field((@at (Center, Center, Center) u * S))
# vS = Field((@at (Center, Center, Center) v * S))
# wS = Field((@at (Center, Center, Center) w * S))

# # Y-averages of velocities and tracers
# u_yavg = Average(u, dims=2)
# v_yavg = Average(v, dims=2)
# w_yavg = Average(w, dims=2)
# uu_yavg = Average(uu, dims=2)
# vv_yavg = Average(vv, dims=2)
# ww_yavg = Average(ww, dims=2)
# T_yavg = Average(T, dims=2)
# S_yavg = Average(S, dims=2)

# # Y-averages of cross-correlations
# uv_yavg = Average(uv, dims=2)
# uw_yavg = Average(uw, dims=2)
# vw_yavg = Average(vw, dims=2)
# uT_yavg = Average(uT, dims=2)
# vT_yavg = Average(vT, dims=2)
# wT_yavg = Average(wT, dims=2)
# uS_yavg = Average(uS, dims=2)
# vS_yavg = Average(vS, dims=2)
# wS_yavg = Average(wS, dims=2)

# output_interval = callback_interval

# simulation.output_writers[:yavg_fields] = NetCDFWriter(
#     model,
#     (; u_yavg, v_yavg, w_yavg, T_yavg, S_yavg,
#         uv_yavg, uw_yavg, vw_yavg,
#         uT_yavg, vT_yavg, wT_yavg,
#         uS_yavg, vS_yavg, wS_yavg),
#     schedule=AveragedTimeInterval(output_interval, window=output_interval),
#     filename="time_yavg_$(run_tag).nc",
#     overwrite_existing=overwrite_existing)

# KE = KineticEnergy(model)
# TKE = 0.5 * ((Field(uu_yavg) - Field(u_yavg) * Field(u_yavg))
#              + (Field(vv_yavg) - Field(v_yavg) * Field(v_yavg))
#              + (Field(ww_yavg) - Field(w_yavg) * Field(w_yavg)))

# # # Domain-integrated quantities (scalar time series)
# ∫KE = Integral(KE)

# # # Y-averages for spatial structure
# KE_yavg = Average(KE, dims=2)
# TKE_yavg = Average(TKE, dims=2)

# simulation.output_writers[:ke_yavg] = NetCDFWriter(
#     model,
#     (; KE_yavg, TKE_yavg,
#         ∫KE),
#     filename="KE_yavg_$(run_tag).nc",
#     schedule=TimeInterval(output_interval),
#     overwrite_existing=overwrite_existing)

# if checkpointing
#     checkpoint_prefix = "checkpoint_$(run_tag)"
#     simulation.output_writers[:checkpointer] = Checkpointer(model,
#         schedule=TimeInterval(5days),
#         prefix=checkpoint_prefix,
#         overwrite_existing=true,
#         cleanup=true)
# end

if gradient_IC
    @inline α_lin(y) = clamp(y / params.Ly, 0.0, 1.0)
    @inline blend(a, b, α) = (1 - α) * a + α * b
    @inline Tᵢ(x, y, z) = blend(T_south_pwl(z), T_north_pwl(z), α_lin(y))
    @inline Sᵢ(x, y, z) = blend(S_south_pwl(z), S_north_pwl(z), α_lin(y))
else
    @inline Tᵢ(x, y, z) = T_south_pwl(z)
    @inline Sᵢ(x, y, z) = S_south_pwl(z)
end

set!(model, v=(x, y, z) -> v∞(x, y, z, params), T=Tᵢ, S=Sᵢ)

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
 gradient_IC:     $(gradient_IC)
 sigmoid_v_bc:    $(sigmoid_v_bc)
 sigmoid_ic:      $(sigmoid_ic)
 is_coriolis:     $(is_coriolis)
 checkpointing:   $(checkpointing)
 shoal_bath:      $(shoal_bath)
 ════════════════════════════════════════════════════════
"""
run!(simulation, pickup=pickup)
