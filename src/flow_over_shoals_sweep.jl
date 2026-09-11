# ═══════════════════════════════════════════════════════════════════════════
# flow_over_shoals_sweep.jl
# ═══════════════════════════════════════════════════════════════════════════
# Modified version of flow_over_shoals.jl for parameter sweeps.
# Reads sweep parameters from environment variables set by sweep_driver.jl:
#   SWEEP_Hs, SWEEP_SHOAL_LENGTH, SWEEP_SHELF_DEPTH, SWEEP_SHELF_BREAK_END,
#   SWEEP_RUN_LABEL, SWEEP_RUN_INDEX
# ═══════════════════════════════════════════════════════════════════════════

using Oceananigans
using Oceananigans.Grids: Periodic, Bounded
using Oceananigans.Units
using Oceananigans.BoundaryConditions: FieldBoundaryConditions
using Oceananigans.TurbulenceClosures
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

# ═══════════════════════════════════════════════════════════════════════════
# Read sweep parameters from environment (set by sweep_driver.jl)
# Falls back to defaults so script can also be run standalone.
# ═══════════════════════════════════════════════════════════════════════════
sweep_Zs = parse(Float64, get(ENV, "SWEEP_Zs", "-5.0"))
sweep_shoal_length = parse(Float64, get(ENV, "SWEEP_SHOAL_LENGTH", "40000.0"))
sweep_sigma = parse(Float64, get(ENV, "SWEEP_SIGMA", "8000.0"))
sweep_Zsh = parse(Float64, get(ENV, "SWEEP_Zsh", "-25.0"))
sweep_shelf_break_end = parse(Float64, get(ENV, "SWEEP_SHELF_BREAK_END", "12000.0"))
sweep_run_label = get(ENV, "SWEEP_RUN_LABEL", "standalone")
sweep_run_index = parse(Int, get(ENV, "SWEEP_RUN_INDEX", "0"))
sweep_wind_stress = parse(Float64, get(ENV, "SWEEP_WIND_STRESS", "0.0"))
sweep_v0 = parse(Float64, get(ENV, "SWEEP_V0", "0.2"))

@info "Sweep parameters: Zs=$sweep_Zs, shoal_length=$sweep_shoal_length, sigma=$sweep_sigma, Zsh=$sweep_Zsh, shelf_break_end=$sweep_shelf_break_end, wind_stress=$sweep_wind_stress, v0=$sweep_v0"

# build
@info "building domain"

# switches
LES = true
mass_flux = true
periodic_y = true
gradient_IC = false
sigmoid_v_bc = true
sigmoid_ic = true
sigmoid_wind = true          # taper wind to zero across the east sponge
is_coriolis = true
checkpointing = false
shoal_bath = true
ramp_wind = true             # ramp τ over ~2 inertial periods

# Vertical mixing closure. :catke is the recommended production choice;
# :ribased and :constant exist so the wind run can be repeated against the
# closure used in shoals38/39 without changing anything else.
closure_choice = :catke      # :catke | :ribased | :constant
if has_cuda_gpu()
    arch = GPU()
else
    arch = CPU()
end
@info "architecture = $(arch)"

# ═══════════════════════════════════════════════════════════════════════════
# Bathymetry (shared with flow_over_shoals.jl)
# ═══════════════════════════════════════════════════════════════════════════
include(joinpath(@__DIR__, "dshoal_vn_param.jl"))

# ═══════════════════════════════════════════════════════════════════════════
# simulation knobs
# ═══════════════════════════════════════════════════════════════════════════
run_number = sweep_run_index
sim_runtime = 50days
callback_interval = 86400seconds
run_tag = "sweep_$(sweep_run_label)"

if LES
    params = (; Lx=150e3, Ly=200e3, Lz=50)
else
    params = (; Lx=150000, Ly=200000, Lz=50)
end
if arch == CPU()
    params = (; params..., Nx=60, Ny=60, Nz=10, νh=1.0, κh=1.0)
else
    params = (; params..., Nx=300, Ny=400, Nz=50, νh=1e-5, κh=1e-5)
end

x, y, z = (0, params.Lx), (0, params.Ly), (-params.Lz, 0)

# grid  
if periodic_y
    grid = RectilinearGrid(arch; size=(params.Nx, params.Ny, params.Nz), halo=(4, 4, 4), x, y, z, topology=(Bounded, Periodic, Bounded))
else
    grid = RectilinearGrid(arch; size=(params.Nx, params.Ny, params.Nz), halo=(4, 4, 4), x, y, z, topology=(Bounded, Bounded, Bounded))
end

# model parameters
if shoal_bath
    slope_bottom = dshoal_param_bottom(params.Ly;
        Zs=sweep_Zs,
        shoal_length=sweep_shoal_length,
        sigma=sweep_sigma,
        Zsh=sweep_Zsh,
        shelf_break_end=sweep_shelf_break_end)
    GFB = GridFittedBottom(slope_bottom)
    ib_grid = ImmersedBoundaryGrid(grid, GFB)
else
    ib_grid = grid
end

@info ib_grid

# ═══════════════════════════════════════════════════════════════════════════
# Stratification and Wind Setup
# ═══════════════════════════════════════════════════════════════════════════
if mass_flux
    v₀ = sweep_v0
else
    v₀ = 0.0
end

v₀ = abs(v₀)

# defaults
T_north_v1, S_north_v1 = 20.5389, 32.6264
T_south_v1, S_south_v1 = 24.5378, 35.5830

params = (; params...,
    v₀=v₀,
    Ls=20e3,
    Le=50e3,
    Lw=10e3,
    τ=6hours,
    T_north_v1=T_north_v1,
    T_south_v1=T_south_v1,
    S_north_v1=S_north_v1,
    S_south_v1=S_south_v1,
    wind_stress=sweep_wind_stress)

# GPU-compatible SMOOTH piecewise linear T/S profiles (from CTD data)
const δ_smooth = 2.5

@inline smooth_step(z, z0) = 0.5 * (1.0 - tanh((z - z0) / δ_smooth))

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

#+++ Drag 
z₀ = 2.5e-4 # roughness length
z₁ = Oceananigans.Grids.minimum_zspacing(grid, Center(), Center(), Center()) / 2
@info "Using z₁ =" z₁

const κᵛᵏ = 0.4 # von Karman constant
c_dz = (κᵛᵏ / log(z₁ / z₀))^2 # quadratic drag coefficient
@info "Defining momentum BCs with Cᴰ =" c_dz
drag = BulkDrag(coefficient=c_dz)
#---
if LES
    @inline tsbc(x, z, t) = T_south_pwl(z, T_south_v1)
    @inline tnbc(x, z, t) = T_north_pwl(z, T_north_v1)
    @inline ssbc(x, z, t) = S_south_pwl(z, S_south_v1)
    @inline snbc(x, z, t) = S_north_pwl(z, S_north_v1)
end

# ═══════════════════════════════════════════════════════════════════════════
# Wind stress
# ═══════════════════════════════════════════════════════════════════════════
ρ₀ = 1024.0
const f₀ = 2 * 7.292115e-5 * sind(35.2480)          # 8.42e-5 s⁻¹
const inertial_period = 2π / f₀                     # 20.74 h
const T_ramp = ramp_wind ? 2 * inertial_period : 0.0
const τ_kinematic = sweep_wind_stress / ρ₀         # m² s⁻²
const Lx_c = params.Lx
const Le_c = params.Le

@info @sprintf("f = %.3e s⁻¹, inertial period = %.2f h, wind ramp = %.1f h",
    f₀, inertial_period / 3600, T_ramp / 3600)

# Offshore taper: the east sponge relaxes v → 0 over the outer Le = 50 km. If
# the wind keeps accelerating v there, the sponge and the forcing fight each
# other permanently. Taper the stress to zero across that same band so the wind
# is uniform over the shelf and slope (0–100 km) and absent where the sponge
# takes over. Set sigmoid_wind = false for a genuinely uniform wind.
@inline wind_shape(x) = 0.5 * (1 - tanh((x - (Lx_c - Le_c)) / (0.25 * Le_c)))
@inline wind_ramp(t) = ifelse(T_ramp > 0, 1 - exp(-t / T_ramp), one(t))

if sigmoid_wind
    @inline wind_v_flux(x, y, t) = -τ_kinematic * wind_ramp(t) * wind_shape(x)
else
    @inline wind_v_flux(x, y, t) = -τ_kinematic * wind_ramp(t)
end

wind_bc_u = FluxBoundaryCondition(0.0)
wind_bc_v = FluxBoundaryCondition(wind_v_flux)

@inline function sigmoidal_s2(x, Lx)
    xS = 65e3
    k2 = 40 / Lx
    return 1 / (1 + exp(k2 * (x - xS)))
end

# velocity function
if sigmoid_v_bc
    @inline function v∞(x, z, t, p)
        xC = 3e3
        k1 = 80 / p.Lx

        s1 = 1 / (1 + exp(-k1 * (x - xC)))
        s2 = sigmoidal_s2(x, p.Lx)
        s = (s1 - 1) + s2
        sc = clamp(s, 0.0, 1.0)
        return p.v₀ * sc
    end
else
    @inline function v∞(x, z, t, p)
        return p.v₀
    end
end

# built-in masks
const south_mask = PiecewiseLinearMask{:y}(center=0.0, width=params.Ls)
const north_mask = PiecewiseLinearMask{:y}(center=params.Ly, width=params.Ls)
const east_mask = PiecewiseLinearMask{:x}(center=params.Lx, width=params.Le)

# targets
const global_params = params
@inline v_target_inflow(x, y, z, t) = v∞(x, z, t, global_params)

@inline T_target_south(x, y, z, t) = T_south_pwl(z, global_params.T_south_v1)
@inline S_target_south(x, y, z, t) = S_south_pwl(z, global_params.S_south_v1)

@inline T_target_north(x, y, z, t) = T_north_pwl(z, global_params.T_north_v1)
@inline S_target_north(x, y, z, t) = S_north_pwl(z, global_params.S_north_v1)

const T_target = T_target_south
const S_target = S_target_south
const inflow_mask = south_mask

# forcing functions — note there is no w sponge: w is diagnosed from continuity
# in the hydrostatic model and cannot (and should not) be relaxed.
if periodic_y
    u_sponge_inflow = Relaxation(; rate=1 / global_params.τ, mask=inflow_mask, target=0.0)
    u_sponge_e = Relaxation(; rate=1 / global_params.τ, mask=east_mask, target=0.0)

    v_sponge_inflow = Relaxation(; rate=1 / global_params.τ, mask=inflow_mask, target=v_target_inflow)
    v_sponge_e = Relaxation(; rate=1 / global_params.τ, mask=east_mask, target=0.0)

    T_sponge_inflow = Relaxation(; rate=1 / global_params.τ, mask=inflow_mask, target=T_target)
    T_sponge_e = Relaxation(; rate=1 / global_params.τ, mask=east_mask, target=T_target)

    S_sponge_inflow = Relaxation(; rate=1 / global_params.τ, mask=inflow_mask, target=S_target)
    S_sponge_e = Relaxation(; rate=1 / global_params.τ, mask=east_mask, target=S_target)

    if mass_flux
        forcings = (u=(u_sponge_inflow, u_sponge_e),
            v=(v_sponge_inflow, v_sponge_e),
            T=(T_sponge_inflow, T_sponge_e),
            S=(S_sponge_inflow, S_sponge_e))
    else
        forcings = (T=(T_sponge_inflow, T_sponge_e), S=(S_sponge_inflow, S_sponge_e))
    end
end

if periodic_y
    T_bcs = FieldBoundaryConditions()
    S_bcs = FieldBoundaryConditions()
    u_bcs = FieldBoundaryConditions(immersed=drag, bottom=drag, top=wind_bc_u)
    v_bcs = FieldBoundaryConditions(immersed=drag, bottom=drag, top=wind_bc_v)
else
    northern_bc = NormalFlowBoundaryCondition(v∞; parameters=params, scheme=PerturbationAdvection(inflow_timescale=0.0, outflow_timescale=0.0))
    southern_bc = NormalFlowBoundaryCondition(v∞; parameters=params, scheme=PerturbationAdvection(inflow_timescale=0.0, outflow_timescale=0.0))
    eastern_bc = NormalFlowBoundaryCondition(0.0; scheme=PerturbationAdvection(inflow_timescale=0.0, outflow_timescale=Inf))

    T_bcs = FieldBoundaryConditions(south=ValueBoundaryCondition(tsbc), north=ValueBoundaryCondition(tnbc))
    S_bcs = FieldBoundaryConditions(south=ValueBoundaryCondition(ssbc), north=ValueBoundaryCondition(snbc))

    u_bcs = FieldBoundaryConditions(immersed=drag, bottom=drag, south=ValueBoundaryCondition(0.0; scheme=PerturbationAdvection(inflow_timescale=Inf, outflow_timescale=0.0)), east=eastern_bc, top=wind_bc_u)
    v_bcs = FieldBoundaryConditions(immersed=drag, bottom=drag, north=northern_bc, south=southern_bc, east=ValueBoundaryCondition(0.0), top=wind_bc_v)
end

bcs = (u=u_bcs, v=v_bcs, T=T_bcs, S=S_bcs)
@info "Boundary Conditions:" bcs

if is_coriolis
    coriolis = FPlane(latitude=35.2480)
else
    coriolis = nothing
end

# ═══════════════════════════════════════════════════════════════════════════
# Turbulence closure
# ═══════════════════════════════════════════════════════════════════════════
if closure_choice === :catke
    horizontal_closure = HorizontalScalarDiffusivity(ν=1e-4, κ=1e-4)
    vertical_closure = CATKEVerticalDiffusivity()
    tracers = (:T, :S)
elseif closure_choice === :ribased
    horizontal_closure = HorizontalScalarDiffusivity(ν=1e-3, κ=1e-3)
    vertical_closure = RiBasedVerticalDiffusivity()
    tracers = (:T, :S)
elseif closure_choice === :constant
    horizontal_closure = HorizontalScalarDiffusivity(ν=1e-3, κ=1e-3)
    vertical_closure = VerticalScalarDiffusivity(ν=1e-4, κ=1e-5)
    tracers = (:T, :S)
else
    error("closure_choice must be :catke, :ribased or :constant; got $(closure_choice)")
end

closure = (horizontal_closure, vertical_closure)
@info "Closure ($(closure_choice)):" closure

# ═══════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════
model = HydrostaticFreeSurfaceModel(ib_grid;
    momentum_advection=WENO(order=5),
    tracer_advection=WENO(order=5),
    free_surface=SplitExplicitFreeSurface(ib_grid; cfl=0.7),
    tracers=tracers,
    buoyancy=SeawaterBuoyancy(),
    coriolis=coriolis,
    closure=closure,
    boundary_conditions=bcs,
    forcing=forcings
)

@info "" model

# Check for environment variable SWEEP_PICKUP or local checkpoint files
sweep_pickup_env = get(ENV, "SWEEP_PICKUP", "")
if !isempty(sweep_pickup_env) && isfile(sweep_pickup_env)
    pickup = sweep_pickup_env
    sim_runtime = 100days
    @info "Picking up from specified checkpoint: $(pickup). Setting runtime to 100 days."
else
    checkpoint_files = filter(f -> startswith(f, "checkpoint_") && endswith(f, ".jld2"), readdir("."))
    if !isempty(checkpoint_files)
        pickup = last(sort(checkpoint_files))
        sim_runtime = 100days
        @info "Found local checkpoint: $(pickup). Setting runtime to 100 days."
    else
        pickup = false
        sim_runtime = 50days
        @info "No checkpoint found. Running 50-day run."
    end
end
overwrite_existing = (pickup === false)

simulation = Simulation(model, Δt=5minutes, stop_time=sim_runtime)
conjure_time_step_wizard!(simulation, cfl=0.7)

progress = TimedMessenger()
simulation.callbacks[:progress] = Callback(progress, TimeInterval(callback_interval))

u, v, w = model.velocities
T = model.tracers.T
S = model.tracers.S
Ro = @at (Center, Center, Center) RossbyNumber(model)
KE = @at (Center, Center, Center) KineticEnergy(model)
b_op = Oceananigans.Models.buoyancy_operation(model)
PV = @at (Center, Center, Center) ErtelPotentialVorticity(model, u, v, w, b_op, model.coriolis)

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

slice_fields = (; u_c, v_c, w_c, T, S, Ro, KE, PV)
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

# Mid-x YZ slice (along-shore transect at domain center)
simulation.output_writers[:midx_slice] = NetCDFWriter(model, slice_fields,
    filename="midx_$(run_tag).nc",
    schedule=TimeInterval(callback_interval),
    indices=(round(Int, params.Nx / 5), :, :),
    overwrite_existing=overwrite_existing)

# (3) 3D Time Averages (10 day window)
simulation.output_writers[:time_avg_3d] = NetCDFWriter(model, tavg_fields,
    filename="time_avg_3d_$(run_tag).nc",
    schedule=AveragedTimeInterval(10days, window=10days),
    overwrite_existing=overwrite_existing)

# ── Save sweep metadata to a small NetCDF file for postprocessing ──────
using NCDatasets
NCDatasets.Dataset("sweep_metadata_$(run_tag).nc", "c") do ds
    ds.attrib["run_label"] = sweep_run_label
    ds.attrib["run_index"] = sweep_run_index
    ds.attrib["Zs"] = sweep_Zs
    ds.attrib["shoal_length"] = sweep_shoal_length
    ds.attrib["Zsh"] = sweep_Zsh
    ds.attrib["shelf_break_end"] = sweep_shelf_break_end
    ds.attrib["wind_stress"] = sweep_wind_stress
    ds.attrib["v0"] = sweep_v0
end

# initial conditions
if pickup === false
    @info "Setting initial conditions"
    if sigmoid_ic
        v_init = (x, y, z) -> v∞(x, z, 0, params)
    else
        v_init = v₀
    end

    if gradient_IC
        @inline α_lin(y) = clamp(y / global_params.Ly, 0.0, 1.0)
        @inline blend(a, b, α) = (1 - α) * a + α * b
        @inline Tᵢ(x, y, z) = blend(T_south_pwl(z, global_params.T_south_v1), T_north_pwl(z, global_params.T_north_v1), α_lin(y))
        @inline Sᵢ(x, y, z) = blend(S_south_pwl(z, global_params.S_south_v1), S_north_pwl(z, global_params.S_north_v1), α_lin(y))
    else
        @inline Tᵢ(x, y, z) = T_south_pwl(z, global_params.T_south_v1)
        @inline Sᵢ(x, y, z) = S_south_pwl(z, global_params.S_south_v1)
    end

    if closure_choice === :catke
        set!(model, u=0.0, v=v_init, T=Tᵢ, S=Sᵢ, e=1e-6)
    else
        set!(model, u=0.0, v=v_init, T=Tᵢ, S=Sᵢ)
    end
else
    @info "Skipping initial conditions setup — loading fields from checkpoint $(pickup)."
end

# run simulation
@info """
════════════════════════════════════════════════════════
 SWEEP SIMULATION: $(run_tag)
════════════════════════════════════════════════════════
 Run label:       $(sweep_run_label)
 Run index:       $(sweep_run_index)
 Runtime:         $(sim_runtime)
 Architecture:    $(arch)

 ── Sweep Parameters ──
 Zs (shoal_depth):$(sweep_Zs) m
 shoal_length:    $(sweep_shoal_length) m
 Zsh (shelf_depth):$(sweep_Zsh) m
 shelf_break_end: $(sweep_shelf_break_end) m
 wind_stress:     $(sweep_wind_stress) N/m^2
 v0:              $(sweep_v0) m/s

 ── Switches ──
 LES:             $(LES)
 mass_flux:       $(mass_flux)
 periodic_y:      $(periodic_y)
 gradient_IC:     $(gradient_IC)
 sigmoid_v_bc:    $(sigmoid_v_bc)
 sigmoid_ic:      $(sigmoid_ic)
 is_coriolis:     $(is_coriolis)
 shoal_bath:      $(shoal_bath)
════════════════════════════════════════════════════════
"""
run!(simulation, pickup=pickup)
