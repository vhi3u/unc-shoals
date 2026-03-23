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

# ═══════════════════════════════════════════════════════════════════════════
# Read sweep parameters from environment (set by sweep_driver.jl)
# Falls back to defaults so script can also be run standalone.
# ═══════════════════════════════════════════════════════════════════════════
sweep_Hs = parse(Float64, get(ENV, "SWEEP_Hs", "15.0"))
sweep_shoal_length = parse(Float64, get(ENV, "SWEEP_SHOAL_LENGTH", "20000.0"))
sweep_sigma = parse(Float64, get(ENV, "SWEEP_SIGMA", "8000.0"))
sweep_shelf_depth = parse(Float64, get(ENV, "SWEEP_SHELF_DEPTH", "-25.0"))
sweep_shelf_break_end = parse(Float64, get(ENV, "SWEEP_SHELF_BREAK_END", "12000.0"))
sweep_run_label = get(ENV, "SWEEP_RUN_LABEL", "standalone")
sweep_run_index = parse(Int, get(ENV, "SWEEP_RUN_INDEX", "0"))

@info "Sweep parameters: Hs=$sweep_Hs, shoal_length=$sweep_shoal_length, sigma=$sweep_sigma, shelf_depth=$sweep_shelf_depth, shelf_break_end=$sweep_shelf_break_end"

# build
@info "building domain"

# switches
LES = true
mass_flux = true
periodic_y = true
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

# ═══════════════════════════════════════════════════════════════════════════
# Bathymetry (shared with flow_over_shoals.jl)
# ═══════════════════════════════════════════════════════════════════════════
include(joinpath(@__DIR__, "dshoal_vn_param.jl"))

# ═══════════════════════════════════════════════════════════════════════════
# simulation knobs
# ═══════════════════════════════════════════════════════════════════════════
run_number = sweep_run_index
sim_runtime = 100days
callback_interval = 86400seconds
run_tag = "sweep_$(sweep_run_label)"

if LES
    params = (; Lx=100e3, Ly=300e3, Lz=50, Nx=30, Ny=30, Nz=10)
else
    params = (; Lx=100000, Ly=300000, Lz=50, Nx=30, Ny=30, Nz=10)
end
if arch == CPU()
    params = (; params..., Nx=30, Ny=60, Nz=10)
else
    params = (; params..., Nx=200, Ny=600, Nz=50)
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
        Hs=sweep_Hs,
        shoal_length=sweep_shoal_length,
        sigma=sweep_sigma,
        shelf_depth=sweep_shelf_depth,
        shelf_break_end=sweep_shelf_break_end)
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

# store parameters for sponge setup
params = (; params...,
    v₀=v₀,
    Ls=50e3,
    Le=60e3,
    τₙ=6hours,
    τₛ=6hours,
    τₑ=20days,
    τ_ts=6hours)

# GPU-compatible SMOOTH piecewise linear T/S profiles (from CTD data)
const δ_smooth = 2.5

@inline smooth_step(z, z0) = 0.5 * (1.0 - tanh((z - z0) / δ_smooth))

@inline function T_north_pwl(z)
    z1, z2, z3 = -5.0, -15.0, -35.0
    v1, v2, v3 = 20.5389, 17.8875, 14.3323
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
cᴰ = 2.5e-3
if LES
    @inline drag_u(x, y, z, t, u, v, p) = -p.cᴰ * √(u^2 + v^2) * u
    @inline drag_v(x, y, z, t, u, v, p) = -p.cᴰ * √(u^2 + v^2) * v
    drag_bc_u = FluxBoundaryCondition(drag_u, field_dependencies=(:u, :v), parameters=(; cᴰ=cᴰ,))
    drag_bc_v = FluxBoundaryCondition(drag_v, field_dependencies=(:u, :v), parameters=(; cᴰ=cᴰ,))
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

# sponge functions
if mass_flux
    if periodic_y
        @inline sponge_u(x, y, z, t, u, p) = -min(
            south_mask(x, y, z, p) * u / p.τₛ,
            north_mask(x, y, z, p) * u / p.τₙ,
            east_mask(x, y, z, p) * u / p.τₑ)

        @inline sponge_v(x, y, z, t, v, p) = -min(
            south_mask(x, y, z, p) * (v - v∞(x, z, t, p)) / p.τₛ,
            north_mask(x, y, z, p) * (v - v∞(x, z, t, p)) / p.τₙ,
            east_mask(x, y, z, p) * v / p.τₑ)

        @inline sponge_w(x, y, z, t, w, p) = -min(
            south_mask(x, y, z, p) * w / p.τₛ,
            north_mask(x, y, z, p) * w / p.τₙ,
            east_mask(x, y, z, p) * w / p.τₑ)

        @inline sponge_T(x, y, z, t, T, p) = -min(
            south_mask(x, y, z, p) * (T - T_south_pwl(z)) / p.τ_ts,
            north_mask(x, y, z, p) * (T - T_north_pwl(z)) / p.τ_ts,
            east_mask(x, y, z, p) * (T - p.Tₑ) / p.τ_ts)

        @inline sponge_S(x, y, z, t, S, p) = -min(
            south_mask(x, y, z, p) * (S - S_south_pwl(z)) / p.τ_ts,
            north_mask(x, y, z, p) * (S - S_north_pwl(z)) / p.τ_ts,
            east_mask(x, y, z, p) * (S - p.Sₑ) / p.τ_ts)
    else
        @inline sponge_u(x, y, z, t, u, p) = -(
            south_mask(x, y, z, p) * u / p.τₛ +
            north_mask(x, y, z, p) * u / p.τₙ +
            east_mask(x, y, z, p) * u / p.τₑ)

        @inline sponge_v(x, y, z, t, v, p) = -(
            south_mask(x, y, z, p) * (v - v∞(x, z, t, p)) / p.τₛ +
            north_mask(x, y, z, p) * (v - v∞(x, z, t, p)) / p.τₙ +
            east_mask(x, y, z, p) * v / p.τₑ)

        @inline sponge_w(x, y, z, t, w, p) = -(
            south_mask(x, y, z, p) * w / p.τₛ +
            north_mask(x, y, z, p) * w / p.τₙ +
            east_mask(x, y, z, p) * w / p.τₑ)

        @inline sponge_T(x, y, z, t, T, p) = -(
            south_mask(x, y, z, p) * (T - T_south_pwl(z)) / p.τ_ts +
            north_mask(x, y, z, p) * (T - T_north_pwl(z)) / p.τ_ts +
            east_mask(x, y, z, p) * (T - p.Tₑ) / p.τ_ts)

        @inline sponge_S(x, y, z, t, S, p) = -(
            south_mask(x, y, z, p) * (S - S_south_pwl(z)) / p.τ_ts +
            north_mask(x, y, z, p) * (S - S_north_pwl(z)) / p.τ_ts +
            east_mask(x, y, z, p) * (S - p.Sₑ) / p.τ_ts)
    end
end

# forcing functions
FT = Forcing(sponge_T, field_dependencies=:T, parameters=params)
FS = Forcing(sponge_S, field_dependencies=:S, parameters=params)
if mass_flux
    Fᵤ = Forcing(sponge_u, field_dependencies=:u, parameters=params)
    Fᵥ = Forcing(sponge_v, field_dependencies=:v, parameters=params)
    F_w = Forcing(sponge_w, field_dependencies=:w, parameters=params)
    forcings = (u=Fᵤ, v=Fᵥ, w=F_w, T=FT, S=FS)
else
    forcings = (T=FT, S=FS)
end

if periodic_y
    T_bcs = FieldBoundaryConditions()
    S_bcs = FieldBoundaryConditions()
    u_bcs = FieldBoundaryConditions(immersed=drag_bc_u)
    v_bcs = FieldBoundaryConditions(immersed=drag_bc_v)
    w_bcs = FieldBoundaryConditions()
else
    open_bc = OpenBoundaryCondition(v∞; parameters=params, scheme=PerturbationAdvection())
    open_zero = OpenBoundaryCondition(0.0)
    T_bcs = FieldBoundaryConditions(south=ValueBoundaryCondition(tsbc), north=ValueBoundaryCondition(tnbc))
    S_bcs = FieldBoundaryConditions(south=ValueBoundaryCondition(ssbc), north=ValueBoundaryCondition(snbc))
    u_bcs = FieldBoundaryConditions(immersed=drag_bc_u)
    v_bcs = FieldBoundaryConditions(immersed=drag_bc_v, north=open_bc, south=open_bc)
    w_bcs = FieldBoundaryConditions()
end

bcs = (u=u_bcs, v=v_bcs, w=w_bcs, T=T_bcs, S=S_bcs)
if is_coriolis
    coriolis = FPlane(latitude=35.2480)
else
    coriolis = nothing
end

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
        pressure_solver=ConjugateGradientPoissonSolver(ib_grid),
        tracers=(:T, :S),
        buoyancy=SeawaterBuoyancy(),
        coriolis=coriolis,
        boundary_conditions=bcs,
        forcing=forcings
    )
end

@info "" model

pickup = false
overwrite_existing = !pickup

simulation = Simulation(model, Δt=15minutes, stop_time=sim_runtime)
conjure_time_step_wizard!(simulation, cfl=0.9, diffusive_cfl=0.8)

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

# Mid-x YZ slice (along-shore transect at domain center)
simulation.output_writers[:midx_slice] = NetCDFWriter(model, slice_fields,
    filename="midx_$(run_tag).nc",
    schedule=TimeInterval(callback_interval),
    indices=(round(Int, params.Nx / 2), :, :),
    overwrite_existing=overwrite_existing)

# (2) 3D snapshots (every 20 days)
simulation.output_writers[:snapshots_3d] = NetCDFWriter(model, slice_fields,
    filename="snapshots_3d_$(run_tag).nc",
    schedule=TimeInterval(20days),
    overwrite_existing=overwrite_existing)

# (3) 3D Time Averages (10 day window)
simulation.output_writers[:time_avg_3d] = NetCDFWriter(model, tavg_fields,
    filename="time_avg_3d_$(run_tag).nc",
    schedule=AveragedTimeInterval(10days, window=10days),
    overwrite_existing=overwrite_existing)

# Domain-integrated KE time series
∫KE = Integral(KE)
simulation.output_writers[:ke] = NetCDFWriter(model, (; ∫KE),
    schedule=TimeInterval(callback_interval),
    filename="KE_$(run_tag).nc",
    overwrite_existing=overwrite_existing)

# ── Save sweep metadata to a small NetCDF file for postprocessing ──────
using NCDatasets
NCDatasets.Dataset("sweep_metadata_$(run_tag).nc", "c") do ds
    ds.attrib["run_label"] = sweep_run_label
    ds.attrib["run_index"] = sweep_run_index
    ds.attrib["Hs"] = sweep_Hs
    ds.attrib["shoal_length"] = sweep_shoal_length
    ds.attrib["shelf_depth"] = sweep_shelf_depth
    ds.attrib["shelf_break_end"] = sweep_shelf_break_end
end

# initial conditions
@info "Setting initial conditions"
uᵢ = 0.005 * rand(size(u)...)
vᵢ = 0.005 * rand(size(v)...)
wᵢ = 0.005 * rand(size(w)...)
uᵢ .-= mean(uᵢ)
vᵢ .-= mean(vᵢ)
wᵢ .-= mean(wᵢ)
uᵢ .+= 0
if sigmoid_ic
    xv, yv, zv = nodes(v, reshape=true)
    vᵢ .+= v∞.(xv, zv, 0, Ref(params))
else
    vᵢ .+= v₀
end

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
 SWEEP SIMULATION: $(run_tag)
════════════════════════════════════════════════════════
 Run label:       $(sweep_run_label)
 Run index:       $(sweep_run_index)
 Runtime:         $(sim_runtime)
 Architecture:    $(arch)

 ── Sweep Parameters ──
 Hs:              $(sweep_Hs) m
 shoal_length:    $(sweep_shoal_length) m
 shelf_depth:     $(sweep_shelf_depth) m
 shelf_break_end: $(sweep_shelf_break_end) m

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
