# ═══════════════════════════════════════════════════════════════════════════
# bounded_flow_over_shoals.jl
# ═══════════════════════════════════════════════════════════════════════════
# Main simulation script for flow over shoals.
# ═══════════════════════════════════════════════════════════════════════════

using Pkg
Pkg.instantiate()

using Oceananigans
using Oceananigans.Grids: Periodic, Bounded
using Oceananigans.Units
using Oceananigans.BoundaryConditions: NormalFlowBoundaryCondition, FieldBoundaryConditions
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

# ═══════════════════════════════════════════════════════════════════════════
# Bathymetry (shared with bounded_flow_over_shoals.jl)
# ═══════════════════════════════════════════════════════════════════════════
include(joinpath(@__DIR__, "dshoal_vn_param.jl"))

# ═══════════════════════════════════════════════════════════════════════════
# simulation knobs
# ═══════════════════════════════════════════════════════════════════════════
run_number = 100
sim_runtime = 20days
callback_interval = 86400seconds
run_tag = "bounded_shoals$(run_number)"

if LES
    params = (; Lx=100e3, Ly=200e3, Lz=50)
else
    params = (; Lx=100000, Ly=200000, Lz=50)
end
if arch == CPU()
    params = (; params..., Nx=50, Ny=50, Nz=10)
else
    params = (; params..., Nx=200, Ny=400, Nz=50)
end

x, y, z = (0, params.Lx), (0, params.Ly), (-params.Lz, 0)

# grid  
grid = RectilinearGrid(arch; size=(params.Nx, params.Ny, params.Nz), halo=(4, 4, 4), x, y, z, topology=(Bounded, Bounded, Bounded))

# model parameters
if shoal_bath
    slope_bottom = dshoal_param_bottom(params.Ly;
        Hs=15.0,
        shoal_length=40000.0,
        sigma=8000.0,
        shelf_depth=-25.0,
        shelf_break_end=12000.0)
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
    v₀ = 0.10
else
    v₀ = 0.0
end

# defaults
const T_north_v1 = 20.5389
const S_north_v1 = 32.6264
const T_south_v1 = 24.5378
const S_south_v1 = 35.5830

params = (; params...,
    v₀=v₀,
    Ls=20e3,
    Le=40e3,
    Lw=10e3,
    τ=1days,
    T_north_v1=T_north_v1,
    T_south_v1=T_south_v1,
    S_north_v1=S_north_v1,
    S_south_v1=S_south_v1,
    wind_stress=0.0)

# GPU-compatible SMOOTH piecewise linear T/S profiles (from CTD data)
@inline function T_north_pwl(z, v1=20.5389)
    v_bot = 14.3323
    return v1 + (v_bot - v1) * (z / -50.0)
end

@inline function T_south_pwl(z, v1=24.5378)
    v_bot = 23.4116
    return v1 + (v_bot - v1) * (z / -50.0)
end

@inline function S_north_pwl(z, v1=32.6264)
    v_bot = 33.2648
    return v1 + (v_bot - v1) * (z / -50.0)
end

@inline function S_south_pwl(z, v1=35.5830)
    v_bot = 36.1776
    return v1 + (v_bot - v1) * (z / -50.0)
end


params = (; params...)

#+++ Drag (Implemented as in https://doi.org/10.1029/2005WR004685)
z₀ = 2.5e-4 # roughness length
z₁ = Oceananigans.Grids.minimum_zspacing(grid, Center(), Center(), Center()) / 2
@info "Using z₁ =" z₁

const κᵛᵏ = 0.4 # von Karman constant
params = (; params..., c_dz=(κᵛᵏ / log(z₁ / z₀))^2) # quadratic drag coefficient
@info "Defining momentum BCs with Cᴰ =" params.c_dz

@inline τᵘ_drag(x, y, z, t, u, v, w, p) = -p.c_dz * u * √(u^2 + v^2 + w^2)
@inline τᵛ_drag(x, y, z, t, u, v, w, p) = -p.c_dz * v * √(u^2 + v^2 + w^2)
@inline τʷ_drag(x, y, z, t, u, v, w, p) = -p.c_dz * w * √(u^2 + v^2 + w^2)

immersed_drag_bc_u = FluxBoundaryCondition(τᵘ_drag, field_dependencies=(:u, :v, :w), parameters=params)
immersed_drag_bc_v = FluxBoundaryCondition(τᵛ_drag, field_dependencies=(:u, :v, :w), parameters=params)
immersed_drag_bc_w = FluxBoundaryCondition(τʷ_drag, field_dependencies=(:u, :v, :w), parameters=params)
#---
if LES
    @inline tsbc(x, z, t) = T_south_pwl(z, T_south_v1)
    @inline tnbc(x, z, t) = T_north_pwl(z, T_north_v1)
    @inline ssbc(x, z, t) = S_south_pwl(z, S_south_v1)
    @inline snbc(x, z, t) = S_north_pwl(z, S_north_v1)
end

# wind stress BC
ρ₀ = 1024.0
wind_bc_v = FluxBoundaryCondition(-0.0 / ρ₀)

@inline function sigmoidal_s2(x, Lx)
    xS = 65e3
    k2 = 20 / Lx
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

# new sponge masks using built-in functions from Oceananigans

const north_mask = PiecewiseLinearMask{:y}(center=params.Ly, width=params.Ls)
const south_mask = PiecewiseLinearMask{:y}(center=0, width=params.Ls)
const east_mask = PiecewiseLinearMask{:x}(center=params.Lx, width=params.Le)
const global_params = params
# We shift the mask evaluation by 15km so that the sponge layer ramps up 
# right after the shelf. This allows eddies to form physically over the shoal 
# but quickly damps anything that propagates offshore into the deep basin!
@inline offshore_mask_uvw(x, y, z) = 1.0 - sigmoidal_s2(x, global_params.Lx)

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

open_bc = NormalFlowBoundaryCondition(v∞; parameters=params, scheme=PerturbationAdvection())
open_zero = NormalFlowBoundaryCondition(0.0; scheme=PerturbationAdvection())
T_bcs = FieldBoundaryConditions(south=ValueBoundaryCondition(tsbc; scheme=PerturbationAdvection()), north=ValueBoundaryCondition(tnbc; scheme=PerturbationAdvection()))
S_bcs = FieldBoundaryConditions(south=ValueBoundaryCondition(ssbc; scheme=PerturbationAdvection()), north=ValueBoundaryCondition(snbc; scheme=PerturbationAdvection()))
u_bcs = FieldBoundaryConditions(immersed=immersed_drag_bc_u, north=open_zero)
v_bcs = FieldBoundaryConditions(immersed=immersed_drag_bc_v, north=open_bc, south=open_bc, top=wind_bc_v)
w_bcs = FieldBoundaryConditions(immersed=immersed_drag_bc_w, north=open_zero)

bcs = (u=u_bcs, v=v_bcs, w=w_bcs, T=T_bcs, S=S_bcs)
if is_coriolis
    coriolis = FPlane(latitude=35.2480)
else
    coriolis = nothing
end

reltol = sqrt(eps(grid))
abstol = sqrt(eps(grid))


model = NonhydrostaticModel(ib_grid;
    timestepper=:RungeKutta3,
    advection=WENO(order=5),
    closure=(HorizontalScalarDiffusivity(ν=1e-5, κ=1e-5), VerticalScalarDiffusivity(ν=1e-6, κ=1e-6)),
    hydrostatic_pressure_anomaly=CenterField(ib_grid),
    pressure_solver=ConjugateGradientPoissonSolver(ib_grid, reltol=reltol, abstol=abstol, maxiter=100),
    tracers=(:T, :S),
    buoyancy=SeawaterBuoyancy(),
    coriolis=coriolis,
    boundary_conditions=bcs,
    forcing=forcings
)

@info "" model

pickup = isfile("checkpoint_$(run_tag).jld2")
overwrite_existing = !pickup

simulation = Simulation(model, Δt=15minutes, stop_time=sim_runtime)
conjure_time_step_wizard!(simulation, cfl=0.8)

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
# simulation.output_writers[:midx_slice] = NetCDFWriter(model, slice_fields,
#     filename="midx_$(run_tag).nc",
#     schedule=TimeInterval(callback_interval),
#     indices=(round(Int, params.Nx / 5), :, :),
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
    simulation.output_writers[:checkpointer] = Checkpointer(model,
        schedule=TimeInterval(5days),
        prefix="checkpoint_$(run_tag)",
        overwrite_existing=true,
        cleanup=true)
end

# initial conditions
@info "Setting initial conditions"
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

# run simulation
@info """
════════════════════════════════════════════════════════
 SIMULATION CONFIGURATION: $(run_tag)
════════════════════════════════════════════════════════
 Run number:      $(run_number)
 Runtime:         $(sim_runtime)
 Architecture:    $(arch)

 ── Model Parameters ──
 Hs:              $(15.0) m
 shoal_length:    $(20000.0) m
 shelf_depth:     $(-25.0) m
 shelf_break_end: $(12000.0) m
 wind_stress:     $(0.0) N/m^2

 ── Switches ──
 LES:             $(LES)
 mass_flux:       $(mass_flux)
 gradient_IC:     $(gradient_IC)
 sigmoid_v_bc:    $(sigmoid_v_bc)
 sigmoid_ic:      $(sigmoid_ic)
 is_coriolis:     $(is_coriolis)
 shoal_bath:      $(shoal_bath)
════════════════════════════════════════════════════════
"""
run!(simulation, pickup=pickup)
