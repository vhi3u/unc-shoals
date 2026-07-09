# ═══════════════════════════════════════════════════════════════════════════
# flow_over_windy_shoals.jl
# ═══════════════════════════════════════════════════════════════════════════
# Periodic simulation of flow over shoals with wind forcing.
# ═══════════════════════════════════════════════════════════════════════════

using Oceananigans
using Oceananigans.Grids: Periodic, Bounded
using Oceananigans.Units
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, FluxBoundaryCondition
using Oceananigans.TurbulenceClosures
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using Oceananigans.Models: buoyancy_operation
using Oceananigans.OutputWriters
using Oceananigans.Forcings: Relaxation
using Statistics: mean
using Oceanostics: RossbyNumber, KineticEnergy
using Oceanostics.ProgressMessengers: TimedMessenger
using SeawaterPolynomials.TEOS10
using Printf: @sprintf
using NCDatasets
using DataFrames
using CUDA: has_cuda_gpu

@info "building domain"

if has_cuda_gpu()
    arch = GPU()
else
    arch = CPU()
end
@info "architecture = $(arch)"

# ═══════════════════════════════════════════════════════════════════════════
# Bathymetry
# ═══════════════════════════════════════════════════════════════════════════
include(joinpath(@__DIR__, "dshoal_vn_param.jl"))

# ═══════════════════════════════════════════════════════════════════════════
# Simulation knobs
# ═══════════════════════════════════════════════════════════════════════════
run_number = 7
sim_runtime = 25days
callback_interval = 86400seconds
run_tag = "periodic_windy_shoals_$(run_number)"

params = (; Lx=150e3, Ly=200e3, Lz=50)
if arch == CPU()
    params = (; params..., Nx=90, Ny=60, Nz=10)
else
    params = (; params..., Nx=300, Ny=400, Nz=50)
end

x, y, z = (0, params.Lx), (0, params.Ly), (-params.Lz, 0)

# grid  
grid = RectilinearGrid(arch; size=(params.Nx, params.Ny, params.Nz), halo=(4, 4, 4), x, y, z, topology=(Bounded, Periodic, Bounded))

# model parameters
slope_bottom = dshoal_param_bottom(params.Ly;
    Hs=15.0,
    shoal_length=40000.0,
    sigma=8000.0,
    shelf_depth=-25.0,
    shelf_break_end=12000.0)
GFB = GridFittedBottom(slope_bottom)
ib_grid = ImmersedBoundaryGrid(grid, GFB)

@info ib_grid

# ═══════════════════════════════════════════════════════════════════════════
# Stratification and Wind Setup
# ═══════════════════════════════════════════════════════════════════════════
params = (; params...,
    v₀=0.10,
    Ls=20e3,   # North nudging region width
    Le=50e3,   # East nudging region width
    τ=1days,
    T_south_v1=24.5378,
    S_south_v1=35.5830,
    wind_stress=0.05) # 0.05 N/m^2 northward

# GPU-compatible SMOOTH piecewise linear T/S profiles (from CTD data)
const δ_smooth = 2.5

@inline smooth_step(z, z0) = 0.5 * (1.0 - tanh((z - z0) / δ_smooth))

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

#+++ Drag (Implemented as in https://doi.org/10.1029/2005WR004685)
z₀ = 2.5e-4 # roughness length
z₁ = Oceananigans.Grids.minimum_zspacing(grid, Center(), Center(), Center()) / 2
@info "Using z₁ =" z₁

const κᵛᵏ = 0.4 # von Karman constant
params = (; params..., c_dz=(κᵛᵏ / log(z₁ / z₀))^2) # quadratic drag coefficient
@info "Defining momentum BCs with Cᴰ =" params.c_dz

bottom_drag = BulkDrag(coefficient=params.c_dz)
#---

# wind stress BC
ρ₀ = 1024.0
wind_bc_v = FluxBoundaryCondition(-params.wind_stress / ρ₀)

@inline function sigmoidal_s2(x, Lx)
    xS = 65e3
    k2 = 20 / Lx
    return 1 / (1 + exp(k2 * (x - xS)))
end

# velocity function
@inline function v∞(x, z, t, p)
    xC = 3e3
    k1 = 80 / p.Lx

    s1 = 1 / (1 + exp(-k1 * (x - xC)))
    s2 = sigmoidal_s2(x, p.Lx)
    s = (s1 - 1) + s2
    sc = clamp(s, 0.0, 1.0)
    return p.v₀ * sc
end

# new sponge masks using built-in functions from Oceananigans
const north_mask = PiecewiseLinearMask{:y}(center=params.Ly, width=params.Ls)
const east_mask = PiecewiseLinearMask{:x}(center=params.Lx, width=params.Le)
const global_params = params

@inline sponge_mask_TS(x, y, z) = min(north_mask(x, y, z), 1.0)
@inline sponge_mask_uvw(x, y, z) = min(north_mask(x, y, z) + east_mask(x, y, z), 1.0)

@inline function T_target(x, y, z, t)
    return T_south_pwl(z, global_params.T_south_v1)
end

@inline function S_target(x, y, z, t)
    return S_south_pwl(z, global_params.S_south_v1)
end

@inline function v_target(x, y, z, t)
    n = north_mask(x, y, z)
    e = east_mask(x, y, z)
    tot = n + e
    if tot > 0
        return (n * v∞(x, z, t, global_params)) / tot
    else
        return 0.0
    end
end

u_nudging = Relaxation(; rate=1 / global_params.τ, mask=sponge_mask_uvw, target=0.0)
v_nudging = Relaxation(; rate=1 / global_params.τ, mask=sponge_mask_uvw, target=v_target)
w_nudging = Relaxation(; rate=1 / global_params.τ, mask=sponge_mask_uvw, target=0.0)
T_nudging = Relaxation(; rate=1 / global_params.τ, mask=sponge_mask_TS, target=T_target)
S_nudging = Relaxation(; rate=1 / global_params.τ, mask=sponge_mask_TS, target=S_target)

forcings = (u=u_nudging, v=v_nudging, w=w_nudging, T=T_nudging, S=S_nudging)

T_bcs = FieldBoundaryConditions()
S_bcs = FieldBoundaryConditions()
u_bcs = FieldBoundaryConditions(bottom=bottom_drag, immersed=bottom_drag)
v_bcs = FieldBoundaryConditions(bottom=bottom_drag, immersed=bottom_drag, top=wind_bc_v)
w_bcs = FieldBoundaryConditions(immersed=bottom_drag)

bcs = (u=u_bcs, v=v_bcs, w=w_bcs, T=T_bcs, S=S_bcs)
coriolis = FPlane(latitude=35.2480)

reltol = sqrt(eps(grid))
abstol = sqrt(eps(grid))

model = NonhydrostaticModel(ib_grid;
    timestepper=:RungeKutta3,
    advection=WENO(order=5),
    closure=(HorizontalScalarDiffusivity(ν=1e-4, κ=1e-4), VerticalScalarDiffusivity(ν=1e-6, κ=1e-6)),
    hydrostatic_pressure_anomaly=CenterField(ib_grid),
    pressure_solver=ConjugateGradientPoissonSolver(ib_grid, reltol=reltol, abstol=abstol, maxiter=100),
    tracers=(:T, :S),
    buoyancy=SeawaterBuoyancy(),
    coriolis=coriolis,
    boundary_conditions=bcs,
    forcing=forcings
)

@info "" model

simulation = Simulation(model, Δt=15minutes, stop_time=sim_runtime)
conjure_time_step_wizard!(simulation, cfl=0.4)

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

u_c = @at (Center, Center, Center) u
v_c = @at (Center, Center, Center) v
w_c = @at (Center, Center, Center) w

slice_fields = (; u_c, v_c, w_c, T, S, Ro, KE)

# 2D snapshots (every 1 day)
simulation.output_writers[:surface_slice] = NetCDFWriter(model, slice_fields,
    filename="top_$(run_tag).nc",
    schedule=TimeInterval(callback_interval),
    indices=(:, :, params.Nz),
    overwrite_existing=true)

simulation.output_writers[:midy_slice] = NetCDFWriter(model, slice_fields,
    filename="midy_$(run_tag).nc",
    schedule=TimeInterval(callback_interval),
    indices=(:, round(Int, params.Ny / 2), :),
    overwrite_existing=true)

# -----------------------------------------------------------------------------
# Momentum Balance Diagnostics
# -----------------------------------------------------------------------------
V_tot = Integral(v)
U_tot = Integral(u)

Coriolis_v = Integral(-coriolis.f * u)

nudging_coeff = CenterField(ib_grid)
set!(nudging_coeff, (x, y, z) -> (-1 / global_params.τ) * sponge_mask_uvw(x, y, z))

target_v_field = CenterField(ib_grid)
set!(target_v_field, (x, y, z) -> v_target(x, y, z, 0.0))

Nudging_v_op = nudging_coeff * (v - target_v_field)
Nudging_v = Integral(Nudging_v_op)

balance_fields = (; V_tot, U_tot, Coriolis_v, Nudging_v)

simulation.output_writers[:momentum_balance] = NetCDFWriter(model, balance_fields,
    filename="momentum_balance_$(run_tag).nc",
    schedule=TimeInterval(callback_interval),
    overwrite_existing=true)


# initial conditions
@info "Setting initial conditions"
@inline Tᵢ(x, y, z) = T_south_pwl(z, params.T_south_v1)
@inline Sᵢ(x, y, z) = S_south_pwl(z, params.S_south_v1)
@inline v_init(x, y, z) = v∞(x, z, 0, params)

set!(model, u=0.0, v=0.0, w=0.0, T=Tᵢ, S=Sᵢ)

# run simulation
@info """
════════════════════════════════════════════════════════
 SIMULATION CONFIGURATION: $(run_tag)
════════════════════════════════════════════════════════
 Run number:      $(run_number)
 Runtime:         $(sim_runtime)
 Architecture:    $(arch)
 Lx, Ly:          $(params.Lx), $(params.Ly)
 Nudging (N/E):   $(params.Ls), $(params.Le)
 Wind stress:     $(params.wind_stress) N/m²
════════════════════════════════════════════════════════
"""
run!(simulation)
