# ═══════════════════════════════════════════════════════════════════════════
# flow_over_shoals_conv.jl
# ═══════════════════════════════════════════════════════════════════════════
# Convergent flow version of flow_over_shoals.jl
#   - Bounded in Y (not periodic)
#   - v = +0.1 m/s at south boundary (northward)
#   - v = -0.1 m/s at north boundary (southward)  → convergent
#   - Eastern boundary is open (zero-gradient outflow)
#   - Uses PerturbationAdvection for all open boundaries
#   - Sponge layers at N, S, and E boundaries nudge toward target profiles
# ═══════════════════════════════════════════════════════════════════════════

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
@info "building convergent-flow domain"

# switches
LES = true
mass_flux = true
periodic_y = false      # ← BOUNDED for convergent flow
gradient_IC = true      # blend T/S from south to north
sigmoid_v_bc = true     # cross-shelf sigmoid profile on each boundary
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
run_number = 1
sim_runtime = 100days
callback_interval = 86400seconds
run_tag = "conv_shoals$(run_number)"

if LES
    params = (; Lx=100e3, Ly=100e3, Lz=50, Nx=30, Ny=30, Nz=10)
else
    params = (; Lx=100000, Ly=100000, Lz=50, Nx=30, Ny=30, Nz=10)
end
if arch == CPU()
    params = (; params..., Nx=50, Ny=50, Nz=10)
else
    params = (; params..., Nx=200, Ny=200, Nz=50)
end

x, y, z = (0, params.Lx), (0, params.Ly), (-params.Lz, 0)

# grid — BOUNDED in Y
grid = RectilinearGrid(arch; size=(params.Nx, params.Ny, params.Nz),
    halo=(4, 4, 4), x, y, z,
    topology=(Bounded, Bounded, Bounded))

# ═══════════════════════════════════════════════════════════════════════════
# Bathymetry
# ═══════════════════════════════════════════════════════════════════════════
if shoal_bath
    Hs = 15.0
    sigma = 8e3
    shoal_length = 20e3
    slope_bottom = dshoal_param_bottom(params.Ly; Hs=Hs, sigma=sigma, shoal_length=shoal_length)
    GFB = GridFittedBottom(slope_bottom)
    ib_grid = ImmersedBoundaryGrid(grid, GFB)
else
    ib_grid = grid
end

@info ib_grid

# ═══════════════════════════════════════════════════════════════════════════
# Convergent velocity magnitude
# ═══════════════════════════════════════════════════════════════════════════
v₀ = 0.1   # magnitude of along-shore inflow at each boundary

params = (; params...,
    v₀=v₀,
    Ls=10e3,        # sponge width at south
    Ln=10e3,        # sponge width at north
    Le=40e3,        # sponge width at east
    τₙ=6hours,      # nudging timescale north
    τₛ=6hours,      # nudging timescale south (symmetric with north)
    τₑ=24hours,     # nudging timescale east
    τ_ts=24hours)   # tracer nudging timescale

# ═══════════════════════════════════════════════════════════════════════════
# T/S profiles (same as parent script)
# ═══════════════════════════════════════════════════════════════════════════
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

# Salinity at East boundary (Offshore) - STRATIFIED
@inline function S_east_pwl(z)
    z1, z2, z3 = -5.0, -25.0, -45.0
    v1, v2, v3 = 36.0, 35.9, 35.8
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

params = (; params...)

# ═══════════════════════════════════════════════════════════════════════════
# Bottom drag
# ═══════════════════════════════════════════════════════════════════════════
cᴰ = 2.5e-3
@inline drag_u(x, y, z, t, u, v, p) = -p.cᴰ * √(u^2 + v^2) * u
@inline drag_v(x, y, z, t, u, v, p) = -p.cᴰ * √(u^2 + v^2) * v
drag_bc_u = FluxBoundaryCondition(drag_u, field_dependencies=(:u, :v), parameters=(; cᴰ=cᴰ,))
drag_bc_v = FluxBoundaryCondition(drag_v, field_dependencies=(:u, :v), parameters=(; cᴰ=cᴰ,))

# T/S boundary condition helpers (Dirichlet at N and S)
# Using B2 (south) profiles at BOTH boundaries → no density-driven geostrophic flow
@inline tsbc(x, z, t) = T_south_pwl(z)
@inline tnbc(x, z, t) = T_north_pwl(z)   # ← same as south (uniform density)
@inline ssbc(x, z, t) = S_south_pwl(z)
@inline snbc(x, z, t) = S_north_pwl(z)   # ← same as south (uniform density)

# No wind stress for convergent experiment
ρ₀ = 1025.0
τ_wind = 0.0
wind_bc_v = FluxBoundaryCondition(-τ_wind / ρ₀)

# ═══════════════════════════════════════════════════════════════════════════
# Convergent velocity targets
# ═══════════════════════════════════════════════════════════════════════════
# South boundary: +v₀ (northward), North boundary: -v₀ (southward)
# Both use the same cross-shelf sigmoid profile to confine flow to the
# shelf region (0–60 km) and taper to zero offshore.
# ═══════════════════════════════════════════════════════════════════════════

if sigmoid_v_bc
    # South boundary target: +v₀ over the shelf
    @inline function v_south(x, z, t, p)
        xC = 3e3
        xS = 60e3
        Lw = p.Lx
        k1 = 80 / Lw
        k2 = 40 / Lw
        s1 = 1 / (1 + exp(-k1 * (x - xC)))
        s2 = 1 / (1 + exp(k2 * (x - xS)))
        s = (s1 - 1) + s2
        sc = clamp(s, 0.0, 1.0)
        return p.v₀ * sc          # positive = northward
    end

    # North boundary target: -v₀ over the shelf
    @inline function v_north(x, z, t, p)
        xC = 3e3
        xS = 60e3
        Lw = p.Lx
        k1 = 80 / Lw
        k2 = 40 / Lw
        s1 = 1 / (1 + exp(-k1 * (x - xC)))
        s2 = 1 / (1 + exp(k2 * (x - xS)))
        s = (s1 - 1) + s2
        sc = clamp(s, 0.0, 1.0)
        return -p.v₀ * sc         # negative = southward
    end
else
    @inline v_south(x, z, t, p) = p.v₀
    @inline v_north(x, z, t, p) = -p.v₀
end

# ═══════════════════════════════════════════════════════════════════════════
# Mask functions for sponge layers
# ═══════════════════════════════════════════════════════════════════════════

@inline smooth_ramp(dt) = sin(0.5 * π * clamp(dt, 0.0, 1.0))^2

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
    y0 = p.Ly - p.Ln
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
    return smooth_ramp((x - x0) / (x1 - x0))
end

# Offshore mask for tracer nudging (shelf break → eastern boundary)
@inline function offshore_mask(x, y, z, p)
    x0 = 60e3
    x1 = p.Lx
    if x <= x0
        return 0.0
    elseif x >= x1
        return 1.0
    else
        return (x - x0) / (x1 - x0)
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# Sponge / nudging functions  (Bounded, convergent)
# ═══════════════════════════════════════════════════════════════════════════
# Velocity:
#   South sponge → nudge v toward v_south (northward)
#   North sponge → nudge v toward v_north (southward)
#   East  sponge → nudge u,v,w toward 0 (let mass escape freely)
# Tracers:
#   South sponge → nudge T,S toward south profiles
#   North sponge → nudge T,S toward north profiles
#   Offshore     → nudge T,S toward east profiles (gentle)
# ═══════════════════════════════════════════════════════════════════════════

@inline sponge_u(x, y, z, t, u, p) = -(
    south_mask(x, y, z, p) * u / p.τₛ +
    north_mask(x, y, z, p) * u / p.τₙ +
    east_mask(x, y, z, p) * u / p.τₑ)

@inline sponge_v(x, y, z, t, v, p) = -(
    south_mask(x, y, z, p) * (v - v_south(x, z, t, p)) / p.τₛ +
    north_mask(x, y, z, p) * (v - v_north(x, z, t, p)) / p.τₙ +
    east_mask(x, y, z, p) * v / p.τₑ)

@inline sponge_w(x, y, z, t, w, p) = -(
    south_mask(x, y, z, p) * w / p.τₛ +
    north_mask(x, y, z, p) * w / p.τₙ +
    east_mask(x, y, z, p) * w / p.τₑ)

@inline sponge_T(x, y, z, t, T, p) = -(
    south_mask(x, y, z, p) * (T - T_south_pwl(z)) / p.τ_ts +
    north_mask(x, y, z, p) * (T - T_south_pwl(z)) / p.τ_ts +
    offshore_mask(x, y, z, p) * (T - T_east_pwl(z)) / (5 * p.τ_ts))

@inline sponge_S(x, y, z, t, S, p) = -(
    south_mask(x, y, z, p) * (S - S_south_pwl(z)) / p.τ_ts +
    north_mask(x, y, z, p) * (S - S_south_pwl(z)) / p.τ_ts +
    offshore_mask(x, y, z, p) * (S - S_east_pwl(z)) / (5 * p.τ_ts))

# ═══════════════════════════════════════════════════════════════════════════
# Forcings
# ═══════════════════════════════════════════════════════════════════════════
FT = Forcing(sponge_T, field_dependencies=:T, parameters=params)
FS = Forcing(sponge_S, field_dependencies=:S, parameters=params)
Fᵤ = Forcing(sponge_u, field_dependencies=:u, parameters=params)
Fᵥ = Forcing(sponge_v, field_dependencies=:v, parameters=params)
F_w = Forcing(sponge_w, field_dependencies=:w, parameters=params)
forcings = (u=Fᵤ, v=Fᵥ, w=F_w, T=FT, S=FS)

# ═══════════════════════════════════════════════════════════════════════════
# Boundary conditions — BOUNDED + CONVERGENT INFLOW
# ═══════════════════════════════════════════════════════════════════════════
# v: OpenBoundaryCondition at N and S with PerturbationAdvection
#    (ValueBoundaryCondition is not allowed on Face fields at Bounded boundaries)
#    v_south = +v₀ (northward), v_north = -v₀ (southward) → convergent
# u: open at east (PerturbationAdvection, zero target — outflow for excess mass)
# T, S: Dirichlet at N and S

open_bc_south = OpenBoundaryCondition(v_south; parameters=params, scheme=PerturbationAdvection())
open_bc_north = OpenBoundaryCondition(v_north; parameters=params, scheme=PerturbationAdvection())
open_bc_east_zero = OpenBoundaryCondition(0.0; scheme=PerturbationAdvection())

T_bcs = FieldBoundaryConditions(south=ValueBoundaryCondition(tsbc),
    north=ValueBoundaryCondition(tnbc))
S_bcs = FieldBoundaryConditions(south=ValueBoundaryCondition(ssbc),
    north=ValueBoundaryCondition(snbc))
u_bcs = FieldBoundaryConditions(immersed=drag_bc_u,
    east=open_bc_east_zero)
v_bcs = FieldBoundaryConditions(immersed=drag_bc_v,
    south=open_bc_south,
    north=open_bc_north,
    top=wind_bc_v)
w_bcs = FieldBoundaryConditions()

bcs = (u=u_bcs, v=v_bcs, w=w_bcs, T=T_bcs, S=S_bcs)

if is_coriolis
    coriolis = FPlane(latitude=35.2480)
else
    coriolis = nothing
end

# ═══════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════
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

@info "" model

# ═══════════════════════════════════════════════════════════════════════════
# Checkpointing
# ═══════════════════════════════════════════════════════════════════════════
if checkpointing
    checkpoint_prefix = "checkpoint_$(run_tag)"
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

overwrite_existing = !pickup

# ═══════════════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════════════
@info "creating output fields"

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

u_c = @at (Center, Center, Center) u
v_c = @at (Center, Center, Center) v
w_c = @at (Center, Center, Center) w

# Cross-correlations for EKE and Fluxes
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

# (2) 3D Time Averages (10 day window)
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

if checkpointing
    simulation.output_writers[:checkpointer] = Checkpointer(model,
        schedule=TimeInterval(5days),
        prefix="checkpoint_$(run_tag)",
        overwrite_existing=true,
        cleanup=true)
end

# ═══════════════════════════════════════════════════════════════════════════
# Initial conditions
# ═══════════════════════════════════════════════════════════════════════════
if !pickup
    @info "No checkpoint found, setting initial conditions"

    uᵢ = 0.005 * rand(size(u)...)
    vᵢ = 0.005 * rand(size(v)...)
    wᵢ = 0.005 * rand(size(w)...)
    uᵢ .-= mean(uᵢ)
    vᵢ .-= mean(vᵢ)
    wᵢ .-= mean(wᵢ)

    if sigmoid_ic
        # Initialize v with a linear blend from v_south to v_north
        xv, yv, zv = nodes(v, reshape=true)
        # Blend factor: 0 at south, 1 at north
        α_y = yv ./ params.Ly
        v_s = v_south.(xv, zv, 0, Ref(params))
        v_n = v_north.(xv, zv, 0, Ref(params))
        vᵢ .+= (1 .- α_y) .* v_s .+ α_y .* v_n
    else
        # Simple linear blend
        xv, yv, zv = nodes(v, reshape=true)
        α_y = yv ./ params.Ly
        vᵢ .+= (1 .- α_y) .* v₀ .+ α_y .* (-v₀)
    end

    # T/S: uniform B2 (south) profile everywhere — no density gradient
    @inline Tᵢ(x, y, z) = T_south_pwl(z)
    @inline Sᵢ(x, y, z) = S_south_pwl(z)

    set!(model, u=uᵢ, v=vᵢ, w=wᵢ, T=Tᵢ, S=Sᵢ)
end

# ═══════════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════════
@info """
════════════════════════════════════════════════════════
 CONVERGENT FLOW SIMULATION: $(run_tag)
════════════════════════════════════════════════════════
 Run number:      $(run_number)
 Runtime:         $(sim_runtime)
 Architecture:    $(arch)

 ── Flow Configuration ──
 South boundary:  v = +$(v₀) m/s (northward)
 North boundary:  v = -$(v₀) m/s (southward)
 East boundary:   open (PerturbationAdvection, zero target)
 Topology:        Bounded × Bounded × Bounded

 ── Sponge Timescales ──
 τ_south:         $(params.τₛ)
 τ_north:         $(params.τₙ)
 τ_east:          $(params.τₑ)
 τ_tracer:        $(params.τ_ts)

 ── Switches ──
 LES:             $(LES)
 gradient_IC:     $(gradient_IC)
 sigmoid_v_bc:    $(sigmoid_v_bc)
 sigmoid_ic:      $(sigmoid_ic)
 is_coriolis:     $(is_coriolis)
 checkpointing:   $(checkpointing)
 shoal_bath:      $(shoal_bath)
 ════════════════════════════════════════════════════════
"""
run!(simulation, pickup=pickup)
