# windy shoals for nonhydrostatic, GPU/CPU, 

using Oceananigans
using Oceananigans.Grids: Periodic, Bounded, minimum_zspacing
using Oceananigans.Units
using Oceanostics: RossbyNumber, ErtelPotentialVorticity,
    KineticEnergy, KineticEnergyDissipationRate, TurbulentKineticEnergy,
    XShearProductionRate, YShearProductionRate, ZShearProductionRate
using Oceananigans.Solvers: ConjugateGradientPoissonSolver, FFTBasedPoissonSolver
using Oceanostics.ProgressMessengers: TimedMessenger
using SeawaterPolynomials.TEOS10
using Printf: @sprintf
using NCDatasets
using DataFrames
using CUDA: has_cuda_gpu, allowscalar
using Statistics: mean

run_number = 1
run_tag = "windy_shoals$(run_number)"  # e.g. "periodic_run1"
callback_interval = 86400seconds
sim_runtime = 100days


if has_cuda_gpu()
    arch = GPU()
else
    arch = CPU()
end
@info "architecture = $(arch)"
include("dshoal_vn_param.jl")

# build
@info "building domain"

params = (; Lx=100e3, Ly=200e3, Lz=50)

if arch == CPU()
    params = (; params..., Nx=50, Ny=100, Nz=10)
else
    params = (; params..., Nx=200, Ny=400, Nz=50)
end

x, y, z = (0, params.Lx), (0, params.Ly), (-params.Lz, 0)

grid = RectilinearGrid(arch; size=(params.Nx, params.Ny, params.Nz), halo=(4, 4, 4), x, y, z, topology=(Bounded, Periodic, Bounded))

Hs = 5.0         # Height of shoal above -25m shelf
sigma = 8e3       # Gaussian width of shoal (half crossover)
shoal_length = 20e3 # Horizontal span of the shoal ridge

slope_bottom = dshoal_param_bottom(params.Ly; Hs=Hs, sigma=sigma, shoal_length=shoal_length)
GFB = GridFittedBottom(slope_bottom)
ib_grid = ImmersedBoundaryGrid(grid, GFB)

@info ib_grid
v₀ = 0.1

params = (; params..., v₀=v₀)

# Smooth transition function: 0 when z >> z0, 1 when z << z0
const δ_smooth = 2.5
@inline smooth_step(z, z0) = 0.5 * (1.0 - tanh((z - z0) / δ_smooth))

# Temperature at North boundary (B1) - SMOOTHED
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

@inline tsbc(x, z, t) = T_south_pwl(z)
@inline tnbc(x, z, t) = T_north_pwl(z)
@inline ssbc(x, z, t) = S_south_pwl(z)
@inline snbc(x, z, t) = S_north_pwl(z)

# Eastern boundary targets are now functions of z
params = (; params...)

# drag BC

z₀ = 2.5e-4 # roughness length
z₁ = Oceananigans.Grids.minimum_zspacing(grid, Center(), Center(), Center()) / 2
@info "Using z₁ =" z₁
const κᵛᵏ = 0.4 # von Karman constant
params = (; params..., c_dz=(κᵛᵏ / log(z₁ / z₀))^2) # quadratic drag coefficient
@info "Defining momentum BCs with Cᴰ =" params.c_dz
drag_bc = BulkDrag(coefficient=params.c_dz)

# surface wind stress
@inline wind_ramp(t) = tanh(t / (2 * 86400.0))
@inline wind_stress_v(x, y, t) = (-0.05 / 1000.0) * wind_ramp(t)
wind_bc = FluxBoundaryCondition(wind_stress_v)

# Explicit mask functions (fully typed, no global variable captures)
@inline north_mask(x, y, z) = clamp((y - 180e3) / 20e3, 0.0, 1.0)
@inline south_mask(x, y, z) = clamp(1.0 - y / 20e3, 0.0, 1.0)
@inline east_mask(x, y, z) = clamp((x - 60e3) / 40e3, 0.0, 1.0)

@inline sponge_mask(x, y, z) = min(north_mask(x, y, z) + south_mask(x, y, z) + east_mask(x, y, z), 1.0)

@inline east_mask_uvw(x, y, z) = clamp((x - 50e3) / 50e3, 0.0, 1.0)
@inline sponge_mask_uvw(x, y, z) = min(north_mask(x, y, z) + south_mask(x, y, z) + east_mask_uvw(x, y, z), 1.0)

@inline function T_target(x, y, z, t)
    n = north_mask(x, y, z)
    s = south_mask(x, y, z)
    e = east_mask(x, y, z)
    tot = n + s + e
    return tot > 0 ? (n * T_north_pwl(z) + s * T_south_pwl(z) + e * T_east_pwl(z)) / tot : T_south_pwl(z)
end

@inline function S_target(x, y, z, t)
    n = north_mask(x, y, z)
    s = south_mask(x, y, z)
    e = east_mask(x, y, z)
    tot = n + s + e
    return tot > 0 ? (n * S_north_pwl(z) + s * S_south_pwl(z) + e * S_east_pwl(z)) / tot : S_south_pwl(z)
end

@inline function v_target(x, y, z, t)
    n = north_mask(x, y, z)
    s = south_mask(x, y, z)
    e = east_mask_uvw(x, y, z)
    tot = n + s + e
    v0 = 0.1 # local constant
    return tot > 0 ? (n * v0 + s * v0 + e * 0.0) / tot : v0
end

u_nudge = Relaxation(; rate=1 / 24hours, mask=sponge_mask_uvw, target=0)
v_nudge = Relaxation(; rate=1 / 24hours, mask=sponge_mask_uvw, target=v_target)
w_nudge = Relaxation(; rate=1 / 24hours, mask=sponge_mask_uvw, target=0)
T_nudge = Relaxation(; rate=1 / 24hours, mask=sponge_mask, target=T_target)
S_nudge = Relaxation(; rate=1 / 24hours, mask=sponge_mask, target=S_target)

forcings = (; u=u_nudge, v=v_nudge, w=w_nudge, T=T_nudge, S=S_nudge)

# setup BCs

u_bcs = FieldBoundaryConditions(immersed=drag_bc)
v_bcs = FieldBoundaryConditions(immersed=drag_bc, top=wind_bc)
w_bcs = FieldBoundaryConditions(immersed=drag_bc)


coriolis = FPlane(latitude=35.2480)

bcs = (u=u_bcs, v=v_bcs, w=w_bcs)

model = NonhydrostaticModel(ib_grid;
    timestepper=:RungeKutta3,
    advection=WENO(order=5),
    closure=VerticalScalarDiffusivity(ν=1e-4, κ=1e-4),
    hydrostatic_pressure_anomaly=CenterField(ib_grid),
    pressure_solver=ConjugateGradientPoissonSolver(ib_grid, maxiter=200),
    tracers=(:T, :S),
    buoyancy=SeawaterBuoyancy(),
    coriolis=coriolis,
    boundary_conditions=bcs,
    forcing=forcings
)

@info "" model

@info "creating output fields"

# Don't overwrite the NetCDF file when picking up from a checkpoint
overwrite_existing = true

cfl_values = Float64[]       # Stores CFL at each step
cfl_times = Float64[]       # Stores model time

simulation = Simulation(model, Δt=15minutes, stop_time=sim_runtime)

conjure_time_step_wizard!(simulation, cfl=0.7, diffusive_cfl=0.7)

progress = TimedMessenger()

simulation.callbacks[:progress] = Callback(progress, TimeInterval(1days))

# Also add a solver iterations callback so we can see if the Poisson solver is grinding
function print_solver_iterations(sim)
    solver = sim.model.pressure_solver
    if hasproperty(solver, :conjugate_gradient_solver)
        cg = solver.conjugate_gradient_solver
        @info @sprintf("Pressure solver: %d CG iterations", cg.iteration)
    end
end
simulation.callbacks[:solver_iters] = Callback(print_solver_iterations, TimeInterval(1days))

u, v, w = model.velocities
T = model.tracers.T
S = model.tracers.S
Ro = @at (Center, Center, Center) RossbyNumber(model)
KE = @at (Center, Center, Center) KineticEnergy(model)

# Centered velocities for consistency
u_c = @at (Center, Center, Center) u
v_c = @at (Center, Center, Center) v
w_c = @at (Center, Center, Center) w

slice_fields = (; u_c, v_c, w_c, T, S, Ro, KE)

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


uᵢ = 0.005 * rand(size(u)...)
vᵢ = 0.005 * rand(size(v)...)
wᵢ = 0.005 * rand(size(w)...)
uᵢ .-= mean(uᵢ)
vᵢ .-= mean(vᵢ)
wᵢ .-= mean(wᵢ)
uᵢ .+= 0
vᵢ .+= v₀

@inline Tᵢ(x, y, z) = T_south_pwl(z)
@inline Sᵢ(x, y, z) = S_south_pwl(z)

set!(model, u=uᵢ, v=vᵢ, w=wᵢ, T=Tᵢ, S=Sᵢ)
run!(simulation)


