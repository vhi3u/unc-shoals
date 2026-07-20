using Pkg
Pkg.instantiate()

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: Bounded
using Oceananigans.BoundaryConditions: NormalFlowBoundaryCondition, FieldBoundaryConditions, PerturbationAdvection
using Printf
using Oceananigans.OutputWriters
using Oceanostics: RossbyNumber, KineticEnergy
using Oceanostics.ProgressMessengers: TimedMessenger
using NCDatasets
using Oceananigans.TurbulenceClosures
using Oceananigans.Forcings
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using CUDA: has_cuda_gpu, allowscalar
using SeawaterPolynomials.TEOS10

# naming 
run_number = 11

# Domain parameters
Lx = 100e3 # 100 km
Ly = 200e3 # 100 km
Lz = 50    # 50 m

if has_cuda_gpu()
    arch = GPU()
    Nx, Ny, Nz = 200, 400, 50
    νh = 1e-2
    κh = 1e-2
else
    arch = CPU()
    Nx, Ny, Nz = 50, 50, 10
    νh = 1.0
    κh = 1.0
end
@info "architecture = $(arch)"

# Grid
# Topology is Bounded in all directions to represent a closed basin with open boundaries in y
grid = RectilinearGrid(arch; size=(Nx, Ny, Nz),
    halo=(4, 4, 4),
    x=(0, Lx),
    y=(0, Ly),
    z=(-Lz, 0),
    topology=(Bounded, Bounded, Bounded))

include(joinpath(@__DIR__, "dshoal_vn_param.jl"))
const slope_bottom = dshoal_param_bottom(Ly;
    Hs=20.0,
    shoal_length=40000.0,
    sigma=5000.0,
    shelf_depth=-25.0,
    shelf_break_end=12000.0)
GFB = GridFittedBottom(slope_bottom)
ib_grid = ImmersedBoundaryGrid(grid, GFB)

# Flow parameters
const v₀ = 0.1 # m/s (northward flow max)
prebalance = true

# zero BC

flux_zero = FluxBoundaryCondition(0.0)
value_zero = ValueBoundaryCondition(0.0; scheme=PerturbationAdvection(inflow_timescale=Inf, outflow_timescale=0.0))

# Spatially-varying inflow proportional to sqrt(depth) to perfectly balance bottom friction
if prebalance
    @inline v_inflow(x, y, t) = v₀ * sqrt(abs(slope_bottom(x, y)) / 50.0)
else
    @inline v_inflow(x, y, t) = v₀
end

northern_bc = NormalFlowBoundaryCondition(v_inflow; scheme=PerturbationAdvection(inflow_timescale=0.0, outflow_timescale=0.0))
southern_bc = NormalFlowBoundaryCondition(v_inflow)
eastern_bc = NormalFlowBoundaryCondition(0.0; scheme=PerturbationAdvection(inflow_timescale=0.0, outflow_timescale=Inf))

# Stratification Profiles (T/S)
const δ_smooth = 2.5
@inline smooth_step_z(z, z0) = 0.5 * (1.0 - tanh((z - z0) / δ_smooth))

@inline function T_south_pwl(z, v1=24.5378)
    z1, z2, z3 = -5.0, -15.0, -30.0
    v2, v3 = 24.3073, 23.4116
    m12 = (v2 - v1) / (z2 - z1)
    m23 = (v3 - v2) / (z3 - z2)
    w1 = smooth_step_z(z, z1)
    w2 = smooth_step_z(z, z2)
    w3 = smooth_step_z(z, z3)
    return v1 * (1 - w1) + (v1 + m12 * (z - z1)) * (w1 - w2) + (v2 + m23 * (z - z2)) * (w2 - w3) + v3 * w3
end

@inline function S_south_pwl(z, v1=35.5830)
    z1, z2, z3 = -5.0, -15.0, -30.0
    v2, v3 = 35.9986, 36.1776
    m12 = (v2 - v1) / (z2 - z1)
    m23 = (v3 - v2) / (z3 - z2)
    w1 = smooth_step_z(z, z1)
    w2 = smooth_step_z(z, z2)
    w3 = smooth_step_z(z, z3)
    return v1 * (1 - w1) + (v1 + m12 * (z - z1)) * (w1 - w2) + (v2 + m23 * (z - z2)) * (w2 - w3) + v3 * w3
end

@inline tsbc(x, z, t) = T_south_pwl(z)
@inline ssbc(x, z, t) = S_south_pwl(z)

T_bcs = FieldBoundaryConditions(south=ValueBoundaryCondition(tsbc), north=flux_zero)
S_bcs = FieldBoundaryConditions(south=ValueBoundaryCondition(ssbc), north=flux_zero)

# Bottom Drag Formulation
z₀ = 2.5e-4 # roughness length
z₁ = Oceananigans.Grids.minimum_zspacing(grid, Center(), Center(), Center()) / 2
const κᵛᵏ = 0.4 # von Karman constant
c_dz = (κᵛᵏ / log(z₁ / z₀))^2 # quadratic drag coefficient  

drag = BulkDrag(coefficient=c_dz)

# Wind stress BC
ρ₀ = 1024.0
wind_stress_u = 0.0
wind_stress_v = 0.05 # N/m²
wind_bc_u = FluxBoundaryCondition(-wind_stress_u / ρ₀)
wind_bc_v = FluxBoundaryCondition(-wind_stress_v / ρ₀)

# Apply the boundary conditions to the velocity fields
u_bcs = FieldBoundaryConditions(immersed=drag, bottom=drag, north=flux_zero, south=value_zero, east=eastern_bc, top=wind_bc_u)
v_bcs = FieldBoundaryConditions(immersed=drag, bottom=drag, north=northern_bc, south=southern_bc, east=flux_zero, top=wind_bc_v)
w_bcs = FieldBoundaryConditions(immersed=drag, north=flux_zero, south=value_zero, east=flux_zero)

bcs = (u=u_bcs, v=v_bcs, w=w_bcs, T=T_bcs, S=S_bcs)

reltol = sqrt(eps(grid))
abstol = sqrt(eps(grid))


# Model
model = NonhydrostaticModel(ib_grid,
    advection=WENO(order=5),
    pressure_solver=ConjugateGradientPoissonSolver(ib_grid, reltol=reltol, abstol=abstol, maxiter=100),
    tracers=(:T, :S),
    buoyancy=SeawaterBuoyancy(),
    closure=(HorizontalScalarDiffusivity(ν=νh, κ=κh), VerticalScalarDiffusivity(ν=1e-6, κ=1e-6)),
    boundary_conditions=bcs,
    coriolis=FPlane(latitude=35.2480)
)

# Set initial conditions
if prebalance
    set!(model, v=(x, y, z) -> v₀ * sqrt(abs(slope_bottom(x, y)) / 50.0), T=(x, y, z) -> T_south_pwl(z), S=(x, y, z) -> S_south_pwl(z))
else
    set!(model, v=v₀, T=(x, y, z) -> T_south_pwl(z), S=(x, y, z) -> S_south_pwl(z))
end

# Simulation setup
simulation = Simulation(model, Δt=15minutes, stop_time=20days)
conjure_time_step_wizard!(simulation, cfl=0.8)

# Logging progress
# Note: callback_interval is defined further below, so we define it here first
callback_interval = 1days

progress = TimedMessenger()
simulation.callbacks[:progress] = Callback(progress, TimeInterval(callback_interval))


# Run the simulation
@info "Starting simulation for a 100km x 200km basin with 0.1 m/s northward flow..."

# Setup outputs
u, v, w = model.velocities
T = model.tracers.T
S = model.tracers.S
Ro = @at (Center, Center, Center) RossbyNumber(model)
KE = @at (Center, Center, Center) KineticEnergy(model)

# Centered velocities for consistency
u_c = @at (Center, Center, Center) u
v_c = @at (Center, Center, Center) v
w_c = @at (Center, Center, Center) w

uu = Field(u_c * u_c)
vv = Field(v_c * v_c)
ww = Field(w_c * w_c)

slice_fields = (; u_c, v_c, w_c, T, S, Ro, KE)
tavg_fields = (; u_c, v_c, w_c, uu, vv, ww)

run_tag = "ib_basin$(run_number)"

simulation.output_writers[:surface_slice] = NetCDFWriter(model, slice_fields,
    filename="top_$(run_tag).nc",
    schedule=TimeInterval(callback_interval),
    indices=(:, :, Nz),
    overwrite_existing=true)

simulation.output_writers[:midy_slice] = NetCDFWriter(model, slice_fields,
    filename="midy_$(run_tag).nc",
    schedule=TimeInterval(callback_interval),
    indices=(:, round(Int, Ny / 2), :),
    overwrite_existing=true)

run!(simulation)
