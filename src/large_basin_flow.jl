using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: Bounded
using Oceananigans.BoundaryConditions: NormalFlowBoundaryCondition, FieldBoundaryConditions, PerturbationAdvection
using Printf
using Oceananigans.OutputWriters
using Oceanostics: RossbyNumber, KineticEnergy
using Oceanostics.ProgressMessengers: TimedMessenger
using NCDatasets
using Oceananigans.Models: SeawaterBuoyancy
using Oceananigans.TurbulenceClosures
using Oceananigans.Forcings
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using CUDA: has_cuda_gpu, allowscalar

# naming 
run_number = 3

# Domain parameters
const Lx = 100e3 # 100 km
const Ly = 200e3 # 200 km
const Lz = 50    # 50 m

if has_cuda_gpu()
    arch = GPU()
    Nx, Ny, Nz = 200, 400, 50
else
    arch = CPU()
    Nx, Ny, Nz = 50, 50, 10
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
slope_bottom = dshoal_param_bottom(Ly;
    Hs=15.0,
    shoal_length=40000.0,
    sigma=8000.0,
    shelf_depth=-25.0,
    shelf_break_end=12000.0)
GFB = GridFittedBottom(slope_bottom)
ib_grid = ImmersedBoundaryGrid(grid, GFB)

# Flow parameters
const v₀ = 0.1 # m/s (northward flow max)

@inline function v_sigmoidal(x, z, t)
    xC = 3e3
    xS = 65e3
    k1 = 80 / Lx
    k2 = 20 / Lx

    s1 = 1 / (1 + exp(-k1 * (x - xC)))
    s2 = 1 / (1 + exp(k2 * (x - xS)))
    s = (s1 - 1) + s2
    sc = clamp(s, 0.0, 1.0)
    return v₀ * sc
end

# T/S profiles
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

# zero BC

flux_zero = FluxBoundaryCondition(0.0)
value_zero = ValueBoundaryCondition(0.0)

northern_bc = NormalFlowBoundaryCondition(v_sigmoidal; scheme=PerturbationAdvection(inflow_timescale=2minutes, outflow_timescale=30minutes))
southern_bc = NormalFlowBoundaryCondition(v_sigmoidal)

# Bottom Drag Formulation
z₀ = 2.5e-4 # roughness length
z₁ = Oceananigans.Grids.minimum_zspacing(grid, Center(), Center(), Center()) / 2
const κᵛᵏ = 0.4 # von Karman constant
c_dz = (κᵛᵏ / log(z₁ / z₀))^2 # quadratic drag coefficient

@inline τᵘ_drag(x, y, z, t, u, v, w, p) = -p.c_dz * u * √(u^2 + v^2 + w^2)
@inline τᵛ_drag(x, y, z, t, u, v, w, p) = -p.c_dz * v * √(u^2 + v^2 + w^2)
@inline τʷ_drag(x, y, z, t, u, v, w, p) = -p.c_dz * w * √(u^2 + v^2 + w^2)

immersed_drag_bc_u = FluxBoundaryCondition(τᵘ_drag, field_dependencies=(:u, :v, :w), parameters=(; c_dz=c_dz))
immersed_drag_bc_v = FluxBoundaryCondition(τᵛ_drag, field_dependencies=(:u, :v, :w), parameters=(; c_dz=c_dz))
immersed_drag_bc_w = FluxBoundaryCondition(τʷ_drag, field_dependencies=(:u, :v, :w), parameters=(; c_dz=c_dz))

# Apply the boundary conditions to the velocity fields
u_bcs = FieldBoundaryConditions(immersed=immersed_drag_bc_u, north=flux_zero, south=value_zero)
v_bcs = FieldBoundaryConditions(south=southern_bc, north=northern_bc, immersed=immersed_drag_bc_v)
w_bcs = FieldBoundaryConditions(immersed=immersed_drag_bc_w, north=flux_zero, south=value_zero)

@inline tsbc(x, z, t) = T_south_pwl(z)
@inline tnbc(x, z, t) = T_north_pwl(z)
@inline ssbc(x, z, t) = S_south_pwl(z)
@inline snbc(x, z, t) = S_north_pwl(z)

T_bcs = FieldBoundaryConditions(south=ValueBoundaryCondition(tsbc), north=ValueBoundaryCondition(tnbc))
S_bcs = FieldBoundaryConditions(south=ValueBoundaryCondition(ssbc), north=ValueBoundaryCondition(snbc))

bcs = (u=u_bcs, v=v_bcs, w=w_bcs, T=T_bcs, S=S_bcs)

reltol = sqrt(eps(grid))
abstol = sqrt(eps(grid))

# Sponge layers for the boundaries to damp out Coriolis-driven waves and adjustment shock
const L_sponge = 20e3 # 20 km width for the sponge layers
const east_mask = PiecewiseLinearMask{:x}(center=Lx, width=L_sponge)
const south_mask = PiecewiseLinearMask{:y}(center=0, width=L_sponge)
const north_mask = PiecewiseLinearMask{:y}(center=Ly, width=L_sponge)

@inline sponge_mask(x, y, z) = min(east_mask(x, y, z) + south_mask(x, y, z) + north_mask(x, y, z), 1.0)
@inline v_target(x, y, z, t) = v_sigmoidal(x, z, t)

const τ_sponge = 24hours # timescale for relaxation

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

u_nudging = Relaxation(rate=1 / τ_sponge, mask=sponge_mask, target=0.0)
v_nudging = Relaxation(rate=1 / τ_sponge, mask=sponge_mask, target=v_target)
w_nudging = Relaxation(rate=1 / τ_sponge, mask=sponge_mask, target=0.0)
T_nudging = Relaxation(rate=1 / τ_sponge, mask=sponge_mask, target=T_target)
S_nudging = Relaxation(rate=1 / τ_sponge, mask=sponge_mask, target=S_target)

forcings = (u=u_nudging, v=v_nudging, w=w_nudging, T=T_nudging, S=S_nudging)

# Model
model = NonhydrostaticModel(ib_grid,
    advection=WENO(order=5),
    tracers=(:T, :S),
    buoyancy=SeawaterBuoyancy(),
    pressure_solver=ConjugateGradientPoissonSolver(ib_grid, reltol=reltol, abstol=abstol, maxiter=100),
    closure=(HorizontalScalarDiffusivity(ν=1e-5, κ=1e-5), VerticalScalarDiffusivity(ν=1e-6, κ=1e-6)),
    boundary_conditions=bcs,
    coriolis=FPlane(latitude=35.2480),
    forcing=forcings)

# Stratification based on blended profiles
@inline α_lin(y) = clamp(y / Ly, 0.0, 1.0)
@inline blend(a, b, α) = (1 - α) * a + α * b
@inline T_initial(x, y, z) = blend(T_south_pwl(z), T_north_pwl(z), α_lin(y))
@inline S_initial(x, y, z) = blend(S_south_pwl(z), S_north_pwl(z), α_lin(y))

# Set initial conditions (start with uniform flow to match boundaries and linear T/S)
@inline v_initial(x, y, z) = v_sigmoidal(x, z, 0.0)
set!(model, v=v_initial, T=T_initial, S=S_initial)

# Simulation setup
simulation = Simulation(model, Δt=15minutes, stop_time=100days)
conjure_time_step_wizard!(simulation, cfl=0.8)

# Logging progress
# Note: callback_interval is defined further below, so we define it here first
callback_interval = 1days

progress = TimedMessenger()
simulation.callbacks[:progress] = Callback(progress, TimeInterval(callback_interval))

# function print_solver_iterations(sim)
#     solver = sim.model.pressure_solver
#     if hasproperty(solver, :conjugate_gradient_solver)
#         cg = solver.conjugate_gradient_solver
#         @info @sprintf("Pressure solver: %d CG iterations (t = %.2f days)",
#             cg.iteration, time(sim) / 86400)
#     end
# end
# simulation.callbacks[:solver_iters] = Callback(print_solver_iterations, TimeInterval(callback_interval))

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
tavg_fields = (; u_c, v_c, w_c, uu, vv, ww, T, S)

run_tag = "bounded_shoals$(run_number)"

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
