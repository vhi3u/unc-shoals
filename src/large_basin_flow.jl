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

# Domain parameters
Lx = 100e3 # 100 km
Ly = 200e3 # 200 km
Lz = 50    # 50 m


Nx, Ny, Nz = 50, 50, 10

# Grid
# Topology is Bounded in all directions to represent a closed basin with open boundaries in y
grid = RectilinearGrid(size=(Nx, Ny, Nz),
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
v₀ = 0.1 # m/s (northward flow max)

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

# zero BC

open_zero = ValueBoundaryCondition(0.0; scheme=PerturbationAdvection())

# Boundary conditions for northward flow
# Use PerturbationAdvection to make the northern boundary purely a relaxation one 
# (outflow_timescale=0.0) and the southern boundary a radiation one (outflow_timescale=Inf).
northern_bc = NormalFlowBoundaryCondition(v_sigmoidal; scheme=PerturbationAdvection())
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
u_bcs = FieldBoundaryConditions(immersed=immersed_drag_bc_u, north=open_zero)
v_bcs = FieldBoundaryConditions(south=southern_bc, north=northern_bc, immersed=immersed_drag_bc_v)
w_bcs = FieldBoundaryConditions(immersed=immersed_drag_bc_w, north=open_zero)
bcs = (u=u_bcs, v=v_bcs, w=w_bcs)

reltol = sqrt(eps(grid))
abstol = sqrt(eps(grid))

# Sponge layer on the eastern boundary to damp out Coriolis-driven wave reflections
const L_sponge = 20e3 # 20 km width for the sponge layer
const east_mask = PiecewiseLinearMask{:x}(center=Lx, width=L_sponge)
const τ_sponge = 24hours # timescale for relaxation

u_nudging = Relaxation(rate=1 / τ_sponge, mask=east_mask, target=0.0)
v_nudging = Relaxation(rate=1 / τ_sponge, mask=east_mask, target=0.0)
w_nudging = Relaxation(rate=1 / τ_sponge, mask=east_mask, target=0.0)

forcings = (u=u_nudging, v=v_nudging, w=w_nudging)

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

# Linear stratification based on South profiles
@inline function T_initial(x, y, z)
    v1 = 24.5378
    v_bot = 23.4116
    return v1 + (v_bot - v1) * (z / -50.0)
end

@inline function S_initial(x, y, z)
    v1 = 35.5830
    v_bot = 36.1776
    return v1 + (v_bot - v1) * (z / -50.0)
end

# Set initial conditions (start with uniform flow to match boundaries and linear T/S)
@inline v_initial(x, y, z) = v_sigmoidal(x, z, 0.0)
set!(model, v=v_initial, T=T_initial, S=S_initial)

# Simulation setup
simulation = Simulation(model, Δt=15minutes, stop_time=10days)
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

run_tag = "large_basin_flow"

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
