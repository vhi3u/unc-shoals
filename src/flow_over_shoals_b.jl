# ═══════════════════════════════════════════════════════════════════════════
# flow_over_shoals_b.jl
# ═══════════════════════════════════════════════════════════════════════════
# Bounded along-shore domain with PerturbationAdvection open boundary conditions.
# Simplified version of flow_over_shoals.jl:
#   - Bounded topology: (Bounded, Bounded, Bounded)
#   - OpenBoundaryCondition with scheme = PerturbationAdvection() for along-shore inflow/outflow
#   - Purely hydrodynamic flow (no tracers/temperature forcing)
#   - No Coriolis
#   - CPU or GPU execution
# ═══════════════════════════════════════════════════════════════════════════

using Oceananigans
using Oceananigans.Grids: Bounded
using Oceananigans.Units
using Oceananigans.BoundaryConditions: OpenBoundaryCondition, FieldBoundaryConditions
using Oceananigans.TurbulenceClosures
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using Oceananigans.OutputWriters
using Statistics: mean
using Printf: @sprintf
using NCDatasets
using Oceanostics.ProgressMessengers: TimedMessenger
using Oceanostics: KineticEnergy
using CUDA: has_cuda_gpu

if has_cuda_gpu()
    arch = GPU()
    Nx, Ny, Nz = 200, 400, 50
    @info "Building bounded GPU domain (Nx=$Nx, Ny=$Ny, Nz=$Nz)"
else
    arch = CPU()
    Nx, Ny, Nz = 50, 100, 10
    @info "Building bounded CPU domain (Nx=$Nx, Ny=$Ny, Nz=$Nz)"
end

# Grid parameters
Lx = 100e3
Ly = 200e3
Lz = 50

x, y, z = (0, Lx), (0, Ly), (-Lz, 0)

# 1. Create Rectilinear Grid (Bounded in X, Y, Z)
grid = RectilinearGrid(arch; size=(Nx, Ny, Nz), halo=(4, 4, 4), x, y, z, topology=(Bounded, Bounded, Bounded))

# 2. Immersed Shoal Bathymetry
include("dshoal_vn_param.jl")
Hs = 5.0            # Height of shoal above -25m shelf
sigma = 8e3         # Gaussian width of shoal (half crossover)
shoal_length = 20e3 # Horizontal span of the shoal ridge

slope_bottom = dshoal_param_bottom(Ly; Hs=Hs, sigma=sigma, shoal_length=shoal_length)
GFB = GridFittedBottom(slope_bottom)
ib_grid = ImmersedBoundaryGrid(grid, GFB)

@info ib_grid

# 3. Model Parameters and Boundary Conditions
v₀ = 0.1

# Immersed boundary drag
cᴰ = 2.5e-3
@inline immersed_drag_u(x, y, z, t, u, v, cᴰ) = -cᴰ * u * sqrt(u^2 + v^2)
@inline immersed_drag_v(x, y, z, t, u, v, cᴰ) = -cᴰ * v * sqrt(u^2 + v^2)
immersed_drag_bc_u = FluxBoundaryCondition(immersed_drag_u, field_dependencies=(:u, :v), parameters=cᴰ)
immersed_drag_bc_v = FluxBoundaryCondition(immersed_drag_v, field_dependencies=(:u, :v), parameters=cᴰ)

# Open Boundary Conditions for v at South and North boundaries using PerturbationAdvection
open_bc = OpenBoundaryCondition(v₀; scheme=PerturbationAdvection())
open_bc_zero = OpenBoundaryCondition(0.0; scheme=PerturbationAdvection())

u_bcs = FieldBoundaryConditions(immersed=immersed_drag_bc_u, east=open_bc_zero)
v_bcs = FieldBoundaryConditions(immersed=immersed_drag_bc_v, south=open_bc, north=open_bc, east=open_bc)
w_bcs = FieldBoundaryConditions(east=open_bc_zero)
bcs = (u=u_bcs, v=v_bcs, w=w_bcs)

# 4. Construct Nonhydrostatic Model 
model = NonhydrostaticModel(ib_grid;
    timestepper=:RungeKutta3,
    advection=WENO(),
    closure=VerticalScalarDiffusivity(ν=1e-5, κ=1e-5),
    pressure_solver=ConjugateGradientPoissonSolver(ib_grid),
    tracers=(:T, :S),
    buoyancy=SeawaterBuoyancy(),
    coriolis=FPlane(latitude=35.2480),
    boundary_conditions=bcs
)

@info "" model

# 5. Set Initial Conditions
set!(model, v=(x, y, z) -> v₀)

# 6. Configure Simulation
sim_runtime = 100days
callback_interval = 86400seconds
run_tag = "bounded_shoals_simple"

simulation = Simulation(model, Δt=15minutes, stop_time=sim_runtime)
conjure_time_step_wizard!(simulation, cfl=0.7, diffusive_cfl=0.7)

# Progress callback
progress = TimedMessenger()
simulation.callbacks[:progress] = Callback(progress, TimeInterval(callback_interval))

# CG Solver Callback
function print_solver_iterations(sim)
    solver = sim.model.pressure_solver
    if hasproperty(solver, :conjugate_gradient_solver)
        cg = solver.conjugate_gradient_solver
        @info @sprintf("Pressure solver: %d CG iterations (t = %s)",
            cg.iteration, prettytime(time(sim)))
    end
end
simulation.callbacks[:solver_iters] = Callback(print_solver_iterations, TimeInterval(callback_interval))

# 7. Output Writers
u, v, w = model.velocities
u_c = @at (Center, Center, Center) u
v_c = @at (Center, Center, Center) v
w_c = @at (Center, Center, Center) w
KE  = KineticEnergy(model)
slice_fields = (; u_c, v_c, w_c, KE)

# Surface XY slice (top layer)
simulation.output_writers[:surface_slice] = NetCDFWriter(model, slice_fields,
    filename="top_$(run_tag).nc",
    schedule=TimeInterval(callback_interval),
    indices=(:, :, Nz),
    overwrite_existing=true)

# Mid-y XZ slice (cross-shore transect at domain center)
simulation.output_writers[:midy_slice] = NetCDFWriter(model, slice_fields,
    filename="midy_$(run_tag).nc",
    schedule=TimeInterval(callback_interval),
    indices=(:, round(Int, Ny / 2), :),
    overwrite_existing=true)

# 8. Run simulation!
@info "Running the simulation..."
run!(simulation)
