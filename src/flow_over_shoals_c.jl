using Oceananigans
using Oceananigans.Grids: Periodic, Bounded
using Oceananigans.Units
using Oceananigans.BoundaryConditions: OpenBoundaryCondition, FieldBoundaryConditions
using Oceananigans.TurbulenceClosures
using Oceananigans.Solvers: ConjugateGradientPoissonSolver, FFTBasedPoissonSolver
using Oceananigans.Models: buoyancy_operation
using Oceananigans.OutputWriters
using Oceananigans.Forcings
using Oceanostics.ProgressMessengers: TimedMessenger
using SeawaterPolynomials.TEOS10
using Printf: @sprintf
using NCDatasets
using DataFrames
using CUDA: has_cuda_gpu, allowscalar
using Base.Threads

if has_cuda_gpu()
    arch = GPU()
    allowscalar(false)
else
    arch = CPU()
end
@info "architecture = $(arch)"
include("dshoal_vn_param.jl")

@info "Julia threads = $(nthreads())"

run_number = 441 # <-- change this for each new run
sim_runtime = 10days
callback_interval = 86400seconds
run_tag = "TEST_clean$(run_number)"  # e.g. "periodic_run1"

params = (; Lx=100e3, Ly=300e3, Lz=50, Nx=30, Ny=30, Nz=10)
params = (; params..., Nx=200, Ny=600, Nz=50)

x, y, z = (0, params.Lx), (0, params.Ly), (-params.Lz, 0)
grid = RectilinearGrid(arch; size=(params.Nx, params.Ny, params.Nz), halo=(4, 4, 4), x, y, z, topology=(Bounded, Periodic, Bounded))

slope_bottom = dshoal_param_bottom(params.Ly) # testing defaults. 
GFB = GridFittedBottom(slope_bottom)
ib_grid = ImmersedBoundaryGrid(grid, GFB)

@info ib_grid

v₀ = 0.10

params = (; params..., v₀=v₀)

reltol = sqrt(eps(ib_grid))
abstol = sqrt(eps(ib_grid))
@info "reltol = $reltol, abstol = $abstol"

model = NonhydrostaticModel(ib_grid;
    timestepper=:RungeKutta3,
    advection=WENO(order=5),
    closure=AnisotropicMinimumDissipation(),
    #hydrostatic_pressure_anomaly=CenterField(ib_grid),
    pressure_solver=ConjugateGradientPoissonSolver(ib_grid, reltol=reltol, abstol=abstol; preconditioner=FFTBasedPoissonSolver(grid)),
    tracers=(:T, :S),
    buoyancy=SeawaterBuoyancy(),
    coriolis=FPlane(latitude=35.2480),
    #boundary_conditions=bcs
)

@info "" model

@info "creating output fields"

@info "u storage = $(typeof(parent(model.velocities.u)))"
@info "v storage = $(typeof(parent(model.velocities.v)))"
@info "w storage = $(typeof(parent(model.velocities.w)))"

cfl_values = Float64[]       # Stores CFL at each step
cfl_times = Float64[]       # Stores model time

simulation = Simulation(model, Δt=15minutes, stop_time=sim_runtime)

conjure_time_step_wizard!(simulation, cfl=0.9)

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
b = buoyancy_operation(model)

u_c = @at (Center, Center, Center) u
v_c = @at (Center, Center, Center) v
w_c = @at (Center, Center, Center) w
T = model.tracers.T
S = model.tracers.S
# Ro = @at (Center, Center, Center) RossbyNumber(model)

slice_fields = (; u_c, v_c, w_c, T, S)

# Surface XY slice (top layer)
simulation.output_writers[:surface_slice] =
    NetCDFWriter(model, slice_fields,
        filename="top_$(run_tag).nc",
        schedule=TimeInterval(callback_interval),
        indices=(:, :, params.Nz),
        overwrite_existing=true)

# Mid-y XZ slice (cross-shore transect at domain center)
simulation.output_writers[:midy_slice] =
    NetCDFWriter(model, slice_fields,
        filename="midy_$(run_tag).nc",
        schedule=TimeInterval(callback_interval),
        indices=(:, round(Int, params.Ny / 2), :),
        overwrite_existing=true)


vᵢ = v₀

set!(model, v=vᵢ)

run!(simulation)