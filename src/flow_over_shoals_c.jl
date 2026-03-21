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
using CUDA: has_cuda_gpu
using Base.Threads

if has_cuda_gpu()
    arch = GPU()
else
    arch = CPU()
end
include("dshoal_vn_param.jl")

run_number = 441 # <-- change this for each new run
sim_runtime = 10days
callback_interval = 86400seconds
run_tag = "TEST_clean$(run_number)"  # e.g. "periodic_run1"

params = (; Lx=100e3, Ly=300e3, Lz=50, Nx=30, Ny=30, Nz=10)
params = (; params..., Nx=200, Ny=600, Nz=50)

x, y, z = (0, params.Lx), (0, params.Ly), (-params.Lz, 0)
grid = RectilinearGrid(arch; size=(params.Nx, params.Ny, params.Nz), halo=(4, 4, 4), x, y, z, topology=(Bounded, Periodic, Bounded))

# Define shoal parameters (align with the new sigmoidal setup)
Hs = 15.0         # Height of shoal above -25m shelf
sigma = 8e3       # Gaussian width of shoal (half crossover)
shoal_length = 20e3 # Horizontal span of the shoal ridge

slope_bottom = dshoal_param_bottom(params.Ly; Hs=Hs, sigma=sigma, shoal_length=shoal_length)
GFB = GridFittedBottom(slope_bottom)
ib_grid = ImmersedBoundaryGrid(grid, GFB)

@info "Immersed boundary grid created with parameterized shoal."
@info ib_grid

@info "Saving bottom_height to NetCDF"
let
    dx = params.Lx / params.Nx
    dy = params.Ly / params.Ny
    xc = collect(range(dx / 2, step=dx, length=params.Nx))
    yc = collect(range(dy / 2, step=dy, length=params.Ny))

    bathy = zeros(params.Nx, params.Ny)
    for i in 1:params.Nx, j in 1:params.Ny
        bathy[i, j] = slope_bottom(xc[i], yc[j])
    end

    ds_bathy = NCDataset("bottom_height_$(run_tag).nc", "c")
    defDim(ds_bathy, "xC", params.Nx)
    defDim(ds_bathy, "yC", params.Ny)
    defVar(ds_bathy, "xC", xc, ("xC",))
    defVar(ds_bathy, "yC", yc, ("yC",))
    defVar(ds_bathy, "bottom_height", bathy, ("xC", "yC"), attrib=Dict("units" => "m"))
    close(ds_bathy)
end
@info "Finished saving bottom_height to NetCDF"

v₀ = 0.10

params = (; params..., v₀=v₀)

reltol = sqrt(eps(ib_grid))
abstol = sqrt(eps(ib_grid))
@info "reltol = $reltol, abstol = $abstol"

cᴰ = 2.5e-3 # dimensionless drag coefficient
@inline drag_u(x, y, t, u, v, p) = -p.cᴰ * √(u^2 + v^2) * u
@inline drag_v(x, y, t, u, v, p) = -p.cᴰ * √(u^2 + v^2) * v
drag_bc_u = FluxBoundaryCondition(drag_u, field_dependencies=(:u, :v), parameters=(; cᴰ=cᴰ,))
drag_bc_v = FluxBoundaryCondition(drag_v, field_dependencies=(:u, :v), parameters=(; cᴰ=cᴰ,))
u_bcs = FieldBoundaryConditions(bottom=drag_bc_u)
v_bcs = FieldBoundaryConditions(bottom=drag_bc_v)
bcs = (u=u_bcs, v=v_bcs)

model = NonhydrostaticModel(ib_grid;
    timestepper=:RungeKutta3,
    advection=WENO(order=5),
    closure=AnisotropicMinimumDissipation(),
    hydrostatic_pressure_anomaly=CenterField(ib_grid),
    pressure_solver=ConjugateGradientPoissonSolver(ib_grid, reltol=reltol, abstol=abstol),
    tracers=(:T, :S),
    buoyancy=SeawaterBuoyancy(),
    coriolis=FPlane(latitude=35.2480),
    boundary_conditions=bcs
)

@info "" model

@info "creating output fields"

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

# YZ slice at x = 25 km
x_idx = round(Int, 25e3 / (params.Lx / params.Nx))
simulation.output_writers[:yz_slice] =
    NetCDFWriter(model, slice_fields,
        filename="yz_$(run_tag).nc",
        schedule=TimeInterval(callback_interval),
        indices=(x_idx, :, :),
        overwrite_existing=true)


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

# vᵢ = v∞

set!(model, v=(x, y, z) -> v∞(x, y, z, params))

run!(simulation)