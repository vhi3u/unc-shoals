using Oceananigans
using Oceananigans.Units
using Oceananigans.BoundaryConditions: GradientBoundaryCondition, ValueBoundaryCondition, FieldBoundaryConditions, OpenBoundaryCondition
using Oceananigans.TurbulenceClosures
using Oceananigans.OutputWriters
using Oceananigans.Diagnostics: CFL
using Oceanostics: RossbyNumber, ErtelPotentialVorticity
using NCDatasets


include("dshoal_vn.jl")  # defines: dshoal(Lx, Ly, σ, Hs, N)


# domain
Lx = 100e3       # [m] along-shoal domain length
Ly = 200e3      # [m] across-shoal domain width
Lz = 50.0       # [m] depth

Nx, Ny, Nz = 100, 100, 20
grid = RectilinearGrid(size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz), topology=(Bounded, Bounded, Bounded))

# bathymetry parameters
σ = 8.0         # [km] Gaussian width for shoal cross-section
Hs = 15.0       # [m] shoal height


# no shoal bathymetry
x_km, y_km, h = dshoalf1(Lx/1e3, Ly/1e3, σ, Hs, Nx)


# create surface plot to check
using Plots

# If x_km and y_km are 1D, create a 2D meshgrid
xgrid = repeat(x_km', length(y_km), 1)  # (Ny, Nx)
ygrid = repeat(y_km, 1, length(x_km))   # (Ny, Nx)


# Grid with immersed bathymetry
ib_grid = ImmersedBoundaryGrid(grid, GridFittedBottom(h))

# buoyancy driven component (uses lu han 2022 paper for stratification and meridional density gradient in y direction)
ρ₀ = 1024.9       # reference density [kg/m³]
g = 9.81          # gravity [m/s²]
drho_dy = -2.2457e-06         #cross shoal gradient [kg/m⁴]
drho_dz_N = -0.064016  # [kg/m⁴] north
drho_dz_S = -0.0354  # [kg/m⁴] south
drho_dz_0 = -0.0311

ρ_inf(z) = ρ₀ + drho_dz_0 * z

function initial_b(x, y, z)
    ρ = ρ₀ + drho_dz_N * z + drho_dy * y
    return -g * (ρ - ρ_inf(z)) / ρ_inf(z)
end

println("→ Buoyancy at z=0 (surface)")
println("South (y=0): ", initial_b(0, 0, 0))
println("North (y=Ly): ", initial_b(0, Ly, 0))

# velocity forcings

# first test case: one directional flow

v_N = 0.10 # [m s^-1] northward (positive v) ; redundant at the moment. 
v_S = 0.10 # [m s^-1] northward (positive v)

# one directional flow case

u_bcs = FieldBoundaryConditions(;
    # west = OpenBoundaryCondition(0.0)
    east = OpenBoundaryCondition(0.0, scheme=PerturbationAdvection())
)

v_bcs = FieldBoundaryConditions(;
    south = OpenBoundaryCondition(v_S, scheme=PerturbationAdvection()),      # inflow at south
    north = OpenBoundaryCondition(0.0, scheme=PerturbationAdvection())
)


# model setup
model = NonhydrostaticModel(
    grid        = ib_grid,
    timestepper = :RungeKutta3,
    closure     = ScalarDiffusivity(ν=1e-3, κ=1e-3),
    pressure_solver = ConjugateGradientPoissonSolver(ib_grid),
    tracers     = (:b,),
    buoyancy    = BuoyancyTracer(),
    coriolis    = FPlane(latitude = 35.2480),
    boundary_conditions = (; u = u_bcs, v = v_bcs)
)

set!(model, b = initial_b)
@show extrema(model.tracers.b)

# outputs

@info "creating output fields"


cfl_values = Float64[]       # Stores CFL at each step
cfl_times  = Float64[]       # Stores model time

simulation = Simulation(model, Δt=1minutes, stop_time=50days)
conjure_time_step_wizard!(simulation, cfl=0.7,
                          diffusive_cfl = 0.8,
                          max_change = 1.8,
                          min_Δt = 2seconds,
                          max_Δt = 90minutes)

u, v, w = model.velocities
b = model.tracers.b
overwrite_existing = true

# create centered velocities for matlab plotting
u_c = @at (Center, Center, Center) u
v_c = @at (Center, Center, Center) v
w_c = @at (Center, Center, Center) w


ω_z = Field(∂x(v) - ∂y(u))
PV = @at (Center, Center, Center) ErtelPotentialVorticity(model, u, v, w, b, model.coriolis)
Ro = @at (Center, Center, Center) RossbyNumber(model)


outputs = (; u, v, w, b, u_c, v_c, w_c, ω_z, PV, Ro)

saved_output_prefix = "vnshoals"

saved_output_filename = saved_output_prefix * ".nc"
checkpointer_prefix = "checkpoint_" * saved_output_prefix


simulation.callbacks[:progress_logger] = Callback(TimeInterval(1days)) do sim
    @info "⏱️ Simulation time = $(prettytime(sim.model.clock.time))"
end

# cfl value every day callback
simulation.callbacks[:cfl_logger] = Callback(TimeInterval(1days)) do sim
    cfl_val = cfl(sim.model)
    @info ("🌀 CFL number = $(cfl_val)")
end

simulation.callbacks[:surface_v_logger] = Callback(TimeInterval(1days)) do sim
    v = sim.model.velocities.v
    grid = sim.model.grid

    # Indices at center in x and y, top level in z
    i = div(size(v, 1), 2)
    j = div(size(v, 2), 2)
    k = size(v, 3)  # surface is at the top of the domain

    v_val = v[i, j, k]
    @info "🌊 Surface v-velocity at center (x=$(i), y=$(j)) = $(v_val) m/s"
end


simulation.output_writers[:fields] = NetCDFWriter(model, outputs, 
                                                        schedule = TimeInterval(86400seconds),
                                                        filename = saved_output_filename,
                                                        overwrite_existing = overwrite_existing)

ccc_scratch = Field{Center, Center, Center}(model.grid) 

uv = Field((@at (Center, Center, Center) u*v))
uw = Field((@at (Center, Center, Center) u*w))
vw = Field((@at (Center, Center, Center) v*w))
uB = Field((@at (Center, Center, Center) u*b))
vB = Field((@at (Center, Center, Center) v*b))
wB = Field((@at (Center, Center, Center) w*b))

avg_interval = 12hours
simulation.output_writers[:averages] = NetCDFWriter(model, (;u, v, w, b, uv, uw, vw, uB, vB, wB),
                                                          schedule = AveragedTimeInterval(avg_interval, window = avg_interval),
                                                          filename = "timeaveragedfields.nc",
                                                          overwrite_existing = overwrite_existing)

cfl = CFL(simulation.Δt)


simulation.callbacks[:cfl_recorder] = Callback(TimeInterval(10minutes)) do sim
    push!(cfl_values, cfl(sim.model))
    push!(cfl_times, sim.model.clock.time)
end

# run simulation
run!(simulation)


plt = Plots.plot(cfl_times ./ 3600, cfl_values,
     xlabel = "Time (hours)",
     ylabel = "CFL number",
     title = "CFL over time",
           legend = false)
display(plt)
