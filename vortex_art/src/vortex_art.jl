using Oceananigans
using Oceananigans.Units
using Printf
using CUDA
using Statistics

# -----------------------------------------------------------------------------
# ARTISTIC VORTEX SIMULATION: "The Shoal Wake"
# -----------------------------------------------------------------------------
# Goal: Create a really high-resolution wake behind a shoal-like obstacle
# with a passive "dye" tracer for stunning visual effect.
# -----------------------------------------------------------------------------

# Resolution: Increase these for "wall-poster" quality!
# At 4096x2048, it will be incredibly detailed.
Nx = 2048
Ny = 1024
Lx = 20.0
Ly = 10.0

# Physics
ν = 1e-4  # Kinematic viscosity (lower = more fine-scale turbulence)
κ = 1e-4  # Tracer diffusivity
v_inflow = 0.5 # Flow speed

# Use GPU if available (strongly recommended for this resolution)
arch = has_cuda() ? GPU() : CPU()
@info "Running on $(arch)"

# 1. Create the Grid
grid = RectilinearGrid(arch;
    size=(Nx, Ny),
    extent=(Lx, Ly),
    topology=(Periodic, Bounded, Flat))

# 2. Add an "Artistic Shoal" (Immersed Boundary)
# A cylinder or a Gaussian bump that triggers the eddies
shoal_radius = 0.5
shoal_x(x, y) = (x - Lx / 4)^2 + (y - Ly / 2)^2 < shoal_radius^2
ib_grid = ImmersedBoundaryGrid(grid, GridFittedBoundary(shoal_x))

# 3. Setup the Model
# We'll use a passive tracer 'c' as the "dye"
model = NonhydrostaticModel(ib_grid;
    advection=WENO(),
    closure=ScalarDiffusivity(ν=ν, κ=κ),
    tracers=(:c,))

# 4. Initial Conditions
# Uniform velocity + a "dye" strip upstream
uᵢ(x, y) = v_inflow + 0.01 * randn()
vᵢ(x, y) = 0.01 * randn()
cᵢ(x, y) = exp(-(x - Lx / 5)^2 / 0.1) # A narrow vertical strip of dye

set!(model, u=uᵢ, v=vᵢ, c=cᵢ)

# 5. Simulation
simulation = Simulation(model, Δt=0.002, stop_time=50.0)

# Adaptive time-stepping for stability
conjure_time_step!(simulation, target_cfl=0.7)

# 6. Output (High-resolution fields for the "Art")
u, v, w = model.velocities
c = model.tracers.c
vorticity = Field(@at (Center, Center, Center) (∂x(v) - ∂y(u)))

# We'll save the tracer and vorticity
# Tracer 'c' usually looks like beautiful smoke/ink
# Vorticity 'ζ' shows the structure of the eddies
outputs = (; c, ζ=vorticity)

simulation.output_writers[:art] = NetCDFOutputWriter(model, outputs,
    filename="vortex_art_highres.nc",
    schedule=IterationInterval(20),
    overwrite_existing=true)

# 7. Progress report
progress(sim) = @info @sprintf("Iter: %d, time: %.2f, Δt: %.3e, max|u|: %.2e",
    iteration(sim), time(sim), sim.Δt, maximum(abs, sim.model.velocities.u))

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

@info "Starting the artistic flow simulation..."
run!(simulation)

# -----------------------------------------------------------------------------
# POST-PROCESSING TIP:
# Use the NetCDF file in a tool like Python (matplotlib/cmocean) or 
# Julia (CairoMakie) to render with premium colormaps:
# - 'cmo.balance' or 'cmo.curl' for vorticity
# - 'cmo.dense' or 'cmo.matter' for the dye tracer
# -----------------------------------------------------------------------------
