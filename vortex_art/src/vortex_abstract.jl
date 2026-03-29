using Oceananigans
using Oceananigans.Units
using Printf
using CUDA
using Statistics
using NCDatasets

# -----------------------------------------------------------------------------
# ARTISTIC VORTEX SIMULATION: "The Oceanic Bickley Jet"
# -----------------------------------------------------------------------------
# Goal: Create a beautiful, oceanographically realistic meander and 
# ring-shedding event (like the Gulf Stream) on a Beta-plane.
# -----------------------------------------------------------------------------

# Resolution - increase this for "limitless detail"!
# 2048x2048 is crisp, 4096x4096 is wall-poster quality.
grid_resolution = 2048

# Simulation parameters
L = 2π
ν = 2e-7 # Lower viscosity = finer eddies (needs high resolution!)

# Use GPU if available
arch = has_cuda() ? GPU() : CPU()
@info "Running on $(arch)"

# 1. Create the Grid (Periodic in X, Bounded in Y for a "Channel")
grid = RectilinearGrid(arch;
    size=(grid_resolution, grid_resolution),
    x=(0, L), y=(-L / 2, L / 2),
    topology=(Periodic, Bounded, Flat))

# 2. Add Coriolis with Beta Effect (f = f₀ + βy)
# This creates Rossby waves and "realistic" oceanic eddies
coriolis = BetaPlane(f₀=1e-4, β=1e-11)

# 3. Setup the Model
# Passive tracer 'c' will show the meanders
model = NonhydrostaticModel(grid;
    advection=WENO(),
    coriolis=coriolis,
    closure=ScalarDiffusivity(ν=ν, κ=ν / 10),
    tracers=(:c,))

# 4. Initial Conditions: The "Bickley Jet"
# A strong central current that naturally breaks into rings
U = 1.0  # Speed
width = 0.2 # Width
u_jet(x, y) = U * sech(y / width)^2
u_noise(x, y) = 0.1 * randn()
v_noise(x, y) = 0.1 * randn()
uᵢ(x, y) = u_jet(x, y) + u_noise(x, y)
vᵢ(x, y) = v_noise(x, y)

# Tracer Front: Half the domain is "cold/fresh" water (dye)
cᵢ(x, y) = y > 0 ? 1.0 : 0.0

set!(model, u=uᵢ, v=vᵢ, c=cᵢ)

# 5. Simulation
simulation = Simulation(model, Δt=0.001, stop_time=150.0)

# Adaptive time-stepping
conjure_time_step_wizard!(simulation, cfl=0.9)

# 6. Output (High-resolution fields for the "Art")
u, v, w = model.velocities
c = model.tracers.c
vorticity = Field(@at (Center, Center, Center) (∂x(v) - ∂y(u)))

simulation.output_writers[:art] = NetCDFWriter(model, (; ζ=vorticity, c=c),
    filename="vortex_abstract_highres.nc",
    schedule=TimeInterval(10.0),
    overwrite_existing=true)

# 6. Progress report
progress(sim) = @info @sprintf("Iter: %d, time: %.2f, Δt: %.3e",
    iteration(sim), time(sim), sim.Δt)

simulation.callbacks[:progress] = Callback(progress, TimeInterval(10.0))

@info "Starting the abstract vortex simulation..."
run!(simulation)

# -----------------------------------------------------------------------------
# VISUALIZATION RECOMMENDATION:
# Use vorticity for structural detail, and dye for "visceral" mixing.
# -----------------------------------------------------------------------------
