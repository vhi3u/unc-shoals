# testing the TKEDissipationVerticalDiffusivity closure
using Oceananigans
using Oceananigans.Units
using Oceanostics.ProgressMessengers: TimedMessenger
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
import Oceananigans.TurbulenceClosures: cell_diffusion_timescale

cell_diffusion_timescale(closure::CATKEVerticalDiffusivity, diffusivities, grid, clock, fields) = Inf
cell_diffusion_timescale(closure::TKEDissipationVerticalDiffusivity, diffusivities, grid, clock, fields) = Inf

arch = CPU()
x, y, z = (0, 100e3), (0, 200e3), (-50, 0)
grid = RectilinearGrid(arch; size=(50, 100, 10), halo=(4, 4, 4), x, y, z, topology=(Bounded, Periodic, Bounded))

bottom(x, y) = -50 + 20 * exp(-((x - 50e3)^2 + (y - 100e3)^2) / 20e3^2)
ib_grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom))

closure = TKEDissipationVerticalDiffusivity(ExplicitTimeDiscretization())

# add drag
cᴰ = 2.5e-4
drag_bc = BulkDrag(coefficient=cᴰ)
u_bcs = FieldBoundaryConditions(immersed=drag_bc)
v_bcs = FieldBoundaryConditions(immersed=drag_bc)
w_bcs = FieldBoundaryConditions(immersed=drag_bc)


model = NonhydrostaticModel(ib_grid,
    closure=closure,
    tracers=(:b, :e, :ϵ),
    pressure_solver=ConjugateGradientPoissonSolver(ib_grid, maxiter=1000),
    buoyancy=BuoyancyTracer(),
    #buoyancy=SeawaterBuoyancy(),
    timestepper=:QuasiAdamsBashforth2,
    boundary_conditions=(; u=u_bcs, v=v_bcs, w=w_bcs)
)

set!(model, u=0.1, v=0.2, w=0.01, b=1e-3, e=1e-3, ϵ=1e-5)

@info "Model successfully instantiated:" model

simulation = Simulation(model, Δt=1.0, stop_time=1days)
conjure_time_step_wizard!(simulation, cfl=0.5, diffusive_cfl=0.5)

progress = TimedMessenger()
simulation.callbacks[:progress] = Callback(progress, TimeInterval(1hours))

@info "Running simulation for 1 day..."
run!(simulation)
@info "Simulation completed successfully! Final Δt = $(simulation.Δt)"
