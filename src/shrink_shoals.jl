using Oceananigans
using Oceananigans.Grids: Periodic, Bounded, minimum_zspacing
using Oceananigans.Units
using Oceananigans.BoundaryConditions: OpenBoundaryCondition, FieldBoundaryConditions
using Oceananigans.TurbulenceClosures
using Oceananigans.Solvers: ConjugateGradientPoissonSolver, FFTBasedPoissonSolver
using Oceananigans.Models: buoyancy_operation
using Oceananigans.OutputWriters
using Oceananigans.Forcings
using Statistics: mean
using Oceanostics: RossbyNumber, ErtelPotentialVorticity,
    KineticEnergy, KineticEnergyDissipationRate, TurbulentKineticEnergy,
    XShearProductionRate, YShearProductionRate, ZShearProductionRate
using Oceanostics.ProgressMessengers: TimedMessenger
using SeawaterPolynomials.TEOS10
using Printf: @sprintf
using NCDatasets
using DataFrames
using CUDA: has_cuda_gpu, allowscalar

# build
@info "building domain"

# switches
LES = true
periodic_y = true
gradient_IC = false
sigmoid_v_bc = true
sigmoid_ic = true
is_coriolis = true
shoal_bath = true
if has_cuda_gpu()
    arch = GPU()
else
    arch = CPU()
end
@info "architecture = $(arch)"
include("dshoal_vn_param_shrink.jl")

# simulation knobs
run_number = 5 # <-- change this for each new run
sim_runtime = 6hours
callback_interval = 10minutes
run_tag = "shrink_test$(run_number)"  # e.g. "shrink_test9999"

if LES
    params = (; Lx=1000, Ly=2000, Lz=50, Nx=200, Ny=400, Nz=50)
else
    params = (; Lx=100000, Ly=200000, Lz=50, Nx=30, Ny=30, Nz=10)
end
if arch == CPU()
    params = (; params..., Nx=25, Ny=50, Nz=10) # keep the same for now
elseif !LES
    params = (; params..., Nx=200, Ny=400, Nz=50)
end

x, y, z = (0, params.Lx), (0, params.Ly), (-params.Lz, 0)

# grid  

if periodic_y
    grid = RectilinearGrid(arch; size=(params.Nx, params.Ny, params.Nz), halo=(4, 4, 4), x, y, z, topology=(Bounded, Periodic, Bounded))
else
    grid = RectilinearGrid(arch; size=(params.Nx, params.Ny, params.Nz), halo=(4, 4, 4), x, y, z, topology=(Bounded, Bounded, Bounded))
end

# model parameters

if shoal_bath
    # Define shoal parameters (align with the new sigmoidal setup)
    Hs = 5.0         # Height of shoal above -25m shelf
    sigma = LES ? 80.0 : 8e3       # Gaussian width of shoal (half crossover)
    shoal_length = LES ? 200.0 : 20e3 # Horizontal span of the shoal ridge

    scale_factor = params.Lx / 100e3
    slope_bottom = dshoal_param_bottom(params.Ly; Hs=Hs, sigma=sigma, shoal_length=shoal_length, scale=scale_factor)
    GFB = GridFittedBottom(slope_bottom)
    ib_grid = ImmersedBoundaryGrid(grid, GFB)
else
    ib_grid = grid
end

@info ib_grid
v₀ = 0.1

params = (; params...,
    v₀=v₀,
    Ls=LES ? 100.0 : 10e3,
    Le=LES ? 400.0 : 40e3,
    x_off=LES ? 600.0 : 60e3,
    σ_off=LES ? 200.0 : 20e3,
    τₙ=6hours,
    τₛ=24hours,
    τₑ=24hours,
    τ_ts=24hours)



# Logarithmic boundary layer drag formulation
Rz = 2.5e-6
z₀ = Rz * params.Lz # roughness length
z₁ = (params.Lz / params.Nz) / 2 # distance to first cell center
κᵛᵏ = 0.4 # von Karman constant
cᴰ = (κᵛᵏ / log(z₁ / z₀))^2
@info "Calculated logarithmic boundary drag Cᴰ =" cᴰ
# bottom drag (z-boundary): signature (x, y, t, field_deps..., params)
@inline drag_u(x, y, t, u, v, cᴰ) = -cᴰ * u * sqrt(u^2 + v^2)
@inline drag_v(x, y, t, u, v, cᴰ) = -cᴰ * v * sqrt(u^2 + v^2)
# immersed drag (immersed boundary): signature (x, y, z, t, field_deps..., params)
@inline immersed_drag_u(x, y, z, t, u, v, cᴰ) = -cᴰ * u * sqrt(u^2 + v^2)
@inline immersed_drag_v(x, y, z, t, u, v, cᴰ) = -cᴰ * v * sqrt(u^2 + v^2)
drag_bc_u = FluxBoundaryCondition(drag_u, field_dependencies=(:u, :v), parameters=cᴰ)
drag_bc_v = FluxBoundaryCondition(drag_v, field_dependencies=(:u, :v), parameters=cᴰ)
immersed_drag_bc_u = FluxBoundaryCondition(immersed_drag_u, field_dependencies=(:u, :v), parameters=cᴰ)
immersed_drag_bc_v = FluxBoundaryCondition(immersed_drag_v, field_dependencies=(:u, :v), parameters=cᴰ)



# velocity function
if sigmoid_v_bc
    @inline function v∞(x, z, t, p)
        xC = p.Lx == 1000 ? 30.0 : 3e3
        xS = p.Lx == 1000 ? 600.0 : 60e3
        Lw = p.Lx
        k1 = 80 / Lw
        k2 = 40 / Lw

        s1 = 1 / (1 + exp(-k1 * (x - xC)))
        s2 = 1 / (1 + exp(k2 * (x - xS)))
        s = (s1 - 1) + s2
        sc = clamp(s, 0.0, 1.0)
        return p.v₀ * sc
    end
else
    @inline function v∞(x, z, t, p)
        return p.v₀
    end
end

forcings = NamedTuple()

if periodic_y
    u_bcs = FieldBoundaryConditions(immersed=immersed_drag_bc_u)
    v_bcs = FieldBoundaryConditions(immersed=immersed_drag_bc_v)
    w_bcs = FieldBoundaryConditions()
else
    open_bc = OpenBoundaryCondition(v∞; parameters=params, scheme=PerturbationAdvection())
    open_zero = OpenBoundaryCondition(0.0)
    u_bcs = FieldBoundaryConditions(immersed=immersed_drag_bc_u)
    v_bcs = FieldBoundaryConditions(immersed=immersed_drag_bc_v, north=open_bc, south=open_bc)
    w_bcs = FieldBoundaryConditions()
end

bcs = (u=u_bcs, v=v_bcs, w=w_bcs)

if is_coriolis
    coriolis = FPlane(latitude=35.2480)
else
    coriolis = nothing
end

if periodic_y
    model = NonhydrostaticModel(ib_grid;
        timestepper=:RungeKutta3,
        advection=WENO(order=5),
        closure=LES ? DynamicSmagorinsky() : AnisotropicMinimumDissipation(),
        coriolis=coriolis,
        pressure_solver=ConjugateGradientPoissonSolver(ib_grid),
        boundary_conditions=bcs
    )
else
    model = NonhydrostaticModel(ib_grid;
        timestepper=:RungeKutta3,
        advection=WENO(order=5),
        closure=LES ? DynamicSmagorinsky() : AnisotropicMinimumDissipation(),
        coriolis=coriolis,
        pressure_solver=ConjugateGradientPoissonSolver(ib_grid),
        boundary_conditions=bcs
    )
end

@info "" model

# output

@info "creating output fields"

overwrite_existing = true

cfl_values = Float64[]       # Stores CFL at each step
cfl_times = Float64[]       # Stores model time

simulation = Simulation(model, Δt=1.0, stop_time=sim_runtime)

conjure_time_step_wizard!(simulation, cfl=0.7, diffusive_cfl=0.7)

progress = TimedMessenger()

simulation.callbacks[:progress] = Callback(progress, TimeInterval(callback_interval))

function print_solver_iterations(sim)
    solver = sim.model.pressure_solver
    if hasproperty(solver, :conjugate_gradient_solver)
        cg = solver.conjugate_gradient_solver
        @info @sprintf("Pressure solver: %d CG iterations (t = %.2f hours)",
            cg.iteration, time(sim) / 3600)
    end
end
simulation.callbacks[:solver_iters] = Callback(print_solver_iterations, TimeInterval(callback_interval))

u, v, w = model.velocities

KE = @at (Center, Center, Center) KineticEnergy(model)


u_c = @at (Center, Center, Center) u
v_c = @at (Center, Center, Center) v
w_c = @at (Center, Center, Center) w


uu = Field(u_c * u_c)
vv = Field(v_c * v_c)
ww = Field(w_c * w_c)
slice_fields = (; u_c, v_c, w_c, KE)
tavg_fields = (; u_c, v_c, w_c, uu, vv, ww)


simulation.output_writers[:surface_slice] = NetCDFWriter(model, slice_fields,
    filename="top_$(run_tag).nc",
    schedule=TimeInterval(callback_interval),
    indices=(:, :, params.Nz),
    overwrite_existing=overwrite_existing)

simulation.output_writers[:midy_slice] = NetCDFWriter(model, slice_fields,
    filename="midy_$(run_tag).nc",
    schedule=TimeInterval(callback_interval),
    indices=(:, round(Int, params.Ny / 2), :),
    overwrite_existing=overwrite_existing)


# simulation.output_writers[:time_avg_3d] = NetCDFWriter(model, tavg_fields,
#     filename="time_avg_3d_$(run_tag).nc",
#     schedule=AveragedTimeInterval(1hours, window=1hours),
#     overwrite_existing=overwrite_existing)



@info "Setting initial conditions"
# Simple function-based initial conditions
uᵢ(x, y, z) = 0.0
vᵢ(x, y, z) = v∞(x, z, 0, params)
wᵢ(x, y, z) = 0.0

set!(model, u=uᵢ, v=vᵢ, w=wᵢ)

# run simulation
@info """
════════════════════════════════════════════════════════
 SIMULATION CONFIGURATION: $(run_tag)
════════════════════════════════════════════════════════
 Run number:      $(run_number)
 Runtime:         $(sim_runtime)
 Architecture:    $(arch)

 ── Switches ──
 LES:             $(LES)
 periodic_y:      $(periodic_y)
 gradient_IC:     $(gradient_IC)
 sigmoid_v_bc:    $(sigmoid_v_bc)
 sigmoid_ic:      $(sigmoid_ic)
 is_coriolis:     $(is_coriolis)
 shoal_bath:      $(shoal_bath)
 ════════════════════════════════════════════════════════
"""
run!(simulation)
