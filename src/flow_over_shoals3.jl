using Oceananigans
using Oceananigans.Grids: Periodic, Bounded
using Oceananigans.Units
using Oceananigans.BoundaryConditions: OpenBoundaryCondition, FieldBoundaryConditions, FluxBoundaryCondition
using Oceananigans.TurbulenceClosures
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using Oceananigans.BuoyancyFormulations: buoyancy
using Oceananigans.OutputWriters
using Oceananigans.Forcings
using Statistics: mean
using Oceananigans.Diagnostics: CFL
using Oceanostics: RossbyNumber, ErtelPotentialVorticity
using SeawaterPolynomials.TEOS10
using Oceanostics.ProgressMessengers: SingleLineMessenger
using NCDatasets
using DataFrames
# using CUDA: has_cuda_gpu, @allowscalar, CuArray

# include shoal function
include("dshoal_vn.jl")

# add run directory [TBD]

# build
@info "building domain"

# switches
LES = true
mass_flux = true
periodic_y = false
gradient_IC = true
sigmoid_v_bc = true
sigmoid_ic = true
is_coriolis = true
shoal_bath = true
# if has_cuda_gpu()
#     arch = GPU()
# else
#     arch = CPU()
# end
arch = CPU()

# simulation knobs
sim_runtime = 50days
callback_interval = 86400seconds

if LES
    params = (; Lx=100e3, Ly=200e3, Lz=50, Nx=30, Ny=30, Nz=10)
else
    params = (; Lx=100000, Ly=200000, Lz=50, Nx=30, Ny=30, Nz=10)
end
if arch == CPU()
    params = (; params..., Nx=30, Ny=30, Nz=10) # keep the same for now
end

x, y, z = (0, params.Lx), (0, params.Ly), (-params.Lz, 0)

# grid  

if periodic_y
    grid = RectilinearGrid(CPU(); size=(params.Nx, params.Ny, params.Nz), halo=(4, 4, 4), x, y, z, topology=(Bounded, Periodic, Bounded))
else
    grid = RectilinearGrid(CPU(); size=(params.Nx, params.Ny, params.Nz), halo=(4, 4, 4), x, y, z, topology=(Bounded, Bounded, Bounded))
end

# model parameters

# bathymetry
σ = 8.0         # [km] Gaussian width for shoal cross-section
Hs = 15.0       # [m] shoal height
if shoal_bath
    x_km, y_km, h = dshoal(params.Lx / 1e3, params.Ly / 1e3, σ, Hs, params.Nx) # feed grid into shoal function
    ib_grid = ImmersedBoundaryGrid(grid, GridFittedBottom(h)) # immersed boundary grid
else
    ib_grid = grid
end

if mass_flux
    v₀ = 0.10 # [m/s] mean flow velocity
else
    v₀ = 0.0
end

# store parameters for sponge setup
params = (; params...,
    Ls=20e3, # sponge layer size (north and south)
    Le=60e3, # sponge layer size (east)
    τₙ=6hours, # relaxation timescale for north sponge
    τₛ=6hours, # relaxation timescale for south sponge
    τₑ=24hours, # relaxation timescale for east sponge
    τ_ts=6hours) # relaxation timescale for temperature and salinity at the north and south boundaries

# temperature and salinity profiles for north and south boundaries
@info "loading B1 and B2 T/S profiles"
using CSV
using Interpolations: extrapolate, interpolate, Gridded, Flat, Linear
if LES
    ctd = CSV.read("src/ctd_smoothed.csv", DataFrame)
    z_data = collect(ctd.z)
    T_south = collect(ctd.T_B2)
    T_north = collect(ctd.T_B1)
    S_south = collect(ctd.S_B2)
    S_north = collect(ctd.S_B1)
    iT_south = extrapolate(interpolate((z_data,), T_south, Gridded(Linear())), Flat()) # Gridded and Flat are imported from Interpolations
    iT_north = extrapolate(interpolate((z_data,), T_north, Gridded(Linear())), Flat())
    iS_south = extrapolate(interpolate((z_data,), S_south, Gridded(Linear())), Flat())
    iS_north = extrapolate(interpolate((z_data,), S_north, Gridded(Linear())), Flat())
    @inline tsbc(x, z, t) = iT_south(z)
    @inline tnbc(x, z, t) = iT_north(z)
    @inline ssbc(x, z, t) = iS_south(z)
    @inline snbc(x, z, t) = iS_north(z)
    @inline tsbc(x, y, z, t) = tsbc(x, z, t) # wrapper function for temperature at south boundary
    @inline tnbc(x, y, z, t) = tnbc(x, z, t) # wrapper function for temperature at north boundary
    @inline ssbc(x, y, z, t) = ssbc(x, z, t) # wrapper function for salinity at south boundary
    @inline snbc(x, y, z, t) = snbc(x, z, t) # wrapper function for salinity at north boundary
end

# temperature and salinity for eastern boundary (constant)
if LES
    Tₑ(x, y, z) = 23.11
    Sₑ(x, y, z) = 35.5
end

# bottom drag parameters
cᴰ = 2.5e-3 # dimensionless drag coefficient
if LES
    @inline drag_u(x, y, t, u, v, p) = -cᴰ * √(u^2 + v^2) * u
    @inline drag_v(x, y, t, u, v, p) = -cᴰ * √(u^2 + v^2) * v
    drag_bc_u = FluxBoundaryCondition(drag_u, field_dependencies=(:u, :v), parameters=(; cᴰ=cᴰ,))
    drag_bc_v = FluxBoundaryCondition(drag_v, field_dependencies=(:v, :v), parameters=(; cᴰ=cᴰ,))
end

# mask functions
@inline function south_mask(x, y, z, p)
    y0 = 0
    y1 = p.Ls
    if y0 <= y <= y1
        return 1 - y / y1
    else
        return 0.0
    end
end

@inline function north_mask(x, y, z, p)
    y0 = p.Ly - p.Ls
    y1 = p.Ly

    if y0 <= y <= y1
        return (y - y0) / (y1 - y0)
    else
        return 0.0
    end
end

@inline function east_mask(x, y, z, p)
    x0 = p.Lx - p.Le
    x1 = p.Lx

    if x0 <= x <= x1
        return (x - x0) / (x1 - x0)
    else
        return 0.0
    end
end

# velocity function
if sigmoid_v_bc
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
        return v₀ * sc
    end
else
    @inline function v∞(x, z, t, p)
        return v₀
    end
end

# sponge functions
if mass_flux
    if periodic_y
        @inline sponge_u(x, y, z, t, u, p) = -min(
            south_mask(x, y, z, p) * u / params.τₛ,
            north_mask(x, y, z, p) * u / params.τₙ,
            east_mask(x, y, z, p) * u / params.τₑ)

        @inline sponge_v(x, y, z, t, v, p) = -min(
            south_mask(x, y, z, p) * (v - v∞(x, z, t, p)) / params.τₛ,
            north_mask(x, y, z, p) * (v - v∞(x, z, t, p)) / params.τₙ,
            east_mask(x, y, z, p) * (v - v∞(x, z, t, p)) / params.τₑ)

        @inline sponge_w(x, y, z, t, w, p) = -min(
            south_mask(x, y, z, p) * w / params.τₙ,
            north_mask(x, y, z, p) * w / params.τₛ,
            east_mask(x, y, z, p) * w / params.τₑ)

        @inline sponge_T(x, y, z, t, T, p) = -min(
            south_mask(x, y, z, p) * (T - tsbc(x, y, z, t)) / params.τₛ,
            north_mask(x, y, z, p) * (T - tnbc(x, y, z, t)) / params.τₛ,
            east_mask(x, y, z, p) * (T - Tₑ(x, y, z)) / params.τₑ)

        @inline sponge_S(x, y, z, t, S, p) = -min(
            south_mask(x, y, z, p) * (S - ssbc(x, y, z, t)) / params.τₛ,
            north_mask(x, y, z, p) * (S - snbc(x, y, z, t)) / params.τₙ,
            east_mask(x, y, z, p) * (S - Sₑ(x, y, z)) / params.τₑ)
    else
        @inline V(x, y, z, t, p) = v∞(x, z, t, p)
        @inline V(x, z, t, p) = v∞(x, z, t, p)
        @inline sponge_u(x, y, z, t, u, p) = -(south_mask(x, y, z, p) * u / params.τₛ + north_mask(x, y, z, p) * u / params.τₙ)
        @inline sponge_v(x, y, z, t, v, p) = -(south_mask(x, y, z, p) * (v - V(x, z, t, p)) / params.τₛ + north_mask(x, y, z, p) * (v - V(x, z, t, p)) / params.τₙ)
        @inline sponge_w(x, y, z, t, w, p) = -(south_mask(x, y, z, p) * w / params.τₛ + north_mask(x, y, z, p) * w / params.τₙ)
        @inline sponge_T(x, y, z, t, T, p) = -(south_mask(x, y, z, p) * (T - tsbc(x, z, t)) / params.τₛ + north_mask(x, y, z, p) * (T - tnbc(x, z, t)) / params.τₛ)
        @inline sponge_S(x, y, z, t, S, p) = -(south_mask(x, y, z, p) * (S - ssbc(x, z, t)) / params.τₛ + north_mask(x, y, z, p) * (S - snbc(x, z, t)) / params.τₙ)
    end
end

# forcing functions
FT = Forcing(sponge_T, field_dependencies=:T, parameters=params)
FS = Forcing(sponge_S, field_dependencies=:S, parameters=params)
if mass_flux
    Fᵤ = Forcing(sponge_u, field_dependencies=:u, parameters=params)
    Fᵥ = Forcing(sponge_v, field_dependencies=:v, parameters=params)
    F_w = Forcing(sponge_w, field_dependencies=:w, parameters=params)
    forcings = (u=Fᵤ, v=Fᵥ, w=F_w, T=FT, S=FS)
else
    forcings = (T=FT, S=FS)
end

if periodic_y
    T_bcs = FieldBoundaryConditions()
    S_bcs = FieldBoundaryConditions()
    u_bcs = FieldBoundaryConditions(bottom=drag_bc_u)
    v_bcs = FieldBoundaryConditions(bottom=drag_bc_v)
    w_bcs = FieldBoundaryConditions()
else
    open_bc = OpenBoundaryCondition(V; parameters=params, scheme=PerturbationAdvection())
    open_zero = OpenBoundaryCondition(0.0)
    T_bcs = FieldBoundaryConditions(south=ValueBoundaryCondition(tsbc), north=ValueBoundaryCondition(tnbc))
    S_bcs = FieldBoundaryConditions(south=ValueBoundaryCondition(ssbc), north=ValueBoundaryCondition(snbc))
    u_bcs = FieldBoundaryConditions(bottom=drag_bc_u, east=open_zero)
    v_bcs = FieldBoundaryConditions(bottom=drag_bc_v, north=open_bc, south=open_bc, east=open_zero)
    w_bcs = FieldBoundaryConditions()
end

bcs = (u=u_bcs, v=v_bcs, w=w_bcs, T=T_bcs, S=S_bcs)
if is_coriolis
    coriolis = FPlane(latitude=35.2480)
else
    coriolis = nothing
end

if periodic_y
    model = NonhydrostaticModel(
        grid=ib_grid,
        timestepper=:RungeKutta3,
        advection=WENO(order=5),
        closure=AnisotropicMinimumDissipation(),
        pressure_solver=ConjugateGradientPoissonSolver(ib_grid),
        tracers=(:T, :S),
        buoyancy=SeawaterBuoyancy(equation_of_state=TEOS10EquationOfState()),
        coriolis=coriolis,
        boundary_conditions=bcs,
        forcing=forcings
    )
else
    model = NonhydrostaticModel(
        grid=ib_grid,
        timestepper=:RungeKutta3,
        advection=WENO(order=5),
        closure=AnisotropicMinimumDissipation(),
        pressure_solver=ConjugateGradientPoissonSolver(ib_grid),
        tracers=(:T, :S),
        buoyancy=SeawaterBuoyancy(),
        coriolis=coriolis,
        boundary_conditions=bcs,
        forcing=forcings
    )
end

# output

@info "creating output fields"

overwrite_existing = true

cfl_values = Float64[]       # Stores CFL at each step
cfl_times = Float64[]       # Stores model time

simulation = Simulation(model, Δt=15minutes, stop_time=sim_runtime)

conjure_time_step_wizard!(simulation, cfl=0.9, diffusive_cfl=0.8)

start_time = time_ns() * 1e-9
progress = SingleLineMessenger()
simulation.callbacks[:progress] = Callback(progress, TimeInterval(callback_interval))

u, v, w = model.velocities
T = model.tracers.T
S = model.tracers.S
b = buoyancy(model)

u_c = @at (Center, Center, Center) u
v_c = @at (Center, Center, Center) v
w_c = @at (Center, Center, Center) w

# vorticity, potential vorticity, rossby number calculations
ω_z = Field(∂x(v_c) - ∂y(u_c))
ξ = @at (Center, Center, Center) ω_z

PV = @at (Center, Center, Center) ErtelPotentialVorticity(model, u, v, w, b, model.coriolis)
Ro = @at (Center, Center, Center) RossbyNumber(model)

outputs = (; u, v, w, T, S, u_c, v_c, w_c, ω_z, ξ, PV, Ro)
if periodic_y
    saved_output_filename = "periodic_over_shoals3.nc"
else
    saved_output_filename = "bounded_over_shoals3.nc"
end

# checkpointer_prefix = "checkpoint_" * saved_output_filename

# NETCDF writers

simulation.output_writers[:fields] = NetCDFWriter(model, outputs,
    schedule=TimeInterval(callback_interval),
    filename=saved_output_filename,
    overwrite_existing=overwrite_existing)

ccc_scratch = Field{Center,Center,Center}(model.grid)

# CFL logger
cfl = CFL(simulation.Δt)

simulation.callbacks[:cfl_recorder] = Callback(TimeInterval(10minutes)) do sim
    push!(cfl_values, cfl(sim.model))
    push!(cfl_times, sim.model.clock.time)
end

# initial conditions
uᵢ = 0.005 * rand(size(u)...)
vᵢ = 0.005 * rand(size(v)...)
wᵢ = 0.005 * rand(size(w)...)
uᵢ .-= mean(uᵢ)
vᵢ .-= mean(vᵢ)
wᵢ .-= mean(wᵢ)
uᵢ .+= 0
if sigmoid_ic
    xv, yv, zv = nodes(v, reshape=true)
    vᵢ .+= v∞.(xv, zv, 0, Ref(params))
else
    vᵢ .+= v₀
end

if gradient_IC
    @inline α_lin(y) = clamp(y / params.Ly, 0.0, 1.0)
    @inline blend(a, b, α) = (1 - α) * a + α * b
    @inline Tᵢ(x, y, z) = blend(iT_south(z), iT_north(z), α_lin(y))
    @inline Sᵢ(x, y, z) = blend(iS_south(z), iS_north(z), α_lin(y))
else
    @inline Tᵢ(x, y, z) = iT_south(z)
    @inline Sᵢ(x, y, z) = iS_south(z)
end

set!(model, u=uᵢ, v=vᵢ, w=wᵢ, T=Tᵢ, S=Sᵢ)

# run simulation
@info "time to run simulation!"
run!(simulation)