using Oceananigans
using Oceananigans.Units
using SeawaterPolynomials: TEOS10EquationOfState
using Oceananigans.Grids: znodes, Center
using Oceananigans.BoundaryConditions: PerturbationAdvection, OpenBoundaryCondition, ValueBoundaryCondition
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using NCDatasets
using Statistics
using Oceanostics.ProgressMessengers: SingleLineMessenger
using Oceanostics: ErtelPotentialVorticity, RossbyNumber
using Oceananigans.BuoyancyFormulations: buoyancy
using DataFrames

@info "building domain"

# parameters for boundary condition timescales (use later)
τ_v = 6hours
τ_ts = 6hours
Lₛ = 10e3

Lx, Ly, Lz = 100e3, 200e3, 50
Nx, Ny, Nz = 30, 30, 10
x, y, z = (0, Lx), (0, Ly), (-Lz, 0)

params = (; Lx=Lx, Ly=Ly, Ls=Lₛ, τ_v=τ_v, τ_ts=τ_ts)

grid = RectilinearGrid(CPU(); size=(Nx, Ny, Nz), halo=(4, 4, 4),
    x, y, z, topology=(Bounded, Bounded, Bounded))

include("dshoal_vn.jl")

σ = 8.0
Hs = 15.0
x_km, y_km, h = dshoal(Lx / 1e3, Ly / 1e3, σ, Hs, Nx)

ib_grid = ImmersedBoundaryGrid(grid, GridFittedBottom(h))

V₂ = 0.1 # m/s
T₂ = 12.421hours

@inline V(x, y, z, t, p) = p.V₂ * sin(2π * t / p.T₂)
@inline V(x, z, t, p) = V(x, zero(x), z, t, p)

H = Lz

open_bc = OpenBoundaryCondition(V; scheme=PerturbationAdvection(inflow_timescale=τ_v, outflow_timescale=τ_v),
    parameters=(; V₂, T₂))

v_bcs = FieldBoundaryConditions(south=open_bc, north=open_bc)

@info "loading B1 and B2 T/S profiles"
using CSV, Interpolations
ctd = CSV.read("src/ctd_smoothed.csv", DataFrame)
z_data = collect(ctd.z)
T_south = collect(ctd.T_B2)
T_north = collect(ctd.T_B1)
S_south = collect(ctd.S_B2)
S_north = collect(ctd.S_B1)

iT_south = extrapolate(interpolate((z_data,), T_south, Gridded(Linear())), Interpolations.Flat())
iT_north = extrapolate(interpolate((z_data,), T_north, Gridded(Linear())), Interpolations.Flat())
iS_south = extrapolate(interpolate((z_data,), S_south, Gridded(Linear())), Interpolations.Flat())
iS_north = extrapolate(interpolate((z_data,), S_north, Gridded(Linear())), Interpolations.Flat())

@inline tsbc(x, z, t) = iT_south(z)
@inline tnbc(x, z, t) = iT_north(z)
@inline ssbc(x, z, t) = iS_south(z)
@inline snbc(x, z, t) = iS_north(z)

@inline tsbc(x, y, z, t) = tsbc(x, z, t)
@inline tnbc(x, y, z, t) = tnbc(x, z, t)
@inline ssbc(x, y, z, t) = ssbc(x, z, t)
@inline snbc(x, y, z, t) = snbc(x, z, t)

Tₑ(x, y, z) = 23.11
Sₑ(x, y, z) = 36.4


@info "setting up boundary conditions"

tempS = ValueBoundaryCondition(tsbc)
salinS = ValueBoundaryCondition(ssbc)
tempN = ValueBoundaryCondition(tnbc)
salinN = ValueBoundaryCondition(snbc)

T_bcs = FieldBoundaryConditions(south=tempS, north=tempN) #, east=tempE)
S_bcs = FieldBoundaryConditions(south=salinS, north=salinN) #, east=salinE)


model = NonhydrostaticModel(; grid=ib_grid, tracers=(:T, :S),
    buoyancy=SeawaterBuoyancy(equation_of_state=LinearEquationOfState()),
    #pressure_solver=(),
    closure=AnisotropicMinimumDissipation(),
    advection=WENO(order=5), coriolis=FPlane(latitude=35.2480),
    boundary_conditions=(; T=T_bcs, v=v_bcs, S=S_bcs))


# make sure T/S profiles are doing their job...
zC = znodes(ib_grid, Center())

cfl_values = Float64[]       # Stores CFL at each step
cfl_times = Float64[]       # Stores model time

simulation = Simulation(model, Δt=5seconds, stop_time=100days)
conjure_time_step_wizard!(simulation, cfl=0.7)

start_time = time_ns() * 1e-9
progress = SingleLineMessenger()
simulation.callbacks[:progress] = Callback(progress, TimeInterval(1days))

prefix = "testvn"
saved_output_filename = prefix * ".nc"

overwrite_existing = true

@info "creating output fields"

u, v, w = model.velocities
T = model.tracers.T
S = model.tracers.S

using Oceanostics: ErtelPotentialVorticity

u_c = @at (Center, Center, Center) u
v_c = @at (Center, Center, Center) v
w_c = @at (Center, Center, Center) w

ω_z = Field(∂x(v) - ∂y(u))
PV = @at (Center, Center, Center) ErtelPotentialVorticity(model, u, v, w, buoyancy(model), model.coriolis)
Ro = @at (Center, Center, Center) RossbyNumber(model)

outputs = (; u, v, w, u_c, v_c, w_c, ω_z, PV, Ro, T, S)

simulation.output_writers[:fields] = NetCDFWriter(model, outputs,
    schedule=TimeInterval(86400seconds), # (86400seconds) (43200seconds)
    filename=saved_output_filename,
    overwrite_existing=true)

cfl = CFL(simulation.Δt)

simulation.callbacks[:cfl_recorder] = Callback(TimeInterval(10minutes)) do sim
    push!(cfl_values, cfl(sim.model))
    push!(cfl_times, sim.model.clock.time)
end

@inline α_lin(y) = clamp(y / Ly, 0.0, 1.0)
@inline blend(a, b, α) = (1 - α) * a + α * b
@inline Tᵢ(x, y, z) = blend(iT_south(z), iT_north(z), α_lin(y))
@inline Sᵢ(x, y, z) = blend(iS_south(z), iS_north(z), α_lin(y))

using Plots
# verify boundary conditions
zc = znodes(ib_grid, Center())

T_south_model = [iT_south(z) for z in zc]
T_north_model = [iT_north(z) for z in zc]
S_south_model = [iS_south(z) for z in zc]
S_north_model = [iS_north(z) for z in zc]

p1 = plot(T_north_model, zc;
    lw=2, label="B1 north",
    xlabel="Temperature [°C]",
    ylabel="z [m]",
    title="Temperature profiles",
    grid=true)
plot!(p1, T_south_model, zc;
    lw=2, label="B2 south")

# ---- Salinity panel ----
p2 = plot(S_north_model, zc;
    lw=2, label="B1 north",
    xlabel="Salinity [psu]",
    ylabel="",
    title="Salinity profiles",
    grid=true)
plot!(p2, S_south_model, zc;
    lw=2, label="B2 south")

# ---- Combine panels ----
plt = plot(p1, p2; layout=(1, 2), size=(600, 300))
display(plt)

set!(model, T=Tᵢ, S=Sᵢ)

println(model)              # prints a structured summary

run!(simulation)

using Oceananigans.Models.NonhydrostaticModels: north_mass_flux, south_mass_flux

north_flux = north_mass_flux(v)
south_flux = south_mass_flux(v)
println("North flux: ", north_flux)
println("South flux: ", south_flux)

