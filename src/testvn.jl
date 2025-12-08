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

Lx, Ly, Lz = 100e3, 200e3, 50
Nx, Ny, Nz = 30, 30, 10
x, y, z = (0, Lx), (0, Ly), (-Lz, 0)

grid = RectilinearGrid(CPU(); size=(Nx, Ny, Nz), halo=(4, 4, 4),
    x, y, z, topology=(Bounded, Bounded, Bounded))

include("dshoal_vn.jl")

σ = 8.0
Hs = 15.0
x_km, y_km, h = dshoal(Lx / 1e3, Ly / 1e3, σ, Hs, Nx)

ib_grid = ImmersedBoundaryGrid(grid, GridFittedBottom(h))

H = Lz

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
Sₑ(x, y, z) = 35.5

@info "setting up boundary conditions"

tempS = ValueBoundaryCondition(tsbc)
salinS = ValueBoundaryCondition(ssbc)
tempN = ValueBoundaryCondition(tnbc)
salinN = ValueBoundaryCondition(snbc)

T_bcs = FieldBoundaryConditions(south=tempS, north=tempN) #, east=tempE)
S_bcs = FieldBoundaryConditions(south=salinS, north=salinN) #, east=salinE)

# forcings

# north and south sponge

Lₛ = 10e3
Lₑ = 10e3
τₛ = 1days
τₙ = 1days
τₑ = 1days
τ_ts = 1days

params = (; Lx=Lx, Ly=Ly, Ls=Lₛ, Le=Lₑ, τₙ=τₙ, τₛ=τₛ, τₑ=τₑ, τ_ts=τ_ts)

# velocity sigmoid
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
    return 0.10 * sc
end

# x sigmoid for north and south masks
@inline function x_sigmoid(x, p)
    xS = 60e3
    Lw = p.Lx
    k = 40 / Lw
    s = 1 / (1 + exp(k * (x - xS)))
    return 0.1 + 0.9 * s
end

@inline function south_mask(x, y, z, p)
    y0 = 0
    y1 = p.Ls
    if y0 <= y <= y1
        yy = (y - y0) / (y1 - y0)
        return x_sigmoid(x, p) * yy
    else
        return 0.0
    end
end

@inline function north_mask(x, y, z, p)
    y0 = p.Ly - p.Ls
    y1 = p.Ly

    if y0 <= y <= y1
        yy = (y - y0) / (y1 - y0)
        return x_sigmoid(x, p) * yy
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

# eastern mask included, v sponge included

@inline sponge_u(x, y, z, t, u, p) = -min(
    south_mask(x, y, z, p) * u / p.τₛ,
    north_mask(x, y, z, p) * u / p.τₙ,
    east_mask(x, y, z, p) * u / p.τₑ)

@inline sponge_v(x, y, z, t, v, p) = -min(
    south_mask(x, y, z, p) * (v - v∞(x, z, t, p)) / p.τₛ,
    north_mask(x, y, z, p) * (v - v∞(x, z, t, p)) / p.τₙ,
    east_mask(x, y, z, p) * (v - v∞(x, z, t, p)) / p.τₑ)

@inline sponge_w(x, y, z, t, w, p) = -min(
    south_mask(x, y, z, p) * w / p.τₙ,
    north_mask(x, y, z, p) * w / p.τₛ,
    east_mask(x, y, z, p) * w / p.τₑ)

@inline sponge_T(x, y, z, t, T, p) = -min(
    south_mask(x, y, z, p) * (T - tsbc(x, y, z, t)) / p.τ_ts,
    north_mask(x, y, z, p) * (T - tnbc(x, y, z, t)) / p.τ_ts,
    east_mask(x, y, z, p) * (T - Tₑ(x, y, z)) / p.τₑ)

@inline sponge_S(x, y, z, t, S, p) = -min(
    south_mask(x, y, z, p) * (S - ssbc(x, y, z, t)) / p.τ_ts,
    north_mask(x, y, z, p) * (S - snbc(x, y, z, p)) / p.τ_ts,
    east_mask(x, y, z, p) * (S - Sₑ(x, y, z)) / p.τₑ)

# build forcing function

FT = Forcing(sponge_T, field_dependencies=:T, parameters=params)
FS = Forcing(sponge_S, field_dependencies=:S, parameters=params)
Fᵤ = Forcing(sponge_u, field_dependencies=:u, parameters=params)
Fᵥ = Forcing(sponge_v, field_dependencies=:v, parameters=params)
F_w = Forcing(sponge_w, field_dependencies=:w, parameters=params)
forcings = (u=Fᵤ, v=Fᵥ, w=F_w, T=FT, S=FS)
# boundary conditions

open_bc = OpenBoundaryCondition(v∞; scheme=PerturbationAdvection(), parameters=params)

v_bcs = FieldBoundaryConditions(north=open_bc, south=open_bc)
u_bcs = FieldBoundaryConditions()
w_bcs = FieldBoundaryConditions()

bcs = (u=u_bcs, v=v_bcs, w=w_bcs, T=T_bcs, S=S_bcs)


model = NonhydrostaticModel(;
    grid=ib_grid,
    timestepper=:RungeKutta3,
    tracers=(:T, :S),
    buoyancy=SeawaterBuoyancy(equation_of_state=LinearEquationOfState()),
    pressure_solver=(ConjugateGradientPoissonSolver(ib_grid)),
    closure=AnisotropicMinimumDissipation(),
    advection=WENO(order=5),
    coriolis=FPlane(latitude=35.2480),
    boundary_conditions=bcs,
    forcing=forcings)


# make sure T/S profiles are doing their job...
zC = znodes(ib_grid, Center())

cfl_values = Float64[]       # Stores CFL at each step
cfl_times = Float64[]       # Stores model time

simulation = Simulation(model, Δt=5seconds, stop_time=20days)
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

# initial conditions
uᵢ = 0.005 * rand(size(u)...)
vᵢ = 0.005 * rand(size(v)...)
wᵢ = 0.005 * rand(size(w)...)

uᵢ .-= mean(uᵢ)
vᵢ .-= mean(vᵢ)
wᵢ .-= mean(wᵢ)
xv, yv, zv = nodes(v, reshape=true)
vᵢ .+= v∞.(xv, zv, 0.0, Ref(params))

@inline α_lin(y) = clamp(y / Ly, 0.0, 1.0)
@inline blend(a, b, α) = (1 - α) * a + α * b
@inline Tᵢ(x, y, z) = blend(iT_south(z), iT_north(z), α_lin(y))
@inline Sᵢ(x, y, z) = blend(iS_south(z), iS_north(z), α_lin(y))

set!(model, u=uᵢ, v=vᵢ, w=wᵢ, T=Tᵢ, S=Sᵢ)

# set!(model, T=Tᵢ, S=Sᵢ, v=V(0, 0, 0, 0, (; V₂, T₂)))

println(model)              # prints a structured summary

run!(simulation)

# using Oceananigans.Models.NonhydrostaticModels: north_mass_flux, south_mass_flux

# north_flux = north_mass_flux(v)
# south_flux = south_mass_flux(v)
# println("North flux: ", north_flux)
# println("South flux: ", south_flux)

