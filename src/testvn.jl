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

params = (; Lx = Lx, Ly = Ly, Ls = Lₛ, τ_v = τ_v, τ_ts = τ_ts)

grid = RectilinearGrid(CPU(); size=(Nx, Ny, Nz), halo=(4, 4, 4),
                       x, y, z, topology=(Bounded, Bounded, Bounded))


include("dshoal_vn.jl")

σ = 8.0       
Hs = 15.0      
x_km, y_km, h = dshoal(Lx/1e3, Ly/1e3, σ, Hs, Nx)

ib_grid = ImmersedBoundaryGrid(grid, GridFittedBottom(h))
# wedge(x, y) = -H *(1 + (y + abs(x)) / δ)
# grid = ImmersedBoundaryGrid(grid, GridFittedBottom(wedge))

V₂ = 0.1 # m/s

@inline V(x, y, z, t, p) = p.V₂
@inline V(x, z, t, p) = V(x, zero(x), z, t, p)
@inline V₀(x, y, z, t) = 0.0
@inline V₀(x, z, t) = V₀(x, zero(x), z, t)

H = Lz

open_bc = OpenBoundaryCondition(V; scheme = PerturbationAdvection(inflow_timescale = τ_v, outflow_timescale = τ_v),
                                parameters=(; V₂))

v_bcs = FieldBoundaryConditions(south = open_bc, north = open_bc)

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

Tᵢ(x, y, z) = 23.11
Sᵢ(x, y, z) = 36.4

@info "setting up boundary conditions"

# south_temp = OpenBoundaryCondition(tsbc; scheme = PerturbationAdvection())
# north_temp = OpenBoundaryCondition(tnbc; scheme = PerturbationAdvection())

# south_salin = OpenBoundaryCondition(ssbc; scheme = PerturbationAdvection())
# north_salin = OpenBoundaryCondition(snbc; scheme = PerturbationAdvection())

# T_bcs = FieldBoundaryConditions(south = south_temp, north = north_temp)
# S_bcs = FieldBoundaryConditions(south = south_salin, north = north_salin)

tempS = ValueBoundaryCondition(tsbc)
salinS = ValueBoundaryCondition(ssbc)
tempN = ValueBoundaryCondition(tnbc)
salinN = ValueBoundaryCondition(snbc)

tempE = ValueBoundaryCondition(Tᵢ)
salinE = ValueBoundaryCondition(Sᵢ)

T_bcs = FieldBoundaryConditions(south = tempS, north = tempN, east = tempE)
S_bcs = FieldBoundaryConditions(south = salinS, north = salinN, east = salinE)

# create same nudgings from periodic case:

@inline function south_mask(x, y, z, p)
    y0 = 0
    y1 = p.Ls
    if y0 <= y <= y1
        return 1 - y/y1
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
    x0 = p.Lx - p.Ls
    x1 = p.Lx

    if x0 <= x <= x1
        return (x - x0) / (x1 - x0)
    else
        return 0.0
    end
end

# sponge functions
@inline sponge_u(x, y, z, t, u, p) = -(south_mask(x, y, z, p) + east_nudge(x, y, z, p)) * u / p.τ 
@inline sponge_v(x, y, z, t, v, p) = -(south_mask(x, y, z, p) + east_nudge(x, y, z, p)) * (v - v∞(x, z, t)) / p.τ 
@inline sponge_w(x, y, z, t, w, p) = -(south_mask(x, y, z, p) + east_nudge(x, y, z, p)) * w / p.τ

# temperature and salinity: we need to nudge south to B2 mooring, and north to B1 mooring. 
@inline sponge_T(x, y, z, t, T, p) =
    -south_mask(x, y, z, p) / p.τ_ts * (T - tsbc(x, y, z, t)) -
    north_mask(x, y, z, p) / p.τ_ts * (T - tnbc(x, y, z, t)) -
    east_mask(x, y, z, p) / p.τ_ts * (T - Tₑ(x, y, z))
@inline sponge_S(x, y, z, t, S, p) =
    -south_mask(x, y, z, p) / p.τ_ts * (S - ssbc(x, y, z, t)) -
    north_mask(x, y, z, p) / p.τ_ts * (S - snbc(x, y, z, t)) -
    east_mask(x, y, z, p) / p.τ_ts * (S - Sₑ(x, y, z))

# add forcings
FT = Forcing(sponge_T, field_dependencies = :T, parameters = params)
FS = Forcing(sponge_S, field_dependencies = :S, parameters = params)
Fᵤ = Forcing(sponge_u, field_dependencies = :u, parameters = params)
Fᵥ = Forcing(sponge_v, field_dependencies = :v, parameters = params)
F_w = Forcing(sponge_w, field_dependencies = :w, parameters = params)

forcings = (u = Fᵤ, v = Fᵥ, w = F_w, T = FT, S = FS)


model = NonhydrostaticModel(; grid = ib_grid, tracers = (:T, :S),
                              buoyancy = SeawaterBuoyancy(equation_of_state = TEOS10EquationOfState()),
                              pressure_solver = ConjugateGradientPoissonSolver(ib_grid),
                              closure = AnisotropicMinimumDissipation(),
                              advection = WENO(order=5), coriolis = FPlane(latitude=35.2480),
                              boundary_conditions = (; T=T_bcs, v = v_bcs, S = S_bcs),
                              forcing = forcings)

# check bcs...
# @show model.velocities.v.boundary_conditions
# @show model.tracers.T.boundary_conditions
# @show model.velocities.u.boundary_conditions
# @show model.tracers.T.boundary_conditions
# @show model.tracers.S.boundary_conditions
# need to blend temperature/salinity profiles from B1 and B2 for the initial conditions.

# @inline α_lin(y) = clamp(y / Ly, 0.0, 1.0)
# @inline blend(a, b, α) = (1 - α) * a + α * b
# @inline Tᵢ(x, y, z) = blend(iT_south(z), iT_north(z), α_lin(y))
# @inline Sᵢ(x, y, z) = blend(iS_south(z), iS_north(z), α_lin(y))


# make sure gradient works...
# @show extrema(model.tracers.T), extrema(model.tracers.S)

# make sure T/S profiles are doing their job...
zC = znodes(ib_grid, Center())

cfl_values = Float64[]       # Stores CFL at each step
cfl_times  = Float64[]       # Stores model time

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
# ζ = ∂x(v) - ∂y(u)

using Oceanostics: ErtelPotentialVorticity
# using Oceananigans.BuoyancyFormulations: buoyancy
# q = Field(ErtelPotentialVorticity(model, model.velocities..., buoyancy(model), model.coriolis))

u_c = @at (Center, Center, Center) u
v_c = @at (Center, Center, Center) v
w_c = @at (Center, Center, Center) w

ω_z = Field(∂x(v) - ∂y(u))
PV = @at (Center, Center, Center) ErtelPotentialVorticity(model, u, v, w, buoyancy(model), model.coriolis)
Ro = @at (Center, Center, Center) RossbyNumber(model)

outputs = (; u, v, w, u_c, v_c, w_c, ω_z, PV, Ro, T, S)

simulation.output_writers[:fields] = NetCDFWriter(model, outputs, 
                                                        schedule = TimeInterval(86400seconds), # (86400seconds) (43200seconds)
                                                        filename = saved_output_filename,
                                                        overwrite_existing = true)

cfl = CFL(simulation.Δt)



simulation.callbacks[:cfl_recorder] = Callback(TimeInterval(10minutes)) do sim
    push!(cfl_values, cfl(sim.model))
    push!(cfl_times, sim.model.clock.time)
end

# using Statistics: mean
# uᵢ = 0.005 * rand(size(u)...)
# vᵢ = 0.005 * rand(size(v)...)
# wᵢ = 0.005 * rand(size(w)...)

# uᵢ .-= mean(uᵢ)
# vᵢ .-= mean(vᵢ)
# wᵢ .-= mean(wᵢ)

# vᵢ .+= 0.10

# initial conditions for temperature and salinity based on cast 77?
# salinity avg = 36.4, temperature avg = 23.11

# set!(model, T = Tᵢ, S = Sᵢ, u = uᵢ, v = vᵢ, w = wᵢ) #, v=V(0, 0, 0, 0, (; V₂)))
set!(model, T = Tᵢ, S = Sᵢ) #, v = V(0, 0, 0, 0, (; V₂)))

# # check velocity
# zC = znodes(ib_grid, Center())
# ktop = length(zC)
# vs_surf = [model.velocities.v[i, 1, ktop] for i in 1:Nx]

# @show mean(vs_surf), extrema(vs_surf)
# @show extrema(model.tracers.T), extrema(model.tracers.S)

println(model)              # prints a structured summary

run!(simulation)
