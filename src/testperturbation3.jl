using Oceananigans
using Oceananigans.Units
using SeawaterPolynomials: TEOS10EquationOfState
using Oceananigans.Grids: znodes, Center
using NCDatasets
using Statistics
using Oceanostics.ProgressMessengers: SingleLineMessenger
using Oceanostics: ErtelPotentialVorticity, RossbyNumber
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using Oceananigans.BuoyancyFormulations: buoyancy
using DataFrames

Lx, Ly, Lz = 100e3, 200e3, 50
Nx, Ny, Nz = 30, 30, 10
# Nx, Ny, Nz = 60, 60, 10
# Nx, Ny, Nz = 100, 100, 10
x, y, z = (0, Lx), (0, Ly), (-Lz, 0)

grid = RectilinearGrid(CPU(); size=(Nx, Ny, Nz), halo=(4, 4, 4),
    x, y, z, topology=(Bounded, Bounded, Bounded))

include("dshoal_vn.jl")

σ = 8.0
Hs = 15.0
x_km, y_km, h = dshoal(Lx / 1e3, Ly / 1e3, σ, Hs, Nx)

ib_grid = ImmersedBoundaryGrid(grid, GridFittedBottom(h))

# boundary condition setup

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

@inline V(x, y, z, t, p) = v∞(x, z, t, p)
@inline V(x, z, t, p) = v∞(x, z, t, p)


H = Lz

open_bc = OpenBoundaryCondition(V; parameters=(; Lx=Lx),
    scheme=PerturbationAdvection())


# bottom drag and surface wind stresses:

cᴰ = 2.5e-3

@inline drag_u(x, y, t, u, v, p) = -cᴰ * √(u^2 + v^2) * u
@inline drag_v(x, y, t, u, v, p) = -cᴰ * √(u^2 + v^2) * v

drag_bc_u = FluxBoundaryCondition(drag_u, field_dependencies=(:u, :v), parameters=(; cᴰ))
drag_bc_v = FluxBoundaryCondition(drag_v, field_dependencies=(:u, :v), parameters=(; cᴰ))

u_bcs = FieldBoundaryConditions(bottom=drag_bc_u)
v_bcs = FieldBoundaryConditions(south=open_bc, north=open_bc, bottom=drag_bc_v)


# temperature and salinity boundary conditions

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

Ls = 10e3
τ = 12hours
params = (; Lx=Lx, Ly=Ly, Ls=Ls, τ=τ, V₂=V₂)

S_bcs = FieldBoundaryConditions(south=ValueBoundaryCondition(ssbc), north=ValueBoundaryCondition(snbc))
T_bcs = FieldBoundaryConditions(south=ValueBoundaryCondition(tsbc), north=ValueBoundaryCondition(tnbc))


# mask function

@inline function south_mask(x, y, z, p)
    y0 = 0
    y1 = p.Ls
    if y0 <= y <= y1
        yy = (y - y0) / (y1 - y0)
        return yy
    else
        return 0.0
    end
end

@inline function north_mask(x, y, z, p)
    y0 = p.Ly - p.Ls
    y1 = p.Ly
    if y0 <= y <= y1
        yy = (y - y0) / (y1 - y0)
        return yy
    else
        return 0.0
    end
end

# sponge functions
@inline sponge_u(x, y, z, t, u, p) = -(south_mask(x, y, z, p) * u / p.τ + north_mask(x, y, z, p) * u / p.τ)
@inline sponge_v(x, y, z, t, v, p) = -(south_mask(x, y, z, p) * (v - V(x, z, t, p)) / p.τ + north_mask(x, y, z, p) * (v - V(x, z, t, p)) / p.τ)
@inline sponge_w(x, y, z, t, w, p) = -(south_mask(x, y, z, p) * w / p.τ + north_mask(x, y, z, p) * w / p.τ)

@inline sponge_T(x, y, z, t, T, p) = -(south_mask(x, y, z, p) * (T - tsbc(x, z, t)) / p.τ + north_mask(x, y, z, p) * (T - tnbc(x, z, t)) / p.τ)
@inline sponge_S(x, y, z, t, S, p) = -(south_mask(x, y, z, p) * (S - ssbc(x, z, t)) / p.τ + north_mask(x, y, z, p) * (S - snbc(x, z, t)) / p.τ)

# forcings
FT = Forcing(sponge_T, field_dependencies=:T, parameters=params)
FS = Forcing(sponge_S, field_dependencies=:S, parameters=params)
Fᵤ = Forcing(sponge_u, field_dependencies=:u, parameters=params)
Fᵥ = Forcing(sponge_v, field_dependencies=:v, parameters=params)
F_w = Forcing(sponge_w, field_dependencies=:w, parameters=params)
forcings = (u=Fᵤ, v=Fᵥ, w=F_w, T=FT, S=FS)


model = NonhydrostaticModel(
    grid=ib_grid,
    tracers=(:T, :S),
    buoyancy=SeawaterBuoyancy(),
    closure=AnisotropicMinimumDissipation(),
    pressure_solver=ConjugateGradientPoissonSolver(ib_grid),
    advection=WENO(order=5),
    coriolis=FPlane(latitude=35.2480),
    boundary_conditions=(; T=T_bcs, S=S_bcs, v=v_bcs),
    forcing=forcings
)

u, v, w = model.velocities
T = model.tracers.T
S = model.tracers.S
b = buoyancy(model)

@inline α_lin(y) = clamp(y / Ly, 0.0, 1.0)
@inline blend(a, b, α) = (1 - α) * a + α * b
@inline Tᵢ(x, y, z) = blend(iT_south(z), iT_north(z), α_lin(y))
@inline Sᵢ(x, y, z) = blend(iS_south(z), iS_north(z), α_lin(y))

uᵢ = 0.005 * rand(size(u)...)
vᵢ = 0.005 * rand(size(v)...)
wᵢ = 0.005 * rand(size(w)...)
uᵢ .-= mean(uᵢ)
vᵢ .-= mean(vᵢ)
wᵢ .-= mean(wᵢ)
uᵢ .+= 0
xv, yv, zv = nodes(v, reshape=true)
vᵢ .+= v∞.(xv, zv, 0, Ref(params))

set!(model, T=Tᵢ, S=Sᵢ, u=uᵢ, v=vᵢ, w=wᵢ)




# DONT CHANGE ANYTHING BELOW THIS LINE

cfl_values = Float64[]
cfl_times = Float64[]

simulation = Simulation(model, Δt=5seconds, stop_time=100days)
conjure_time_step_wizard!(simulation, cfl=0.9)

start_time = time_ns() * 1e-9
progress = SingleLineMessenger()
simulation.callbacks[:progress] = Callback(progress, TimeInterval(1days))

prefix = "testperturbation3"
saved_output_filename = prefix * ".nc"

overwrite_existing = true

u_c = @at (Center, Center, Center) u
v_c = @at (Center, Center, Center) v
w_c = @at (Center, Center, Center) w
ω_z = Field(∂x(v) - ∂y(u))
ζ = @at (Center, Center, Center) ω_z
Ro = @at (Center, Center, Center) RossbyNumber(model)
PV = @at (Center, Center, Center) ErtelPotentialVorticity(model, u, v, w, b, model.coriolis)
outputs = (; u, v, w, T, S, u_c, v_c, w_c, ω_z, ζ, PV, Ro)

simulation.output_writers[:fields] = NetCDFWriter(model, outputs,
    schedule=TimeInterval(86400seconds), # (86400seconds) (43200seconds)
    filename=saved_output_filename,
    overwrite_existing=true)

cfl = CFL(simulation.Δt)

simulation.callbacks[:cfl_recorder] = Callback(TimeInterval(10minutes)) do sim
    push!(cfl_values, cfl(sim.model))
    push!(cfl_times, sim.model.clock.time)
end

println(model)

run!(simulation)

# using Oceananigans.Models.NonhydrostaticModels: north_mass_flux, south_mass_flux

# north_flux = north_mass_flux(v)
# south_flux = south_mass_flux(v)
# println("North flux: ", north_flux)
# println("South flux: ", south_flux)

