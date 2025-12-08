using Oceananigans
using Oceananigans.Units
using SeawaterPolynomials: TEOS10EquationOfState
using Oceananigans.Grids: znodes, Center
using NCDatasets
using Statistics
using Oceanostics.ProgressMessengers: SingleLineMessenger
using Oceanostics: ErtelPotentialVorticity, RossbyNumber
using DataFrames

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

# boundary condition setup

V₂ = 0.1
@inline V(x, z, t, p) = p.V₂   # p always holds V₂

H = Lz

open_bc = OpenBoundaryCondition(V; parameters=(; V₂),
    scheme=PerturbationAdvection())

v_bcs = FieldBoundaryConditions(south=open_bc, north=open_bc)


# temperature and salinity boundary conditions

Tₙ(x, z, t) = 20.5
Tₛ(x, z, t) = 24.5
Sₙ(x, z, t) = 32.6
Sₛ(x, z, t) = 35.5

Ls = 10e3
τ = 1hours
params = (; Lx=Lx, Ly=Ly, Ls=Ls, τ=τ, V₂=V₂)

S_bcs = FieldBoundaryConditions(south=ValueBoundaryCondition(Sₛ), north=ValueBoundaryCondition(Sₙ))
T_bcs = FieldBoundaryConditions(south=ValueBoundaryCondition(Tₛ), north=ValueBoundaryCondition(Tₙ))


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

@inline sponge_T(x, y, z, t, T, p) = -(south_mask(x, y, z, p) * (T - Tₛ(x, z, t)) / p.τ + north_mask(x, y, z, p) * (T - Tₙ(x, z, t)) / p.τ)
@inline sponge_S(x, y, z, t, S, p) = -(south_mask(x, y, z, p) * (S - Sₛ(x, z, t)) / p.τ + north_mask(x, y, z, p) * (S - Sₙ(x, z, t)) / p.τ)

# forcings
FT = Forcing(sponge_T, field_dependencies=:T, parameters=params)
FS = Forcing(sponge_S, field_dependencies=:S, parameters=params)
Fᵤ = Forcing(sponge_u, field_dependencies=:u, parameters=params)
Fᵥ = Forcing(sponge_v, field_dependencies=:v, parameters=params)
F_w = Forcing(sponge_w, field_dependencies=:w, parameters=params)
forcings = (u=Fᵤ, v=Fᵥ, w=F_w, T=FT, S=FS)

# @inline ambient_temperature(y, z, t, H) = 12 + 4z / H
# @inline ambient_temperature(x, y, z, t, H) = ambient_temperature(y, z, t, H)
# ambient_temperature_bc = ValueBoundaryCondition(ambient_temperature; parameters=H)
# T_bcs = FieldBoundaryConditions(south=ambient_temperature_bc, north=ambient_temperature_bc)

# ambient_salinity_bc = ValueBoundaryCondition(32)
# S_bcs = FieldBoundaryConditions(south=ambient_salinity_bc, north=ambient_salinity_bc)

model = NonhydrostaticModel(
    grid=ib_grid,
    tracers=(:T, :S),
    buoyancy=SeawaterBuoyancy(),
    closure=AnisotropicMinimumDissipation(),
    advection=WENO(order=5),
    coriolis=FPlane(latitude=35.2480),
    boundary_conditions=(; T=T_bcs, S=S_bcs, v=v_bcs),
    #forcing=forcings
)


Tᵢ(x, y, z) = Tₛ(x, y, z)
Sᵢ(x, y, z) = Sₛ(x, y, z)

uᵢ = 0.005 * rand(size(u)...)
vᵢ = 0.005 * rand(size(v)...)
wᵢ = 0.005 * rand(size(w)...)
uᵢ .-= mean(uᵢ)
vᵢ .-= mean(vᵢ)
wᵢ .-= mean(wᵢ)
vᵢ .+= 0.10

set!(model, T=Tᵢ, S=Sᵢ, v=vᵢ)
cfl_values = Float64[]
cfl_times = Float64[]

simulation = Simulation(model, Δt=5seconds, stop_time=20days)
conjure_time_step_wizard!(simulation, cfl=0.7)

start_time = time_ns() * 1e-9
progress = SingleLineMessenger()
simulation.callbacks[:progress] = Callback(progress, TimeInterval(1days))

prefix = "testperturbation"
saved_output_filename = prefix * ".nc"

overwrite_existing = true

u, v, w = model.velocities
@show model.velocities.v.boundary_conditions
@show minimum(v)
@show maximum(v)
using Oceanostics: ErtelPotentialVorticity

u_c = @at (Center, Center, Center) u
v_c = @at (Center, Center, Center) v
w_c = @at (Center, Center, Center) w
ω_z = Field(∂x(v) - ∂y(u))
Ro = @at (Center, Center, Center) RossbyNumber(model)

outputs = (; u, v, w, u_c, v_c, w_c, ω_z, Ro)

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

using Oceananigans.Models.NonhydrostaticModels: north_mass_flux, south_mass_flux

north_flux = north_mass_flux(v)
south_flux = south_mass_flux(v)
println("North flux: ", north_flux)
println("South flux: ", south_flux)

