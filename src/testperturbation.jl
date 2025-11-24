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

T₂ = 12.421hours
V₂ = 0.1
@inline V(x, y, z, t, p) = p.V₂ * sin(2π * t / p.T₂)
@inline V(x, z, t, p) = V(x, zero(x), z, t, p)

# @inline U(x, y, z, t, p) = p.U₂
# @inline U(y, z, t, p) = U(zero(y), y, z, t, p)

H = Lz

open_bc = OpenBoundaryCondition(V; parameters=(; V₂, T₂),
    scheme=PerturbationAdvection())

v_bcs = FieldBoundaryConditions(south=open_bc, north=open_bc)

@inline ambient_temperature(y, z, t, H) = 12 + 4z / H
@inline ambient_temperature(x, y, z, t, H) = ambient_temperature(y, z, t, H)
ambient_temperature_bc = ValueBoundaryCondition(ambient_temperature; parameters=H)
T_bcs = FieldBoundaryConditions(south=ambient_temperature_bc, north=ambient_temperature_bc)

ambient_salinity_bc = ValueBoundaryCondition(32)
S_bcs = FieldBoundaryConditions(south=ambient_salinity_bc, north=ambient_salinity_bc)

# u_bcs = FieldBoundaryConditions(west = open_bc, east = open_bc)

model = NonhydrostaticModel(; grid=ib_grid,
    tracers=(:T, :S),
    buoyancy=SeawaterBuoyancy(),
    closure=AnisotropicMinimumDissipation(),
    advection=WENO(order=5),
    coriolis=FPlane(latitude=0),
    boundary_conditions=(; T=T_bcs, S=S_bcs, v=v_bcs))

Tᵢ(x, y, z) = ambient_temperature(x, y, z, 0, H)

set!(model, T=Tᵢ, S=32, v=V(0, 0, 0, 0, (; V₂, T₂)))

cfl_values = Float64[]
cfl_times = Float64[]

simulation = Simulation(model, Δt=5seconds, stop_time=100days)
conjure_time_step_wizard!(simulation, cfl=0.7)

start_time = time_ns() * 1e-9
progress = SingleLineMessenger()
simulation.callbacks[:progress] = Callback(progress, TimeInterval(1days))

prefix = "testperturbation"
saved_output_filename = prefix * ".nc"

overwrite_existing = true

u, v, w = model.velocities
@show model.velocities.v.boundary_conditions
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

