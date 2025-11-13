using Oceananigans
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

include("dshoal_vn.jl")

@info "building domain"

# setup domain
Lx = 100e3
Ly = 400e3
Lz = 50.0
x, y, z = (0, Lx), (0, Ly), (-Lz, 0)

# Nx, Ny, Nz = 100, 100, 24 # do this one if you have the time.
# Nx, Ny, Nz = 60, 60, 20
Nx, Ny, Nz = 30, 30, 10

# Plan: closed boundary conditions on east and west, periodic flow from south to north.

grid = RectilinearGrid(CPU(); size=(Nx, Ny, Nz), halo=(4, 4, 4),
                       x, y, z, topology=(Bounded, Periodic, Bounded))

# bathymetry parameters
σ = 8.0         # [km] Gaussian width for shoal cross-section
Hs = 15.0       # [m] shoal height

# bathymetry
x_km, y_km, h = dshoal(Lx/1e3, Ly/1e3, σ, Hs, Nx)
@show size(h) == (Nx, Ny)

# plt = surface(x_km, y_km, h)
# display(plt)


# If x_km and y_km are 1D, create a 2D meshgrid
xgrid = repeat(x_km', length(y_km), 1)  # (Ny, Nx)
ygrid = repeat(y_km, 1, length(x_km))   # (Ny, Nx)

# Grid with immersed bathymetry
ib_grid = ImmersedBoundaryGrid(grid, GridFittedBottom(h))

# temperature and salinity profiles from B1 and B2 moorings

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

# turn 3 argument to 4 argument (model seems to want this)
# CHANGE: make original functions 4 argument from the start (then we can remove this)
@inline tsbc(x, y, z, t) = tsbc(x, z, t)
@inline tnbc(x, y, z, t) = tnbc(x, z, t)
@inline ssbc(x, y, z, t) = ssbc(x, z, t)
@inline snbc(x, y, z, t) = snbc(x, z, t)

Tᵢ(x, y, z) = 23.11
Sᵢ(x, y, z) = 36.4


@info "setting up boundary conditions"

# prescribed inflow (we want 0.10 m/s flow from south to north i.e. positive)

v∞(x, z, t) = 0.10

# north and south sponges 

Lₛ = 50e3
τ = 6hours
τ_ts = 6days
τ_e = 6days
# params = (; Lx = Lx, Ls = Lₛ, τ = τ, τ_ts = τ_ts)
params = (; Lx = Lx, Ly = Ly, Ls = Lₛ, τ = τ, τ_ts = τ_ts, τ_e = τ_e)

# linear masks 0 at interior 1 at boundary for north and south. 
# extent of this mask should be from y = 0 (south boundary) to y = Lₛ = 20 km 
@inline function south_mask(x, y, z, p)
    y0 = 0
    y1 = p.Ls
    if y0 <= y <= y1
        return 1 - y/y1
    else
        return 0.0
    end
end

# extent of this mask should be from Ly - Ls to y = Ly ? 
@inline function north_mask(x, y, z, p) 
    y0 = p.Ly - p.Ls
    y1 = p.Ly

    if y0 <= y <= y1
        return (y - y0) / (y1 - y0)
    else
        return 0.0
    end
end

# @inline function east_mask(x, y, z, p)
#     x0 = p.Lx - p.Ls
#     x1 = p.Lx

#     if x0 <= x <= x1
#         return (x - x0) / (x1 - x0)
#     else
#         return 0.0 
#     end
# end

# add sigmoid taper past x = 60 km

@inline function east_mask(x, y, z, p)
    x0 = 60e3
    Lw = p.Lx
    k = 8 / Lw
    s = 1 / (1 + exp(-k * (x - x0)))
    return clamp(s, 1.0, 0.0)
end


# sponge functions
@inline sponge_u(x, y, z, t, u, p) = -(south_mask(x, y, z, p) + east_mask(x, y, z, p)) * u / p.τ 
@inline sponge_v(x, y, z, t, v, p) = -(south_mask(x, y, z, p) + east_mask(x, y, z, p)) * (v - v∞(x, z, t)) / p.τ 
@inline sponge_w(x, y, z, t, w, p) = -(south_mask(x, y, z, p) + east_mask(x, y, z, p)) * w / p.τ

# temperature and salinity: we need to nudge south to B2 mooring, and north to B1 mooring. 
@inline sponge_T(x, y, z, t, T, p) =
    -south_mask(x, y, z, p) / p.τ_ts * (T - tsbc(x, y, z, t)) -
    north_mask(x, y, z, p) / p.τ_ts * (T - tnbc(x, y, z, t)) -
    east_mask(x, y, z, p) / p.τ_e * (T - Tᵢ(x, y, z))
@inline sponge_S(x, y, z, t, S, p) =
    -south_mask(x, y, z, p) / p.τ_ts * (S - ssbc(x, y, z, t)) -
    north_mask(x, y, z, p) / p.τ_ts * (S - snbc(x, y, z, t)) -
    east_mask(x, y, z, p) / p.τ_e * (S - Sᵢ(x, y, z))

# add forcings
FT = Forcing(sponge_T, field_dependencies = :T, parameters = params)
FS = Forcing(sponge_S, field_dependencies = :S, parameters = params)
Fᵤ = Forcing(sponge_u, field_dependencies = :u, parameters = params)
Fᵥ = Forcing(sponge_v, field_dependencies = :v, parameters = params)
F_w = Forcing(sponge_w, field_dependencies = :w, parameters = params)

forcings = (u = Fᵤ, v = Fᵥ, w = F_w, T = FT, S = FS)
# forcings = (T = FT, S = FS)

T_bcs = FieldBoundaryConditions()
S_bcs = FieldBoundaryConditions()

u_bcs = FieldBoundaryConditions()
v_bcs = FieldBoundaryConditions()
w_bcs = FieldBoundaryConditions()

bcs = (u = u_bcs, v = v_bcs, w = w_bcs, T = T_bcs, S = S_bcs)

model = NonhydrostaticModel(
    grid = ib_grid,
    timestepper = :RungeKutta3,
    advection = WENO(order=5),
    closure = AnisotropicMinimumDissipation(),
    pressure_solver = ConjugateGradientPoissonSolver(ib_grid),
    tracers = (:T, :S),
    buoyancy = SeawaterBuoyancy(),
    coriolis = FPlane(latitude = 35.2480),
    boundary_conditions = bcs,
    forcing = forcings
)

# output stuff
@info "creating output fields"

overwrite_existing = true

cfl_values = Float64[]       # Stores CFL at each step
cfl_times  = Float64[]       # Stores model time

simulation = Simulation(model, Δt=15minutes, stop_time=100days)
# conjure_time_step_wizard!(simulation, cfl=0.8,
#                           diffusive_cfl = 0.9,
#                           max_change = 5.0,
#                           min_Δt = 2seconds,
#                           max_Δt = 3hours
#                           )

conjure_time_step_wizard!(simulation, cfl = 0.7, diffusive_cfl = 0.8)

start_time = time_ns() * 1e-9
progress = SingleLineMessenger()
simulation.callbacks[:progress] = Callback(progress, TimeInterval(86400seconds))

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
saved_output_prefix = "vnshoals_ts_extended"

saved_output_filename = saved_output_prefix * ".nc"
checkpointer_prefix = "checkpoint_" * saved_output_prefix

# NETCDF writers

simulation.output_writers[:fields] = NetCDFWriter(model, outputs, 
                                                        schedule = TimeInterval(86400seconds),
                                                        filename = saved_output_filename,
                                                        overwrite_existing = overwrite_existing)

ccc_scratch = Field{Center, Center, Center}(model.grid) 

# CFL logger
cfl = CFL(simulation.Δt)


simulation.callbacks[:cfl_recorder] = Callback(TimeInterval(10minutes)) do sim
    push!(cfl_values, cfl(sim.model))
    push!(cfl_times, sim.model.clock.time)
end

println(model)              # prints a structured summary


# impose initial conditions
uᵢ = 0.005 * rand(size(u)...)
vᵢ = 0.005 * rand(size(v)...)
wᵢ = 0.005 * rand(size(w)...)

uᵢ .-= mean(uᵢ)
vᵢ .-= mean(vᵢ)
wᵢ .-= mean(wᵢ)
vᵢ .+= 0.10

# Tᵢ(x, y, z) = iT_south(z)
# Sᵢ(x, y, z) = iS_south(z)

# @inline α_lin(y) = clamp(y / Ly, 0.0, 1.0)
# @inline blend(a, b, α) = (1 - α) * a + α * b
# @inline Tᵢ(x, y, z) = blend(iT_south(z), iT_north(z), α_lin(y))
# @inline Sᵢ(x, y, z) = blend(iS_south(z), iS_north(z), α_lin(y))

# # test initial condition clamp...
# using Oceananigans.Grids: xnodes, ynodes, znodes, Center
# xs = xnodes(ib_grid, Center()); ys = ynodes(ib_grid, Center()); zs = znodes(ib_grid, Center())
# xmid, ymid, z0 = xs[div(end,2)], ys[div(end,2)], zs[end]

# Tx1 = Tᵢ(xs[1], ymid, z0);    Tx2 = Tᵢ(xs[end], ymid, z0)   # vary x only
# Ty1 = Tᵢ(xmid, ys[1], z0);    Ty2 = Tᵢ(xmid, ys[end], z0)   # vary y only
# @show Tx1 Tx2 Ty1 Ty2  (Tx2-Tx1)  (Ty2-Ty1)

set!(model, u = uᵢ, v = vᵢ, w = wᵢ, T = Tᵢ, S = Sᵢ)
# set!(model; u=uᵢ, v=vᵢ, w=wᵢ)

# diagnostics before running...

dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz


@info "time to run simulation!"
# run simulation
run!(simulation)

