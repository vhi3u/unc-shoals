# using Pkg
# Pkg.instantiate() # Only need to do this once when you started the repo in another machine
# Pkg.resolve()
# import Pkg;
# Pkg.add("Oceananigans");
# Pkg.add("NCDatasets");
# Pkg.add("DataFrames");
# Pkg.add("Interpolations");
# Pkg.add("CUDA");
# Pkg.add("Oceanostics");
# Pkg.add("CSV");
# Pkg.add("Statistics");
# Pkg.add("SeawaterPolynomials");
# import Pkg;
# # Pkg.add("Rasters");
# Pkg.instantiate() # Only need to do this once when you started the repo in another machine
# Pkg.resolve()

using Oceananigans
using Oceananigans.Grids: Periodic, Bounded
using Oceananigans.Units
using Oceananigans.BoundaryConditions: OpenBoundaryCondition, FieldBoundaryConditions, FluxBoundaryCondition
using Oceananigans.TurbulenceClosures
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using Oceananigans.Models: buoyancy_operation
using Oceananigans.OutputWriters
using Oceananigans.Forcings
using Statistics: mean
using Oceananigans.Diagnostics: CFL
using Oceanostics: RossbyNumber, ErtelPotentialVorticity
using SeawaterPolynomials.TEOS10
using Oceanostics.ProgressMessengers: SingleLineMessenger
using NCDatasets
using DataFrames
using CUDA: has_cuda_gpu

# include shoal function
include("dshoal_vn.jl")
# include("dshoal_sigmoid_vn.jl")

# add run directory [TBD]

# build
@info "building domain"

# switches
LES = true
mass_flux = true
periodic_y = true
gradient_IC = true
sigmoid_v_bc = true
sigmoid_ic = true
is_coriolis = true
shoal_bath = true
if has_cuda_gpu()
    arch = GPU()
else
    arch = CPU()
end
# arch = CPU()

# simulation knobs
sim_runtime = 100days
callback_interval = 86400seconds

if LES
    params = (; Lx=100e3, Ly=300e3, Lz=50, Nx=30, Ny=30, Nz=10)
else
    params = (; Lx=100000, Ly=300000, Lz=50, Nx=30, Ny=30, Nz=10)
end
if arch == CPU()
    params = (; params..., Nx=30, Ny=30, Nz=10) # keep the same for now
else
    params = (; params..., Nx=60, Ny=60, Nz=20)
end

x, y, z = (0, params.Lx), (0, params.Ly), (-params.Lz, 0)

# grid  

if periodic_y
    grid = RectilinearGrid(arch; size=(params.Nx, params.Ny, params.Nz), halo=(4, 4, 4), x, y, z, topology=(Bounded, Periodic, Bounded))
else
    grid = RectilinearGrid(arch; size=(params.Nx, params.Ny, params.Nz), halo=(4, 4, 4), x, y, z, topology=(Bounded, Bounded, Bounded))
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
    v₀=v₀, # add v₀ to params for GPU compatibility
    Ls=50e3, # sponge layer size (north and south)
    Le=60e3, # sponge layer size (east)
    τₙ=6hours, # relaxation timescale for north sponge
    τₛ=6hours, # relaxation timescale for south sponge
    τₑ=24hours, # relaxation timescale for east sponge
    τ_ts=3hours) # relaxation timescale for temperature and salinity at the north and south boundaries

# GPU-compatible SMOOTH piecewise linear T/S profiles (from CTD data)
# B1 = North, B2 = South
# Uses tanh blending for smooth transitions (equivalent to MATLAB smoothdata)
# δ = smoothing length scale (meters), set to ~2.5m for 5m effective smoothing

const δ_smooth = 2.5  # smoothing length scale in meters

# Smooth transition function: 0 when z >> z0, 1 when z << z0
@inline smooth_step(z, z0) = 0.5 * (1.0 - tanh((z - z0) / δ_smooth))

# Temperature at North boundary (B1) - SMOOTHED
@inline function T_north_pwl(z)
    z1, z2, z3 = -5.0, -15.0, -35.0
    v1, v2, v3 = 20.5389, 17.8875, 14.3323
    m12 = (v2 - v1) / (z2 - z1)
    m23 = (v3 - v2) / (z3 - z2)
    # Piecewise linear values
    val1 = v1
    val2 = v1 + m12 * (z - z1)
    val3 = v2 + m23 * (z - z2)
    val4 = v3
    # Smooth blending between segments
    w1 = smooth_step(z, z1)  # 0 above z1, 1 below z1
    w2 = smooth_step(z, z2)  # 0 above z2, 1 below z2
    w3 = smooth_step(z, z3)  # 0 above z3, 1 below z3
    # Blend: start with val1, transition to val2 at z1, to val3 at z2, to val4 at z3
    return val1 * (1 - w1) + val2 * (w1 - w2) + val3 * (w2 - w3) + val4 * w3
end

# Temperature at South boundary (B2) - SMOOTHED
@inline function T_south_pwl(z)
    z1, z2, z3 = -5.0, -15.0, -30.0
    v1, v2, v3 = 24.5378, 24.3073, 23.4116
    m12 = (v2 - v1) / (z2 - z1)
    m23 = (v3 - v2) / (z3 - z2)
    val1 = v1
    val2 = v1 + m12 * (z - z1)
    val3 = v2 + m23 * (z - z2)
    val4 = v3
    w1 = smooth_step(z, z1)
    w2 = smooth_step(z, z2)
    w3 = smooth_step(z, z3)
    return val1 * (1 - w1) + val2 * (w1 - w2) + val3 * (w2 - w3) + val4 * w3
end

# Salinity at North boundary (B1) - SMOOTHED
@inline function S_north_pwl(z)
    z1, z2, z3 = -5.0, -15.0, -35.0
    v1, v2, v3 = 32.6264, 33.7062, 33.2648
    m12 = (v2 - v1) / (z2 - z1)
    m23 = (v3 - v2) / (z3 - z2)
    val1 = v1
    val2 = v1 + m12 * (z - z1)
    val3 = v2 + m23 * (z - z2)
    val4 = v3
    w1 = smooth_step(z, z1)
    w2 = smooth_step(z, z2)
    w3 = smooth_step(z, z3)
    return val1 * (1 - w1) + val2 * (w1 - w2) + val3 * (w2 - w3) + val4 * w3
end

# Salinity at South boundary (B2) - SMOOTHED
@inline function S_south_pwl(z)
    z1, z2, z3 = -5.0, -15.0, -30.0
    v1, v2, v3 = 35.5830, 35.9986, 36.1776
    m12 = (v2 - v1) / (z2 - z1)
    m23 = (v3 - v2) / (z3 - z2)
    val1 = v1
    val2 = v1 + m12 * (z - z1)
    val3 = v2 + m23 * (z - z2)
    val4 = v3
    w1 = smooth_step(z, z1)
    w2 = smooth_step(z, z2)
    w3 = smooth_step(z, z3)
    return val1 * (1 - w1) + val2 * (w1 - w2) + val3 * (w2 - w3) + val4 * w3
end

# Eastern boundary T/S constants
Tₑ_val = 23.11
Sₑ_val = 35.5
params = (; params..., Tₑ=Tₑ_val, Sₑ=Sₑ_val)

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
        return p.v₀ * sc
    end
else
    @inline function v∞(x, z, t, p)
        return p.v₀
    end
end

# sponge functions
if mass_flux
    if periodic_y
        @inline sponge_u(x, y, z, t, u, p) = -min(
            south_mask(x, y, z, p) * u / p.τₛ,
            north_mask(x, y, z, p) * u / p.τₙ,
            east_mask(x, y, z, p) * u / p.τₑ)

        @inline sponge_v(x, y, z, t, v, p) = -min(
            south_mask(x, y, z, p) * (v - v∞(x, z, t, p)) / p.τₛ,
            north_mask(x, y, z, p) * (v - v∞(x, z, t, p)) / p.τₙ,
            east_mask(x, y, z, p) * v / p.τₑ)

        @inline sponge_w(x, y, z, t, w, p) = -min(
            south_mask(x, y, z, p) * w / p.τₙ,
            north_mask(x, y, z, p) * w / p.τₛ,
            east_mask(x, y, z, p) * w / p.τₑ)

        @inline sponge_T(x, y, z, t, T, p) = -min(
            south_mask(x, y, z, p) * (T - T_south_pwl(z)) / p.τₛ,
            north_mask(x, y, z, p) * (T - T_north_pwl(z)) / p.τₛ,
            east_mask(x, y, z, p) * (T - p.Tₑ) / p.τₑ)

        @inline sponge_S(x, y, z, t, S, p) = -min(
            south_mask(x, y, z, p) * (S - S_south_pwl(z)) / p.τₛ,
            north_mask(x, y, z, p) * (S - S_north_pwl(z)) / p.τₙ,
            east_mask(x, y, z, p) * (S - p.Sₑ) / p.τₑ)
    else
        @inline V(x, y, z, t, p) = v∞(x, z, t, p)
        @inline V(x, z, t, p) = v∞(x, z, t, p)
        @inline sponge_u(x, y, z, t, u, p) = -(south_mask(x, y, z, p) * u / p.τₛ + north_mask(x, y, z, p) * u / p.τₙ)
        @inline sponge_v(x, y, z, t, v, p) = -(south_mask(x, y, z, p) * (v - V(x, z, t, p)) / p.τₛ + north_mask(x, y, z, p) * (v - V(x, z, t, p)) / p.τₙ)
        @inline sponge_w(x, y, z, t, w, p) = -(south_mask(x, y, z, p) * w / p.τₛ + north_mask(x, y, z, p) * w / p.τₙ)
        @inline sponge_T(x, y, z, t, T, p) = -(south_mask(x, y, z, p) * (T - T_south_pwl(z)) / p.τₛ + north_mask(x, y, z, p) * (T - T_north_pwl(z)) / p.τₛ)
        @inline sponge_S(x, y, z, t, S, p) = -(south_mask(x, y, z, p) * (S - S_south_pwl(z)) / p.τₛ + north_mask(x, y, z, p) * (S - S_north_pwl(z)) / p.τₙ)
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
    model = NonhydrostaticModel(ib_grid;
        #grid=ib_grid,
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
    model = NonhydrostaticModel(ib_grid;
        #grid=ib_grid,
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
b = buoyancy_operation(model)

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
    saved_output_filename = "periodic_over_shoals2.nc"
else
    saved_output_filename = "bounded_over_shoals2.nc"
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
    @inline Tᵢ(x, y, z) = blend(T_south_pwl(z), T_north_pwl(z), α_lin(y))
    @inline Sᵢ(x, y, z) = blend(S_south_pwl(z), S_north_pwl(z), α_lin(y))
else
    @inline Tᵢ(x, y, z) = T_south_pwl(z)
    @inline Sᵢ(x, y, z) = S_south_pwl(z)
end

set!(model, u=uᵢ, v=vᵢ, w=wᵢ, T=Tᵢ, S=Sᵢ)

# run simulation
@info "time to run simulation!"
run!(simulation)

# --- Plot bathymetry with GLMakie ---
# using GLMakie

# fig = Figure(size=(800, 600))
# ax = Axis3(fig[1, 1],
#     xlabel="x (km)",
#     ylabel="y (km)",
#     zlabel="Depth (m)",
#     title="Shoal Bathymetry")
# surf = surface!(ax, collect(x_km), collect(y_km), h, colormap=:viridis)
# Colorbar(fig[1, 2], surf, label="Depth (m)")
# display(fig)