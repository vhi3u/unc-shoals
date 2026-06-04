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
using Oceananigans.Grids: Periodic, Bounded, minimum_zspacing
using Oceananigans.Units
using Oceananigans.BoundaryConditions: OpenBoundaryCondition, FieldBoundaryConditions
using Oceananigans.TurbulenceClosures
using Oceananigans.Models.HydrostaticFreeSurfaceModels: HydrostaticFreeSurfaceModel
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
mass_flux = true
periodic_y = true
gradient_IC = false
sigmoid_v_bc = true
sigmoid_ic = true
is_coriolis = true
checkpointing = false
shoal_bath = true
if has_cuda_gpu()
    arch = GPU()
else
    arch = CPU()
end
@info "architecture = $(arch)"
include("dshoal_vn_param.jl")

# simulation knobs
run_number = 0000 # <-- change this for each new run
sim_runtime = 10days
callback_interval = 86400seconds
run_tag = (periodic_y ? "periodic" : "bounded") * "_hydro_shoals$(run_number)"  # e.g. "periodic_hydro_shoals0000"

if LES
    params = (; Lx=100e3, Ly=200e3, Lz=50, Nx=30, Ny=30, Nz=10)
else
    params = (; Lx=100000, Ly=200000, Lz=50, Nx=30, Ny=30, Nz=10)
end
if arch == CPU()
    params = (; params..., Nx=50, Ny=100, Nz=10) # keep the same for now
else
    params = (; params..., Nx=200, Ny=400, Nz=50)
end

x, y, z = (0, params.Lx), (0, params.Ly), (-params.Lz, 0)

# grid  

if periodic_y
    grid = RectilinearGrid(arch; size=(params.Nx, params.Ny, params.Nz), halo=(4, 4, 4), x, y, z, topology=(Bounded, Periodic, Bounded))
else
    grid = RectilinearGrid(arch; size=(params.Nx, params.Ny, params.Nz), halo=(4, 4, 4), x, y, z, topology=(Bounded, Bounded, Bounded))
end

if shoal_bath
    # Define shoal parameters (align with the new sigmoidal setup)
    Hs = 5.0         # Height of shoal above -25m shelf
    sigma = 8e3       # Gaussian width of shoal (half crossover)
    shoal_length = 20e3 # Horizontal span of the shoal ridge

    slope_bottom = dshoal_param_bottom(params.Ly; Hs=Hs, sigma=sigma, shoal_length=shoal_length)
    GFB = GridFittedBottom(slope_bottom)
    ib_grid = ImmersedBoundaryGrid(grid, GFB)
else
    ib_grid = grid
end

@info ib_grid
v₀ = 0.1

params = (; params...,
    v₀=v₀,
    Ls=10e3,
    Le=40e3,
    τₙ=6hours,
    τₛ=24hours,
    τₑ=24hours,
    τ_ts=24hours)

const δ_smooth = 2.5  # smoothing length scale in meters

# Smooth transition function: 0 when z >> z0, 1 when z << z0
@inline smooth_step(z, z0) = 0.5 * (1.0 - tanh((z - z0) / δ_smooth))

# Temperature at North boundary (B1) - SMOOTHED
@inline function T_north_pwl(z)
    z1, z2, z3 = -5.0, -15.0, -35.0
    v1, v2, v3 = 20.5389, 17.8875, 14.3323
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

# Temperature at East boundary (Offshore) - STRATIFIED
@inline function T_east_pwl(z)
    z1, z2, z3 = -5.0, -25.0, -45.0
    v1, v2, v3 = 25.0, 23.0, 21.0
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

# Salinity at East boundary (Offshore) - STABLE (Saltier at depth)
@inline function S_east_pwl(z)
    z1, z2, z3 = -5.0, -25.0, -45.0
    v1, v2, v3 = 35.8, 36.0, 36.2      # Flipped: 35.8 at surface, 36.2 at bottom
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

params = (; params...)

# T/S boundary condition helpers
cᴰ = 2.5e-3
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
if LES
    @inline tsbc(x, z, t) = T_south_pwl(z)
    @inline tnbc(x, z, t) = T_north_pwl(z)
    @inline ssbc(x, z, t) = S_south_pwl(z)
    @inline snbc(x, z, t) = S_north_pwl(z)
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

# offshore mask: sigmoid (tanh) taper centered at the shelf break
# σ_off controls the half-width of the transition (larger = wider, more gradual)
const σ_off = 20e3   # half-width of sigmoid taper (m)
@inline offshore_mask(x, y, z, p) = 0.5 * (1.0 + tanh((x - 60e3) / σ_off))

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
        @inline sponge_u(x, y, z, t, u, p) = -(
            north_mask(x, y, z, p) * u / p.τₙ +
            offshore_mask(x, y, z, p) * u / p.τₑ)

        @inline sponge_v(x, y, z, t, v, p) = -(
            north_mask(x, y, z, p) * (v - v∞(x, z, t, p)) / p.τₙ +
            offshore_mask(x, y, z, p) * (v - v∞(x, z, t, p)) / p.τₑ)

        @inline sponge_T(x, y, z, t, T, p) = -(
            north_mask(x, y, z, p) * (T - T_south_pwl(z)) / p.τ_ts +
            offshore_mask(x, y, z, p) * (T - T_east_pwl(z)) / (5 * p.τ_ts))

        @inline sponge_S(x, y, z, t, S, p) = -(
            north_mask(x, y, z, p) * (S - S_south_pwl(z)) / p.τ_ts +
            offshore_mask(x, y, z, p) * (S - S_east_pwl(z)) / (5 * p.τ_ts))
    else
        # Bounded case
        @inline sponge_u(x, y, z, t, u, p) = -(
            north_mask(x, y, z, p) * u / p.τₙ +
            offshore_mask(x, y, z, p) * u / p.τₑ)

        @inline sponge_v(x, y, z, t, v, p) = -(
            north_mask(x, y, z, p) * (v - v∞(x, z, t, p)) / p.τₙ +
            offshore_mask(x, y, z, p) * (v - v∞(x, z, t, p)) / p.τₑ)

        @inline sponge_T(x, y, z, t, T, p) = -(
            north_mask(x, y, z, p) * (T - T_north_pwl(z)) / p.τ_ts +
            offshore_mask(x, y, z, p) * (T - T_east_pwl(z)) / p.τ_ts)

        @inline sponge_S(x, y, z, t, S, p) = -(
            north_mask(x, y, z, p) * (S - S_north_pwl(z)) / p.τ_ts +
            offshore_mask(x, y, z, p) * (S - S_east_pwl(z)) / p.τ_ts)
    end
end

# forcing functions
FT = Forcing(sponge_T, field_dependencies=:T, parameters=params)
FS = Forcing(sponge_S, field_dependencies=:S, parameters=params)
if mass_flux
    Fᵤ = Forcing(sponge_u, field_dependencies=:u, parameters=params)
    Fᵥ = Forcing(sponge_v, field_dependencies=:v, parameters=params)
    forcings = (u=Fᵤ, v=Fᵥ, T=FT, S=FS)
else
    forcings = (T=FT, S=FS)
end

if periodic_y
    T_bcs = FieldBoundaryConditions()
    S_bcs = FieldBoundaryConditions()
    u_bcs = FieldBoundaryConditions(immersed=immersed_drag_bc_u)
    v_bcs = FieldBoundaryConditions(immersed=immersed_drag_bc_v)
else
    open_bc = OpenBoundaryCondition(v∞; parameters=params, scheme=PerturbationAdvection())
    open_zero = OpenBoundaryCondition(0.0)
    T_bcs = FieldBoundaryConditions(south=ValueBoundaryCondition(tsbc), north=ValueBoundaryCondition(tnbc))
    S_bcs = FieldBoundaryConditions(south=ValueBoundaryCondition(ssbc), north=ValueBoundaryCondition(snbc))
    u_bcs = FieldBoundaryConditions(immersed=immersed_drag_bc_u)
    v_bcs = FieldBoundaryConditions(immersed=immersed_drag_bc_v, north=open_bc, south=open_bc)
end

bcs = (u=u_bcs, v=v_bcs, T=T_bcs, S=S_bcs)

if is_coriolis
    coriolis = FPlane(latitude=35.2480)
else
    coriolis = nothing
end

closure = CATKEVerticalDiffusivity()
tracers = (:T, :S)

if periodic_y
    model = HydrostaticFreeSurfaceModel(ib_grid;
        momentum_advection=WENO(order=5),
        tracer_advection=WENO(order=5),
        closure=closure,
        tracers=tracers,
        buoyancy=SeawaterBuoyancy(),
        coriolis=coriolis,
        boundary_conditions=bcs,
        forcing=forcings
    )
else
    model = HydrostaticFreeSurfaceModel(ib_grid;
        momentum_advection=WENO(order=5),
        tracer_advection=WENO(order=5),
        closure=closure,
        tracers=tracers,
        buoyancy=SeawaterBuoyancy(),
        coriolis=coriolis,
        boundary_conditions=bcs,
        forcing=forcings
    )
end

@info "" model

# Check for existing checkpoint to determine if we should pickup or start fresh
if checkpointing
    checkpoint_prefix = periodic_y ? "checkpoint_$(run_tag)" : "checkpoint_$(run_tag)"
    checkpoint_files = filter(f -> startswith(f, checkpoint_prefix) && endswith(f, ".jld2"), readdir("."))
    if !isempty(checkpoint_files)
        @info "Found checkpoint file(s) - will restore when simulation runs"
        pickup = true
    else
        pickup = false
    end
else
    pickup = false
end

# output

@info "creating output fields"

# Don't overwrite the NetCDF file when picking up from a checkpoint
overwrite_existing = !pickup

cfl_values = Float64[]       # Stores CFL at each step
cfl_times = Float64[]       # Stores model time

simulation = Simulation(model, Δt=15minutes, stop_time=sim_runtime)

conjure_time_step_wizard!(simulation, cfl=0.7)

progress = TimedMessenger()

simulation.callbacks[:progress] = Callback(progress, TimeInterval(callback_interval))

u, v, w = model.velocities
T = model.tracers.T
S = model.tracers.S

Ro = @at (Center, Center, Center) RossbyNumber(model)
KE = @at (Center, Center, Center) KineticEnergy(model)

# Centered velocities for consistency
u_c = @at (Center, Center, Center) u
v_c = @at (Center, Center, Center) v
w_c = @at (Center, Center, Center) w

uu = Field(u_c * u_c)
vv = Field(v_c * v_c)
ww = Field(w_c * w_c)
uT = Field(u_c * T)
uS = Field(u_c * S)
vT = Field(v_c * T)
vS = Field(v_c * S)
wT = Field(w_c * T)
wS = Field(w_c * S)

slice_fields = (; u_c, v_c, w_c, T, S, Ro, KE)
tavg_fields = (; u_c, v_c, w_c, uu, vv, ww, T, S, uT, uS, vT, vS, wT, wS)

# (1) 2D snapshots (every 1 day)
# Surface XY slice (top layer)
simulation.output_writers[:surface_slice] = NetCDFWriter(model, slice_fields,
    filename="top_$(run_tag).nc",
    schedule=TimeInterval(callback_interval),
    indices=(:, :, params.Nz),
    overwrite_existing=overwrite_existing)

# Mid-y XZ slice (cross-shore transect at domain center)
simulation.output_writers[:midy_slice] = NetCDFWriter(model, slice_fields,
    filename="midy_$(run_tag).nc",
    schedule=TimeInterval(callback_interval),
    indices=(:, round(Int, params.Ny / 2), :),
    overwrite_existing=overwrite_existing)

# (3) 3D Time Averages (10 day window)
simulation.output_writers[:time_avg_3d] = NetCDFWriter(model, tavg_fields,
    filename="time_avg_3d_$(run_tag).nc",
    schedule=AveragedTimeInterval(10days, window=10days),
    overwrite_existing=overwrite_existing)

if checkpointing
    checkpoint_prefix = periodic_y ? "checkpoint_$(run_tag)" : "checkpoint_$(run_tag)"
    simulation.output_writers[:checkpointer] = Checkpointer(model,
        schedule=TimeInterval(5days),
        prefix=checkpoint_prefix,
        overwrite_existing=true,
        cleanup=true)
end

if !pickup
    @info "No checkpoint found, setting initial conditions"
    # initial conditions
    uᵢ = 0.005 * rand(size(u)...)
    vᵢ = 0.005 * rand(size(v)...)
    uᵢ .-= mean(uᵢ)
    vᵢ .-= mean(vᵢ)
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

    eᵢ(x, y, z) = 1e-6

    set!(model, u=uᵢ, v=vᵢ, T=Tᵢ, S=Sᵢ, e=eᵢ)
end

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
 mass_flux:       $(mass_flux)
 periodic_y:      $(periodic_y)
 gradient_IC:     $(gradient_IC)
 sigmoid_v_bc:    $(sigmoid_v_bc)
 sigmoid_ic:      $(sigmoid_ic)
 is_coriolis:     $(is_coriolis)
 checkpointing:   $(checkpointing)
 shoal_bath:      $(shoal_bath)
 ════════════════════════════════════════════════════════
"""
run!(simulation, pickup=pickup)
