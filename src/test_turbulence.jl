# ═══════════════════════════════════════════════════════════════════════════
# test_turbulence.jl
# ═══════════════════════════════════════════════════════════════════════════
# Main simulation script for flow over shoals.
# ═══════════════════════════════════════════════════════════════════════════

using Oceananigans
using Oceananigans.Grids: Periodic, Bounded
using Oceananigans.Units
using Oceananigans.BoundaryConditions: OpenBoundaryCondition, FieldBoundaryConditions
using Oceananigans.TurbulenceClosures
using Oceananigans.Solvers: ConjugateGradientPoissonSolver, FFTBasedPoissonSolver
using Oceananigans.Models: buoyancy_operation
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

# ═══════════════════════════════════════════════════════════════════════════
# Bathymetry (shared with flow_over_shoals.jl)
# ═══════════════════════════════════════════════════════════════════════════
include(joinpath(@__DIR__, "dshoal_vn_param.jl"))

# ═══════════════════════════════════════════════════════════════════════════
# Global Helper Functions
# ═══════════════════════════════════════════════════════════════════════════

# GPU-compatible SMOOTH piecewise linear T/S profiles (from CTD data)
const δ_smooth = 2.5

@inline smooth_step(z, z0) = 0.5 * (1.0 - tanh((z - z0) / δ_smooth))

@inline function T_north_pwl(z, v1=20.5389)
    z1, z2, z3 = -5.0, -15.0, -35.0
    v2, v3 = 17.8875, 14.3323
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

@inline function T_south_pwl(z, v1=24.5378)
    z1, z2, z3 = -5.0, -15.0, -30.0
    v2, v3 = 24.3073, 23.4116
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

@inline function S_north_pwl(z, v1=32.6264)
    z1, z2, z3 = -5.0, -15.0, -35.0
    v2, v3 = 33.7062, 33.2648
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

@inline function S_south_pwl(z, v1=35.5830)
    z1, z2, z3 = -5.0, -15.0, -30.0
    v2, v3 = 35.9986, 36.1776
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

@inline function sigmoidal_s2(x, Lx)
    xS = 65e3
    k2 = 20 / Lx
    return 1 / (1 + exp(k2 * (x - xS)))
end

# ═══════════════════════════════════════════════════════════════════════════
# simulation knobs
# ═══════════════════════════════════════════════════════════════════════════
function run_turbulence_test(ν_val, κ_val)

    run_number = 5
    sim_runtime = 10days
    callback_interval = 86400seconds
    run_tag = @sprintf("periodic_shoals%d_nu%.1e_ka%.1e", run_number, ν_val, κ_val)

    if LES
        params = (; Lx=200e3, Ly=200e3, Lz=50)
    else
        params = (; Lx=100000, Ly=200000, Lz=50)
    end
    if arch == CPU()
        params = (; params..., Nx=60, Ny=60, Nz=10)
    else
        params = (; params..., Nx=400, Ny=400, Nz=50)
    end

    x, y, z = (0, params.Lx), (0, params.Ly), (-params.Lz, 0)

    # grid  
    if periodic_y
        grid = RectilinearGrid(arch; size=(params.Nx, params.Ny, params.Nz), halo=(4, 4, 4), x, y, z, topology=(Bounded, Periodic, Bounded))
    else
        grid = RectilinearGrid(arch; size=(params.Nx, params.Ny, params.Nz), halo=(4, 4, 4), x, y, z, topology=(Bounded, Bounded, Bounded))
    end

    # model parameters
    if shoal_bath
        slope_bottom = dshoal_param_bottom(params.Ly;
            Hs=15.0,
            shoal_length=20000.0,
            sigma=8000.0,
            shelf_depth=-25.0,
            shelf_break_end=12000.0)
        GFB = GridFittedBottom(slope_bottom)
        ib_grid = ImmersedBoundaryGrid(grid, GFB)
    else
        ib_grid = grid
    end

    @info ib_grid

    # ═══════════════════════════════════════════════════════════════════════════
    # Stratification and Wind Setup
    # ═══════════════════════════════════════════════════════════════════════════
    if mass_flux
        v₀ = 0.10
    else
        v₀ = 0.0
    end

    # defaults
    T_north_v1, S_north_v1 = 20.5389, 32.6264
    T_south_v1, S_south_v1 = 24.5378, 35.5830

    params = (; params...,
        v₀=v₀,
        Ls=40e3,
        Le=100e3,
        Lw=10e3,
        τ=1days,
        T_north_v1=T_north_v1,
        T_south_v1=T_south_v1,
        S_north_v1=S_north_v1,
        S_south_v1=S_south_v1,
        wind_stress=0.0)

    # Eastern boundary targets are now functions of z
    params = (; params...)

    #+++ Drag (Implemented as in https://doi.org/10.1029/2005WR004685)
    z₀ = 2.5e-4 # roughness length
    z₁ = Oceananigans.Grids.minimum_zspacing(grid, Center(), Center(), Center()) / 2
    @info "Using z₁ =" z₁

    κᵛᵏ = 0.4 # von Karman constant
    params = (; params..., c_dz=(κᵛᵏ / log(z₁ / z₀))^2) # quadratic drag coefficient
    @info "Defining momentum BCs with Cᴰ =" params.c_dz

    @inline τᵘ_drag(x, y, z, t, u, v, w, p) = -p.c_dz * u * √(u^2 + v^2 + w^2)
    @inline τᵛ_drag(x, y, z, t, u, v, w, p) = -p.c_dz * v * √(u^2 + v^2 + w^2)
    @inline τʷ_drag(x, y, z, t, u, v, w, p) = -p.c_dz * w * √(u^2 + v^2 + w^2)

    immersed_drag_bc_u = FluxBoundaryCondition(τᵘ_drag, field_dependencies=(:u, :v, :w), parameters=params)
    immersed_drag_bc_v = FluxBoundaryCondition(τᵛ_drag, field_dependencies=(:u, :v, :w), parameters=params)
    immersed_drag_bc_w = FluxBoundaryCondition(τʷ_drag, field_dependencies=(:u, :v, :w), parameters=params)
    #---
    if LES
        @inline tsbc(x, z, t) = T_south_pwl(z, T_south_v1)
        @inline tnbc(x, z, t) = T_north_pwl(z, T_north_v1)
        @inline ssbc(x, z, t) = S_south_pwl(z, S_south_v1)
        @inline snbc(x, z, t) = S_north_pwl(z, S_north_v1)
    end

    # wind stress BC
    ρ₀ = 1024.0
    wind_bc_v = FluxBoundaryCondition(-0.0 / ρ₀)

    # velocity function
    local v∞
    if sigmoid_v_bc
        v∞ = (x, z, t, p) -> begin
            xC = 3e3
            k1 = 80 / p.Lx

            s1 = 1 / (1 + exp(-k1 * (x - xC)))
            s2 = sigmoidal_s2(x, p.Lx)
            s = (s1 - 1) + s2
            sc = clamp(s, 0.0, 1.0)
            return p.v₀ * sc
        end
    else
        v∞ = (x, z, t, p) -> p.v₀
    end

    # new sponge masks using built-in functions from Oceananigans

    north_mask = PiecewiseLinearMask{:y}(center=params.Ly, width=params.Ls)
    south_mask = PiecewiseLinearMask{:y}(center=0, width=params.Ls)
    east_mask = PiecewiseLinearMask{:x}(center=params.Lx, width=params.Le)
    global_params = params
    # We shift the mask evaluation by 30km so that the sponge layer only turns on 
    # *after* the velocity has safely tapered to 0. This prevents artificial vorticity generation!
    @inline offshore_mask_uvw(x, y, z) = 1.0 - sigmoidal_s2(x - 80e3, global_params.Lx)

    local sponge_mask, sponge_mask_uvw, T_target, S_target, v_target
    if periodic_y
        sponge_mask = (x, y, z) -> min(north_mask(x, y, z) + south_mask(x, y, z), 1.0)
        sponge_mask_uvw = (x, y, z) -> min(north_mask(x, y, z) + south_mask(x, y, z) + offshore_mask_uvw(x, y, z), 1.0)

        T_target = (x, y, z, t) -> T_south_pwl(z)
        S_target = (x, y, z, t) -> S_south_pwl(z)
        v_target = (x, y, z, t) -> v∞(x, z, t, global_params)
    else
        sponge_mask = (x, y, z) -> min(north_mask(x, y, z) + south_mask(x, y, z), 1.0)
        sponge_mask_uvw = (x, y, z) -> min(north_mask(x, y, z) + south_mask(x, y, z) + offshore_mask_uvw(x, y, z), 1.0)

        T_target = (x, y, z, t) -> begin
            n = north_mask(x, y, z)
            s = south_mask(x, y, z)
            tot = n + s
            return tot > 0 ? (n * T_north_pwl(z) + s * T_south_pwl(z)) / tot : T_south_pwl(z)
        end

        S_target = (x, y, z, t) -> begin
            n = north_mask(x, y, z)
            s = south_mask(x, y, z)
            tot = n + s
            return tot > 0 ? (n * S_north_pwl(z) + s * S_south_pwl(z)) / tot : S_south_pwl(z)
        end

        v_target = (x, y, z, t) -> v∞(x, z, t, global_params)
    end

    u_nudging = Relaxation(; rate=1 / global_params.τ, mask=sponge_mask_uvw, target=0.0)
    v_nudging = Relaxation(; rate=1 / global_params.τ, mask=sponge_mask_uvw, target=v_target)
    w_nudging = Relaxation(; rate=1 / global_params.τ, mask=sponge_mask_uvw, target=0.0)
    T_nudging = Relaxation(; rate=1 / global_params.τ, mask=sponge_mask, target=T_target)
    S_nudging = Relaxation(; rate=1 / global_params.τ, mask=sponge_mask, target=S_target)

    # forcing functions
    if mass_flux
        forcings = (u=u_nudging, v=v_nudging, w=w_nudging, T=T_nudging, S=S_nudging)
    else
        forcings = (T=T_nudging, S=S_nudging)
    end

    if periodic_y
        T_bcs = FieldBoundaryConditions()
        S_bcs = FieldBoundaryConditions()
        u_bcs = FieldBoundaryConditions(immersed=immersed_drag_bc_u)
        v_bcs = FieldBoundaryConditions(immersed=immersed_drag_bc_v, top=wind_bc_v)
        w_bcs = FieldBoundaryConditions(immersed=immersed_drag_bc_w)
    else
        open_bc = OpenBoundaryCondition(v∞; parameters=params, scheme=PerturbationAdvection())
        open_zero = OpenBoundaryCondition(0.0)
        T_bcs = FieldBoundaryConditions(south=ValueBoundaryCondition(tsbc), north=ValueBoundaryCondition(tnbc))
        S_bcs = FieldBoundaryConditions(south=ValueBoundaryCondition(ssbc), north=ValueBoundaryCondition(snbc))
        u_bcs = FieldBoundaryConditions(immersed=immersed_drag_bc_u)
        v_bcs = FieldBoundaryConditions(immersed=immersed_drag_bc_v, north=open_bc, south=open_bc, top=wind_bc_v)
        w_bcs = FieldBoundaryConditions(immersed=immersed_drag_bc_w)
    end

    bcs = (u=u_bcs, v=v_bcs, w=w_bcs, T=T_bcs, S=S_bcs)
    if is_coriolis
        coriolis = FPlane(latitude=35.2480)
    else
        coriolis = nothing
    end

    reltol = sqrt(eps(grid))
    abstol = sqrt(eps(grid))


    if periodic_y
        model = NonhydrostaticModel(ib_grid;
            timestepper=:RungeKutta3,
            advection=WENO(order=5),
            closure=VerticalScalarDiffusivity(ν=ν_val, κ=κ_val),
            hydrostatic_pressure_anomaly=CenterField(ib_grid),
            pressure_solver=ConjugateGradientPoissonSolver(ib_grid, reltol=reltol, abstol=abstol, maxiter=100),
            tracers=(:T, :S),
            buoyancy=SeawaterBuoyancy(),
            coriolis=coriolis,
            boundary_conditions=bcs,
            forcing=forcings
        )
    else
        model = NonhydrostaticModel(ib_grid;
            timestepper=:RungeKutta3,
            advection=WENO(order=5),
            closure=VerticalScalarDiffusivity(ν=ν_val, κ=κ_val),
            hydrostatic_pressure_anomaly=CenterField(ib_grid),
            pressure_solver=ConjugateGradientPoissonSolver(ib_grid, reltol=reltol, abstol=abstol, maxiter=100),
            tracers=(:T, :S),
            buoyancy=SeawaterBuoyancy(),
            coriolis=coriolis,
            boundary_conditions=bcs,
            forcing=forcings
        )
    end

    @info "" model

    pickup = isfile("checkpoint_$(run_tag).jld2")
    overwrite_existing = !pickup

    simulation = Simulation(model, Δt=15minutes, stop_time=sim_runtime)
    conjure_time_step_wizard!(simulation, cfl=0.4)

    progress = TimedMessenger()
    simulation.callbacks[:progress] = Callback(progress, TimeInterval(callback_interval))

    function print_solver_iterations(sim)
        solver = sim.model.pressure_solver
        if hasproperty(solver, :conjugate_gradient_solver)
            cg = solver.conjugate_gradient_solver
            @info @sprintf("Pressure solver: %d CG iterations (t = %.2f days)",
                cg.iteration, time(sim) / 86400)
        end
    end
    simulation.callbacks[:solver_iters] = Callback(print_solver_iterations, TimeInterval(callback_interval))

    u, v, w = model.velocities
    T = model.tracers.T
    S = model.tracers.S
    Ro = @at (Center, Center, Center) RossbyNumber(model)
    KE = @at (Center, Center, Center) KineticEnergy(model)

    # Centered velocities for consistency
    u_c = @at (Center, Center, Center) u
    v_c = @at (Center, Center, Center) v
    w_c = @at (Center, Center, Center) w

    slice_fields = (; u_c, v_c, w_c, T, S, Ro, KE)

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


    if checkpointing
        checkpoint_prefix = periodic_y ? "checkpoint_$(run_tag)" : "checkpoint_$(run_tag)"
        simulation.output_writers[:checkpointer] = Checkpointer(model,
            schedule=TimeInterval(5days),
            prefix=checkpoint_prefix,
            overwrite_existing=true,
            cleanup=true)
    end

    # initial conditions
    @info "Setting initial conditions"
    if sigmoid_ic
        v_init = (x, y, z) -> v∞(x, z, 0, params)
    else
        v_init = v₀
    end

    local Tᵢ, Sᵢ
    if gradient_IC
        α_lin = y -> clamp(y / params.Ly, 0.0, 1.0)
        blend = (a, b, α) -> (1 - α) * a + α * b
        Tᵢ = (x, y, z) -> blend(T_south_pwl(z, T_south_v1), T_north_pwl(z, T_north_v1), α_lin(y))
        Sᵢ = (x, y, z) -> blend(S_south_pwl(z, S_south_v1), S_north_pwl(z, S_north_v1), α_lin(y))
    else
        Tᵢ = (x, y, z) -> T_south_pwl(z, T_south_v1)
        Sᵢ = (x, y, z) -> S_south_pwl(z, S_south_v1)
    end

    set!(model, u=0.0, v=v_init, w=0.0, T=Tᵢ, S=Sᵢ)

    # run simulation
    @info """
    ════════════════════════════════════════════════════════
     SIMULATION CONFIGURATION: $(run_tag)
    ════════════════════════════════════════════════════════
     Run number:      $(run_number)
     Runtime:         $(sim_runtime)
     Architecture:    $(arch)

     ── Model Parameters ──
     Hs:              $(15.0) m
     shoal_length:    $(20000.0) m
     shelf_depth:     $(-25.0) m
     shelf_break_end: $(12000.0) m
     wind_stress:     $(0.0) N/m^2

     ── Switches ──
     LES:             $(LES)
     mass_flux:       $(mass_flux)
     periodic_y:      $(periodic_y)
     gradient_IC:     $(gradient_IC)
     sigmoid_v_bc:    $(sigmoid_v_bc)
     sigmoid_ic:      $(sigmoid_ic)
     is_coriolis:     $(is_coriolis)
     shoal_bath:      $(shoal_bath)
    ════════════════════════════════════════════════════════
    """
    run!(simulation, pickup=pickup)

end # function run_turbulence_test

test_pairs = [
    (1e-7, 1e-7),
    (1e-5, 1e-5),
    (1e-3, 1e-3)
]

for (ν, κ) in test_pairs
    run_turbulence_test(ν, κ)
end
