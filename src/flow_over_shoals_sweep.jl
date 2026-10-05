# ═══════════════════════════════════════════════════════════════════════════
# flow_over_shoals_sweep.jl
# ═══════════════════════════════════════════════════════════════════════════
# Modified version of flow_over_shoals.jl for parameter sweeps.
# Reads sweep parameters from environment variables set by sweep_driver.jl:
#   SWEEP_Hs, SWEEP_SHOAL_LENGTH, SWEEP_SHELF_DEPTH, SWEEP_SHELF_BREAK_END,
#   SWEEP_RUN_LABEL, SWEEP_RUN_INDEX
# ═══════════════════════════════════════════════════════════════════════════

using Oceananigans
using Oceananigans.Grids: Periodic, Bounded, znode
using Oceananigans.Units
using Oceananigans.BoundaryConditions: FieldBoundaryConditions
using Oceananigans.Fields: interior
using Oceananigans.TurbulenceClosures
using Oceananigans.OutputWriters
using Oceananigans.Forcings
using Statistics: mean
# NOTE: YShearProductionRate / ZShearProductionRate are not exported by
# Oceanostics (verified against both 0.18.0 and 0.21.2), so this `using` was
# failing at load. Trimmed to the three diagnostics the script actually uses.
using Oceanostics: RossbyNumber, ErtelPotentialVorticity, KineticEnergy
using Oceanostics.ProgressMessengers: TimedMessenger
using SeawaterPolynomials.TEOS10
using Printf: @sprintf
using NCDatasets
using DataFrames
using CUDA: has_cuda_gpu, allowscalar

# ═══════════════════════════════════════════════════════════════════════════
# Read sweep parameters from environment (set by sweep_driver.jl)
# Falls back to defaults so script can also be run standalone.
# ═══════════════════════════════════════════════════════════════════════════
sweep_Zs = parse(Float64, get(ENV, "SWEEP_Zs", "-5.0"))
sweep_shoal_length = parse(Float64, get(ENV, "SWEEP_SHOAL_LENGTH", "40000.0"))
sweep_sigma = parse(Float64, get(ENV, "SWEEP_SIGMA", "8000.0"))
sweep_Zsh = parse(Float64, get(ENV, "SWEEP_Zsh", "-25.0"))
sweep_shelf_break_end = parse(Float64, get(ENV, "SWEEP_SHELF_BREAK_END", "12000.0"))
sweep_run_label = get(ENV, "SWEEP_RUN_LABEL", "standalone")
sweep_run_index = parse(Int, get(ENV, "SWEEP_RUN_INDEX", "0"))
sweep_wind_stress = parse(Float64, get(ENV, "SWEEP_WIND_STRESS", "0.0"))
sweep_v0 = parse(Float64, get(ENV, "SWEEP_V0", "0.2"))

@info "Sweep parameters: Zs=$sweep_Zs, shoal_length=$sweep_shoal_length, sigma=$sweep_sigma, Zsh=$sweep_Zsh, shelf_break_end=$sweep_shelf_break_end, wind_stress=$sweep_wind_stress, v0=$sweep_v0"

# build
@info "building domain"

# switches
LES = true
mass_flux = true
periodic_y = true
gradient_IC = false
sigmoid_v_bc = true
sigmoid_ic = true
sigmoid_wind = false         # uniform wind. The taper's stress gradient drives
# shear instability at the taper line (seen in runs 51/52), and with open_east
# there is no sponge for it to protect.
is_coriolis = true
checkpointing = false
shoal_bath = true
ramp_wind = true             # ramp τ over ~2 inertial periods
open_east = true             # radiative east boundary instead of wall + sponge

# Vertical mixing closure. :catke is the recommended production choice;
# :ribased and :constant exist so the wind run can be repeated against the
# closure used in shoals38/39 without changing anything else.
closure_choice = :catke      # :catke | :ribased | :constant
# Equation of state. :teos10 is the physically correct choice (the linear
# default's alpha = 1.67e-4 is a ~10 C value; the true value here is 2.94e-4).
# :linear exists so a blow-up can be tested against the EOS as a single
# controlled variable without editing the model constructor.
eos_choice = :linear         # :teos10 | :linear
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
# simulation knobs
# ═══════════════════════════════════════════════════════════════════════════
run_number = sweep_run_index
sim_runtime = 50days
callback_interval = 86400seconds
# Sub-inertial output. The inertial period is 20.74 h, so DAILY snapshots alias
# it to a spurious 6.35-day oscillation sitting squarely in the mesoscale band.
snapshot_interval = 6hours
run_tag = "sweep_$(sweep_run_label)"

# Lx stays at 150 km. The 250 km domain in flow_over_shoals_hydrostatic.jl was
# needed only to push the east SPONGE away from the shelf; with open_east there
# is no sponge. The bathymetry's offshore ramp is fixed at 63.5 km and no sweep
# parameter moves it, so 150 km leaves 87 km of offshore room — more than the
# 37 km the sponged 250 km domain actually left usable.
if LES
    params = (; Lx=150e3, Ly=200e3, Lz=50)
else
    params = (; Lx=150000, Ly=200000, Lz=50)
end

# Horizontal grid spacing is the knob, not Nx/Ny, so changing Lx can never
# silently change the resolution (and vice versa).
Δh = 500.0   # m
if arch == CPU()
    params = (; params..., Nx=60, Ny=60, Nz=10)
else
    params = (; params...,
        Nx=round(Int, params.Lx / Δh),
        Ny=round(Int, params.Ly / Δh),
        Nz=50)
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
        Zs=sweep_Zs,
        shoal_length=sweep_shoal_length,
        sigma=sweep_sigma,
        Zsh=sweep_Zsh,
        shelf_break_end=sweep_shelf_break_end)
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
    v₀ = sweep_v0
else
    v₀ = 0.0
end

v₀ = abs(v₀)

# defaults
T_north_v1, S_north_v1 = 20.5389, 32.6264
T_south_v1, S_south_v1 = 24.5378, 35.5830

params = (; params...,
    v₀=v₀,
    Ls=20e3,
    Le=50e3,
    Lw=10e3,
    τ=6hours,
    T_north_v1=T_north_v1,
    T_south_v1=T_south_v1,
    S_north_v1=S_north_v1,
    S_south_v1=S_south_v1,
    wind_stress=sweep_wind_stress)

# GPU-compatible SMOOTH piecewise linear T/S profiles (from CTD data)
const δ_smooth = 2.5

@inline smooth_step_z(z, z0) = 0.5 * (1.0 - tanh((z - z0) / δ_smooth))

@inline function T_north_pwl(z, v1=20.5389)
    z1, z2, z3 = -5.0, -15.0, -35.0
    v2, v3 = 17.8875, 14.3323
    m12 = (v2 - v1) / (z2 - z1)
    m23 = (v3 - v2) / (z3 - z2)
    val1 = v1
    val2 = v1 + m12 * (z - z1)
    val3 = v2 + m23 * (z - z2)
    val4 = v3
    w1 = smooth_step_z(z, z1)
    w2 = smooth_step_z(z, z2)
    w3 = smooth_step_z(z, z3)
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
    w1 = smooth_step_z(z, z1)
    w2 = smooth_step_z(z, z2)
    w3 = smooth_step_z(z, z3)
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
    w1 = smooth_step_z(z, z1)
    w2 = smooth_step_z(z, z2)
    w3 = smooth_step_z(z, z3)
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
    w1 = smooth_step_z(z, z1)
    w2 = smooth_step_z(z, z2)
    w3 = smooth_step_z(z, z3)
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
    w1 = smooth_step_z(z, z1)
    w2 = smooth_step_z(z, z2)
    w3 = smooth_step_z(z, z3)
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
    w1 = smooth_step_z(z, z1)
    w2 = smooth_step_z(z, z2)
    w3 = smooth_step_z(z, z3)
    return val1 * (1 - w1) + val2 * (w1 - w2) + val3 * (w2 - w3) + val4 * w3
end

# Eastern boundary targets are now functions of z
params = (; params...)

#+++ Drag 
z₀ = 2.5e-4 # roughness length
z₁ = Oceananigans.Grids.minimum_zspacing(grid, Center(), Center(), Center()) / 2
@info "Using z₁ =" z₁

const κᵛᵏ = 0.4 # von Karman constant
c_dz = (κᵛᵏ / log(z₁ / z₀))^2 # quadratic drag coefficient
@info "Defining momentum BCs with Cᴰ =" c_dz
drag = BulkDrag(coefficient=c_dz)
# Bottom facet only: with a stair-stepped bottom, applying drag to the vertical
# side faces of the steps is unphysical and is itself a grid-scale vorticity
# source. `immersed=drag` (the old form) does apply it to all faces.
immersed_drag = ImmersedBoundaryCondition(bottom=drag)
#---
if LES
    @inline tsbc(x, z, t) = T_south_pwl(z, T_south_v1)
    @inline tnbc(x, z, t) = T_north_pwl(z, T_north_v1)
    @inline ssbc(x, z, t) = S_south_pwl(z, S_south_v1)
    @inline snbc(x, z, t) = S_north_pwl(z, S_north_v1)
end

# ═══════════════════════════════════════════════════════════════════════════
# Wind stress
# ═══════════════════════════════════════════════════════════════════════════
ρ₀ = 1024.0
const f₀ = 2 * 7.292115e-5 * sind(35.2480)          # 8.42e-5 s⁻¹
const inertial_period = 2π / f₀                     # 20.74 h
const T_ramp = ramp_wind ? 2 * inertial_period : 0.0
const τ_kinematic = sweep_wind_stress / ρ₀         # m² s⁻²
const Lx_c = params.Lx
const Le_c = params.Le

@info @sprintf("f = %.3e s⁻¹, inertial period = %.2f h, wind ramp = %.1f h",
    f₀, inertial_period / 3600, T_ramp / 3600)

# Offshore taper: the east sponge relaxes v → 0 over the outer Le = 50 km. If
# the wind keeps accelerating v there, the sponge and the forcing fight each
# other permanently. Taper the stress to zero across that same band so the wind
# is uniform over the shelf and slope (0–100 km) and absent where the sponge
# takes over. Set sigmoid_wind = false for a genuinely uniform wind.
@inline wind_shape(x) = 0.5 * (1 - tanh((x - (Lx_c - Le_c)) / (0.25 * Le_c)))
@inline wind_ramp(t) = ifelse(T_ramp > 0, 1 - exp(-t / T_ramp), one(t))

if sigmoid_wind
    @inline wind_v_flux(x, y, t) = -τ_kinematic * wind_ramp(t) * wind_shape(x)
else
    @inline wind_v_flux(x, y, t) = -τ_kinematic * wind_ramp(t)
end

wind_bc_u = FluxBoundaryCondition(0.0)
wind_bc_v = FluxBoundaryCondition(wind_v_flux)

# Pinned, NOT tied to Lx. Tying k to Lx means changing the domain width
# silently rebroadens the inflow profile and breaks comparability between runs.
const k1_fixed = 80 / 150e3     # coastal sigmoid, e-folding width 1875 m
const k2_fixed = 40 / 150e3     # offshore sigmoid, e-folding width 3750 m

@inline function sigmoidal_s2(x)
    xS = 65e3
    return 1 / (1 + exp(k2_fixed * (x - xS)))
end

# velocity function
if sigmoid_v_bc
    @inline function v∞(x, z, t, p)
        xC = 3e3
        k1 = k1_fixed

        s1 = 1 / (1 + exp(-k1 * (x - xC)))
        s2 = sigmoidal_s2(x)
        s = (s1 - 1) + s2
        sc = clamp(s, 0.0, 1.0)
        return p.v₀ * sc
    end
else
    @inline function v∞(x, z, t, p)
        return p.v₀
    end
end

# built-in masks
const south_mask = PiecewiseLinearMask{:y}(center=0.0, width=params.Ls)
const north_mask = PiecewiseLinearMask{:y}(center=params.Ly, width=params.Ls)
const east_mask = PiecewiseLinearMask{:x}(center=params.Lx, width=params.Le)

# targets
const global_params = params
@inline v_target_inflow(x, y, z, t) = v∞(x, z, t, global_params)

@inline T_target_south(x, y, z, t) = T_south_pwl(z, global_params.T_south_v1)
@inline S_target_south(x, y, z, t) = S_south_pwl(z, global_params.S_south_v1)

@inline T_target_north(x, y, z, t) = T_north_pwl(z, global_params.T_north_v1)
@inline S_target_north(x, y, z, t) = S_north_pwl(z, global_params.S_north_v1)

const T_target = T_target_south
const S_target = S_target_south
const inflow_mask = south_mask

# forcing functions — note there is no w sponge: w is diagnosed from continuity
# in the hydrostatic model and cannot (and should not) be relaxed.
if periodic_y
    u_sponge_inflow = Relaxation(; rate=1 / global_params.τ, mask=inflow_mask, target=0.0)
    u_sponge_e = Relaxation(; rate=1 / global_params.τ, mask=east_mask, target=0.0)

    v_sponge_inflow = Relaxation(; rate=1 / global_params.τ, mask=inflow_mask, target=v_target_inflow)
    v_sponge_e = Relaxation(; rate=1 / global_params.τ, mask=east_mask, target=0.0)

    T_sponge_inflow = Relaxation(; rate=1 / global_params.τ, mask=inflow_mask, target=T_target)
    T_sponge_e = Relaxation(; rate=1 / global_params.τ, mask=east_mask, target=T_target)

    S_sponge_inflow = Relaxation(; rate=1 / global_params.τ, mask=inflow_mask, target=S_target)
    S_sponge_e = Relaxation(; rate=1 / global_params.τ, mask=east_mask, target=S_target)

    # With open_east the east sponges are NOT applied. A sponge relaxing u -> 0
    # at an open boundary fights the radiation condition directly, and the T/S
    # sponge is redundant because the open BC prescribes the exterior on inflow.
    if mass_flux
        forcings = open_east ?
                   (u=u_sponge_inflow, v=v_sponge_inflow,
                    T=T_sponge_inflow, S=S_sponge_inflow) :
                   (u=(u_sponge_inflow, u_sponge_e),
                    v=(v_sponge_inflow, v_sponge_e),
                    T=(T_sponge_inflow, T_sponge_e),
                    S=(S_sponge_inflow, S_sponge_e))
    else
        forcings = open_east ?
                   (T=T_sponge_inflow, S=S_sponge_inflow) :
                   (T=(T_sponge_inflow, T_sponge_e), S=(S_sponge_inflow, S_sponge_e))
    end
end

if periodic_y && open_east
    # ═══════════════════════════════════════════════════════════════════════
    # Radiative eastern boundary
    # ═══════════════════════════════════════════════════════════════════════
    # A closed wall + sponge cannot pass the wind-driven Ekman transport; it is
    # forced downward instead, bending isopycnals at the sponge edge. An open
    # boundary lets it leave.
    #
    # outflow_timescale = Inf is PURE RADIATION. Any FINITE value applies
    #   dφ/dt + c dφ/dn = -(φ - φ_ext)/τ
    # i.e. it relaxes the boundary velocity toward the exterior value (0),
    # suppressing the very outflow we are trying to permit.
    #
    # target_transport (radiation schemes, Oceananigans >= 0.113) pins the NET
    # flux by shifting u uniformly along the boundary each step. The west
    # boundary is a closed coast and y is periodic, so the net must be ~0; the
    # vertical profile is left free, giving Ekman out at the surface and a
    # return inflow at depth.
    # outflow_timescale: FINITE, not Inf. Inf means literally no restoring term
    # on outflow, so the boundary value can drift without bound — and the whole
    # point of the Marchesiello et al. (2001) nudging is to prevent that. The
    # docstring's own example uses 360 days rather than Inf. At 30 days the
    # steady-state suppression is phi_b/phi_1 = Cn/(Cn + Dt/tau) >= 98% even for
    # a small phase-speed Courant number Cn, so the Ekman outflow is preserved
    # while the value stays bounded. (6 h, used in run 59, keeps only ~26% at
    # small Cn — that is why it passed just 11% of the Ekman transport.)
    east_radiation = NormalRadiation(outflow_timescale=30days,
        inflow_timescale=1days,
        target_transport=0)

    # Exterior T/S on the inflow branch. Must vary with depth, or the deep
    # return flow imports surface properties and destroys the stratification at
    # the boundary. Discrete form (j, k, grid, clock, fields) — the only
    # function-valued exterior state exercised in Oceananigans' open-boundary
    # tests. NOTE: this uses the SOUTH profile, matching the interior water
    # mass. T_east_pwl / S_east_pwl above define a distinct offshore water mass
    # and are currently unused; swapping them in would impose a cross-shore
    # front, which is a deliberate experiment rather than a default.
    # Literals, not T_south_v1/S_south_v1: those are non-const globals, and a
    # kernel function that closes over one gets it boxed, which can trigger
    # InvalidIRError on GPU.
    @inline T_east_bc(j, k, grid, clock, fields) = T_south_pwl(znode(k, grid, Center()), 24.5378)
    @inline S_east_bc(j, k, grid, clock, fields) = S_south_pwl(znode(k, grid, Center()), 35.5830)

    T_bcs = FieldBoundaryConditions(east=ValueBoundaryCondition(T_east_bc; scheme=east_radiation, discrete_form=true))
    S_bcs = FieldBoundaryConditions(east=ValueBoundaryCondition(S_east_bc; scheme=east_radiation, discrete_form=true))
    u_bcs = FieldBoundaryConditions(immersed=immersed_drag, bottom=drag, top=wind_bc_u,
        east=NormalFlowBoundaryCondition(0; scheme=east_radiation))
    v_bcs = FieldBoundaryConditions(immersed=immersed_drag, bottom=drag, top=wind_bc_v)

    # Barotropic transport: exterior (U, η) = (0, 0). This is what holds the net
    # mass flux at zero. Omit it and the baroclinic mode radiates while the
    # barotropic mode reflects off a closed boundary — worse than either choice.
    U_bcs = FieldBoundaryConditions(ib_grid, (Face(), Center(), nothing);
        east=GravityWaveRadiationBoundaryCondition((0.0, 0.0)))
elseif periodic_y
    T_bcs = FieldBoundaryConditions()
    S_bcs = FieldBoundaryConditions()
    u_bcs = FieldBoundaryConditions(immersed=immersed_drag, bottom=drag, top=wind_bc_u)
    v_bcs = FieldBoundaryConditions(immersed=immersed_drag, bottom=drag, top=wind_bc_v)
else
    northern_bc = NormalFlowBoundaryCondition(v∞; parameters=params, scheme=PerturbationAdvection(inflow_timescale=0.0, outflow_timescale=0.0))
    southern_bc = NormalFlowBoundaryCondition(v∞; parameters=params, scheme=PerturbationAdvection(inflow_timescale=0.0, outflow_timescale=0.0))
    eastern_bc = NormalFlowBoundaryCondition(0.0; scheme=PerturbationAdvection(inflow_timescale=0.0, outflow_timescale=Inf))

    T_bcs = FieldBoundaryConditions(south=ValueBoundaryCondition(tsbc), north=ValueBoundaryCondition(tnbc))
    S_bcs = FieldBoundaryConditions(south=ValueBoundaryCondition(ssbc), north=ValueBoundaryCondition(snbc))

    u_bcs = FieldBoundaryConditions(immersed=immersed_drag, bottom=drag, south=ValueBoundaryCondition(0.0; scheme=PerturbationAdvection(inflow_timescale=Inf, outflow_timescale=0.0)), east=eastern_bc, top=wind_bc_u)
    v_bcs = FieldBoundaryConditions(immersed=immersed_drag, bottom=drag, north=northern_bc, south=southern_bc, east=ValueBoundaryCondition(0.0), top=wind_bc_v)
end

bcs = (periodic_y && open_east) ? (u=u_bcs, v=v_bcs, T=T_bcs, S=S_bcs, U=U_bcs) :
      (u=u_bcs, v=v_bcs, T=T_bcs, S=S_bcs)
@info "Boundary Conditions:" bcs

if is_coriolis
    coriolis = FPlane(latitude=35.2480)
else
    coriolis = nothing
end

# ═══════════════════════════════════════════════════════════════════════════
# Turbulence closure
# ═══════════════════════════════════════════════════════════════════════════
# Unused: see the closure assignment below.
# horizontal_closure = HorizontalScalarDiffusivity(ν=1e-3, κ=1e-3)

if closure_choice === :catke
    vertical_closure = CATKEVerticalDiffusivity()
    tracers = (:T, :S)
elseif closure_choice === :ribased
    vertical_closure = RiBasedVerticalDiffusivity()
    tracers = (:T, :S)
elseif closure_choice === :constant
    vertical_closure = VerticalScalarDiffusivity(ν=1e-4, κ=1e-5)
    tracers = (:T, :S)
else
    error("closure_choice must be :catke, :ribased or :constant; got $(closure_choice)")
end

# Vertical only, matching flow_over_shoals_hydrostatic.jl. At dx = 500 m with
# Rd ~ 1.4 km the scale separation is 2.7x, so no biharmonic nu4 damps 2dx
# without also damping the eddies; the harmonic nu = 1e-3 that used to be here
# was negligible in any case (e-folding at 2dx of ~30 years).
closure = vertical_closure
@info "Closure ($(closure_choice)):" closure

# ═══════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════
# TEOS10 rather than the default LinearEquationOfState, whose alpha = 1.67e-4 is
# a ~10 C value; at this site's 24-26 C the true thermal expansion is 2.94e-4.
# It barely moves N^2 here (salinity dominates drho/dz) but gets the SIGN of any
# horizontal T contrast wrong.
if eos_choice === :teos10
    seawater_buoyancy = SeawaterBuoyancy(equation_of_state=TEOS10EquationOfState(reference_density=ρ₀))
elseif eos_choice === :linear
    seawater_buoyancy = SeawaterBuoyancy()
else
    error("eos_choice must be :teos10 or :linear; got $(eos_choice)")
end
@info "Equation of state ($(eos_choice)):" seawater_buoyancy

model = HydrostaticFreeSurfaceModel(ib_grid;
    momentum_advection=WENO(order=5),
    tracer_advection=WENO(order=5),
    free_surface=SplitExplicitFreeSurface(ib_grid; cfl=0.7),
    tracers=tracers,
    buoyancy=seawater_buoyancy,
    coriolis=coriolis,
    closure=closure,
    boundary_conditions=bcs,
    forcing=forcings
)

@info "" model

# Check for environment variable SWEEP_PICKUP or local checkpoint files
sweep_pickup_env = get(ENV, "SWEEP_PICKUP", "")
if !isempty(sweep_pickup_env) && isfile(sweep_pickup_env)
    pickup = sweep_pickup_env
    sim_runtime = 100days
    @info "Picking up from specified checkpoint: $(pickup). Setting runtime to 100 days."
else
    checkpoint_files = filter(f -> startswith(f, "checkpoint_") && endswith(f, ".jld2"), readdir("."))
    if !isempty(checkpoint_files)
        pickup = last(sort(checkpoint_files))
        sim_runtime = 100days
        @info "Found local checkpoint: $(pickup). Setting runtime to 100 days."
    else
        pickup = false
        sim_runtime = 50days
        @info "No checkpoint found. Running 50-day run."
    end
end
overwrite_files = (pickup === false)

# initial_Δt was 5 minutes with NO max_Δt cap. flow_over_shoals_hydrostatic.jl
# uses 1 minute and caps at 10 minutes. The unbalanced start plus a radiative
# east boundary makes the first few hundred steps the riskiest phase, and an
# uncapped wizard can walk Δt up while the barotropic substep count follows it.
simulation = Simulation(model, Δt=1minutes, stop_time=sim_runtime)
conjure_time_step_wizard!(simulation, cfl=0.7, max_Δt=10minutes)

progress = TimedMessenger()
simulation.callbacks[:progress] = Callback(progress, TimeInterval(callback_interval))

# ═══════════════════════════════════════════════════════════════════════════
# Health check
# ═══════════════════════════════════════════════════════════════════════════
# Without this a blow-up surfaces as `InexactError: Int64(NaN)` from
# calculate_substeps, deep inside the free-surface substepping. That only says
# Δt had already become NaN — not which field failed, where, or when. This
# reports the offending field and the cross-shore column it first appears in.
function report_health(sim)
    m = sim.model
    flds = (u=m.velocities.u, v=m.velocities.v, w=m.velocities.w,
        T=m.tracers.T, S=m.tracers.S)
    bad = String[]
    for (name, f) in pairs(flds)
        colmax = Array(vec(maximum(abs, interior(f), dims=(2, 3))))
        if !all(isfinite, colmax)
            i = findfirst(!isfinite, colmax)
            xkm = round((i - 0.5) * params.Lx / params.Nx / 1e3, digits=1)
            push!(bad, string(name, " first at i=", i, " (x ≈ ", xkm, " km)"))
        end
    end
    if !isempty(bad)
        error(string("BLOW-UP, not a solver bug.",
            "\n  iteration = ", iteration(sim),
            ", t = ", round(time(sim) / 3600, digits=2), " h",
            ", Δt = ", round(sim.Δt, digits=1), " s",
            "\n  non-finite: ", join(bad, "; "),
            "\n  run_label = ", sweep_run_label,
            ", tau = ", sweep_wind_stress,
            ", Zs = ", sweep_Zs,
            ", Ls = ", sweep_shoal_length,
            ", Zsh = ", sweep_Zsh))
    end
    @info @sprintf("health: iter %6d | t = %7.3f d | Δt = %6.1f s | max|u| = %.4f | max|v| = %.4f | max|w| = %.2e",
        iteration(sim), time(sim) / 86400, sim.Δt,
        maximum(abs, interior(m.velocities.u)),
        maximum(abs, interior(m.velocities.v)),
        maximum(abs, interior(m.velocities.w)))
end
simulation.callbacks[:health] = Callback(report_health, IterationInterval(25))

u, v, w = model.velocities
T = model.tracers.T
S = model.tracers.S
Ro = @at (Center, Center, Center) RossbyNumber(model)
KE = @at (Center, Center, Center) KineticEnergy(model)
b_op = Oceananigans.Models.buoyancy_operation(model)
PV = @at (Center, Center, Center) ErtelPotentialVorticity(model, u, v, w, b_op, model.coriolis)

# Centered velocities for consistency
u_c = @at (Center, Center, Center) u
v_c = @at (Center, Center, Center) v
w_c = @at (Center, Center, Center) w

# Cross-correlations for EKE and Fluxes
# EKE = 0.5 * (⟨uu⟩ - ⟨u⟩² + ⟨vv⟩ - ⟨v⟩² + ⟨ww⟩ - ⟨w⟩²)  — computed in post-processing from tavg_fields
uu = Field(u_c * u_c)
vv = Field(v_c * v_c)
ww = Field(w_c * w_c)
uT = Field(u_c * T)
uS = Field(u_c * S)
vT = Field(v_c * T)
vS = Field(v_c * S)
wT = Field(w_c * T)
wS = Field(w_c * S)

slice_fields = (; u_c, v_c, w_c, T, S, Ro, KE, PV)
tavg_fields = (; u_c, v_c, w_c, uu, vv, ww, T, S, uT, uS, vT, vS, wT, wS)

# (1) 2D snapshots (every 1 day)
# Surface XY slice (top layer)
simulation.output_writers[:surface_slice] = NetCDFWriter(model, slice_fields,
    filename="top_$(run_tag).nc",
    schedule=TimeInterval(snapshot_interval),
    indices=(:, :, params.Nz),
    overwrite_files=overwrite_files)

# Mid-y XZ slice (cross-shore transect at domain center)
simulation.output_writers[:midy_slice] = NetCDFWriter(model, slice_fields,
    filename="midy_$(run_tag).nc",
    schedule=TimeInterval(snapshot_interval),
    indices=(:, round(Int, params.Ny / 2), :),
    overwrite_files=overwrite_files)

# Mid-x YZ slice (along-shore transect at domain center)
simulation.output_writers[:midx_slice] = NetCDFWriter(model, slice_fields,
    filename="midx_$(run_tag).nc",
    schedule=TimeInterval(snapshot_interval),
    indices=(round(Int, params.Nx / 5), :, :),
    overwrite_files=overwrite_files)

# (3) 3D Time Averages (10 day window)
simulation.output_writers[:time_avg_3d] = NetCDFWriter(model, tavg_fields,
    filename="time_avg_3d_$(run_tag).nc",
    schedule=AveragedTimeInterval(10days, window=10days),
    overwrite_files=overwrite_files)

# ── Save sweep metadata to a small NetCDF file for postprocessing ──────
using NCDatasets
NCDatasets.Dataset("sweep_metadata_$(run_tag).nc", "c") do ds
    ds.attrib["run_label"] = sweep_run_label
    ds.attrib["run_index"] = sweep_run_index
    ds.attrib["Zs"] = sweep_Zs
    ds.attrib["shoal_length"] = sweep_shoal_length
    ds.attrib["Zsh"] = sweep_Zsh
    ds.attrib["shelf_break_end"] = sweep_shelf_break_end
    ds.attrib["wind_stress"] = sweep_wind_stress
    ds.attrib["v0"] = sweep_v0
end

# initial conditions
if pickup === false
    @info "Setting initial conditions"
    if sigmoid_ic
        v_init = (x, y, z) -> v∞(x, z, 0, params)
    else
        v_init = v₀
    end

    if gradient_IC
        @inline α_lin(y) = clamp(y / global_params.Ly, 0.0, 1.0)
        @inline blend(a, b, α) = (1 - α) * a + α * b
        @inline Tᵢ(x, y, z) = blend(T_south_pwl(z, global_params.T_south_v1), T_north_pwl(z, global_params.T_north_v1), α_lin(y))
        @inline Sᵢ(x, y, z) = blend(S_south_pwl(z, global_params.S_south_v1), S_north_pwl(z, global_params.S_north_v1), α_lin(y))
    else
        @inline Tᵢ(x, y, z) = T_south_pwl(z, global_params.T_south_v1)
        @inline Sᵢ(x, y, z) = S_south_pwl(z, global_params.S_south_v1)
    end

    if closure_choice === :catke
        set!(model, u=0.0, v=v_init, T=Tᵢ, S=Sᵢ, e=1e-6)
    else
        set!(model, u=0.0, v=v_init, T=Tᵢ, S=Sᵢ)
    end
else
    @info "Skipping initial conditions setup — loading fields from checkpoint $(pickup)."
end

# run simulation
@info """
════════════════════════════════════════════════════════
 SWEEP SIMULATION: $(run_tag)
════════════════════════════════════════════════════════
 Run label:       $(sweep_run_label)
 Run index:       $(sweep_run_index)
 Runtime:         $(sim_runtime)
 Architecture:    $(arch)
 Domain:          $(params.Lx/1e3) x $(params.Ly/1e3) km x $(params.Lz) m
 Grid:            $(params.Nx) x $(params.Ny) x $(params.Nz)  (Δx = Δy = $(params.Lx/params.Nx) m, Δz = $(params.Lz/params.Nz) m)
 EOS:             TEOS10 (ρ_ref = $(ρ₀))
 east boundary:   $(open_east ? "radiative (NormalRadiation, target_transport=0)" : "wall + sponge")
 snapshot every:  $(snapshot_interval/3600) h  (inertial period 20.74 h)

 ── Sweep Parameters ──
 Zs (shoal_depth):$(sweep_Zs) m
 shoal_length:    $(sweep_shoal_length) m
 Zsh (shelf_depth):$(sweep_Zsh) m
 shelf_break_end: $(sweep_shelf_break_end) m
 wind_stress:     $(sweep_wind_stress) N/m^2
 v0:              $(sweep_v0) m/s

 ── Switches ──
 LES:             $(LES)
 mass_flux:       $(mass_flux)
 periodic_y:      $(periodic_y)
 gradient_IC:     $(gradient_IC)
 sigmoid_v_bc:    $(sigmoid_v_bc)
 sigmoid_ic:      $(sigmoid_ic)
 sigmoid_wind:    $(sigmoid_wind)
 is_coriolis:     $(is_coriolis)
 shoal_bath:      $(shoal_bath)
 open_east:       $(open_east)
════════════════════════════════════════════════════════
"""
run!(simulation, pickup=pickup)
