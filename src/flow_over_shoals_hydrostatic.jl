# ═══════════════════════════════════════════════════════════════════════════
# flow_over_shoals_hydrostatic2.jl
# ═══════════════════════════════════════════════════════════════════════════
# Alternative route to the same goal as flow_over_shoals_hydrostatic.jl — a
# stable, artifact-free, grid-noise-free M2-tidal flow over the shoal.
#
# Identical physics and numerics EXCEPT the horizontal dissipation strategy:
#   • flow_over_shoals_hydrostatic.jl: flux-form WENO momentum advection damped by
#     a heavy *explicit* stack — Laplacian (ν_h) + biharmonic (ν₄) hyperviscosity.
#   • THIS script: WENOVectorInvariant momentum advection, whose vorticity-flux
#     upwinding supplies scale-selective dissipation intrinsically, plus only a
#     *light* Laplacian background (no biharmonic) — a less heavy-handed closure.
#
# Both share: HydrostaticFreeSurfaceModel + SplitExplicitFreeSurface, CATKE
# vertical mixing, BulkDrag on bottom + immersed seafloor, a body-force M2 tide
# (ramped) with gentle north/offshore reset sponges, and the tightened CFL.
#
# Geometry/forcing read from SWEEP_* env vars (standalone defaults); resolution
# and runtime overridable via NX/NY/NZ and SIM_RUNTIME_DAYS. See
# src/NUMERICAL_ARTIFACTS.md for the dissipation/closure rationale.
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

#+++ Hacks
import Oceananigans.TurbulenceClosures: cell_diffusion_timescale

cell_diffusion_timescale(closure::CATKEVerticalDiffusivity, diffusivities, grid, clock, fields) = Inf
#---

# ═══════════════════════════════════════════════════════════════════════════
# Read sweep parameters from environment (set by sweep_driver.jl)
# Falls back to defaults so script can also be run standalone.
# ═══════════════════════════════════════════════════════════════════════════
sweep_Hs = parse(Float64, get(ENV, "SWEEP_Hs", "15.0"))
sweep_shoal_length = parse(Float64, get(ENV, "SWEEP_SHOAL_LENGTH", "20000.0"))
sweep_sigma = parse(Float64, get(ENV, "SWEEP_SIGMA", "8000.0"))
sweep_shelf_depth = parse(Float64, get(ENV, "SWEEP_SHELF_DEPTH", "-25.0"))
sweep_shelf_break_end = parse(Float64, get(ENV, "SWEEP_SHELF_BREAK_END", "12000.0"))
sweep_run_label = get(ENV, "SWEEP_RUN_LABEL", "standalone")
sweep_run_index = parse(Int, get(ENV, "SWEEP_RUN_INDEX", "0"))
sweep_strat = get(ENV, "SWEEP_STRAT", "default")
sweep_wind_stress = parse(Float64, get(ENV, "SWEEP_WIND_STRESS", "0.0"))

@info "Sweep parameters: Hs=$sweep_Hs, shoal_length=$sweep_shoal_length, sigma=$sweep_sigma, shelf_depth=$sweep_shelf_depth, shelf_break_end=$sweep_shelf_break_end, strat=$sweep_strat, wind_stress=$sweep_wind_stress"

# build
@info "building domain"

# switches
mass_flux = true
gradient_IC = false
sigmoid_v_bc = true
sigmoid_ic = true
checkpointing = true
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
sim_runtime = parse(Float64, get(ENV, "SIM_RUNTIME_DAYS", "30")) * days
callback_interval = parse(Float64, get(ENV, "CALLBACK_HOURS", "24")) * hours
run_tag = "hydrostatic2_$(sweep_run_label)"

params = (; Lx=100e3, Ly=200e3, Lz=50)
if arch == CPU()
    params = (; params..., Nx=30, Ny=60, Nz=10)
else
    params = (; params..., Nx=200, Ny=400, Nz=50)
end
# Optional resolution overrides (NX/NY/NZ env vars) for resolution testing.
params = (; params...,
    Nx=parse(Int, get(ENV, "NX", string(params.Nx))),
    Ny=parse(Int, get(ENV, "NY", string(params.Ny))),
    Nz=parse(Int, get(ENV, "NZ", string(params.Nz))))

x, y, z = (0, params.Lx), (0, params.Ly), (-params.Lz, 0)
# halo=(7,7,5): WENOVectorInvariant's 9th-order vorticity reconstruction needs ≥7 horizontal halo points.
grid = RectilinearGrid(arch; size=(params.Nx, params.Ny, params.Nz), halo=(7, 7, 5), x, y, z, topology=(Bounded, Periodic, Bounded))

# model parameters
slope_bottom = dshoal_param_bottom(params.Ly;
    Hs=sweep_Hs,
    shoal_length=sweep_shoal_length,
    sigma=sweep_sigma,
    shelf_depth=sweep_shelf_depth,
    shelf_break_end=sweep_shelf_break_end)
GFB = GridFittedBottom(slope_bottom)
ib_grid = ImmersedBoundaryGrid(grid, GFB)

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

if sweep_strat == "winter"
    T_north_v1, S_north_v1 = 13.17, 34.54
    T_south_v1, S_south_v1 = 20.37, 36.28
elseif sweep_strat == "summer"
    T_north_v1, S_north_v1 = 24.45, 32.74
    T_south_v1, S_south_v1 = 27.34, 35.83
end

params = (; params...,
    v₀=v₀,
    Ls=10e3,
    Le=40e3,
    Lw=10e3,
    τₙ=24hours,    # gentle north reset (was 6h — too fast, drove a NE-corner instability)
    τₛ=24hours,
    τₑ=24hours,
    τw=24hours,
    τ_ts=24hours,
    T_north_v1=T_north_v1,
    T_south_v1=T_south_v1,
    S_north_v1=S_north_v1,
    S_south_v1=S_south_v1,
    wind_stress=sweep_wind_stress,
    ω_M2=2π / 12.4206hours,   # M2 tidal angular frequency (period ≈ 12.42 h)
    τ_ramp=2days)             # tidal spin-up ramp timescale

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

# Eastern boundary targets are now functions of z
params = (; params...)

#+++ Drag (Implemented as in https://doi.org/10.1029/2005WR004685)
z₀ = 2.5e-4 # roughness length
z₁ = Oceananigans.Grids.minimum_zspacing(grid, Center(), Center(), Center()) / 2
@info "Using z₁ =" z₁

const κᵛᵏ = 0.4 # von Karman constant
params = (; params..., c_dz=(κᵛᵏ / log(z₁ / z₀))^2) # quadratic drag coefficient
@info "Defining momentum BCs with Cᴰ =" params.c_dz

# Quadratic bulk drag with the law-of-the-wall coefficient c_dz from above.
# `BulkDrag` returns a FluxBoundaryCondition whose direction is inferred from
# each velocity's location; it is applied below on both the domain bottom (the
# offshore seafloor at z = -Lz) and the immersed boundary (shelf/shoal topography).
drag = BulkDrag(coefficient=params.c_dz)
#---
# wind stress BC
ρ₀ = 1024.0
wind_bc_v = FluxBoundaryCondition(-sweep_wind_stress / ρ₀)

# ─────────────────────────────────────────────────────────────────────────
# Nudging (sponge) masks — Oceananigans `PiecewiseLinearMask{D}(center, width)`
# is a tent along direction D: value 1 at `center`, ramping linearly to 0 at
# `center ± width`, and 0 beyond. Each mask is centered ON the boundary it
# nudges toward, with `width` equal to the same sponge thickness used before,
# so only the in-domain half of the tent acts — i.e. a one-sided ramp from 0
# in the interior to 1 at the boundary, over the same distances as before.
# Called as mask(x, y, z) (no parameters argument).
# ─────────────────────────────────────────────────────────────────────────
const south_mask    = PiecewiseLinearMask{:y}(center=0.0,       width=params.Ls)  # y ∈ [0, Ls]
const north_mask    = PiecewiseLinearMask{:y}(center=params.Ly, width=params.Ls)  # y ∈ [Ly-Ls, Ly]
const west_mask     = PiecewiseLinearMask{:x}(center=0.0,       width=params.Lw)  # x ∈ [0, Lw]
const east_mask     = PiecewiseLinearMask{:x}(center=params.Lx, width=params.Le)  # x ∈ [Lx-Le, Lx]
# offshore: linear ramp over the eastern/offshore zone, from the shelf break
# (x = Lx - Le = 60 km) up to the offshore boundary (x = Lx = 100 km).
const offshore_mask = PiecewiseLinearMask{:x}(center=params.Lx, width=params.Le)

# velocity function — barotropic M2 tide.
# `sc_shape(x)` is the cross-shore amplitude shape (≈1 over the shelf, → 0
# offshore). The tide is v∞ = v₀·sc·sin(ω t). Rather than imposing this tide with
# a thin, fast sponge — which drove a NE-corner grid-scale instability (the thin
# strip tried to track the 12.4 h tide at τₙ=6h; see src/NUMERICAL_ARTIFACTS.md) —
# we drive it with a smooth, domain-wide body force F_tide = dv∞/dt =
# v₀·ω·sc·cos(ω t), and keep the sponges as *gentle* reset layers (slow τ).
if sigmoid_v_bc
    @inline function sc_shape(x, p)
        xC = 3e3
        xS = 60e3
        Lw = p.Lx
        k1 = 40 / Lw
        k2 = 20 / Lw
        s1 = 1 / (1 + exp(-k1 * (x - xC)))
        s2 = 1 / (1 + exp(k2 * (x - xS)))
        return clamp((s1 - 1) + s2, 0.0, 1.0)
    end
else
    @inline sc_shape(x, p) = 1.0
end
# Smooth spin-up ramp (≈ 1 - e^{-t/τ_ramp}) so the tide eases in over τ_ramp ≫ a
# tidal period instead of hitting full amplitude in the first cycles.
@inline tide_amp(t, p)    = p.v₀ * (1 - exp(-t / p.τ_ramp))
@inline v∞(x, z, t, p)    = tide_amp(t, p) * sc_shape(x, p) * sin(p.ω_M2 * t)            # tidal velocity
@inline F_tide_v(x, t, p) = tide_amp(t, p) * p.ω_M2 * sc_shape(x, p) * cos(p.ω_M2 * t)   # body force ≈ dv∞/dt

# T_0 / S_0: the initial-condition T and S profiles (the CTD-derived "south"
# profile — the same stratification imposed as the initial condition below). The
# northern sponge nudges the flow back to these.
@inline T_0(z) = T_south_pwl(z, T_south_v1)
@inline S_0(z) = S_south_pwl(z, S_south_v1)

# Sponge nudging at the northern and eastern (offshore) edges. Two
# PiecewiseLinearMask ramps drive it: `north_mask` (0 → 1 over the last Ls at
# y = Ly) and `offshore_mask` (0 → 1 over [Lx-Le, Lx], i.e. 60 → 100 km). Each
# relaxes the flow toward a reference state:
#   • north (timescales τₙ / τ_ts): the initial condition — u → 0, v → v∞,
#     T → T_0, S → S_0 — so the shoal wake doesn't recirculate through the
#     periodic southern boundary;
#   • east  (timescales τₑ / τ_ts): the offshore state — u → 0, v → v∞,
#     T → T_east, S → S_east — a Davies layer that absorbs disturbances before
#     the eastern wall.
# T/S north targets use p.T_south_v1 / p.S_south_v1 (= T_0/S_0) for GPU safety;
# T_east_pwl / S_east_pwl are plain functions of z.
if mass_flux
    @inline sponge_u(x, y, z, t, u, p) = -(
        north_mask(x, y, z) * u / p.τₙ +
        offshore_mask(x, y, z) * u / p.τₑ)
    @inline sponge_v(x, y, z, t, v, p) = F_tide_v(x, t, p) - (
        north_mask(x, y, z) * (v - v∞(x, z, t, p)) / p.τₙ +
        offshore_mask(x, y, z) * (v - v∞(x, z, t, p)) / p.τₑ)
    @inline sponge_T(x, y, z, t, T, p) = -(
        north_mask(x, y, z) * (T - T_south_pwl(z, p.T_south_v1)) / p.τ_ts +
        offshore_mask(x, y, z) * (T - T_east_pwl(z)) / p.τ_ts)
    @inline sponge_S(x, y, z, t, S, p) = -(
        north_mask(x, y, z) * (S - S_south_pwl(z, p.S_south_v1)) / p.τ_ts +
        offshore_mask(x, y, z) * (S - S_east_pwl(z)) / p.τ_ts)
end

# forcing functions
FT = Forcing(sponge_T, field_dependencies=:T, parameters=params)
FS = Forcing(sponge_S, field_dependencies=:S, parameters=params)
if mass_flux
    # No w forcing: w is diagnostic in the hydrostatic model.
    Fᵤ = Forcing(sponge_u, field_dependencies=:u, parameters=params)
    Fᵥ = Forcing(sponge_v, field_dependencies=:v, parameters=params)
    forcings = (u=Fᵤ, v=Fᵥ, T=FT, S=FS)
else
    forcings = (T=FT, S=FS)
end

# y is periodic — no north/south boundary conditions. The eastern (offshore) and
# western boundaries are closed walls; the eastern wall is fronted by the offshore
# sponge above (a Davies nudging layer), and T/S are no-flux on the walls.
# Quadratic bulk drag acts on the domain bottom (offshore seafloor) and the
# immersed seafloor (shelf/shoal).
T_bcs = FieldBoundaryConditions()
S_bcs = FieldBoundaryConditions()
u_bcs = FieldBoundaryConditions(bottom=drag, immersed=drag)
v_bcs = FieldBoundaryConditions(bottom=drag, immersed=drag, top=wind_bc_v)

# No w boundary conditions: w is diagnostic in the hydrostatic model.
bcs = (u=u_bcs, v=v_bcs, T=T_bcs, S=S_bcs)
coriolis = FPlane(latitude=35.2480)

#+++ Horizontal dissipation (alternative "lighter" route) — see src/NUMERICAL_ARTIFACTS.md
# Where flow_over_shoals_hydrostatic.jl damps grid-scale (2Δx) noise with a heavy
# *explicit* stack (Laplacian + biharmonic), this variant leans on the advection
# scheme instead: WENOVectorInvariant momentum advection (below) upwinds the
# vorticity flux and supplies scale-selective dissipation intrinsically (its
# default vorticity reconstruction is 9th order). We therefore keep only a *light*
# Laplacian background — half of the other script's ν_h and no biharmonic — to
# quiet low-shear regions / internal waves, plus CATKE for vertical mixing.
# ν_h scales ∝ Δx² from the guide's Δx = 1.7 km baseline (ν_h = 20 → here 10 m²/s).
Δh = params.Lx / params.Nx                       # ≈ Δy (square cells)
ν_h = 10.0 * (Δh / 1700)^2                        # light Laplacian viscosity (m²/s)
@info "Horizontal closure (light Laplacian + WENOVectorInvariant)" Δh ν_h
closure = (HorizontalScalarDiffusivity(ν=ν_h, κ=ν_h / 4),   # light always-on background; Pr_h = 4
           CATKEVerticalDiffusivity())                      # vertical mixing
#---

#+++ Create model
free_surface = SplitExplicitFreeSurface(ib_grid; cfl=0.5)
model = HydrostaticFreeSurfaceModel(ib_grid;
    timestepper = :QuasiAdamsBashforth2,
    momentum_advection = WENOVectorInvariant(),   # vorticity-form WENO: built-in scale-selective dissipation
    tracer_advection = WENO(order=5),
    free_surface = free_surface,
    tracers = (:T, :S),
    buoyancy = SeawaterBuoyancy(),
    coriolis = coriolis,
    boundary_conditions = bcs,
    closure = closure,
    forcing = forcings
)
@info "" model
#---

#+++ Create simulation
pickup = isfile("checkpoint_$(run_tag).jld2")
overwrite_existing = !pickup

simulation = Simulation(model, Δt=2minutes, stop_time=sim_runtime)
conjure_time_step_wizard!(simulation, IterationInterval(10); cfl=0.15, max_Δt=15minutes)

progress = TimedMessenger()
simulation.callbacks[:progress] = Callback(progress, TimeInterval(callback_interval))
#---

#+++ Output: a single writer with all state variables, every 12 hours
# State variables: velocities (u, v, w) + tracers (T, S) + free-surface η.
η = model.free_surface.displacement
state_fields = merge(model.velocities, model.tracers, (; η))
simulation.output_writers[:fields] = JLD2Writer(model, state_fields,
    filename="fields_$(run_tag).jld2",
    schedule=TimeInterval(3hours),
    overwrite_existing=overwrite_existing)
#---

#+++ initial conditions
@info "Setting initial conditions"
v_init = sigmoid_ic ? ((x, y, z) -> v∞(x, z, 0, params)) : v₀

if gradient_IC
    @inline α_lin(y) = clamp(y / params.Ly, 0.0, 1.0)
    @inline blend(a, b, α) = (1 - α) * a + α * b
    @inline Tᵢ(x, y, z) = blend(T_south_pwl(z, T_south_v1), T_north_pwl(z, T_north_v1), α_lin(y))
    @inline Sᵢ(x, y, z) = blend(S_south_pwl(z, S_south_v1), S_north_pwl(z, S_north_v1), α_lin(y))
else
    @inline Tᵢ(x, y, z) = T_0(z)
    @inline Sᵢ(x, y, z) = S_0(z)
end

# w is diagnostic (recomputed from continuity), so it is not set here.
set!(model, u=0.0, v=v_init, T=Tᵢ, S=Sᵢ)
#---

#+++ run simulation
@info """
════════════════════════════════════════════════════════
 HYDROSTATIC SIMULATION: $(run_tag)
 (HydrostaticFreeSurfaceModel + SplitExplicitFreeSurface + WENOVectorInvariant)
════════════════════════════════════════════════════════
 Run label:       $(sweep_run_label)
 Run index:       $(sweep_run_index)
 Runtime:         $(sim_runtime)
 Architecture:    $(arch)

 ── Sweep Parameters ──
 Hs:              $(sweep_Hs) m
 shoal_length:    $(sweep_shoal_length) m
 shelf_depth:     $(sweep_shelf_depth) m
 shelf_break_end: $(sweep_shelf_break_end) m
 strat:           $(sweep_strat)
 wind_stress:     $(sweep_wind_stress) N/m^2

 ── Switches ──
 mass_flux:       $(mass_flux)
 gradient_IC:     $(gradient_IC)
 sigmoid_v_bc:    $(sigmoid_v_bc)
 sigmoid_ic:      $(sigmoid_ic)
════════════════════════════════════════════════════════
"""
run!(simulation, pickup=pickup)
#---

#+++ Animate the output
if get(ENV, "SKIP_PLOT", "false") != "true"
    try
        include(joinpath(@__DIR__, "plot_hydrostatic_simulation.jl"))
    catch err
        @warn "Skipped animation (expected on headless/GPU nodes without Plots)" exception = (err, catch_backtrace())
    end
end
#---