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
using Oceananigans.Grids: Periodic, Bounded, xnode, ynode, znode
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
sim_runtime = parse(Float64, get(ENV, "SIM_RUNTIME_DAYS", "20")) * days
run_tag = "hydrostatic2_$(sweep_run_label)"

params = (; Lx=200e3, Ly=400e3, Lz=50)
if arch == CPU()
    params = (; params..., Nx=60, Ny=120, Nz=10)
else
    params = (; params..., Nx=400, Ny=800, Nz=50)   # Δx=Δy=0.5 km (same resolution as the old 100×200 km domain)
end
# Optional resolution overrides (NX/NY/NZ env vars) for resolution testing.
params = (; params...,
    Nx=parse(Int, get(ENV, "NX", string(params.Nx))),
    Ny=parse(Int, get(ENV, "NY", string(params.Ny))),
    Nz=parse(Int, get(ENV, "NZ", string(params.Nz))))

# Resolution-aware horizontal-closure controls (see the closure block below).
#   Re₀          base Reynolds-number coefficient (tunable; larger ⇒ lower ν_h)
#   closure_dims 2 ⇒ p=2 (2-D scaling), 3 ⇒ p=4/3 (3-D scaling)
#   U_closure    characteristic horizontal velocity scale (m/s) ≈ tidal+mean flow
# Default Re₀=32 ⇒ ν_h ≈ 0.0078 m²/s at 200×400×50 (κ_h ≈ 0.002). Chosen from the
# src/tune_closure.jl sweep: it sits at the resolved-enstrophy plateau (not
# overdamped — unlike ν≈1) yet ~110× below the previous hard-coded 0.865 m²/s, with
# WENOVectorInvariant providing the bulk of the scale-selective dissipation. Lower
# ν changes the solution by <1% and removes the stability margin.
params = (; params...,
    Re₀=parse(Float64, get(ENV, "RE0", "32.0")),
    closure_dims=parse(Int, get(ENV, "CLOSURE_DIMS", "2")),
    U_closure=0.1)

x, y, z = (0, params.Lx), (0, params.Ly), (-params.Lz, 0)
# halo=(7,7,5): WENOVectorInvariant's 9th-order vorticity reconstruction needs ≥7 horizontal halo points.
grid = RectilinearGrid(arch; size=(params.Nx, params.Ny, params.Nz), halo=(7, 7, 5), x, y, z, topology=(Bounded, Periodic, Bounded))

# model parameters
# Keep the shoal at its ORIGINAL physical location even though the along-shore domain
# is now 400 km. The shoal y-center is Ly_bathy/2, so we pass the original 200 km length
# (⇒ center at y = 100 km), not params.Ly (which would re-center it at y = 200 km). The
# cross-shore profile is defined in absolute km, so the wider Lx leaves it unaffected.
Ly_bathy = 200e3
slope_bottom = dshoal_param_bottom(Ly_bathy;
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

# Master sponge relaxation timescale, env-tunable for nudging-strength sweeps.
τ_sponge = parse(Float64, get(ENV, "SPONGE_TAU_HOURS", "24")) * hours
params = (; params...,
    v₀=v₀,
    Ls=100e3,   # north sponge half-width: mask centered at Ly-Ls=300 km, width Ls ⇒ active y∈[200,400] km
    Le=100e3,   # offshore (east) sponge width: mask centered at Lx=200 km, width Le ⇒ active x∈[100,200] km
    Lw=10e3,
    τₙ=τ_sponge,
    τₛ=τ_sponge,
    τₑ=τ_sponge,
    τw=τ_sponge,
    τ_ts=τ_sponge,
    τ_eta=τ_sponge,
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
# `center ± width`, and 0 beyond. Two masks are active here (called as
# mask(x, y, z), no parameters argument):
#
#   • north_mask: a y-tent CENTERED at y = Ly - Ls = 300 km, width Ls = 100 km.
#     It is nonzero on y ∈ [200, 400] km, PEAKS at y = 300 km, and returns to ZERO
#     at the periodic seam (y = 0 ≡ Ly = 400 km) and at y = 200 km. Peaking in the
#     interior (rather than on the boundary) keeps the forcing continuous across the
#     periodic-y seam — a boundary-peaked one-sided ramp would jump from full strength
#     to zero across the seam and seed grid-scale noise there.
#   • offshore_mask: an x-tent centered on the offshore wall (x = Lx = 200 km), width
#     Le = 100 km ⇒ a one-sided ramp 0 → 1 over x ∈ [100, 200] km (a Davies layer in
#     front of the eastern wall). The shoal (x ≈ 8–38 km) sits far inshore of it.
# ─────────────────────────────────────────────────────────────────────────
const north_mask    = PiecewiseLinearMask{:y}(center=params.Ly - params.Ls, width=params.Ls)
const offshore_mask = PiecewiseLinearMask{:x}(center=params.Lx,              width=params.Le)

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
        Lw = 100e3   # fixed reference length (the original Lx) so the tidal cross-shore
                     # shape is unchanged when the offshore domain is widened
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

# Sponge nudging in the northern strip and the offshore (eastern) band, via the two
# masks above. BOTH layers relax toward the SAME initial-condition reference state —
# this is the key change from the earlier setup, where the offshore layer pulled T/S
# toward a *different* offshore profile (T_east/S_east) and parked a spurious standing
# front at its inner edge:
#   • u → 0                     (IC: the fluid starts at rest)
#   • v → v∞(x, z, t)           (the prescribed background barotropic tide; the IC is v∞
#                                at t=0, i.e. this is the IC generalized in time — it
#                                absorbs perturbations without damping the imposed tide,
#                                and v∞ ≈ 0 in the offshore band so v → 0 there)
#   • T → T_0(z)=T_south_pwl,   S → S_0(z)=S_south_pwl   (the IC stratification)
#   • η → 0                     (IC free surface; see sponge_eta below)
# Timescales: τₙ/τₑ (momentum), τ_ts (tracers), τ_eta (η) — all = τ_sponge here.
# T/S targets use p.T_south_v1 / p.S_south_v1 (= T_0/S_0), passed via params for GPU safety.
#
# ⚠ GPU note: these are written in *discrete form* (discrete_form=true), i.e.
# f(i, j, k, grid, clock, model_fields, p), indexing the field directly as
# model_fields.<f>[i,j,k] and recovering (x,y,z) with xnode/ynode/znode at the
# field's own location. The natural "continuous" form (field_dependencies=:u,…)
# does NOT compile on the GPU here: ContinuousForcing's field-interpolation path
# (and the built-in Relaxation, which shares it) makes the momentum-tendency
# kernel type-unstable → InvalidIRError (gpu_gc_pool_alloc). Discrete form sidesteps
# that machinery entirely while reproducing identical physics. Each forcing gets a
# *minimal* params NamedTuple (only the fields it needs); masks are const globals.
@inline sponge_T(i, j, k, grid, clock, mf, p) = begin
    x = xnode(i, grid, Center()); y = ynode(j, grid, Center()); z = znode(k, grid, Center())
    T = @inbounds mf.T[i, j, k]
    Tᵢ = T_south_pwl(z, p.T_south_v1)   # initial-condition profile — both sponges relax to it
    -(north_mask(x, y, z) * (T - Tᵢ) / p.τ_ts +
      offshore_mask(x, y, z) * (T - Tᵢ) / p.τ_ts)
end
@inline sponge_S(i, j, k, grid, clock, mf, p) = begin
    x = xnode(i, grid, Center()); y = ynode(j, grid, Center()); z = znode(k, grid, Center())
    S = @inbounds mf.S[i, j, k]
    Sᵢ = S_south_pwl(z, p.S_south_v1)   # initial-condition profile — both sponges relax to it
    -(north_mask(x, y, z) * (S - Sᵢ) / p.τ_ts +
      offshore_mask(x, y, z) * (S - Sᵢ) / p.τ_ts)
end
if mass_flux
    @inline sponge_u(i, j, k, grid, clock, mf, p) = begin
        x = xnode(i, grid, Face()); y = ynode(j, grid, Center()); z = znode(k, grid, Center())
        u = @inbounds mf.u[i, j, k]
        -(north_mask(x, y, z) * u / p.τₙ +
          offshore_mask(x, y, z) * u / p.τₑ)
    end
    @inline sponge_v(i, j, k, grid, clock, mf, p) = begin
        x = xnode(i, grid, Center()); y = ynode(j, grid, Face()); z = znode(k, grid, Center())
        t = clock.time
        v = @inbounds mf.v[i, j, k]
        F_tide_v(x, t, p) - (
            north_mask(x, y, z) * (v - v∞(x, z, t, p)) / p.τₙ +
            offshore_mask(x, y, z) * (v - v∞(x, z, t, p)) / p.τₑ)
    end
end

# Free-surface displacement sponge: nudge η → 0 (its initial condition) in the same
# north + offshore regions. In the split-explicit solver the η forcing is called as
# F(i, j, k_top, grid, clock, (; η, U, V)) — only η/U/V are in scope — so it MUST be
# discrete-form and reference only mf.η. η lives at (Center, Center); masks ignore z.
@inline sponge_eta(i, j, k, grid, clock, mf, p) = begin
    x = xnode(i, grid, Center()); y = ynode(j, grid, Center())
    η = @inbounds mf.η[i, j, k]
    -(north_mask(x, y, 0.0) * η / p.τ_eta + offshore_mask(x, y, 0.0) * η / p.τ_eta)
end

# Minimal per-forcing params (small NamedTuples keep the GPU kernels lean).
T_force_params   = (; τ_ts=params.τ_ts, T_south_v1=params.T_south_v1)
S_force_params   = (; τ_ts=params.τ_ts, S_south_v1=params.S_south_v1)
u_force_params   = (; τₙ=params.τₙ, τₑ=params.τₑ)
v_force_params   = (; τₙ=params.τₙ, τₑ=params.τₑ, v₀=params.v₀,
                      τ_ramp=params.τ_ramp, ω_M2=params.ω_M2, Lx=params.Lx)
eta_force_params = (; τ_eta=params.τ_eta)

# forcing functions (discrete form — see GPU note above)
FT = Forcing(sponge_T, discrete_form=true, parameters=T_force_params)
FS = Forcing(sponge_S, discrete_form=true, parameters=S_force_params)
Fη = Forcing(sponge_eta, discrete_form=true, parameters=eta_force_params)
if mass_flux
    # No w forcing: w is diagnostic in the hydrostatic model.
    Fᵤ = Forcing(sponge_u, discrete_form=true, parameters=u_force_params)
    Fᵥ = Forcing(sponge_v, discrete_form=true, parameters=v_force_params)
    forcings = (u=Fᵤ, v=Fᵥ, T=FT, S=FS, η=Fη)
else
    forcings = (T=FT, S=FS, η=Fη)
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

#+++ Horizontal dissipation — resolution-aware Reynolds-number scaling
# This variant leans on the advection scheme for the bulk of the grid-scale
# damping: WENOVectorInvariant momentum advection (below) upwinds the vorticity
# flux and supplies scale-selective dissipation intrinsically (its default
# vorticity reconstruction is 9th order). On top of that we add only a *light*
# Laplacian background to quiet low-shear regions / internal waves, plus CATKE
# for vertical mixing.
#
# Rather than fixing an absolute viscosity (which would have to be re-picked by
# hand at every resolution), we let ν_h follow from a base Reynolds-number
# coefficient Re₀ that is held fixed across resolutions, while the *effective*
# Reynolds number is allowed to grow with the horizontal grid count N:
#
#     Re(N) = Re₀ · N^p ,   p = 2   (2-D enstrophy-cascade scaling)
#                           p = 4/3 (3-D Kolmogorov scaling)
#
# The viscosity is the inverse of that Reynolds number times a characteristic
# momentum scale U·L, so it shrinks automatically as the grid is refined (more
# grid points ⇒ higher Re ⇒ lower ν_h):
#
#     ν_h = U·L / (Re₀ · N^p)
#
# We never form an explicit Reynolds number — Re₀ is just the tunable
# proportionality constant (larger Re₀ ⇒ lower ν_h, i.e. less damping, closer to
# the marginally-stable limit). N is the cross-shore grid count and L = Lx, so
# L/N = Δx exactly (cells are square here, Δx = Δy). The default Re₀ is the
# lowest-dissipation value found to stay stable + noise-free at the production
# resolution (see src/tune_closure.jl). κ_h = ν_h / Pr_h locks tracer diffusivity
# to viscosity (Pr_h = 4). p is hydrostatic-ambiguous; 2-D is the default (it
# matches the proven ν ∝ Δx² scaling), CLOSURE_DIMS=3 selects the 4/3 exponent.
# NOTE: Re₀ is normalised against the bare N^p, so it is NOT comparable between
# the two exponents — at N=200, N² = 4e4 but N^(4/3) ≈ 1.2e3, so the same Re₀
# gives ~34× more viscosity under 3-D. Switching CLOSURE_DIMS ⇒ re-tune Re₀.
p_closure = params.closure_dims == 3 ? (4 // 3) : 2
N_closure = params.Nx                              # cross-shore grid count; L = Lx ⇒ L/N = Δx
ν_h = params.U_closure * params.Lx / (params.Re₀ * float(N_closure)^p_closure)
Δh = params.Lx / params.Nx                         # ≈ Δy (square cells), for reference
@info "Horizontal closure (Re-scaled Laplacian + WENOVectorInvariant)" Δh params.Re₀ p_closure ν_h ν_h/4
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

#+++ Create simulation
pickup = isfile("checkpoint_$(run_tag).jld2")
overwrite_existing = !pickup

# Cap Δt to resolve internal gravity waves: max_Δt = 0.5/√(N²max).
# Compute N² from a single fully-wet offshore column (i = Nx, deepest water). The IC is
# horizontally homogeneous, so any wet column has the correct stratification — and this
# avoids immersed shelf/shoal cells contaminating a horizontal average (which inflates
# N²max ~500× and would pin Δt ~25× too small).
bfield = Oceananigans.Models.buoyancy_field(model)
compute!(bfield)
zc = [znode(k, grid, Center()) for k in 1:params.Nz]
bcol = Array(interior(bfield, params.Nx, max(1, params.Ny ÷ 2), :))
N²_max = maximum(diff(bcol) ./ diff(zc))
max_Δt = 0.5 / √(max(N²_max, eps()))
@info "Δt cap from buoyancy frequency (offshore wet column)" N²_max max_Δt

simulation = Simulation(model, Δt=2minutes, stop_time=sim_runtime)
conjure_time_step_wizard!(simulation, IterationInterval(5); cfl=0.4, max_Δt)

progress = TimedMessenger()
simulation.callbacks[:progress] = Callback(progress, TimeInterval(24hours))
#---

#+++ Output: a single writer with all state variables.
# State variables: velocities (u, v, w) + tracers (T, S) + free-surface η.
# Output cadence is OUTPUT_INTERVAL_HOURS (default 1 h). At 200×400×50 each frame
# is ~160 MB, so for long runs use a coarser interval (e.g. 3 h) to bound file size.
output_interval = parse(Float64, get(ENV, "OUTPUT_INTERVAL_HOURS", "1")) * hours
η = model.free_surface.displacement
state_fields = merge(model.velocities, model.tracers, (; η))
simulation.output_writers[:fields] = JLD2Writer(model, state_fields,
    filename="fields_$(run_tag).jld2",
    schedule=TimeInterval(output_interval),
    overwrite_existing=overwrite_existing)
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
