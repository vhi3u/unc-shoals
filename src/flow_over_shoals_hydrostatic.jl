# ═══════════════════════════════════════════════════════════════════════════
# flow_over_shoals_hydrostatic.jl
# ═══════════════════════════════════════════════════════════════════════════
# Hydrostatic counterpart of flow_over_shoals.jl.
#
# Why hydrostatic:
#   Δx = Δy = 500 m, H = 50 m, Rd ≈ 3.2 km. Nonhydrostatic corrections scale
#   as (H/L)²; even at the smallest resolved scale (2Δx = 1 km) that is
#   (50/1000)² ≈ 2.5e-3, and against Rd it is ~2e-4. The Oceananigans docs
#   place NonhydrostaticModel at O(1 m) grid spacing (LES/DNS) and
#   HydrostaticFreeSurfaceModel at O(30 m)–O(100 km). Dropping the
#   ConjugateGradientPoissonSolver removes a 3D elliptic solve per stage on a
#   grid with a 900:1 cell aspect ratio (Δx=500 m vs Δz_min=0.556 m), which is
#   both the dominant cost and a noise source.
#
# Differences from flow_over_shoals.jl (all deliberate, all switchable):
#   1. HydrostaticFreeSurfaceModel + SplitExplicitFreeSurface (no CG solver).
#   2. w is diagnostic: no w boundary conditions, no w sponge. Relaxing w in a
#      nonhydrostatic model forces the pressure projection to undo it every
#      step; here the problem simply does not exist.
#   3. CATKE replaces RiBasedVerticalDiffusivity (see closure block below).
#   4. Explicit scale-selective horizontal viscosity. The old script set
#      params.νh/κh but never used them — RiBasedVerticalDiffusivity is
#      vertical-only, so there was no lateral dissipation at all.
#   5. The wind stress is ramped over ~2 inertial periods instead of switched
#      on as a step at t = 0.
#   6. Bottom-facet-only immersed drag to avoid unphysical side-face forces.
# ═══════════════════════════════════════════════════════════════════════════

using Oceananigans
using Oceananigans.Grids: Periodic, Bounded
using Oceananigans.Units
using Oceananigans.BoundaryConditions: FieldBoundaryConditions
using Oceananigans.Fields: interior
using Oceananigans.TurbulenceClosures
using Oceananigans.OutputWriters
using Oceananigans.Forcings
using Statistics: mean
using Oceanostics: RossbyNumber, ErtelPotentialVorticity, KineticEnergy
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
sigmoid_wind = true          # taper wind to zero across the east sponge
is_coriolis = true
checkpointing = false
shoal_bath = true
ramp_wind = true             # ramp τ over ~2 inertial periods

# Vertical mixing closure. :catke is the recommended production choice;
# :ribased and :constant exist so the wind run can be repeated against the
# closure used in shoals38/39 without changing anything else.
closure_choice = :catke      # :catke | :ribased | :constant

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

# simulation knobs
run_number = 46
callback_interval = 1days
snapshot_interval = 1days          # sub-inertial: inertial period is 20.74 h,
                                    # daily output aliases it into fake bands
run_tag = (periodic_y ? "periodic" : "bounded") * "_shoals$(run_number)_hydro"

pickup = false
wind_stress = 0.0                  # N m⁻²
sim_runtime = 25days
@info "Starting hydrostatic run $(run_tag)."

if LES
    params = (; Lx=150e3, Ly=200e3, Lz=50)
else
    params = (; Lx=100000, Ly=200000, Lz=50)
end
if arch == CPU()
    params = (; params..., Nx=50, Ny=50, Nz=10)
else
    params = (; params..., Nx=600, Ny=800, Nz=50)
end

x, y, z = (0, params.Lx), (0, params.Ly), (-params.Lz, 0)

# # "Warped" height coordinate
# refinement = 1.8
# stretching = 12
# Nz_grid = params.Nz

# # Normalized height ranging from 0 to 1 (0 at bottom, 1 at top)
# h_grid(k) = (k - 1) / Nz_grid

# # Linear near-surface generator
# ζ₀(k) = 1 + (h_grid(k) - 1) / refinement

# # Bottom-intensified stretching function
# Σ(k) = (1 - exp(-stretching * h_grid(k))) / (1 - exp(-stretching))

# # Generating function (maps k=1 to -params.Lz and k=Nz_grid+1 to 0)
# z_faces(k) = params.Lz * (ζ₀(k) * Σ(k) - 1)

# grid
if periodic_y
    grid = RectilinearGrid(arch; size=(params.Nx, params.Ny, params.Nz), halo=(4, 4, 4), x, y, z, topology=(Bounded, Periodic, Bounded))
else
    grid = RectilinearGrid(arch; size=(params.Nx, params.Ny, params.Nz), halo=(4, 4, 4), x, y, z, topology=(Bounded, Bounded, Bounded))
end

# model parameters
if shoal_bath
    slope_bottom = dshoal_param_bottom(params.Ly;
        Zs=-5.0,
        shoal_length=40000.0,
        sigma=8000.0,
        Zsh=-25.0,
        shelf_break_end=12000.0)
    immersed_boundary = GridFittedBottom(slope_bottom)
    ib_grid = ImmersedBoundaryGrid(grid, immersed_boundary)
else
    ib_grid = grid
end

@info ib_grid

# ═══════════════════════════════════════════════════════════════════════════
# Stratification and Wind Setup
# ═══════════════════════════════════════════════════════════════════════════
if mass_flux
    v₀ = 0.20
else
    v₀ = 0.0
end

# defaults
T_north_v1, S_north_v1 = 20.5389, 32.6264
T_south_v1, S_south_v1 = 24.5378, 35.5830

params = (; params...,
    v₀=v₀,
    Ls=20e3,
    Le=50e3,
    τ=6hours,
    T_north_v1=T_north_v1,
    T_south_v1=T_south_v1,
    S_north_v1=S_north_v1,
    S_south_v1=S_south_v1)

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

#+++ Drag
# NOTE: z₁ is the half-height of the *smallest* cell in the column, which on
# this surface-refined grid is the top cell (0.556 m), not the bottom one
# (up to 4.86 m). Kept identical to flow_over_shoals.jl so the hydrostatic and
# nonhydrostatic runs are comparable; Cᴰ ≈ 3.3e-3 is a plausible coastal value
# regardless, but revisit if you tune the bottom boundary layer.
z₀ = 2.5e-4 # roughness length
z₁ = Oceananigans.Grids.minimum_zspacing(grid, Center(), Center(), Center()) / 2
@info "Using z₁ =" z₁

const κᵛᵏ = 0.4 # von Karman constant
c_dz = (κᵛᵏ / log(z₁ / z₀))^2 # quadratic drag coefficient
@info "Defining momentum BCs with Cᴰ =" c_dz
drag = BulkDrag(coefficient=c_dz)
# Bottom facet only: with a stair-stepped/partial-cell bottom, applying drag to
# the vertical side faces of the steps is unphysical and is itself a grid-scale
# vorticity source.
immersed_drag = ImmersedBoundaryCondition(bottom=drag)

# With GridFittedBottom there is no thin-cell stability issue, so max_Δt is
# only limited by the CFL and the time-step wizard's tuning.
max_Δt = 10minutes
initial_Δt = 1minutes
#---

# ═══════════════════════════════════════════════════════════════════════════
# Wind stress
# ═══════════════════════════════════════════════════════════════════════════
# A step-function wind at t = 0 over a 30,000 km² f-plane injects a
# domain-filling inertial oscillation that has nothing to decay against (no β,
# no interior dissipation) and just scatters off the shoal every 20.74 h.
# Ramping with an e-folding time T ≫ 1/f suppresses that free inertial mode by
# roughly 1/(fT)²: with T = 2 inertial periods, fT ≈ 12.6, so ~1%.
ρ₀ = 1024.0
const f₀ = 2 * 7.292115e-5 * sind(35.2480)          # 8.42e-5 s⁻¹
const inertial_period = 2π / f₀                     # 20.74 h
const T_ramp = ramp_wind ? 2 * inertial_period : 0.0
const τ_kinematic = wind_stress / ρ₀                # m² s⁻²
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

# velocity function
@inline function sigmoidal_s2(x, Lx)
    xS = 65e3
    k2 = 40 / Lx
    return 1 / (1 + exp(k2 * (x - xS)))
end

if sigmoid_v_bc
    @inline function v∞(x, z, t, p)
        xC = 3e3
        k1 = 80 / p.Lx

        s1 = 1 / (1 + exp(-k1 * (x - xC)))
        s2 = sigmoidal_s2(x, p.Lx)
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
@inline T_target_south(x, y, z, t) = T_south_pwl(z, 24.5378)
@inline S_target_south(x, y, z, t) = S_south_pwl(z, 35.5830)

const T_target = T_target_south
const S_target = S_target_south
const inflow_mask = south_mask

# forcing functions — note there is no w sponge: w is diagnosed from continuity
# in the hydrostatic model and cannot (and should not) be relaxed.

sponge_scaling = 1 # use this if you want the southern inflow sponge nudging to be a lot stronger (to dissipate the downstream wake into the periodic boundary)
u_sponge_inflow = Relaxation(; rate=1 / (global_params.τ * inflow_scaling), mask=inflow_mask, target=0.0)
u_sponge_e = Relaxation(; rate=1 / global_params.τ, mask=east_mask, target=0.0)

v_sponge_inflow = Relaxation(; rate=1 / (global_params.τ * inflow_scaling), mask=inflow_mask, target=v_target_inflow)
v_sponge_e = Relaxation(; rate=1 / global_params.τ, mask=east_mask, target=0.0)

T_sponge_inflow = Relaxation(; rate=1 / (global_params.τ * inflow_scaling), mask=inflow_mask, target=T_target)
T_sponge_e = Relaxation(; rate=1 / global_params.τ, mask=east_mask, target=T_target)

S_sponge_inflow = Relaxation(; rate=1 / (global_params.τ * inflow_scaling), mask=inflow_mask, target=S_target)
S_sponge_e = Relaxation(; rate=1 / global_params.τ, mask=east_mask, target=S_target)

forcings = (u=(u_sponge_inflow, u_sponge_e),
    v=(v_sponge_inflow, v_sponge_e),
    T=(T_sponge_inflow, T_sponge_e),
    S=(S_sponge_inflow, S_sponge_e))

T_bcs = FieldBoundaryConditions()
S_bcs = FieldBoundaryConditions()
u_bcs = FieldBoundaryConditions(immersed=immersed_drag, bottom=drag, top=wind_bc_u)
v_bcs = FieldBoundaryConditions(immersed=immersed_drag, bottom=drag, top=wind_bc_v)

bcs = (u=u_bcs, v=v_bcs, T=T_bcs, S=S_bcs)
@info "Boundary Conditions:" bcs

if is_coriolis
    coriolis = FPlane(latitude=35.2480)
else
    coriolis = nothing
end

# ═══════════════════════════════════════════════════════════════════════════
# Turbulence closure
# ═══════════════════════════════════════════════════════════════════════════
# VERTICAL — CATKE.
#   CATKE carries a prognostic TKE tracer (:e), so the wind-driven surface
#   boundary layer deepens on a physical timescale set by the TKE budget. It
#   was calibrated against a suite of 35 LES, and is the closure used for
#   hydrostatic ocean configurations in Oceananigans.
#
#   RiBasedVerticalDiffusivity was a poor fit for this problem on three counts:
#     (a) Oceananigans warns at construction that it is "experimental …
#         unvalidated and whose default parameters are not calibrated for
#         realistic ocean conditions or for use in a three-dimensional
#         simulation";
#     (b) with defaults ν₀ = 0.7 m² s⁻¹ and the tanh taper, ν reaches
#         ~0.43 m² s⁻¹ where Ri → 0, giving an Ekman depth √(2ν/f) ≈ 101 m in a
#         50 m water column — the wind stress is mixed through the whole column
#         instead of a surface boundary layer, which destroys the vertical
#         shear and stratification that the eddies feed on;
#     (c) ν and κ are functions of the *pointwise* Ri, which on a 500 m grid
#         under near-inertial shear swings two orders of magnitude between
#         adjacent columns. (If you do go back to it, use
#         horizontal_Ri_filter = FivePointHorizontalFilter() — the docstring
#         says it "can help alleviate noise" — plus maximum_viscosity /
#         maximum_diffusivity caps.)
#
# HORIZONTAL — biharmonic, scale-selective.
#   The old script defined params.νh/κh and never used them, so there was no
#   lateral dissipation at all and nothing removed energy at 2–4Δx except
#   WENO's (weak, by design) implicit dissipation.
#
#   Biharmonic rather than harmonic because Rd ≈ 3.2 km is only 6.5 grid points:
#   a harmonic ν large enough to damp 2Δx would also damp the eddies. With
#   ν₄ = 1e5 m⁴ s⁻¹ and damping rate ν₄k⁴:
#       2Δx = 1 km   →  e-folding ≈ 1.8 h    (kills grid noise)
#       Rd  = 3.2 km →  e-folding ≈ 7.8 days (leaves eddies alone)
#   A harmonic ν tuned to the same 2Δx rate would damp Rd in ~20 h.
#
#   κ₄ = 0 (the default): WENO-5 tracer advection already supplies tracer-scale
#   dissipation, and adding biharmonic κ on top over-diffuses the fronts.
# const ν₄ = 1e5   # m⁴ s⁻¹ — scale as Δx⁴ if you change resolution

# horizontal_closure = HorizontalScalarBiharmonicDiffusivity(ν=ν₄)

if closure_choice === :catke
    # NOTE: do NOT list :e here. CATKE registers it as an auxiliary tracer and
    # the model constructor throws if you also name it explicitly. It still
    # shows up as model.tracers.e after construction.
    horizontal_closure = HorizontalScalarDiffusivity(ν=1e-3, κ=1e-3)
    vertical_closure = CATKEVerticalDiffusivity()
    tracers = (:T, :S)
elseif closure_choice === :ribased
    horizontal_closure = HorizontalScalarDiffusivity(ν=1e-3, κ=1e-3)
    vertical_closure = RiBasedVerticalDiffusivity()
    tracers = (:T, :S)
elseif closure_choice === :constant
    # Control: the closure used in shoals38, so the wind run can be repeated
    # with only the wind changing.
    horizontal_closure = HorizontalScalarDiffusivity(ν=1e-3, κ=1e-3)
    vertical_closure = VerticalScalarDiffusivity(ν=1e-4, κ=1e-5)
    tracers = (:T, :S)
else
    error("closure_choice must be :catke, :ribased or :constant; got $(closure_choice)")
end

closure = (horizontal_closure, vertical_closure)
@info "Closure ($(closure_choice)):" closure

# ═══════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════
# SplitExplicitFreeSurface with a cfl (and no fixed_Δt) recomputes its
# barotropic substep count each step, so it stays consistent with the
# time-step wizard.
model = HydrostaticFreeSurfaceModel(ib_grid;
    momentum_advection=WENO(order=5),
    tracer_advection=WENO(order=5),
    free_surface=SplitExplicitFreeSurface(ib_grid; cfl=0.7),
    tracers=tracers,
    buoyancy=SeawaterBuoyancy(),
    coriolis=coriolis,
    closure=closure,
    boundary_conditions=bcs,
    forcing=forcings
)

@info "" model

overwrite_existing = (pickup === false)

simulation = Simulation(model, Δt=initial_Δt, stop_time=sim_runtime)
# max_Δt comes from the bottom-drag stability limit computed above, not a guess.
conjure_time_step_wizard!(simulation, cfl=0.7, max_Δt=max_Δt)

progress = TimedMessenger()
simulation.callbacks[:progress] = Callback(progress, TimeInterval(callback_interval))

# # Cheap health check. Without it, a blow-up surfaces as `InexactError:
# # Int64(NaN)` from deep inside the free-surface substepping, which says nothing
# # about the cause. This reports Δt and the velocity extrema on the way there.
# function report_health(sim)
#     ui = interior(sim.model.velocities.u)
#     vi = interior(sim.model.velocities.v)
#     umax, vmax = maximum(abs, ui), maximum(abs, vi)
#     @info @sprintf("t = %7.3f d | Δt = %6.1f s | max|u| = %.4f | max|v| = %.4f",
#         time(sim) / 86400, sim.Δt, umax, vmax)
#     if !isfinite(umax) || !isfinite(vmax)
#         error("Velocity field is not finite at t = $(time(sim)) s — blow-up, not a solver bug.")
#     end
# end
# simulation.callbacks[:health] = Callback(report_health, IterationInterval(200))

u, v, w = model.velocities
T = model.tracers.T
S = model.tracers.S
PV = @at (Center, Center, Center) ErtelPotentialVorticity(model, tracer_name=:T)
Ro = @at (Center, Center, Center) RossbyNumber(model)
KE = @at (Center, Center, Center) KineticEnergy(model)

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

slice_fields = (; u_c, v_c, w_c, T, S, Ro, KE)
if closure_choice === :catke
    slice_fields = (; slice_fields..., e=model.tracers.e)
end
tavg_fields = (; u_c, v_c, w_c, uu, vv, ww, T, S, uT, uS, vT, vS, wT, wS)

# (1) 2D snapshots — every 6 h, i.e. below the 20.74 h inertial period, so a
# radiating wave can be told apart from a daily aliasing artifact.
simulation.output_writers[:surface_slice] = NetCDFWriter(model, slice_fields,
    filename="top_$(run_tag).nc",
    schedule=TimeInterval(snapshot_interval),
    indices=(:, :, params.Nz),
    overwrite_existing=overwrite_existing)

# Mid-y XZ slice (cross-shore transect at domain center)
simulation.output_writers[:midy_slice] = NetCDFWriter(model, slice_fields,
    filename="midy_$(run_tag).nc",
    schedule=TimeInterval(snapshot_interval),
    indices=(:, round(Int, params.Ny / 2), :),
    overwrite_existing=overwrite_existing)

# (2) 3D Time Averages (10 day window)
simulation.output_writers[:time_avg_3d] = NetCDFWriter(model, tavg_fields,
    filename="time_avg_3d_$(run_tag).nc",
    schedule=AveragedTimeInterval(10days, window=10days),
    overwrite_existing=overwrite_existing)

if checkpointing
    simulation.output_writers[:checkpointer] = Checkpointer(model,
        schedule=TimeInterval(5days),
        prefix="checkpoint_$(run_tag)",
        overwrite_existing=true,
        cleanup=true)
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
        @inline α_lin(y) = clamp(y / params.Ly, 0.0, 1.0)
        @inline blend(a, b, α) = (1 - α) * a + α * b
        @inline Tᵢ(x, y, z) = blend(T_south_pwl(z, T_south_v1), T_north_pwl(z, T_north_v1), α_lin(y))
        @inline Sᵢ(x, y, z) = blend(S_south_pwl(z, S_south_v1), S_north_pwl(z, S_north_v1), α_lin(y))
    else
        @inline Tᵢ(x, y, z) = T_south_pwl(z, T_south_v1)
        @inline Sᵢ(x, y, z) = S_south_pwl(z, S_south_v1)
    end

    if closure_choice === :catke
        set!(model, u=0.0, v=v_init, T=Tᵢ, S=Sᵢ, e=1e-6)
    else
        set!(model, u=0.0, v=v_init, T=Tᵢ, S=Sᵢ)
    end
else
    @info "Skipping initial conditions setup — picking up fields from $(pickup)."
end

# run simulation
@info """
════════════════════════════════════════════════════════
 SIMULATION CONFIGURATION: $(run_tag)   [HYDROSTATIC]
════════════════════════════════════════════════════════
 Run number:      $(run_number)
 Runtime:         $(sim_runtime)
 Architecture:    $(arch)

 ── Model Parameters ──
 Zs (shoal_depth): $(-5.0) m
 shoal_length:     $(40000.0) m
 Zsh (shelf_depth):$(-25.0) m
 shelf_break_end:  $(12000.0) m
 wind_stress:      $(wind_stress) N/m^2
 wind ramp:        $(T_ramp / 3600) h
 vertical closure: $(closure_choice)
 horizontal ν₄:    $(ν₄) m^4/s
 Cᴰ:               $(c_dz)
 max_Δt:           $(max_Δt) s

 ── Switches ──
 LES:              $(LES)
 mass_flux:        $(mass_flux)
 periodic_y:       $(periodic_y)
 gradient_IC:      $(gradient_IC)
 sigmoid_v_bc:     $(sigmoid_v_bc)
 sigmoid_ic:       $(sigmoid_ic)
 sigmoid_wind:     $(sigmoid_wind)
 is_coriolis:      $(is_coriolis)
 shoal_bath:       $(shoal_bath)
 ramp_wind:        $(ramp_wind)
════════════════════════════════════════════════════════
"""
run!(simulation, pickup=pickup)
