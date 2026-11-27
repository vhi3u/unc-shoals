# ═══════════════════════════════════════════════════════════════════════════
# flow_over_shoals_sweep.jl
# ═══════════════════════════════════════════════════════════════════════════
# Modified version of flow_over_shoals.jl for parameter sweeps.
# Reads sweep parameters from environment variables set by sweep_driver.jl:
#   SWEEP_Hs, SWEEP_SHOAL_LENGTH, SWEEP_SHELF_DEPTH, SWEEP_SHELF_BREAK_END,
#   SWEEP_RUN_LABEL, SWEEP_RUN_INDEX
# ═══════════════════════════════════════════════════════════════════════════

using Oceananigans
using Oceananigans.Grids: Periodic, Bounded, Closed, Open, znode, xnode, static_column_depthᶜᶜᵃ
using Oceananigans.BoundaryConditions: ObliqueRadiation, TracerReservoir
using Oceananigans.Units
using Oceananigans.BoundaryConditions: FieldBoundaryConditions
using Oceananigans.Fields: interior
using Oceananigans.TurbulenceClosures
using Oceananigans.OutputWriters
using Oceananigans.Forcings
using Statistics: mean
# No Oceanostics: every release that supports Oceananigans 0.118 requires
# Julia 1.12, and Casper's module is 1.11. Its three diagnostics (Rossby
# number, kinetic energy, Ertel PV) and the progress printer are a few lines
# each with Oceananigans' own operators; see the output section.
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
sweep_rise_mid = parse(Float64, get(ENV, "SWEEP_RISE_MID", "63500.0"))
sweep_rise_width = parse(Float64, get(ENV, "SWEEP_RISE_WIDTH", "5000.0"))
# 0.0 reproduces the shoal-welded-to-the-coast geometry exactly.
sweep_shoal_start = parse(Float64, get(ENV, "SWEEP_SHOAL_START", "0.0"))
sweep_shoal_inner_width = parse(Float64, get(ENV, "SWEEP_SHOAL_INNER_WIDTH", "0.0"))
# Flat offshore depth. Must be deeper than Zsh or no shelf break exists.
sweep_h_deep = parse(Float64, get(ENV, "SWEEP_H_DEEP", "-50.0"))
# Domain depth and vertical spacing. Nz is derived, so changing Lz cannot
# silently change the vertical resolution.
sweep_Lz = parse(Float64, get(ENV, "SWEEP_LZ", "50.0"))
sweep_dz = parse(Float64, get(ENV, "SWEEP_DZ", "1.0"))
# Stratification below the deepest T/S breakpoint, as a FRACTION of the
# gradient in the segment above it. The profiles were constant below their
# last breakpoint, which left the bottom 40% of a 50 m column unstratified --
# 10 to 15 of 50 levels in the offshore reference column, every run, which is
# what forces ape_wedge.m to floor N^2 and makes eta and APE_qg unreliable
# there. A deeper domain would make that worse, not better. 0.0 reproduces
# the old constant tail exactly, so existing runs are unchanged.
const DEEP_STRAT_FRAC = parse(Float64, get(ENV, "SWEEP_DEEP_STRAT_FRAC", "0.0"))
sweep_run_label = get(ENV, "SWEEP_RUN_LABEL", "standalone")
sweep_run_index = parse(Int, get(ENV, "SWEEP_RUN_INDEX", "0"))
sweep_wind_stress = parse(Float64, get(ENV, "SWEEP_WIND_STRESS", "0.0"))
sweep_v0 = parse(Float64, get(ENV, "SWEEP_V0", "0.2"))
sweep_dx = parse(Float64, get(ENV, "SWEEP_DX", "500.0"))
sweep_sim_days = parse(Float64, get(ENV, "SWEEP_SIM_DAYS", "50.0"))
# Alongshore domain length and the shoal's alongshore centre. 200 km with the
# shoal in the middle reproduces every earlier run. With open y boundaries the
# domain no longer has to be long enough for the wake to recirculate, so it can
# be shorter, with the shoal placed upstream to leave room for its wake.
const sweep_Ly = parse(Float64, get(ENV, "SWEEP_LY", "200000.0"))
const sweep_shoal_y0 = parse(Float64, get(ENV, "SWEEP_SHOAL_Y0", string(sweep_Ly / 2)))
# crest_length > 0: the flat crest is this long, starting where the coastal
# slope reaches Zs; shoal_length is then derived (see dshoal_param_bottom).
# 0 (the default) keeps shoal_length as given, as in every earlier run.
const sweep_crest_length = parse(Float64, get(ENV, "SWEEP_CREST_LENGTH", "0.0"))

# ── Time-varying wind ─────────────────────────────────────────────────────
# A constant wind and a constant jet settle into STEADY meanders around the
# shoal, and in every steady run so far those standing meanders carry 4-20x
# the buoyancy flux of the transient eddies. Real shelf wind comes in 2-10 day
# events, and fronts sharpened during an event go unstable when it relaxes.
#   steady     tau(t) = SWEEP_WIND_STRESS                      (default; as before)
#   pulsed     on/off events: SWEEP_WIND_STRESS is the PEAK, held for a fraction
#              SWEEP_WIND_DUTY of each SWEEP_WIND_PERIOD_DAYS, zero otherwise
#   reversing  +peak for half of each period, -peak for the other half (mean 0)
# Edges are smoothed over about half an inertial period. Choose a period that
# divides the 10-day averaging window (5 or 10 d), so every average spans
# whole cycles. The time-mean stress is written to sweep_metadata as
# wind_stress_mean; the MATLAB tools use it for the Ekman transport.
const WIND_MODE_NAME = get(ENV, "SWEEP_WIND_MODE", "steady")
const WIND_MODE_ID = WIND_MODE_NAME == "steady" ? 0 :
                     WIND_MODE_NAME == "pulsed" ? 1 :
                     WIND_MODE_NAME == "reversing" ? 2 :
                     error("SWEEP_WIND_MODE must be steady, pulsed or reversing, not '$WIND_MODE_NAME'")
const WIND_PERIOD_DAYS = parse(Float64, get(ENV, "SWEEP_WIND_PERIOD_DAYS", "10.0"))
const WIND_DUTY = parse(Float64, get(ENV, "SWEEP_WIND_DUTY", "0.5"))
WIND_MODE_ID == 1 && !(0 < WIND_DUTY < 1) && error("SWEEP_WIND_DUTY must lie in (0, 1), not $WIND_DUTY")
WIND_MODE_ID > 0 && !(WIND_PERIOD_DAYS > 0) && error("SWEEP_WIND_PERIOD_DAYS must be positive")

# ── Cross-shelf salinity front ────────────────────────────────────────────
# The interior starts horizontally uniform and the inflow strip is nudged
# back to that one profile, so nothing maintains a cross-shelf density
# contrast except what wind and the shoal build. Zhao et al.'s eddies feed on
# a continuously supplied light coastal water mass. This adds one: shelf water
# FRESHER by SWEEP_FRONT_DS psu inshore of SWEEP_FRONT_X, with a tanh front of
# half-width SWEEP_FRONT_WIDTH:
#   S(x, z) = S_ambient(z) - DS * 1/2 [1 - tanh((x - X) / W)] * G(z)
#   G(z) = exp(z / SWEEP_FRONT_DEPTH)   (surface-trapped), or 1 if that is 0
#                                       (full-depth, the default)
# Applied to the initial condition AND the inflow nudging target, so the
# inflow keeps resupplying it. Without the second the nudging erases the
# front within one pass of the periodic channel. The east open boundary is
# 130 km offshore of any front used here, where the anomaly is < 1e-10 psu,
# so its exterior state is unchanged. Salinity only, with temperature
# unchanged, so salinity doubles as Zhao et al.'s freshwater tracer:
# FWF = int u (S_ref - S)/S_ref. DS = 0 (the default) reproduces every
# earlier run exactly.
const FRONT_DS = parse(Float64, get(ENV, "SWEEP_FRONT_DS", "0.0"))
const FRONT_X = parse(Float64, get(ENV, "SWEEP_FRONT_X", "20000.0"))
const FRONT_W = parse(Float64, get(ENV, "SWEEP_FRONT_WIDTH", "4000.0"))
const FRONT_H = parse(Float64, get(ENV, "SWEEP_FRONT_DEPTH", "0.0"))
FRONT_DS != 0 && !(FRONT_W > 0) && error("SWEEP_FRONT_WIDTH must be positive")
@inline front_xshape(x) = 0.5 * (1 - tanh((x - FRONT_X) / FRONT_W))
@inline front_zshape(z) = ifelse(FRONT_H > 0, exp(z / max(FRONT_H, 1e-6)), one(z))
@inline S_front(x, z) = -FRONT_DS * front_xshape(x) * front_zshape(z)

@info "Wind mode: $WIND_MODE_NAME (period $WIND_PERIOD_DAYS d, duty $WIND_DUTY); front: DS=$FRONT_DS psu at x=$FRONT_X m, width $FRONT_W m, depth scale $FRONT_H m"
@info "Sweep parameters: Zs=$sweep_Zs, shoal_length=$sweep_shoal_length, shoal_start=$sweep_shoal_start, shoal_inner_width=$sweep_shoal_inner_width, sigma=$sweep_sigma, Zsh=$sweep_Zsh, shelf_break_end=$sweep_shelf_break_end, wind_stress=$sweep_wind_stress, v0=$sweep_v0"

# build
@info "building domain"

# switches
LES = true
mass_flux = true
# y boundaries. Periodic (the default, every run before 2026-11-25): water
# leaving the north re-enters the south through the nudging strip. Open
# (SWEEP_OPEN_Y = true, Oceananigans >= 0.118): the jet enters through the
# south boundary and leaves through the north one, so the shoal's wake is not
# recirculated. See the boundary-condition section and src/test_open_boundaries.jl,
# where this configuration was developed and tested.
const open_y = get(ENV, "SWEEP_OPEN_Y", "false") == "true"
periodic_y = !open_y
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
# Oceanostics' progress messengers, copied into the repo (see the file header)
include(joinpath(@__DIR__, "ProgressMessengers.jl"))
using .ProgressMessengers: TimedMessenger

# ═══════════════════════════════════════════════════════════════════════════
# simulation knobs
# ═══════════════════════════════════════════════════════════════════════════
run_number = sweep_run_index
sim_runtime = sweep_sim_days * days
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
    params = (; Lx=150e3, Ly=sweep_Ly, Lz=sweep_Lz)
else
    params = (; Lx=150000, Ly=sweep_Ly, Lz=sweep_Lz)
end

# Horizontal grid spacing is the knob, not Nx/Ny, so changing Lx can never
# silently change the resolution (and vice versa).
# Set per run from the sweep table (SWEEP_DX), default 500 m. Nx/Ny are
# derived, so changing dx cannot silently change the domain.
Δh = sweep_dx
if arch == CPU()
    params = (; params..., Nx=60, Ny=60, Nz=10)
else
    params = (; params...,
        Nx=round(Int, params.Lx / Δh),
        Ny=round(Int, params.Ly / Δh),
        Nz=round(Int, params.Lz / sweep_dz))
end

x, y, z = (0, params.Lx), (0, params.Ly), (-params.Lz, 0)

# grid
# Oceananigans >= 0.118 sets topology per SIDE, and an open-boundary scheme is
# accepted only on an `Open` side: x is a coast wall at x = 0 and, with
# open_east, open at x = Lx. y is Periodic or Open at both ends.
x_topology = open_east ? (Closed, Open) : Bounded
y_topology = open_y ? Open : Periodic
grid = RectilinearGrid(arch; size=(params.Nx, params.Ny, params.Nz), halo=(4, 4, 4), x, y, z,
                       topology=(x_topology, y_topology, Bounded))

# model parameters
if shoal_bath
    slope_bottom = dshoal_param_bottom(params.Ly;
        Zs=sweep_Zs,
        shoal_length=sweep_shoal_length,
        sigma=sweep_sigma,
        Zsh=sweep_Zsh,
        shelf_break_end=sweep_shelf_break_end,
        rise_mid=sweep_rise_mid,
        rise_width=sweep_rise_width,
        shoal_start=sweep_shoal_start,
        shoal_inner_width=sweep_shoal_inner_width,
        h_deep=sweep_h_deep,
        crest_length=(sweep_crest_length > 0 ? sweep_crest_length : nothing),
        shoal_y0=sweep_shoal_y0)
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
    val4 = v3 + DEEP_STRAT_FRAC * m23 * (z - z3)
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
    val4 = v3 + DEEP_STRAT_FRAC * m23 * (z - z3)
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
    val4 = v3 + DEEP_STRAT_FRAC * m23 * (z - z3)
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
    val4 = v3 + DEEP_STRAT_FRAC * m23 * (z - z3)
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
    val4 = v3 + DEEP_STRAT_FRAC * m23 * (z - z3)
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
    val4 = v3 + DEEP_STRAT_FRAC * m23 * (z - z3)
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

# Time envelope of the stress (SWEEP_WIND_MODE). s = sin(2 pi t / P) is above
# cos(pi d) for a fraction d of each period, so thresholding it gives duty d;
# tanh with sharpness P / (2 pi T_edge) makes each edge last about T_edge.
const WIND_PERIOD = WIND_PERIOD_DAYS * 86400.0
const WIND_EDGE = inertial_period / 2
const WIND_THRESH = cospi(WIND_DUTY)
const WIND_SHARP = WIND_PERIOD / (2π * WIND_EDGE)
@inline function wind_envelope(t)
    s = sin(2π * t / WIND_PERIOD)
    return ifelse(WIND_MODE_ID == 0, one(t),
           ifelse(WIND_MODE_ID == 1, 0.5 * (1 + tanh((s - WIND_THRESH) * WIND_SHARP)),
                                     tanh(s * WIND_SHARP)))
end
# Time-mean of the envelope over one period, for the metadata (Ekman transport
# in the MATLAB tools). Exact for steady, ~duty for pulsed, ~0 for reversing.
const WIND_MEAN_FACTOR = WIND_MODE_ID == 0 ? 1.0 :
    mean(wind_envelope(t) for t in range(0, WIND_PERIOD, length=20001)[1:end-1])
@info @sprintf("wind envelope: mode %s, mean factor %.4f -> time-mean stress %.4g N/m^2",
    WIND_MODE_NAME, WIND_MEAN_FACTOR, sweep_wind_stress * WIND_MEAN_FACTOR)

if sigmoid_wind
    @inline wind_v_flux(x, y, t) = -τ_kinematic * wind_ramp(t) * wind_envelope(t) * wind_shape(x)
else
    @inline wind_v_flux(x, y, t) = -τ_kinematic * wind_ramp(t) * wind_envelope(t)
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
# + S_front: the inflow strip resupplies the cross-shelf front (zero when
# SWEEP_FRONT_DS = 0).
@inline S_target_south(x, y, z, t) = S_south_pwl(z, global_params.S_south_v1) + S_front(x, z)

@inline T_target_north(x, y, z, t) = T_north_pwl(z, global_params.T_north_v1)
@inline S_target_north(x, y, z, t) = S_north_pwl(z, global_params.S_north_v1)

const T_target = T_target_south
const S_target = S_target_south
const inflow_mask = south_mask

# forcing functions — note there is no w sponge: w is diagnosed from continuity
# in the hydrostatic model and cannot (and should not) be relaxed.
# The south strip is kept with open y boundaries too. An open boundary passes
# what the interior sends it, but cannot make the interior accept the profile
# imposed on inflow (a depth-uniform jet that bottom drag shears within a few
# km): without the strip, test_open_boundaries.jl grew a 0.34 m/s
# recirculation along the inflow boundary. The analysis excludes the strip.
if true
    u_sponge_inflow = Relaxation(; rate=1 / global_params.τ, mask=inflow_mask, target=0.0)
    u_sponge_e = Relaxation(; rate=1 / global_params.τ, mask=east_mask, target=0.0)

    v_sponge_inflow = Relaxation(; rate=1 / global_params.τ, mask=inflow_mask, target=v_target_inflow)
    v_sponge_e = Relaxation(; rate=1 / global_params.τ, mask=east_mask, target=0.0)

    T_sponge_inflow = Relaxation(; rate=1 / global_params.τ, mask=inflow_mask, target=T_target)
    T_sponge_e = Relaxation(; rate=1 / global_params.τ, mask=east_mask, target=T_target)

    S_sponge_inflow = Relaxation(; rate=1 / global_params.τ, mask=inflow_mask, target=S_target)
    S_sponge_e = Relaxation(; rate=1 / global_params.τ, mask=east_mask, target=S_target)

    # Open y: a weak TRACER-only sponge at the outflow end (1 day, 20 km).
    # Without it the 2.5 km test overshot S by 0.05 psu where the outflow
    # boundary meets the coast. Velocities are left to the radiation condition,
    # so eddies and the wake still leave freely. Exclude y > Ly - 20 km in the
    # analysis.
    if open_y
        north_strip = PiecewiseLinearMask{:y}(center=params.Ly, width=params.Ls)
        T_sponge_inflow = (T_sponge_inflow, Relaxation(; rate=1 / 1day, mask=north_strip, target=T_target))
        S_sponge_inflow = (S_sponge_inflow, Relaxation(; rate=1 / 1day, mask=north_strip, target=S_target))
    end

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

if open_y
    # ═══════════════════════════════════════════════════════════════════════
    # Open south (inflow) and north (outflow) boundaries
    # ═══════════════════════════════════════════════════════════════════════
    # Developed in src/test_open_boundaries.jl. Per field:
    #   v (normal)       ObliqueRadiation, exterior v∞(x): imposed on inflow over
    #                    1 h at the south; at the north, radiated on outflow
    #                    (weak 30-day pull) and nudged over 1 day on backflow.
    #                    Oblique, not Normal, radiation because eddies and the
    #                    wake reach the boundary at an angle.
    #   u (tangential)   ObliqueRadiation toward 0
    #   V (barotropic)   Flather with the transport PINNED to the jet's: in at
    #                    the south = out at the north, so the mean free surface
    #                    cannot drift. The exterior η is the GEOSTROPHIC one,
    #                    g dη/dx = f v∞ (~10 cm across the shelf): with η_ext = 0
    #                    Flather fights the jet's own balance.
    #   T, S             TracerReservoir: the exterior water mass (south profile,
    #                    plus the front if one is set) enters on inflow.
    @inline v_in(x) = v∞(x, 0.0, 0.0, global_params)
    @inline v_ext_bc(x, z, t) = v_in(x)

    # Net inflow transport over the wet part of the y = 0 section, computed on
    # the actual grid so the pinned target is exactly what the interior holds.
    xc_cpu = Array(xnodes(grid, Center()))
    zc_cpu = Array(znodes(grid, Center()))
    Δx_in = params.Lx / params.Nx
    Δz_in = params.Lz / params.Nz
    Q_in = sum(v_in(xi) * Δx_in * Δz_in * count(zk -> zk > slope_bottom(xi, 0.0), zc_cpu) for xi in xc_cpu)
    @info @sprintf("open y: inflow transport pinned at %.3e m^3/s", Q_in)

    # Discrete form (i, k, grid, clock, fields), as Flather's (U, η) tuple
    # requires; column depth from the immersed grid, which is GPU-safe.
    @inline V_ext_bc(i, k, grid, clock, fields) = v_in(xnode(i, grid, Center())) * static_column_depthᶜᶜᵃ(i, 1, grid)
    # ∫ of the sigmoid jet (k1_fixed, k2_fixed, xC = 3 km, xS = 65 km); its clamp
    # is never active, so the integral is exact.
    @inline Fjet(x) = log1p(exp(k1_fixed * (x - 3e3))) / k1_fixed - x - log1p(exp(-k2_fixed * (x - 65e3))) / k2_fixed
    @inline η_geo(x) = f₀ / 9.80665 * global_params.v₀ * (Fjet(x) - Fjet(0.0))
    @inline η_ext_bc(i, k, grid, clock, fields) = η_geo(xnode(i, grid, Center()))

    @inline T_ext_y(x, z, t) = T_south_pwl(z, 24.5378)
    @inline S_ext_y(x, z, t) = S_south_pwl(z, 35.5830) + S_front(x, z)

    south_scheme = ObliqueRadiation(inflow_timescale=1hour, outflow_timescale=30days)
    north_scheme = ObliqueRadiation(inflow_timescale=1day, outflow_timescale=30days)
    reservoir = TracerReservoir(inflow_length_scale=10e3, outflow_length_scale=10e3)

    # South/north conditions per field, splatted into the constructors below.
    vy = (south=NormalFlowBoundaryCondition(v_ext_bc; scheme=south_scheme),
          north=NormalFlowBoundaryCondition(v_ext_bc; scheme=north_scheme))
    uy = (south=ValueBoundaryCondition(0.0; scheme=south_scheme),
          north=ValueBoundaryCondition(0.0; scheme=north_scheme))
    Ty = (south=ValueBoundaryCondition(T_ext_y; scheme=reservoir),
          north=ValueBoundaryCondition(T_ext_y; scheme=reservoir))
    Sy = (south=ValueBoundaryCondition(S_ext_y; scheme=reservoir),
          north=ValueBoundaryCondition(S_ext_y; scheme=reservoir))
    V_bcs = FieldBoundaryConditions(south=GravityWaveRadiationBoundaryCondition((V_ext_bc, η_ext_bc); target_transport=Q_in),
                                    north=GravityWaveRadiationBoundaryCondition((V_ext_bc, η_ext_bc); target_transport=Q_in))
else
    vy = uy = Ty = Sy = NamedTuple()
    V_bcs = nothing
end



if open_east
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

    T_bcs = FieldBoundaryConditions(; east=ValueBoundaryCondition(T_east_bc; scheme=east_radiation, discrete_form=true), Ty...)
    S_bcs = FieldBoundaryConditions(; east=ValueBoundaryCondition(S_east_bc; scheme=east_radiation, discrete_form=true), Sy...)
    u_bcs = FieldBoundaryConditions(; immersed=immersed_drag, bottom=drag, top=wind_bc_u,
        east=NormalFlowBoundaryCondition(0; scheme=east_radiation), uy...)
    # v is tangential at the east boundary; with open_y it also carries the
    # south/north conditions.
    v_bcs = FieldBoundaryConditions(; immersed=immersed_drag, bottom=drag, top=wind_bc_v, vy...)

    # Barotropic transport: exterior (U, η) = (0, 0). This is what holds the net
    # mass flux at zero. Omit it and the baroclinic mode radiates while the
    # barotropic mode reflects off a closed boundary — worse than either choice.
    U_bcs = FieldBoundaryConditions(ib_grid, (Face(), Center(), nothing);
        east=GravityWaveRadiationBoundaryCondition((0.0, 0.0)))
else
    T_bcs = FieldBoundaryConditions(; Ty...)
    S_bcs = FieldBoundaryConditions(; Sy...)
    u_bcs = FieldBoundaryConditions(; immersed=immersed_drag, bottom=drag, top=wind_bc_u, uy...)
    v_bcs = FieldBoundaryConditions(; immersed=immersed_drag, bottom=drag, top=wind_bc_v, vy...)
    U_bcs = nothing
end

bcs = (; u=u_bcs, v=v_bcs, T=T_bcs, S=S_bcs)
isnothing(U_bcs) || (bcs = (; bcs..., U=U_bcs))
isnothing(V_bcs) || (bcs = (; bcs..., V=V_bcs))
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
        sim_runtime = sweep_sim_days * days
        @info "No checkpoint found. Running $(sweep_sim_days)-day run."
    end
end
overwrite_files = (pickup === false)

# initial_Δt was 5 minutes with NO max_Δt cap. flow_over_shoals_hydrostatic.jl
# uses 1 minute and caps at 10 minutes. The unbalanced start plus a radiative
# east boundary makes the first few hundred steps the riskiest phase, and an
# uncapped wizard can walk Δt up while the barotropic substep count follows it.
#
# max_Δt MUST scale with Δh. The 10 min ceiling is a CFL number tuned at
# Δh = 500 m -- hydrostatic.jl says so outright: "only limited by the CFL and
# the time-step wizard's tuning" -- and advective CFL is u Δt / Δx. Frozen at
# 600 s it means:
#
#     Δh = 500 m, |u| = 0.3 m/s  ->  CFL = 0.36   (headroom)
#     Δh = 250 m, |u| = 0.3 m/s  ->  CFL = 0.72   (already past the target)
#
# so a 250 m run has far less reserve than its 500 m twin. Scaling holds the
# CFL ceiling fixed instead of the second count.
#
# This ceiling did NOT cause the R250_Zs15 blow-up -- res250_6.log shows Δt
# peaking at 305.8 s, never near 600 s, so the wizard was already CFL-limited
# well below the cap and this scaling would have changed nothing. It is kept
# because a resolution-blind CFL ceiling is wrong on its own terms in a sweep
# that varies Δx. The real cause was vertical: see min_change below.
max_Δt = 10minutes * (Δh / 500)

simulation = Simulation(model, Δt=1minutes, stop_time=sim_runtime)
# Schedule 5, not the default 10, and min_change 0.2, not the default 0.5.
#
# min_change is what killed R250_Zs15 at t = 0.497 d. Δz = Lz/Nz = 1 m while
# Δx = 250 m, an aspect ratio of 250:1, so |w|/Δz enters the advective CFL on
# equal terms with |u|/Δx even though w is ~250x smaller than u. From the log:
#
#   iter 200:  |w|=1.14e-3  ->  vertical share of the CFL budget = 35%
#   iter 225:  |w|=1.49e-2  ->  vertical share = 88%,  required Δt = 41 s
#
# The wizard had Δt = 168.2 s and needed 41 s. With min_change = 0.5 the most
# it could do was 84 s, then 42 s -- ten more iterations away, by which time w
# had grown again. It was chasing a runaway it could not catch. min_change =
# 0.2 lets one check cover that whole descent.
#
# min_change only binds when the wizard wants to cut by more than 2x, which a
# healthy run never does, so the finished R500 runs stay comparable.
#
# For context on whether that w is physics: u|dh/dx| = 0.25 * 0.00346 =
# 8.7e-4 m/s, which is the ambient w in the log. 1.49e-2 is 17x that, so the
# spike is grid-scale noise, not a resolved vertical velocity.
conjure_time_step_wizard!(simulation, IterationInterval(5),
    cfl=0.7, max_Δt=max_Δt, min_change=0.2)
@info @sprintf("Time step: initial 60 s, max %.0f s  (Δh = %.0f m -> CFL = %.2f at |u| = 0.3 m/s)",
    max_Δt, Δh, 0.3 * max_Δt / Δh)

# Progress line: Oceanostics' TimedMessenger (iteration, % done, model time, Δt,
# wall time, wall time per step, max |u|,|v|,|w|, advective and diffusive CFL,
# max viscosity), from the vendored src/ProgressMessengers.jl.
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
    # Where w lives, and how the advective CFL budget splits. Δz = 1 m against
    # Δx = 250 m means |w|/Δz can dominate the CFL while |w| itself still looks
    # negligible next to |u| -- which is exactly how R250_Zs15 died, and the
    # old one-line health report could not show it. The columnwise reduction
    # is a GPU reduction plus an Nx-element transfer, so it is cheap.
    umax = maximum(abs, interior(m.velocities.u))
    vmax = maximum(abs, interior(m.velocities.v))
    wcol = Array(vec(maximum(abs, interior(m.velocities.w), dims=(2, 3))))
    wmax, iw = findmax(wcol)
    xw = (iw - 0.5) * params.Lx / params.Nx / 1e3

    tx = umax / Δh
    ty = vmax / Δh
    tz = wmax / (params.Lz / params.Nz)
    tot = tx + ty + tz

    # Co-located upper bound: the three maxima need not share a cell, so this
    # brackets the wizard's own measurement from above rather than matching it.
    @info @sprintf("health: iter %6d | t = %7.3f d | Δt = %6.1f s | max|u| = %.4f | max|v| = %.4f | max|w| = %.2e at x = %5.1f km | CFL <= %.2f (vert %2.0f%%)",
        iteration(sim), time(sim) / 86400, sim.Δt,
        umax, vmax, wmax, xw, sim.Δt * tot, 100 * tz / tot)
end
simulation.callbacks[:health] = Callback(report_health, IterationInterval(10))

u, v, w = model.velocities
T = model.tracers.T
S = model.tracers.S
# Same definitions as Oceanostics' RossbyNumber, KineticEnergy and
# ErtelPotentialVorticity on an f-plane (Coriolis only in the vertical
# component), all interpolated to cell centres:
#   Ro = ζ / f,  ζ = ∂x v - ∂y u
#   KE = (u² + v² + w²) / 2
#   PV = (∂y w - ∂z v) ∂x b + (∂z u - ∂x w) ∂y b + (f + ζ) ∂z b
b_op = Oceananigans.Models.buoyancy_operation(model)
Ro = @at (Center, Center, Center) (∂x(v) - ∂y(u)) / f₀
KE = @at (Center, Center, Center) (u^2 + v^2 + w^2) / 2
PV = @at (Center, Center, Center) ((∂y(w) - ∂z(v)) * ∂x(b_op) + (∂z(u) - ∂x(w)) * ∂y(b_op) +
                                   (f₀ + ∂x(v) - ∂y(u)) * ∂z(b_op))

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
    # Through the shoal's centre line (shoal_y0), not the domain centre: they
    # differ once shoal_y0 != Ly/2. Same index as before when they coincide.
    indices=(:, clamp(round(Int, params.Ny * sweep_shoal_y0 / params.Ly), 1, params.Ny), :),
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
    ds.attrib["shoal_length"] = sweep_crest_length > 0 ?
        shoal_crest_start(sweep_Zs, sweep_Zsh, sweep_shelf_break_end, sweep_rise_mid, sweep_rise_width, sweep_h_deep) + sweep_crest_length :
        sweep_shoal_length                                # where the crest ENDS (derived if crest_length set)
    ds.attrib["crest_length"] = sweep_crest_length        # 0 = not declared; length = shoal_length - crest_start
    ds.attrib["crest_start"] = shoal_crest_start(sweep_Zs, sweep_Zsh, sweep_shelf_break_end, sweep_rise_mid, sweep_rise_width, sweep_h_deep)
    ds.attrib["Ly"] = sweep_Ly
    ds.attrib["shoal_y0"] = sweep_shoal_y0
    ds.attrib["sigma"] = sweep_sigma
    ds.attrib["shoal_start"] = sweep_shoal_start
    ds.attrib["shoal_inner_width"] = sweep_shoal_inner_width
    ds.attrib["Zsh"] = sweep_Zsh
    ds.attrib["shelf_break_end"] = sweep_shelf_break_end
    ds.attrib["rise_mid"] = sweep_rise_mid
    ds.attrib["rise_width"] = sweep_rise_width
    ds.attrib["h_deep"] = sweep_h_deep
    ds.attrib["Lz"] = sweep_Lz
    ds.attrib["dx"] = sweep_dx
    ds.attrib["wind_stress"] = sweep_wind_stress          # peak for pulsed/reversing
    ds.attrib["wind_mode"] = WIND_MODE_NAME
    ds.attrib["wind_period_days"] = WIND_PERIOD_DAYS
    ds.attrib["wind_duty"] = WIND_DUTY
    ds.attrib["wind_stress_mean"] = sweep_wind_stress * WIND_MEAN_FACTOR
    ds.attrib["front_ds"] = FRONT_DS
    ds.attrib["front_x"] = FRONT_X
    ds.attrib["front_width"] = FRONT_W
    ds.attrib["front_depth"] = FRONT_H
    ds.attrib["v0"] = sweep_v0
    ds.attrib["open_y"] = Int(open_y)
    # ape_wedge.m rebuilds the initial T/S profile (T_south_pwl/S_south_pwl)
    # as a fixed reference for g'_h, and the tail below the last breakpoint
    # depends on this. Absent in older files, which all ran with 0.
    ds.attrib["deep_strat_frac"] = DEEP_STRAT_FRAC
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
        @inline Sᵢ(x, y, z) = blend(S_south_pwl(z, global_params.S_south_v1), S_north_pwl(z, global_params.S_north_v1), α_lin(y)) + S_front(x, z)
    else
        @inline Tᵢ(x, y, z) = T_south_pwl(z, global_params.T_south_v1)
        @inline Sᵢ(x, y, z) = S_south_pwl(z, global_params.S_south_v1) + S_front(x, z)
    end

    if closure_choice === :catke
        set!(model, u=0.0, v=v_init, T=Tᵢ, S=Sᵢ, e=1e-6)
        open_y && set!(model, η=(x, y, z) -> η_geo(x))   # start in balance with the boundary η
    else
        set!(model, u=0.0, v=v_init, T=Tᵢ, S=Sᵢ)
        open_y && set!(model, η=(x, y, z) -> η_geo(x))
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
 max_Δt:          $(max_Δt) s  (scaled from 600 s at Δh = 500 m)
 EOS:             $(eos_choice) (ρ_ref = $(ρ₀))
 east boundary:   $(open_east ? "radiative (NormalRadiation, target_transport=0)" : "wall + sponge")
 snapshot every:  $(snapshot_interval/3600) h  (inertial period 20.74 h)

 ── Sweep Parameters ──
 Zs (shoal_depth):$(sweep_Zs) m
 shoal_length:    $(sweep_shoal_length) m
 Zsh (shelf_depth):$(sweep_Zsh) m
 shelf_break_end: $(sweep_shelf_break_end) m
 rise_mid:        $(sweep_rise_mid) m  (shelf $(round((sweep_rise_mid - sweep_shelf_break_end)/1e3, digits=1)) km)
 wind_stress:     $(sweep_wind_stress) N/m^2
 v0:              $(sweep_v0) m/s

 ── Switches ──
 LES:             $(LES)
 mass_flux:       $(mass_flux)
 periodic_y:      $(periodic_y)   (open_y = $(open_y))
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
