# ═══════════════════════════════════════════════════════════════════════════
# test_open_boundaries.jl — a non-periodic channel with open boundaries
# ═══════════════════════════════════════════════════════════════════════════
# Standalone test of the Oceananigans >= 0.118 open-boundary machinery for the
# shoal problem, before it goes into flow_over_shoals_sweep.jl.
#
# The sweep runs a y-PERIODIC channel: water leaving the north re-enters the
# south, where a 20 km strip nudges it back to the inflow jet and profile every
# 6 h. The shoal's wake therefore recirculates, and the inflow is "clean" only
# because the nudging scrubs it. This script replaces that with real open
# boundaries:
#
#   west  (x = 0)   coast, a wall                       topology (Closed, Open) in x
#   south (y = 0)   INFLOW: the jet v∞(x, z) and the ambient water (plus an
#                   optional cross-shelf front) enter through the boundary
#   north (y = Ly)  OUTFLOW: the jet, the wake and any eddies leave
#   east  (x = Lx)  open ocean, zero net transport
#
# Boundary conditions, per field:
#   v, u (boundary-normal)  ObliqueRadiation (Raymond & Kuo 1984, Marchesiello
#                           et al. 2001 nudging). Unlike NormalRadiation it
#                           radiates disturbances that reach the boundary at an
#                           angle, which is what eddies and a meandering wake do.
#                           On inflow it relaxes to the exterior value over
#                           inflow_timescale; on outflow it radiates, with a weak
#                           pull toward the exterior over outflow_timescale.
#   tangential u, v         ObliqueRadiation as Value conditions
#   U, V (barotropic)       Flather (GravityWaveRadiation): external transport
#                           prescribed, surface gravity waves radiate. The net
#                           transport is PINNED (target_transport): in at the
#                           south = out at the north, zero at the east, so the
#                           mean free surface cannot drift.
#   η                       SurfaceWaveRadiation (paired with Flather by default)
#   T, S                    TracerReservoir (MOM6-style): the exterior water mass
#                           enters on inflow, the interior value leaves on outflow
#
# Usage (from the project root):
#   julia --project=. src/test_open_boundaries.jl                 # CPU smoke test
#   OB_DX=500 OB_DAYS=30 julia --project=. src/test_open_boundaries.jl   # GPU
# Environment knobs (all optional):
#   OB_DX (m, default 2500)   OB_DZ (m, 2)      OB_DAYS (5)    OB_V0 (0.2 m/s)
#   OB_WIND (N/m^2, 0)        OB_FRONT_DS (psu, 0)  OB_FRONT_X (m, 20000)
#   OB_FRONT_WIDTH (m, 4000)  OB_OUTPUT (true/false, true)
#
# What to check in the log: no NaN; |η| mean stays ~0; south inflow and north
# outflow transport both equal the target; T and S stay inside the range set
# by the profiles (no boundary overshoot); and, in the output, the wake leaving
# the north boundary without a reflection back into the domain.
# ═══════════════════════════════════════════════════════════════════════════

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: Closed
using Oceananigans.BoundaryConditions: ObliqueRadiation, TracerReservoir, GravityWaveRadiationBoundaryCondition
using Oceananigans.Fields: interior
using Printf
using Statistics: mean
using CUDA: has_cuda_gpu
using NCDatasets   # loads the NetCDFWriter extension

# ── knobs ──────────────────────────────────────────────────────────────────
env(k, d) = parse(Float64, get(ENV, k, string(d)))
const Δh       = env("OB_DX", 2500.0)
const Δz       = env("OB_DZ", 2.0)
const sim_days = env("OB_DAYS", 5.0)
const v₀       = env("OB_V0", 0.2)
const τ_wind   = env("OB_WIND", 0.0)
const FRONT_DS = env("OB_FRONT_DS", 0.0)
const FRONT_X  = env("OB_FRONT_X", 20000.0)
const FRONT_W  = env("OB_FRONT_WIDTH", 4000.0)
const write_output = get(ENV, "OB_OUTPUT", "true") == "true"
# Control: OB_PERIODIC=true makes y periodic (no south/north boundaries, no
# nudging), everything else identical -- separates boundary artifacts from
# interior ones.
const periodic_control = get(ENV, "OB_PERIODIC", "false") == "true"
# Inflow sponge: relax u -> 0, v -> v∞, T/S -> exterior over the first
# OB_SPONGE_WIDTH metres (default 20 km, the same strip the sweep nudges and
# the MATLAB tools exclude) on a 6 h timescale. 0 turns it off. See the
# comment at the forcing for why open boundaries still want it.
const sponge_width = env("OB_SPONGE_WIDTH", 20000.0)
# Outflow tracer sponge at the north end (width in m, 0 = off; timescale 1 day).
# Tracers only: velocities and eddies must still leave through the radiation
# condition, which a velocity sponge would fight.
const north_sponge_width = env("OB_NORTH_SPONGE_WIDTH", 20000.0)

arch = has_cuda_gpu() ? GPU() : CPU()
const Lx, Ly, Lz = 150e3, 200e3, 50.0
Nx, Ny, Nz = round(Int, Lx / Δh), round(Int, Ly / Δh), round(Int, Lz / Δz)
@info "Open-boundary test on $arch: $(Nx)x$(Ny)x$(Nz), dx = $Δh m, $sim_days days, v0 = $v₀, wind = $τ_wind, front dS = $FRONT_DS"

# ── grid: coast wall at x = 0, everything else open ────────────────────────
grid = RectilinearGrid(arch; size=(Nx, Ny, Nz), halo=(4, 4, 4),
                       x=(0, Lx), y=(0, Ly), z=(-Lz, 0),
                       topology=((Closed, Open), periodic_control ? Periodic : Open, Bounded))

# Bathymetry: the sweep's own, at the Zs05_edge geometry (shoal on the shelf,
# crest 20-52 km). The shoal is centred at Ly/2 with a Gaussian of sigma = 8 km
# alongshore, so both open boundaries see the plain shelf.
include(joinpath(@__DIR__, "dshoal_vn_param.jl"))
bottom = dshoal_param_bottom(Ly; Zs=-5.0, shoal_length=52e3, sigma=8e3, Zsh=-25.0,
                             shelf_break_end=12e3, rise_mid=76e3, rise_width=5e3,
                             shoal_start=20e3, shoal_inner_width=8700.0, h_deep=-50.0)
ib_grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom))

# ── water masses: the sweep's south profile, plus an optional front ────────
const δ_smooth = 2.5
@inline smooth_step_z(z, z0) = 0.5 * (1 - tanh((z - z0) / δ_smooth))
@inline function pwl(z, v1, v2, v3)
    z1, z2, z3 = -5.0, -15.0, -30.0
    m12 = (v2 - v1) / (z2 - z1); m23 = (v3 - v2) / (z3 - z2)
    w1, w2, w3 = smooth_step_z(z, z1), smooth_step_z(z, z2), smooth_step_z(z, z3)
    return v1 * (1 - w1) + (v1 + m12 * (z - z1)) * (w1 - w2) + (v2 + m23 * (z - z2)) * (w2 - w3) + v3 * w3
end
@inline T_amb(z) = pwl(z, 24.5378, 24.3073, 23.4116)
@inline S_amb(z) = pwl(z, 35.5830, 35.9986, 36.1776)
@inline S_front(x, z) = -FRONT_DS * 0.5 * (1 - tanh((x - FRONT_X) / FRONT_W))
@inline T_ext(x, z) = T_amb(z)
@inline S_ext(x, z) = S_amb(z) + S_front(x, z)

# ── the inflow jet (the sweep's v∞): zero at the coast, v0 across the shelf,
#    zero offshore of ~65 km ─────────────────────────────────────────────────
const k1, k2 = 80 / 150e3, 40 / 150e3
@inline function v∞(x)
    s = (1 / (1 + exp(-k1 * (x - 3e3))) - 1) + 1 / (1 + exp(k2 * (x - 65e3)))
    return v₀ * clamp(s, 0.0, 1.0)
end

# Net transport the jet carries, integrated over the WET part of the inflow
# section (y = 0, plain shelf), computed on the actual grid so the target the
# boundary conditions pin is exactly what the interior can hold.
xc = collect(xnodes(grid, Center()))
zc = collect(znodes(grid, Center()))
Δzc = Lz / Nz
depth_south(x) = -bottom(x, 0.0)
Q_in = sum(v∞(x) * Δh * Δzc * count(z -> z > -depth_south(x), zc) for x in xc)
@info @sprintf("inflow transport pinned at %.3e m^3/s (%.1f mSv)", Q_in, Q_in / 1e3)

# ── boundary conditions ────────────────────────────────────────────────────
# Timescales follow the Marchesiello et al. (2001) recipe: strong nudging
# where water ENTERS (it must carry the prescribed properties), weak where it
# LEAVES (just enough to keep the boundary value from wandering).
const τ_in_strong = 1hour     # south: the jet and its water mass are imposed
const τ_in_weak   = 1day      # north/east: occasional backflow, weakly nudged
const τ_out       = 30days    # outflow: radiate, with a weak pull

inflow_scheme  = ObliqueRadiation(inflow_timescale=τ_in_strong, outflow_timescale=τ_out)
outflow_scheme = ObliqueRadiation(inflow_timescale=τ_in_weak,   outflow_timescale=τ_out)

@inline v_ext(x, z, t) = v∞(x)
@inline zero_ext(ξ, z, t) = 0.0


# Barotropic transport per unit width: V∞ = v∞ * local depth. The target pins
# the boundary integral, so this sets only the SHAPE Flather starts from.
# Functions inside the Flather (U, η) tuple are called in DISCRETE form,
# (i, k, grid, clock, fields), with i the along-boundary index.
# Column depth from the immersed grid (GPU-safe; calling the bathymetry
# closure inside a boundary kernel is not guaranteed to compile for GPU).
using Oceananigans.Grids: xnode, static_column_depthᶜᶜᵃ
@inline V_ext(i, k, grid, clock, fields) = v∞(xnode(i, grid, Center())) * static_column_depthᶜᶜᵃ(i, 1, grid)

# The exterior free surface MUST be the one that balances the jet. A 0.2 m/s
# jet needs g dη/dx = f v, ~10 cm across the shelf; with η_ext = 0 the Flather
# condition fights that balance, and the first version of this script grew a
# spurious 0.3 m/s current along the south boundary that upwelled 10 m water
# to the surface (and WENO then overshot S at its edge). η_geo is the analytic
# integral of the sigmoid jet (its clamp is never active: s stays in (0, 1)),
#   ∫σ(k1(x - a)) dx = log(1 + e^{k1(x-a)})/k1,  ∫σ(-k2(x - b)) dx = -log(1 + e^{-k2(x-b)})/k2
const g_grav = 9.80665
const f_cor = 2 * 7.292115e-5 * sind(35.2480)
@inline Fjet(x) = log1p(exp(k1 * (x - 3e3))) / k1 - x - log1p(exp(-k2 * (x - 65e3))) / k2
@inline η_geo(x) = f_cor / g_grav * v₀ * (Fjet(x) - Fjet(0.0))
@inline η_ext(i, k, grid, clock, fields) = η_geo(xnode(i, grid, Center()))
V_bcs = FieldBoundaryConditions(south=GravityWaveRadiationBoundaryCondition((V_ext, η_ext); target_transport=Q_in),
                                north=GravityWaveRadiationBoundaryCondition((V_ext, η_ext); target_transport=Q_in))
# East: the jet is ~0 there, so η_geo is flat at its offshore value.
const η_east = η_geo(Lx)
U_bcs = FieldBoundaryConditions(east=GravityWaveRadiationBoundaryCondition((0.0, η_east); target_transport=0.0))

# Tracers: the exterior water mass enters on inflow; the reservoir relaxes to
# it over inflow_length_scale and to the interior over outflow_length_scale.
# Both must be FINITE here. Offshore of the jet the boundary-normal velocity is
# ~0 and flips sign, so a memoryless reservoir (length 0) jumps between the
# exterior and interior values at every flip, and WENO overshoots at the jump:
# with 0 / 0 the 2.5 km test went 0.035 psu below the freshest water anywhere
# in the profiles within 2 days. OB_RESERVOIR_L sets both (m).
const L_res = env("OB_RESERVOIR_L", 10000.0)
reservoir = TracerReservoir(inflow_length_scale=L_res, outflow_length_scale=L_res)
@inline T_ext_y(x, z, t) = T_ext(x, z)
@inline S_ext_y(x, z, t) = S_ext(x, z)
@inline T_ext_x(y, z, t) = T_ext(Lx, z)
@inline S_ext_x(y, z, t) = S_ext(Lx, z)
T_bcs = FieldBoundaryConditions(south=ValueBoundaryCondition(T_ext_y; scheme=reservoir),
                                north=ValueBoundaryCondition(T_ext_y; scheme=reservoir),
                                east=ValueBoundaryCondition(T_ext_x; scheme=reservoir))
S_bcs = FieldBoundaryConditions(south=ValueBoundaryCondition(S_ext_y; scheme=reservoir),
                                north=ValueBoundaryCondition(S_ext_y; scheme=reservoir),
                                east=ValueBoundaryCondition(S_ext_x; scheme=reservoir))

# Bottom drag and wind, as in the sweep.
z₀ = 2.5e-4
c_d = (0.4 / log((Δz / 2) / z₀))^2
drag = BulkDrag(coefficient=c_d)
immersed_drag = ImmersedBoundaryCondition(bottom=drag)
const τ_kin = τ_wind / 1024.0
@inline wind_v(x, y, t) = -τ_kin * (1 - exp(-t / (2 * 2π / 8.42e-5)))

v_bcs = FieldBoundaryConditions(south=NormalFlowBoundaryCondition(v_ext; scheme=inflow_scheme),
                                north=NormalFlowBoundaryCondition(v_ext; scheme=outflow_scheme),
                                east=ValueBoundaryCondition(0.0; scheme=outflow_scheme),
                                immersed=immersed_drag, bottom=drag, top=FluxBoundaryCondition(wind_v))
u_bcs = FieldBoundaryConditions(east=NormalFlowBoundaryCondition(0.0; scheme=outflow_scheme),
                                south=ValueBoundaryCondition(0.0; scheme=inflow_scheme),
                                north=ValueBoundaryCondition(0.0; scheme=outflow_scheme),
                                immersed=immersed_drag, bottom=drag)

if periodic_control
    v_bcs = FieldBoundaryConditions(east=ValueBoundaryCondition(0.0; scheme=outflow_scheme),
                                    immersed=immersed_drag, bottom=drag, top=FluxBoundaryCondition(wind_v))
    u_bcs = FieldBoundaryConditions(east=NormalFlowBoundaryCondition(0.0; scheme=outflow_scheme),
                                    immersed=immersed_drag, bottom=drag)
    T_bcs = FieldBoundaryConditions(east=ValueBoundaryCondition(T_ext_x; scheme=reservoir))
    S_bcs = FieldBoundaryConditions(east=ValueBoundaryCondition(S_ext_x; scheme=reservoir))
    boundary_conditions = (u=u_bcs, v=v_bcs, U=U_bcs, T=T_bcs, S=S_bcs)
else
    boundary_conditions = (u=u_bcs, v=v_bcs, U=U_bcs, V=V_bcs, T=T_bcs, S=S_bcs)
end

# ── inflow sponge ──────────────────────────────────────────────────────────
# An open boundary passes what the interior sends it, but it cannot make the
# interior ACCEPT the profile imposed on inflow. Here the jet is pushed in
# depth-uniform, while bottom drag shears it within a few km. Without a
# sponge, the 2.5 km test grew a 0.34 m/s recirculation along the south
# boundary (water leaving just offshore of the jet and turning offshore)
# that kept growing over 2 days and upwelled 10 m water to the surface.
# A sponge strip at the inflow absorbs the adjustment (Marchesiello et al.
# 2001), and it is the same strip the sweep already has and its analysis
# already excludes (y < 20 km).
if sponge_width > 0
    inflow_strip = PiecewiseLinearMask{:y}(center=0.0, width=sponge_width)
    @inline v_target(x, y, z, t) = v∞(x)
    @inline T_target(x, y, z, t) = T_ext(x, z)
    @inline S_target(x, y, z, t) = S_ext(x, z)
    u_sp = Relaxation(rate=1 / 6hours, mask=inflow_strip, target=0.0)
    v_sp = Relaxation(rate=1 / 6hours, mask=inflow_strip, target=v_target)
    T_sp = Relaxation(rate=1 / 6hours, mask=inflow_strip, target=T_target)
    S_sp = Relaxation(rate=1 / 6hours, mask=inflow_strip, target=S_target)
    if north_sponge_width > 0
        outflow_strip = PiecewiseLinearMask{:y}(center=Ly, width=north_sponge_width)
        T_sp = (T_sp, Relaxation(rate=1 / 1day, mask=outflow_strip, target=T_target))
        S_sp = (S_sp, Relaxation(rate=1 / 1day, mask=outflow_strip, target=S_target))
    end
    forcing = (u=u_sp, v=v_sp, T=T_sp, S=S_sp)
else
    forcing = NamedTuple()
end

# ── model ──────────────────────────────────────────────────────────────────
model = HydrostaticFreeSurfaceModel(ib_grid;
    momentum_advection=WENO(order=5),
    tracer_advection=WENO(order=5),
    free_surface=SplitExplicitFreeSurface(ib_grid; cfl=0.7),
    tracers=(:T, :S),
    buoyancy=SeawaterBuoyancy(),
    coriolis=FPlane(latitude=35.2480),
    closure=CATKEVerticalDiffusivity(),
    forcing,
    boundary_conditions)

set!(model, v=(x, y, z) -> v∞(x), η=(x, y, z) -> η_geo(x),
            T=(x, y, z) -> T_ext(x, z), S=(x, y, z) -> S_ext(x, z))
@info @sprintf("balanced free surface: η_geo rises %.3f m from the coast to the offshore edge", η_geo(Lx))

Δt = min(10minutes, 0.3 * Δh / max(v₀, 0.05))
simulation = Simulation(model; Δt, stop_time=sim_days * days)
wizard = TimeStepWizard(cfl=0.5, max_Δt=Δt, max_change=1.1)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

# ── diagnostics ────────────────────────────────────────────────────────────
T_lo, T_hi = extrema(T_ext(x, z) for x in xc, z in zc)
S_lo, S_hi = extrema(S_ext(x, z) for x in xc, z in zc)
function transports(model)
    v = Array(interior(model.velocities.v))
    u = Array(interior(model.velocities.u))
    south = sum(v[:, 1, :]) * Δh * Δzc
    north = sum(v[:, end, :]) * Δh * Δzc
    east  = sum(u[end, :, :]) * Δh * Δzc
    return south, north, east
end
function progress(sim)
    m = sim.model
    T = Array(interior(m.tracers.T)); S = Array(interior(m.tracers.S))
    wetT = T[T .!= 0]; wetS = S[S .!= 0]
    η = Array(interior(m.free_surface.displacement))
    s, n, e = transports(m)
    umax = maximum(abs, interior(m.velocities.u)); vmax = maximum(abs, interior(m.velocities.v))
    @printf("t = %6.2f d  Δt = %5.1f s  |u|,|v| max = %.3f, %.3f  <η> = %+.2e  η range %+.3f..%+.3f  Q south/north/east = %6.1f / %6.1f / %+6.1f mSv  T %.3f..%.3f  S %.3f..%.3f\n",
            m.clock.time / day, sim.Δt, umax, vmax, mean(η), extrema(η)..., s / 1e3, n / 1e3, e / 1e3,
            extrema(wetT)..., extrema(wetS)...)
    if any(isnan, η) || umax > 5
        error("blew up at t = $(m.clock.time / day) days")
    end
    return nothing
end
simulation.callbacks[:progress] = Callback(progress, TimeInterval(max(sim_days / 20, 0.25) * days))
@info @sprintf("profile ranges: T %.3f..%.3f, S %.3f..%.3f -- a boundary overshoot shows up as values outside these", T_lo, T_hi, S_lo, S_hi)

if write_output
    outdir = joinpath(@__DIR__, "..", "out", "open_boundaries")
    mkpath(outdir)
    tag = @sprintf("ob_dx%d%s", round(Int, Δh), periodic_control ? "_periodic" : "")
    snap = TimeInterval(max(sim_days / 40, 0.25) * days)
    simulation.output_writers[:surface] = NetCDFWriter(model, (u=model.velocities.u, v=model.velocities.v,
                                                               T=model.tracers.T, S=model.tracers.S);
        filename=joinpath(outdir, "$(tag)_surface.nc"), indices=(:, :, Nz),
        schedule=snap, overwrite_files=true)
    # η lives on the top face (k = Nz + 1), so it cannot share the k = Nz slice.
    simulation.output_writers[:eta] = NetCDFWriter(model, (; η=model.free_surface.displacement);
        filename=joinpath(outdir, "$(tag)_eta.nc"), schedule=snap, overwrite_files=true)
    # A mid-depth section across the channel near each open end, for reflections.
    simulation.output_writers[:sections] = NetCDFWriter(model, (v=model.velocities.v, S=model.tracers.S);
        filename=joinpath(outdir, "$(tag)_xz_north.nc"), indices=(:, Ny - 1, :),
        schedule=snap, overwrite_files=true)
    @info "surface snapshots -> $(joinpath(outdir, "$(tag)_surface.nc"))"
end

run!(simulation)
@info "finished"
