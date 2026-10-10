using Pkg
Pkg.instantiate()
# ═══════════════════════════════════════════════════════════════════════════
# flow_over_shoals_buoyancy.jl
# ═══════════════════════════════════════════════════════════════════════════
# Single-buoyancy-tracer counterpart of flow_over_shoals_hydrostatic.jl.
# Same bathymetry, domain, resolution, wind and boundary treatment; the only
# physics change is T and S -> a single prognostic buoyancy tracer b.
#
# Why drop T/S:
#   1. With no water-mass labelling in play, two tracers earn nothing. One
#      tracer and no equation-of-state evaluation is cheaper and simpler.
#   2. It removes the equation of state entirely. Under TEOS10 the Boussinesq
#      buoyancy b = -g ρ′(Θ,Sᴬ,Z)/ρ_ref makes a PERFECTLY MIXED column weakly
#      unstable (N² = -3.8e-6 s⁻²), because the instability comes from the Z
#      term and so survives mixing. With CATKE convectively adjusting on any
#      N² < 0 that is a feedback with no fixed point, and it is what was
#      blowing up the sweep. A prognostic b cannot reproduce it: a mixed
#      column is exactly neutral by construction.
#
# Stratification (see strat_choice below):
#   :uniform  — b(z) = N² z with N² = 2e-4 s⁻². Within the observed range
#               (CTD column-mean 1.5e-4, pycnocline peak 3.9e-4), the value
#               used by Zhao et al., and it gives Rd = 2.67 km = 5.3 cells at
#               Δx = 500 m — the first configuration here in which the deep
#               eddies are actually resolved. It is an IDEALISATION: the real
#               profile is surface-intensified with a ~20 m unstratified
#               bottom layer, and uniform N² has neither.
#   :observed — the CTD profile reduced to b(z) using a linear EOS with
#               coefficients tuned to 24-26 °C (α = 2.94e-4, β = 7.24e-4).
#               Reproduces the TEOS10 deformation radius to 1% (1.72 vs
#               1.70 km), so it is directly comparable with runs 54-61 while
#               still being a single tracer with no EOS.
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
using Oceanostics: RossbyNumber, KineticEnergy
using Oceanostics.ProgressMessengers: TimedMessenger
using Printf: @sprintf
using NCDatasets
using CUDA: has_cuda_gpu

@info "building domain"

# switches
LES = true
mass_flux = true
periodic_y = true
sigmoid_v_bc = true
sigmoid_ic = true
sigmoid_wind = false         # uniform wind: a tapered stress gradient drives
# shear instability at the taper line (runs 51/52)
is_coriolis = true
shoal_bath = true
ramp_wind = true             # ramp τ over ~2 inertial periods
open_east = true             # radiative east boundary instead of wall + sponge

strat_choice = :uniform      # :uniform | :observed
closure_choice = :catke      # :catke | :constant

if has_cuda_gpu()
    arch = GPU()
else
    arch = CPU()
end
@info "architecture = $(arch)"

# ═══════════════════════════════════════════════════════════════════════════
# Bathymetry
# ═══════════════════════════════════════════════════════════════════════════
# dshoal_vn_param_coastal.jl, NOT dshoal_vn_param.jl: the original pinned the
# coast at 5 m depth in two places (h0 = -5.0, and a min(-5.0, ...) clamp), so
# lowering Zs and Zsh moved the shoal and shelf but left a 5 m nearshore strip
# that funnelled the alongshore flow into a coastal jet. Zc moves it.
include(joinpath(@__DIR__, "dshoal_vn_param_coastal.jl"))

# Defined once and used in BOTH the constructor and the configuration banner,
# so the two cannot drift apart (the banner was already stale against the
# constructor before this).
Zc_bath = -25.0              # depth at the coast, x = 0
Zs_bath = -30.0              # shoal crest
Zsh_bath = -50.0             # shelf
shoal_length_bath = 40000.0
sigma_bath = 8000.0
shelf_break_end_bath = 12000.0
coastal_drop_bath = 2.0      # drop across the 0-5 km ramp; preserves the slope

# simulation knobs
run_number = 73
callback_interval = 1days
snapshot_interval = 6hours   # sub-inertial: the inertial period is 20.74 h, so
# DAILY output would alias it to a fake 6.35-day band
run_tag = "periodic_buoy$(run_number)_hydro"

pickup = false
wind_stress = 0.05           # N m⁻², northward (upwelling-favourable here)
sim_runtime = 25days
@info "Starting buoyancy-tracer run $(run_tag)."

if LES
    params = (; Lx=250e3, Ly=200e3, Lz=50)
else
    params = (; Lx=100000, Ly=200000, Lz=50)
end

# Horizontal grid spacing is the knob, not Nx/Ny, so changing Lx can never
# silently change the resolution (and vice versa).
Δh = 500.0   # m
if arch == CPU()
    params = (; params..., Nx=50, Ny=50, Nz=10)
else
    params = (; params...,
        Nx=round(Int, params.Lx / Δh),
        Ny=round(Int, params.Ly / Δh),
        Nz=50)
end

x, y, z = (0, params.Lx), (0, params.Ly), (-params.Lz, 0)

if periodic_y
    grid = RectilinearGrid(arch; size=(params.Nx, params.Ny, params.Nz), halo=(4, 4, 4), x, y, z, topology=(Bounded, Periodic, Bounded))
else
    grid = RectilinearGrid(arch; size=(params.Nx, params.Ny, params.Nz), halo=(4, 4, 4), x, y, z, topology=(Bounded, Bounded, Bounded))
end

if shoal_bath
    slope_bottom = dshoal_param_bottom_coastal(params.Ly;
        Zs=Zs_bath,
        shoal_length=shoal_length_bath,
        sigma=sigma_bath,
        Zsh=Zsh_bath,
        shelf_break_end=shelf_break_end_bath,
        Zc=Zc_bath,
        coastal_drop=coastal_drop_bath)
    immersed_boundary = GridFittedBottom(slope_bottom)
    ib_grid = ImmersedBoundaryGrid(grid, immersed_boundary)
else
    ib_grid = grid
end

@info ib_grid

# ═══════════════════════════════════════════════════════════════════════════
# Stratification
# ═══════════════════════════════════════════════════════════════════════════
const N²₀ = 2e-4             # s⁻², uniform background stratification
const g_earth = 9.80665

# CTD-derived profiles, kept so :observed stays available. NOTE the name
# smooth_step_z: dshoal_vn_param.jl already defines a 3-argument smooth_step,
# and giving this one the same name invites a silent collision.
const δ_smooth = 2.5
@inline smooth_step_z(z, z0) = 0.5 * (1.0 - tanh((z - z0) / δ_smooth))

@inline function T_south_pwl(z)
    z1, z2, z3 = -5.0, -15.0, -30.0
    v1, v2, v3 = 24.5378, 24.3073, 23.4116
    m12 = (v2 - v1) / (z2 - z1)
    m23 = (v3 - v2) / (z3 - z2)
    w1, w2, w3 = smooth_step_z(z, z1), smooth_step_z(z, z2), smooth_step_z(z, z3)
    return v1 * (1 - w1) + (v1 + m12 * (z - z1)) * (w1 - w2) + (v2 + m23 * (z - z2)) * (w2 - w3) + v3 * w3
end

@inline function S_south_pwl(z)
    z1, z2, z3 = -5.0, -15.0, -30.0
    v1, v2, v3 = 35.5830, 35.9986, 36.1776
    m12 = (v2 - v1) / (z2 - z1)
    m23 = (v3 - v2) / (z3 - z2)
    w1, w2, w3 = smooth_step_z(z, z1), smooth_step_z(z, z2), smooth_step_z(z, z3)
    return v1 * (1 - w1) + (v1 + m12 * (z - z1)) * (w1 - w2) + (v2 + m23 * (z - z2)) * (w2 - w3) + v3 * w3
end

# α and β tuned to 24-26 °C. The Oceananigans LinearEquationOfState default
# α = 1.67e-4 is a ~10 °C value and is 76% too small here.
const α_b = 2.94e-4
const β_b = 7.24e-4

# Resolved at parse time, so the resulting b_profile is a plain function with
# no runtime branch — which is what makes it safe to call inside a GPU kernel.
if strat_choice === :uniform
    @inline b_profile(z) = N²₀ * z
elseif strat_choice === :observed
    @inline b_profile(z) = g_earth * (α_b * (T_south_pwl(z) - 24.5378) - β_b * (S_south_pwl(z) - 35.5830))
else
    error("strat_choice must be :uniform or :observed; got $(strat_choice)")
end

@info @sprintf("Stratification (%s): b(0) = %.5f, b(-%.0f) = %.5f m s⁻², Δb = %.5f m s⁻²",
    strat_choice, b_profile(0.0), params.Lz, b_profile(-params.Lz),
    b_profile(0.0) - b_profile(-params.Lz))

#+++ Drag
z₀ = 2.5e-4
z₁ = Oceananigans.Grids.minimum_zspacing(grid, Center(), Center(), Center()) / 2
@info "Using z₁ =" z₁

const κᵛᵏ = 0.4
c_dz = (κᵛᵏ / log(z₁ / z₀))^2
@info "Defining momentum BCs with Cᴰ =" c_dz
drag = BulkDrag(coefficient=c_dz)
# Bottom facet only: on a stair-stepped bottom, drag on the vertical side faces
# of the steps is unphysical and is itself a grid-scale vorticity source.
immersed_drag = ImmersedBoundaryCondition(bottom=drag)

max_Δt = 10minutes
initial_Δt = 1minutes
#---

# ═══════════════════════════════════════════════════════════════════════════
# Wind stress
# ═══════════════════════════════════════════════════════════════════════════
# A step-function wind at t = 0 injects a domain-filling inertial oscillation
# with nothing to decay against. Ramping with T ≫ 1/f suppresses that free mode
# by ~1/(fT)²: at T = 2 inertial periods, fT ≈ 12.6, so ~1%.
ρ₀ = 1024.0
const f₀ = 2 * 7.292115e-5 * sind(35.2480)          # 8.42e-5 s⁻¹
const inertial_period = 2π / f₀                     # 20.74 h
const T_ramp = ramp_wind ? 2 * inertial_period : 0.0
const τ_kinematic = wind_stress / ρ₀                # m² s⁻²

@info @sprintf("f = %.3e s⁻¹, inertial period = %.2f h, wind ramp = %.1f h",
    f₀, inertial_period / 3600, T_ramp / 3600)

@inline wind_ramp(t) = ifelse(T_ramp > 0, 1 - exp(-t / T_ramp), one(t))
# A negative top flux is a POSITIVE-direction stress in Oceananigans, so this is
# northward — the same sense as v₀.
@inline wind_v_flux(x, y, t) = -τ_kinematic * wind_ramp(t)

wind_bc_u = FluxBoundaryCondition(0.0)
wind_bc_v = FluxBoundaryCondition(wind_v_flux)

# ═══════════════════════════════════════════════════════════════════════════
# Prescribed inflow
# ═══════════════════════════════════════════════════════════════════════════
if mass_flux
    v₀ = 0.20
else
    v₀ = 0.0
end

params = (; params..., v₀=v₀, Ls=20e3, Le=50e3, τ=6hours)

# Pinned, NOT tied to Lx: tying k to Lx means changing the domain width
# silently rebroadens the inflow profile and breaks run-to-run comparability.
const k1_fixed = 80 / 150e3     # coastal sigmoid, e-folding width 1875 m
const k2_fixed = 40 / 150e3     # offshore sigmoid, e-folding width 3750 m

@inline sigmoidal_s2(x) = 1 / (1 + exp(k2_fixed * (x - 65e3)))

if sigmoid_v_bc
    @inline function v∞(x, z, t, p)
        s1 = 1 / (1 + exp(-k1_fixed * (x - 3e3)))
        s = (s1 - 1) + sigmoidal_s2(x)
        return p.v₀ * clamp(s, 0.0, 1.0)
    end
else
    @inline v∞(x, z, t, p) = p.v₀
end

# ═══════════════════════════════════════════════════════════════════════════
# Sponges — southern inflow only; the east boundary is radiative
# ═══════════════════════════════════════════════════════════════════════════
const south_mask = PiecewiseLinearMask{:y}(center=0.0, width=params.Ls)
const global_params = params

@inline v_target_inflow(x, y, z, t) = v∞(x, z, t, global_params)
@inline b_target_inflow(x, y, z, t) = b_profile(z)

u_sponge_inflow = Relaxation(; rate=1 / global_params.τ, mask=south_mask, target=0.0)
v_sponge_inflow = Relaxation(; rate=1 / global_params.τ, mask=south_mask, target=v_target_inflow)
b_sponge_inflow = Relaxation(; rate=1 / global_params.τ, mask=south_mask, target=b_target_inflow)

forcings = (u=u_sponge_inflow, v=v_sponge_inflow, b=b_sponge_inflow)

# ═══════════════════════════════════════════════════════════════════════════
# Eastern boundary
# ═══════════════════════════════════════════════════════════════════════════
# A closed wall cannot pass the wind-driven Ekman transport; it is forced
# downward instead and bends the isopycnals at the boundary. An open boundary
# lets it leave, with the net pinned to zero because the west is a closed coast
# and y is periodic — so Ekman exits at the surface and returns at depth.
#
# outflow_timescale is FINITE, not Inf. Inf means literally no restoring term,
# so the boundary value can drift unbounded. At 30 days the steady-state
# suppression is Cn/(Cn + Δt/τ) ≥ 98% even for a small phase-speed Courant
# number, so the outflow is preserved while the value stays bounded.
if open_east
    east_radiation = NormalRadiation(outflow_timescale=30days,
        inflow_timescale=1days,
        target_transport=0)

    # Discrete form, (j, k) for an x-boundary: the only function-valued
    # exterior state exercised in Oceananigans' open-boundary tests. It must
    # vary with depth, or the deep return flow imports surface buoyancy and
    # destroys the stratification at the boundary.
    @inline b_east(j, k, grid, clock, fields) = b_profile(znode(k, grid, Center()))

    u_bcs = FieldBoundaryConditions(immersed=immersed_drag, bottom=drag, top=wind_bc_u,
        east=NormalFlowBoundaryCondition(0; scheme=east_radiation))
    v_bcs = FieldBoundaryConditions(immersed=immersed_drag, bottom=drag, top=wind_bc_v)
    b_bcs = FieldBoundaryConditions(east=ValueBoundaryCondition(b_east; scheme=east_radiation, discrete_form=true))

    # Barotropic transport: exterior (U, η) = (0, 0). This is what holds the
    # net mass flux at zero. Omit it and the baroclinic mode radiates while the
    # barotropic mode reflects off a closed boundary — worse than either.
    U_bcs = FieldBoundaryConditions(ib_grid, (Face(), Center(), nothing);
        east=GravityWaveRadiationBoundaryCondition((0.0, 0.0)))

    bcs = (u=u_bcs, v=v_bcs, b=b_bcs, U=U_bcs)
else
    u_bcs = FieldBoundaryConditions(immersed=immersed_drag, bottom=drag, top=wind_bc_u)
    v_bcs = FieldBoundaryConditions(immersed=immersed_drag, bottom=drag, top=wind_bc_v)
    b_bcs = FieldBoundaryConditions()
    bcs = (u=u_bcs, v=v_bcs, b=b_bcs)
end
@info "Boundary Conditions:" bcs

coriolis = is_coriolis ? FPlane(latitude=35.2480) : nothing

# ═══════════════════════════════════════════════════════════════════════════
# Turbulence closure
# ═══════════════════════════════════════════════════════════════════════════
# Vertical only. At Δx = 500 m with Rd = 2.67 km the scale separation from 2Δx
# is 2.67, so no biharmonic ν₄ damps grid noise without also damping the
# eddies; WENO-5's implicit dissipation is the lateral sink.
if closure_choice === :catke
    closure = CATKEVerticalDiffusivity()
    tracers = (:b,)          # CATKE registers :e itself — do NOT list it here
elseif closure_choice === :constant
    closure = VerticalScalarDiffusivity(ν=1e-4, κ=1e-5)
    tracers = (:b,)
else
    error("closure_choice must be :catke or :constant; got $(closure_choice)")
end
@info "Closure ($(closure_choice)):" closure

# ═══════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════
model = HydrostaticFreeSurfaceModel(ib_grid;
    momentum_advection=WENO(order=5),
    tracer_advection=WENO(order=5),
    free_surface=SplitExplicitFreeSurface(ib_grid; cfl=0.7),
    tracers=tracers,
    buoyancy=BuoyancyTracer(),
    coriolis=coriolis,
    closure=closure,
    boundary_conditions=bcs,
    forcing=forcings
)

@info "" model

overwrite_files = (pickup === false)

simulation = Simulation(model, Δt=initial_Δt, stop_time=sim_runtime)
conjure_time_step_wizard!(simulation, cfl=0.7, max_Δt=max_Δt)

progress = TimedMessenger()
simulation.callbacks[:progress] = Callback(progress, TimeInterval(callback_interval))

# Health check. Without it a blow-up surfaces as `InexactError: Int64(NaN)`
# from calculate_substeps, which only says Δt already became NaN — not which
# field failed, where, or when.
function report_health(sim)
    m = sim.model
    flds = (u=m.velocities.u, v=m.velocities.v, w=m.velocities.w, b=m.tracers.b)
    bad = String[]
    for (name, f) in pairs(flds)
        colmax = Array(vec(maximum(abs, interior(f), dims=(2, 3))))
        if !all(isfinite, colmax)
            i = findfirst(!isfinite, colmax)
            push!(bad, string(name, " first at i=", i,
                " (x ≈ ", round((i - 0.5) * Δh / 1e3, digits=1), " km)"))
        end
    end
    isempty(bad) || error(string("BLOW-UP, not a solver bug.",
        "\n  iteration = ", iteration(sim),
        ", t = ", round(time(sim) / 3600, digits=2), " h",
        ", Δt = ", round(sim.Δt, digits=1), " s",
        "\n  non-finite: ", join(bad, "; ")))
    return nothing
end
# simulation.callbacks[:health] = Callback(report_health, IterationInterval(50))

# ═══════════════════════════════════════════════════════════════════════════
# Outputs — surface and mid-y slices only
# ═══════════════════════════════════════════════════════════════════════════
u, v, w = model.velocities
b = model.tracers.b

Ro = @at (Center, Center, Center) RossbyNumber(model)
KE = @at (Center, Center, Center) KineticEnergy(model)
u_c = @at (Center, Center, Center) u
v_c = @at (Center, Center, Center) v
w_c = @at (Center, Center, Center) w

slice_fields = (; u_c, v_c, w_c, b, Ro, KE)
if closure_choice === :catke
    slice_fields = (; slice_fields..., e=model.tracers.e)
end

simulation.output_writers[:surface_slice] = NetCDFWriter(model, slice_fields,
    filename="top_$(run_tag).nc",
    schedule=TimeInterval(snapshot_interval),
    indices=(:, :, params.Nz),
    overwrite_files=overwrite_files)

simulation.output_writers[:midy_slice] = NetCDFWriter(model, slice_fields,
    filename="midy_$(run_tag).nc",
    schedule=TimeInterval(snapshot_interval),
    indices=(:, round(Int, params.Ny / 2), :),
    overwrite_files=overwrite_files)

# ═══════════════════════════════════════════════════════════════════════════
# Initial conditions
# ═══════════════════════════════════════════════════════════════════════════
if pickup === false
    @info "Setting initial conditions"
    v_init = sigmoid_ic ? ((x, y, z) -> v∞(x, z, 0, params)) : v₀
    bᵢ(x, y, z) = b_profile(z)

    if closure_choice === :catke
        set!(model, u=0.0, v=v_init, b=bᵢ, e=1e-6)
    else
        set!(model, u=0.0, v=v_init, b=bᵢ)
    end
else
    @info "Skipping initial conditions setup — picking up from $(pickup)."
end

# Mode-1 WKB radius from the actual b_profile, so it is right for either
# strat_choice rather than only for :uniform.
function deformation_radius(H, n=4000)
    dz = H / n
    tot = 0.0
    for i in 1:n
        zc = -H + dz * (i - 0.5)
        N² = (b_profile(zc + 0.25) - b_profile(zc - 0.25)) / 0.5
        tot += sqrt(max(N², 0.0)) * dz
    end
    return tot / π / f₀
end
Rd_deep = deformation_radius(params.Lz)
Rd_shelf = deformation_radius(25.0)

@info """
════════════════════════════════════════════════════════
 SIMULATION: $(run_tag)   [HYDROSTATIC, BUOYANCY TRACER]
════════════════════════════════════════════════════════
 Run number:      $(run_number)
 Runtime:         $(sim_runtime)
 Architecture:    $(arch)
 Domain:          $(params.Lx/1e3) x $(params.Ly/1e3) km x $(params.Lz) m
 Grid:            $(params.Nx) x $(params.Ny) x $(params.Nz)  (Δx = Δy = $(Δh) m, Δz = $(params.Lz/params.Nz) m)

 ── Physics ──
 buoyancy:        BuoyancyTracer (single tracer b, no equation of state)
 stratification:  $(strat_choice)$(strat_choice === :uniform ? ", N² = $(N²₀) s⁻²" : " (CTD profile, tuned linear EOS)")
 Rd (H = 50 m):   $(round(Rd_deep/1e3, digits=2)) km = $(round(Rd_deep/Δh, digits=1)) cells
 Rd (H = 25 m):   $(round(Rd_shelf/1e3, digits=2)) km = $(round(Rd_shelf/Δh, digits=1)) cells
 wind_stress:     $(wind_stress) N/m² northward
 wind ramp:       $(round(T_ramp/3600, digits=1)) h
 vertical closure:$(closure_choice)
 east boundary:   $(open_east ? "radiative (NormalRadiation, target_transport=0)" : "closed wall")
 Cᴰ:              $(c_dz)
 max_Δt:          $(max_Δt) s
 snapshot every:  $(snapshot_interval/3600) h  (inertial period 20.74 h)

 ── Bathymetry (dshoal_vn_param_coastal.jl) ──
 Zc (coast, x=0):   $(Zc_bath) m
 coastal_drop:      $(coastal_drop_bath) m across the 0-5 km ramp
 Zs (shoal crest):  $(Zs_bath) m
 Zsh (shelf):       $(Zsh_bath) m
 shoal_length:      $(shoal_length_bath) m
 shelf_break_end:   $(shelf_break_end_bath) m
 shallowest point:  $(round(-max(Zc_bath, Zs_bath), digits=1)) m = $(round(Int, -max(Zc_bath, Zs_bath))) cells at Δz = $(params.Lz/params.Nz) m
════════════════════════════════════════════════════════
"""
run!(simulation, pickup=pickup)
