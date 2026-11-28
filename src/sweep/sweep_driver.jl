# ═══════════════════════════════════════════════════════════════════════════
# Sweep Driver — reads params.toml, selects one run by PBS_ARRAY_INDEX,
# then includes the simulation script with the right parameters injected.
# ═══════════════════════════════════════════════════════════════════════════

# Make sure the environment is materialised before anything tries to load
# Oceananigans. This is cheap and idempotent when the depot is already
# populated, and it turns "Package X is required but does not seem to be
# installed" -- which is what you get when JULIA_DEPOT_PATH points somewhere
# that was never instantiated -- into a self-healing step.
#
# IMPORTANT: still run `Pkg.precompile()` once on a LOGIN NODE before
# submitting an array. instantiate() is safe to run concurrently; precompile()
# is not -- six array jobs compiling into one depot fight over pidfiles and
# the whole array fails messily. See the res250_1.log from 2026-11-12.
using Pkg
Pkg.instantiate()

using TOML

# ── Parse arguments ─────────────────────────────────────────────────────
if isempty(ARGS)
    error("""
    Usage: julia --project=. sweep/sweep_driver.jl <RUN_INDEX>

    RUN_INDEX is the 1-based index into the [[runs]] array in sweep/params.toml.
    On Casper this comes from PBS_ARRAY_INDEX automatically.
    """)
end

run_index = parse(Int, ARGS[1])
@info "Sweep driver starting for run index = $run_index"

# ── Load parameter table ───────────────────────────────────────────────
# Which table to run. Defaults to the main geometry/wind sweep; override with
# SWEEP_PARAMS so the resolution check and weak-wind runs can share this driver.
params_name = get(ENV, "SWEEP_PARAMS", "params_HsLs.toml")
params_file = joinpath(@__DIR__, params_name)
@info "Parameter table: $params_file"
config = TOML.parsefile(params_file)
runs = config["runs"]

if run_index < 1 || run_index > length(runs)
    error("Run index $run_index is out of range. $params_file has $(length(runs)) entries.")
end

this_run = runs[run_index]

# ── Extract sweep parameters (optional parameters use defaults) ──────────
sweep_shoal_length = Float64(get(this_run, "shoal_length", 40000.0))   # unused when crest_length is set
sweep_sigma = Float64(get(this_run, "sigma", 8000.0))
sweep_Zsh = Float64(get(this_run, "Zsh", get(this_run, "shelf_depth", -25.0)))
if haskey(this_run, "Zs")
    sweep_Zs = Float64(this_run["Zs"])
elseif haskey(this_run, "Hs")
    sweep_Zs = sweep_Zsh + Float64(this_run["Hs"])
else
    error("Must provide either 'Zs' or 'Hs'")
end
sweep_shelf_break_end = Float64(get(this_run, "shelf_break_end", 12000.0))

# Offshore end of the flat shelf, so shelf length is rise_mid - shelf_break_end.
# Needed because shelf_break_end alone cannot separate shelf length from break
# steepness: widening the break to soften it also eats into the shelf.
sweep_rise_mid = Float64(get(this_run, "rise_mid", 63500.0))
sweep_rise_width = Float64(get(this_run, "rise_width", 5000.0))

# Shoreward edge of the shoal. Default 0.0 welds it to the coast at full
# amplitude, which is what every sweep before 2026-11-13 did -- and is why Zs
# also set the crest length and the inshore flank slope (see the comment in
# src/dshoal_vn_param.jl). Set shoal_start to put the shoal on the shelf, and
# shoal_inner_width to choose the flank slope rather than inherit it.
sweep_shoal_start = Float64(get(this_run, "shoal_start", 0.0))
sweep_shoal_inner_width = Float64(get(this_run, "shoal_inner_width", 0.0))

# Flat offshore depth, domain depth, vertical spacing, and the stratification
# below the deepest T/S breakpoint. All four default to the values every run
# before 2025-11 used, so an existing table is unchanged by their presence.
#
# h_deep was hardcoded at -50.0 in dshoal_vn_param.jl, which is also the old
# domain floor -- so Zsh = -50 had no shelf break, and the usable shelf-depth
# range stopped around -40 m. Deepening the basin is what unblocks it.
sweep_h_deep = Float64(get(this_run, "h_deep", -50.0))
sweep_Lz = Float64(get(this_run, "Lz", 50.0))
sweep_dz = Float64(get(this_run, "dz", 1.0))
sweep_deep_strat = Float64(get(this_run, "deep_strat_frac", 0.0))
# Scale the T and S profiles' departures from their surface values.
sweep_strat_scale_T = Float64(get(this_run, "strat_scale_T", 1.0))
sweep_strat_scale_S = Float64(get(this_run, "strat_scale_S", 1.0))
# Bounds-preserving tracer advection (see BOUNDED_TRACERS in the sweep script).
sweep_bounded_tracers = Bool(get(this_run, "bounded_tracers", false))

if sweep_h_deep >= sweep_Zsh
    error("h_deep ($(sweep_h_deep)) must be deeper than Zsh ($(sweep_Zsh)); " *
          "otherwise the offshore ramp is zero and there is no shelf break.")
end
if sweep_Lz < abs(sweep_h_deep)
    error("Lz ($(sweep_Lz)) is shallower than |h_deep| ($(abs(sweep_h_deep))); " *
          "the offshore basin would be cut off by the domain floor.")
end

sweep_run_label = String(this_run["run_label"])

# New parameters
sweep_strat = String(get(this_run, "strat", "default"))
sweep_wind_stress = Float64(get(this_run, "wind_stress", 0.0))
sweep_v0 = Float64(get(this_run, "v0", 0.2))

# Resolution and duration as sweep parameters. dx must be settable per run so a
# resolution check can put 500 m and 250 m versions of the same case in one
# table; sim_days because a 250 m run of the full 50 days is ~48 h of walltime
# and Casper caps at 24 h.
sweep_dx = Float64(get(this_run, "dx", 500.0))
sweep_sim_days = Float64(get(this_run, "sim_days", 50.0))

# Time-varying wind and a cross-shelf salinity front (see the comments at the
# top of flow_over_shoals_sweep.jl). Defaults reproduce every earlier run.
sweep_wind_mode = String(get(this_run, "wind_mode", "steady"))
sweep_wind_period = Float64(get(this_run, "wind_period_days", 10.0))
sweep_wind_duty = Float64(get(this_run, "wind_duty", 0.5))
sweep_front_ds = Float64(get(this_run, "front_ds", 0.0))
sweep_front_x = Float64(get(this_run, "front_x", 20000.0))
sweep_front_width = Float64(get(this_run, "front_width", 4000.0))
sweep_front_depth = Float64(get(this_run, "front_depth", 0.0))
# Open south/north boundaries instead of a periodic channel (Oceananigans >= 0.118).
sweep_open_y = Bool(get(this_run, "open_y", false))
# Alongshore domain length, the shoal's alongshore centre, and the crest length
# (0 = not declared: shoal_length is used as given, as in every earlier run).
run_Ly = Float64(get(this_run, "Ly", 200000.0))
run_Lx = Float64(get(this_run, "Lx", 150000.0))
run_shoal_y0 = Float64(get(this_run, "shoal_y0", run_Ly / 2))
run_crest_length = Float64(get(this_run, "crest_length", 0.0))
# Disturbance added to the inflow nudging targets, as a fraction of v0
# (0 = none, every earlier run). See INFLOW_NOISE in flow_over_shoals_sweep.jl.
run_inflow_noise = Float64(get(this_run, "inflow_noise", 0.0))
# Bottom drag only where the bottom is shelf-deep (default false: everywhere).
run_offshore_free_slip = Bool(get(this_run, "offshore_free_slip", false))
# Bathymetry: "param" (default, every earlier run) or "diamond" (src/dshoal_diamond.jl)
# and the diamond-only keys; defaults are the GEBCO 2026 Diamond Shoals values.
run_bathy = String(get(this_run, "bathy", "param"))
run_tip_depth = Float64(get(this_run, "tip_depth", -13.0))
run_taper_length = Float64(get(this_run, "taper_length", 3000.0))
run_shoreface_width = Float64(get(this_run, "shoreface_width", 10000.0))
run_break_distance = Float64(get(this_run, "break_distance", 22000.0))
run_cape_length = Float64(get(this_run, "cape_length", 0.0))

@info """
════════════════════════════════════════════════════════
 SWEEP PARAMETERS (run $run_index / $(length(runs)))
════════════════════════════════════════════════════════
 Label:           $sweep_run_label
 Zs (shoal_depth):$sweep_Zs m
 shoal_length:    $sweep_shoal_length m  (crest $(round((sweep_shoal_length - sweep_shoal_start)/1e3, digits=1)) km)
 shoal_start:     $sweep_shoal_start m
 shoal_inner_w:   $sweep_shoal_inner_width m
 h_deep:          $sweep_h_deep m  (break slope $(round((abs(sweep_h_deep) - abs(sweep_Zsh))/(2*sweep_rise_width), sigdigits=3)))
 Lz x dz:         $sweep_Lz m / $sweep_dz m  ($(round(Int, sweep_Lz/sweep_dz)) levels)
 deep_strat_frac: $sweep_deep_strat
 strat_scale T/S: $sweep_strat_scale_T / $sweep_strat_scale_S$(sweep_bounded_tracers ? "  (bounded tracer advection)" : "")
 sigma:           $sweep_sigma m
 Zsh (shelf_depth):$sweep_Zsh m
 shelf_break_end: $sweep_shelf_break_end m
 rise_mid:        $sweep_rise_mid m  (shelf length $(round((sweep_rise_mid - sweep_shelf_break_end)/1e3, digits=1)) km)
 Stratification:  $sweep_strat
 Wind Stress:     $sweep_wind_stress N/m^2  ($sweep_wind_mode, period $sweep_wind_period d, duty $sweep_wind_duty)
 y boundaries:    $(sweep_open_y ? "open (inflow south, outflow north)" : "periodic")
 Lx x Ly:         $(run_Lx / 1e3) x $(run_Ly / 1e3) km, shoal at y = $(run_shoal_y0 / 1e3) km
 crest_length:    $(run_crest_length > 0 ? "$(run_crest_length / 1e3) km" : "not declared (shoal_length used)")
 inflow_noise:    $(run_inflow_noise) x v0
 bottom drag:     $(run_offshore_free_slip ? "shelf only (free slip offshore)" : "everywhere")
 bathymetry:      $(run_bathy)$(run_bathy == "diamond" ? " (cape $(run_cape_length / 1e3) km, tip depth $(run_tip_depth) m, taper $(run_taper_length / 1e3) km, break $(run_break_distance / 1e3) km past the tip)" : "")
 Front:           dS = $sweep_front_ds psu at x = $sweep_front_x m, width $sweep_front_width m, depth scale $sweep_front_depth m
 v0:              $sweep_v0 m/s
 dx:              $sweep_dx m
 sim_days:        $sweep_sim_days days
════════════════════════════════════════════════════════
"""

# ── Locate the project root ────────────────────────────────────────────
# Do NOT use dirname(@__DIR__): this file moved from <root>/sweep to
# <root>/src/sweep in the Nov 2026 migration, so that expression now resolves
# to <root>/src and would put sweep_output/ one level too deep. Nothing would
# error -- the sweep would just write a terabyte somewhere the MATLAB tools do
# not look, and every figure would silently show the previous sweep instead.
# Walk up until we find Project.toml so the answer is independent of nesting.
function project_root(start)
    d = start
    for _ in 1:6
        isfile(joinpath(d, "Project.toml")) && return d
        parent = dirname(d)
        parent == d && break
        d = parent
    end
    error("Could not locate Project.toml above $start — cannot resolve the project root.")
end
const ROOT = project_root(@__DIR__)
@info "Project root: $ROOT"

# ── Create output directory for this run ───────────────────────────────
# Run output goes to scratch on Casper: /glade/work is quota-limited and
# sweep_output outgrew it. Off Casper (a local test) it goes to <root>/sweep_output.
const SCRATCH_OUTPUT = "/glade/derecho/scratch/nguyen/unc-shoals/sweep_output"
const OUTPUT_ROOT = isdir("/glade/derecho/scratch/nguyen") ? SCRATCH_OUTPUT : joinpath(ROOT, "sweep_output")
output_dir = joinpath(OUTPUT_ROOT, "sweep_$(lpad(run_index, 3, '0'))_$(sweep_run_label)")
mkpath(output_dir)
cd(output_dir)
@info "Output directory: $output_dir"

# ── Set environment variables so sim script can read them ──────────────
# This avoids modifying the simulation script's argument parsing.
ENV["SWEEP_Zs"] = string(sweep_Zs)
ENV["SWEEP_SHOAL_LENGTH"] = string(sweep_shoal_length)
ENV["SWEEP_SIGMA"] = string(sweep_sigma)
ENV["SWEEP_Zsh"] = string(sweep_Zsh)
ENV["SWEEP_SHELF_BREAK_END"] = string(sweep_shelf_break_end)
ENV["SWEEP_RISE_MID"] = string(sweep_rise_mid)
ENV["SWEEP_RISE_WIDTH"] = string(sweep_rise_width)
ENV["SWEEP_SHOAL_START"] = string(sweep_shoal_start)
ENV["SWEEP_SHOAL_INNER_WIDTH"] = string(sweep_shoal_inner_width)
ENV["SWEEP_H_DEEP"] = string(sweep_h_deep)
ENV["SWEEP_LZ"] = string(sweep_Lz)
ENV["SWEEP_DZ"] = string(sweep_dz)
ENV["SWEEP_DEEP_STRAT_FRAC"] = string(sweep_deep_strat)
ENV["SWEEP_STRAT_SCALE_T"] = string(sweep_strat_scale_T)
ENV["SWEEP_STRAT_SCALE_S"] = string(sweep_strat_scale_S)
ENV["SWEEP_BOUNDED_TRACERS"] = string(sweep_bounded_tracers)
ENV["SWEEP_STRAT"] = sweep_strat
ENV["SWEEP_WIND_STRESS"] = string(sweep_wind_stress)
ENV["SWEEP_V0"] = string(sweep_v0)
ENV["SWEEP_WIND_MODE"] = sweep_wind_mode
ENV["SWEEP_WIND_PERIOD_DAYS"] = string(sweep_wind_period)
ENV["SWEEP_WIND_DUTY"] = string(sweep_wind_duty)
ENV["SWEEP_FRONT_DS"] = string(sweep_front_ds)
ENV["SWEEP_FRONT_X"] = string(sweep_front_x)
ENV["SWEEP_FRONT_WIDTH"] = string(sweep_front_width)
ENV["SWEEP_FRONT_DEPTH"] = string(sweep_front_depth)
ENV["SWEEP_OPEN_Y"] = string(sweep_open_y)
ENV["SWEEP_LY"] = string(run_Ly)
ENV["SWEEP_LX"] = string(run_Lx)
ENV["SWEEP_SHOAL_Y0"] = string(run_shoal_y0)
ENV["SWEEP_CREST_LENGTH"] = string(run_crest_length)
ENV["SWEEP_INFLOW_NOISE"] = string(run_inflow_noise)
ENV["SWEEP_OFFSHORE_FREE_SLIP"] = string(run_offshore_free_slip)
ENV["SWEEP_BATHY"] = run_bathy
ENV["SWEEP_TIP_DEPTH"] = string(run_tip_depth)
ENV["SWEEP_TAPER_LENGTH"] = string(run_taper_length)
ENV["SWEEP_SHOREFACE_WIDTH"] = string(run_shoreface_width)
ENV["SWEEP_BREAK_DISTANCE"] = string(run_break_distance)
ENV["SWEEP_CAPE_LENGTH"] = string(run_cape_length)
ENV["SWEEP_DX"] = string(sweep_dx)
ENV["SWEEP_SIM_DAYS"] = string(sweep_sim_days)
ENV["SWEEP_RUN_LABEL"] = sweep_run_label
ENV["SWEEP_RUN_INDEX"] = string(run_index)

# ── Include the simulation script ──────────────────────────────────────
sim_script = joinpath(ROOT, "src", "flow_over_shoals_sweep.jl")
@info "Including simulation script: $sim_script"
include(sim_script)
