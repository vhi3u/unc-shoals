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
sweep_shoal_length = Float64(this_run["shoal_length"])
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

@info """
════════════════════════════════════════════════════════
 SWEEP PARAMETERS (run $run_index / $(length(runs)))
════════════════════════════════════════════════════════
 Label:           $sweep_run_label
 Zs (shoal_depth):$sweep_Zs m
 shoal_length:    $sweep_shoal_length m  (crest $(round((sweep_shoal_length - sweep_shoal_start)/1e3, digits=1)) km)
 shoal_start:     $sweep_shoal_start m
 shoal_inner_w:   $sweep_shoal_inner_width m
 sigma:           $sweep_sigma m
 Zsh (shelf_depth):$sweep_Zsh m
 shelf_break_end: $sweep_shelf_break_end m
 rise_mid:        $sweep_rise_mid m  (shelf length $(round((sweep_rise_mid - sweep_shelf_break_end)/1e3, digits=1)) km)
 Stratification:  $sweep_strat
 Wind Stress:     $sweep_wind_stress N/m^2
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
output_dir = joinpath(ROOT, "sweep_output", "sweep_$(lpad(run_index, 3, '0'))_$(sweep_run_label)")
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
ENV["SWEEP_STRAT"] = sweep_strat
ENV["SWEEP_WIND_STRESS"] = string(sweep_wind_stress)
ENV["SWEEP_V0"] = string(sweep_v0)
ENV["SWEEP_DX"] = string(sweep_dx)
ENV["SWEEP_SIM_DAYS"] = string(sweep_sim_days)
ENV["SWEEP_RUN_LABEL"] = sweep_run_label
ENV["SWEEP_RUN_INDEX"] = string(run_index)

# ── Include the simulation script ──────────────────────────────────────
sim_script = joinpath(ROOT, "src", "flow_over_shoals_sweep.jl")
@info "Including simulation script: $sim_script"
include(sim_script)
