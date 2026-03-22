# ═══════════════════════════════════════════════════════════════════════════
# Sweep Driver — reads params.toml, selects one run by PBS_ARRAY_INDEX,
# then includes the simulation script with the right parameters injected.
# ═══════════════════════════════════════════════════════════════════════════

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
params_file = joinpath(@__DIR__, "params.toml")
config = TOML.parsefile(params_file)
runs = config["runs"]

if run_index < 1 || run_index > length(runs)
    error("Run index $run_index is out of range. params.toml has $(length(runs)) entries.")
end

this_run = runs[run_index]

# ── Extract sweep parameters (shelf_depth & shelf_break_end are optional) ─
sweep_Hs = Float64(this_run["Hs"])
sweep_shoal_length = Float64(this_run["shoal_length"])
sweep_shelf_depth = Float64(get(this_run, "shelf_depth", -25.0))
sweep_shelf_break_end = Float64(get(this_run, "shelf_break_end", 12000.0))
sweep_run_label = String(this_run["run_label"])

@info """
════════════════════════════════════════════════════════
 SWEEP PARAMETERS (run $run_index / $(length(runs)))
════════════════════════════════════════════════════════
 Label:           $sweep_run_label
 Hs:              $sweep_Hs m
 shoal_length:    $sweep_shoal_length m
 shelf_depth:     $sweep_shelf_depth m
 shelf_break_end: $sweep_shelf_break_end m
════════════════════════════════════════════════════════
"""

# ── Create output directory for this run ───────────────────────────────
output_dir = joinpath(dirname(@__DIR__), "sweep_output", "sweep_$(lpad(run_index, 3, '0'))_$(sweep_run_label)")
mkpath(output_dir)
cd(output_dir)
@info "Output directory: $output_dir"

# ── Set environment variables so sim script can read them ──────────────
# This avoids modifying the simulation script's argument parsing.
ENV["SWEEP_Hs"] = string(sweep_Hs)
ENV["SWEEP_SHOAL_LENGTH"] = string(sweep_shoal_length)
ENV["SWEEP_SHELF_DEPTH"] = string(sweep_shelf_depth)
ENV["SWEEP_SHELF_BREAK_END"] = string(sweep_shelf_break_end)
ENV["SWEEP_RUN_LABEL"] = sweep_run_label
ENV["SWEEP_RUN_INDEX"] = string(run_index)

# ── Include the simulation script ──────────────────────────────────────
sim_script = joinpath(dirname(@__DIR__), "flow_over_shoals_sweep.jl")
@info "Including simulation script: $sim_script"
include(sim_script)
