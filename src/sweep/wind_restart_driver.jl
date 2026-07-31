# ═══════════════════════════════════════════════════════════════════════════
# wind_restart_driver.jl
# ═══════════════════════════════════════════════════════════════════════════
# Driver script that restarts wind-driven parameter sweeps from a spun-up 50-day
# reference case checkpoint file.
# ═══════════════════════════════════════════════════════════════════════════

using TOML

# ── Parse arguments ─────────────────────────────────────────────────────
if isempty(ARGS)
    error("""
    Usage: julia --project=. sweep/wind_restart_driver.jl <RUN_INDEX>

    RUN_INDEX is the 1-based index into the [[runs]] array in sweep/params_HsLs.toml.
    On Casper this comes from PBS_ARRAY_INDEX automatically.
    """)
end

run_index = parse(Int, ARGS[1])
@info "Wind restart driver starting for run index = $run_index"

# ── Load parameter table ───────────────────────────────────────────────
params_file = joinpath(@__DIR__, "params_HsLs.toml")
config = TOML.parsefile(params_file)
runs = config["runs"]

if run_index < 1 || run_index > length(runs)
    error("Run index $run_index is out of range. $params_file has $(length(runs)) entries.")
end

this_run = runs[run_index]

# ── Extract parameters ────────────────────────────────────────────────
sweep_Hs = Float64(get(this_run, "Hs", 20.0))
sweep_shoal_length = Float64(get(this_run, "shoal_length", 40000.0))
sweep_sigma = Float64(get(this_run, "sigma", 8000.0))
sweep_shelf_depth = Float64(get(this_run, "shelf_depth", -25.0))
sweep_shelf_break_end = Float64(get(this_run, "shelf_break_end", 12000.0))
sweep_run_label = String(this_run["run_label"])
sweep_strat = String(get(this_run, "strat", "default"))
sweep_wind_stress = Float64(get(this_run, "wind_stress", 0.0))
sweep_v0 = Float64(get(this_run, "v0", 0.2))

@info """
════════════════════════════════════════════════════════
 WIND RESTART SWEEP PARAMETERS (run $run_index / $(length(runs)))
════════════════════════════════════════════════════════
 Label:           $sweep_run_label
 Hs:              $sweep_Hs m
 shoal_length:    $sweep_shoal_length m
 Wind Stress:     $sweep_wind_stress N/m^2
 v0:              $sweep_v0 m/s
════════════════════════════════════════════════════════
"""

# ── Locate Reference Spin-Up Checkpoint ──────────────────────────────
project_root = dirname(dirname(@__DIR__))
search_dirs = [
    project_root,
    joinpath(project_root, "sweep_output", "sweep_001_Hs20_Ls40k"),
    joinpath(@__DIR__, "..", "sweep_output"),
    pwd()
]

ref_checkpoint = ""
for d in search_dirs
    if isdir(d)
        ck_files = filter(f -> startswith(f, "checkpoint_") && endswith(f, ".jld2") && !endswith(f, "iteration0.jld2"), readdir(d))
        if !isempty(ck_files)
            ref_checkpoint = joinpath(d, last(sort(ck_files)))
            break
        end
    end
end

# ── Create output directory for this run ───────────────────────────────
output_dir = joinpath(project_root, "sweep_output", "sweep_$(lpad(run_index, 3, '0'))_$(sweep_run_label)")
mkpath(output_dir)

target_checkpoint = ""
if !isempty(ref_checkpoint)
    target_checkpoint = joinpath(output_dir, basename(ref_checkpoint))
    if !isfile(target_checkpoint)
        @info "Copying reference spin-up checkpoint from: $ref_checkpoint"
        @info "Target destination: $target_checkpoint"
        cp(ref_checkpoint, target_checkpoint)
    end
else
    @warn "No reference checkpoint found in search paths. Run will start from t=0 initial conditions."
end

cd(output_dir)
@info "Output directory: $output_dir"

# ── Set environment variables ──────────────────────────────────────────
ENV["SWEEP_Hs"] = string(sweep_Hs)
ENV["SWEEP_SHOAL_LENGTH"] = string(sweep_shoal_length)
ENV["SWEEP_SIGMA"] = string(sweep_sigma)
ENV["SWEEP_SHELF_DEPTH"] = string(sweep_shelf_depth)
ENV["SWEEP_SHELF_BREAK_END"] = string(sweep_shelf_break_end)
ENV["SWEEP_STRAT"] = sweep_strat
ENV["SWEEP_WIND_STRESS"] = string(sweep_wind_stress)
ENV["SWEEP_V0"] = string(sweep_v0)
ENV["SWEEP_RUN_LABEL"] = sweep_run_label
ENV["SWEEP_RUN_INDEX"] = string(run_index)

if !isempty(target_checkpoint) && isfile(target_checkpoint)
    ENV["SWEEP_PICKUP"] = target_checkpoint
end

# ── Include simulation script ──────────────────────────────────────────
sim_script = joinpath(dirname(@__DIR__), "flow_over_shoals_sweep.jl")
@info "Including simulation script: $sim_script"
include(sim_script)
