# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Idealized [Oceananigans](https://github.com/CliMA/Oceananigans.jl) LES of stratified coastal flow over a shoal/seamount. A `NonhydrostaticModel` on a rectilinear grid with a `GridFittedBottom` immersed boundary (the shoal), driven by an along-shore inflow and relaxed toward CTD-derived T/S profiles via sponge layers. Targets NCAR Casper GPUs (A100); falls back to a smaller CPU grid automatically.

## Commands

Julia project; deps are pinned in `Manifest.toml` (Julia 1.12 locally; NCAR HPC loads `julia/1.11.2`). Always use `--project`.

```bash
# One-time setup on a new machine
julia --project -e 'using Pkg; Pkg.instantiate()'

# Run the main simulation (auto-detects GPU via has_cuda_gpu(); CPU otherwise)
julia --project src/flow_over_shoals.jl

# Performance ablation benchmark (levels 0–8, each adds one model component)
julia --project=. src/benchmark_shoals.jl [--gpu] [--full] [--steps N]

# Single sweep run by 1-based index into the params TOML
julia --project=. src/sweep/sweep_driver.jl <RUN_INDEX>
```

HPC submission is via PBS on Casper: `qsub submit_benchmark.pbs`, `qsub src/sweep/submit_sweep.pbs` (a job array; `#PBS -J 1-N` must match the number of `[[runs]]` entries). PBS scripts set `JULIA_DEPOT_PATH` to a glade work path and load modules — edit `<NAME>`/`<EMAIL>`/paths before submitting.

There is no test suite. Validation is empirical: run, then inspect NetCDF output. The `plot_*.jl` / `diagnostic_*.jl` scripts (use `Plots`, run locally, not on GPU) visualize bathymetry, sponge masks, and slices.

## Architecture

**Entrypoints are self-contained, not modular.** Each simulation script is a flat top-to-bottom program that builds grid → bathymetry → BCs/forcings → model → outputs → `run!`. The only shared `include` is `src/dshoal_vn_param.jl` (bathymetry). Everything else — the T/S profile functions, sponge masks, sponge forcings, drag BCs, inflow velocity profile — is **copy-pasted across entrypoints**. If you change physics (e.g. a T/S profile or sponge formulation), you must propagate it to every file that defines it: `flow_over_shoals.jl`, `flow_over_shoals_sweep.jl`, `flow_over_shoals_conv.jl`, and `benchmark_shoals.jl`.

**Entrypoint variants** (`src/`):
- `flow_over_shoals.jl` — the primary, actively-developed run. Periodic-in-y (along-shore) by default with a `periodic_y` switch to a fully-bounded domain.
- `flow_over_shoals_conv.jl` — convergent flow: bounded in y, opposing inflow at north/south, open eastern boundary (`PerturbationAdvection`).
- `flow_over_shoals_sweep.jl` — sweep variant; reads geometry/forcing from `SWEEP_*` environment variables set by `sweep_driver.jl`.
- `flow_over_shoals_clean.jl`, `flow_over_shoals_c.jl` — older snapshots/variants; not the current line of work.

**Configuration is by editing source, not flags.** The main script's behavior is set by a block of boolean switches near the top (`LES`, `mass_flux`, `periodic_y`, `is_coriolis`, `shoal_bath`, `checkpointing`, …) and a `run_number` / `run_tag` used to name output files. Bump `run_number` per run. The `tc/hydrostatic` branch and the `TEST N` commit series are an ongoing manual sweep over turbulence closures and diffusivities to find a stable configuration — expect the closure/`Δt`/diffusivity lines to be the thing being changed commit-to-commit.

**Bathymetry** (`src/dshoal_vn_param.jl`): `dshoal_param_bottom(Ly; Hs, sigma, shoal_length, shelf_depth, shelf_break_end, …)` returns a `bottom(x, y)` closure for `GridFittedBottom`. Background is a cross-shore piecewise depth profile (coastal ramp → shelf break → shelf slope → offshore); the shoal is a Gaussian-in-y, cosine-tapered-in-x bump on top. **Transitions are deliberately smoothed with `tanh`/`smooth_max` for C1 continuity** — this is to help the conjugate-gradient Poisson solver converge over the immersed boundary; keep that in mind before "simplifying" the geometry to sharp corners. `dshoal_vn_param_OLD.jl` is a superseded version.

**Pressure solver:** immersed boundaries require `ConjugateGradientPoissonSolver` (the FFT solver only works on the bare flat grid, as in benchmark Level 0). CG iteration count is the dominant cost and is logged per callback.

**Forcing / boundary structure:**
- *Sponge layers* — `Forcing` functions (`sponge_u/v/w/T/S`) multiply a spatial `*_mask` (linear ramps near north/south boundaries; a `tanh` `offshore_mask` near the shelf break) by relaxation toward a target (inflow velocity `v∞`, or a T/S profile) over a timescale `τ`. This is how inflow and stratification are imposed in the periodic case.
- *Inflow velocity* `v∞(x, z, t, p)` — sigmoid in x giving zero at the coast, `v₀` over the shelf, tapering offshore.
- *Bottom drag* — quadratic drag applied as a flux BC on the immersed boundary (`immersed_drag_u/v`).
- *T/S profiles* — `T_north_pwl`, `T_south_pwl`, `S_*`, `T_east_pwl`, etc.: GPU-friendly piecewise-linear profiles with `tanh` smoothing, hard-coded from CTD casts. `src/ctd_smoothed.csv` is the source data; the `_pwl` functions are the digitized result actually used at runtime.

**Outputs:** `NetCDFWriter`s write to the current working directory, filenames keyed by `run_tag` (`top_*.nc` surface slice, `midy_*.nc` cross-shore transect, `time_avg_3d_*.nc` time-averaged fields incl. cross-correlations for EKE/fluxes computed in post-processing). Diagnostics come from `Oceanostics` (`RossbyNumber`, `KineticEnergy`, …). All `*.nc`/`*.jld2`/`*.png`/`*.gif` are gitignored.

**Sweeps:** `sweep_driver.jl` reads a run table from `src/sweep/params_HsLs.toml` (note: it reads `params_HsLs.toml`, not the example `params.toml`), selects one row by `PBS_ARRAY_INDEX`, exports its values as `SWEEP_*` env vars, `cd`s into a per-run `sweep_output/` directory, and `include`s `flow_over_shoals_sweep.jl`.

**`vortex_art/`** is an unrelated standalone aesthetic-visualization side project (high-res vortex wake), not part of the science pipeline.
