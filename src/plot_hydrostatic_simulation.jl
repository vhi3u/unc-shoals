# ═══════════════════════════════════════════════════════════════════════════
# plot_hydrostatic_simulation.jl
# ═══════════════════════════════════════════════════════════════════════════
# Animate the output of flow_over_shoals_hydrostatic.jl.
#
# That script writes a single JLD2 file, `fields_<run_tag>.jld2`, holding the
# state variables as FieldTimeSeries: u, v, w (velocities), T, S (tracers) and
# η (free-surface elevation), saved every 12 hours.
#
# This produces a 4-panel GIF that evolves in time:
#   ┌──────────────────────────┬──────────────────────────┐
#   │ surface v (along-shore)  │ surface T                │   (x–y, top layer)
#   ├──────────────────────────┼──────────────────────────┤
#   │ v cross-shore transect   │ free surface η           │   (x–z mid-y) │ (x–y)
#   └──────────────────────────┴──────────────────────────┘
#
# Runs locally with Plots (not on the GPU). Usage:
#   julia --project src/plot_hydrostatic_simulation.jl [path/to/fields_*.jld2]
#
# All work is wrapped in `animate_hydrostatic_output` so that, when this file is
# `include`d at the end of the simulation script, none of its locals collide
# with the simulation's globals (e.g. `free_surface`, `η`, `grid`, …).
# ═══════════════════════════════════════════════════════════════════════════

using Oceananigans
using Printf: @sprintf
using Plots

function animate_hydrostatic_output(filename)
    isfile(filename) || error("Output file not found: $(filename)\n" *
                              "Pass the JLD2 path as an argument, e.g.\n" *
                              "  julia --project src/plot_hydrostatic_simulation.jl fields_hydrostatic_standalone.jld2")
    @info "Animating simulation output from $(filename)"

    # load state-variable time series (OnDisk: read one frame at a time)
    load_fts(name) = FieldTimeSeries(filename, name; backend=OnDisk())
    v_ts = load_fts("v")
    T_ts = load_fts("T")
    η_ts = load_fts("η")

    times = T_ts.times
    Nt = length(times)
    Nx, Ny, Nz = size(T_ts.grid)
    jmid = max(1, Ny ÷ 2)            # mid-y index for the cross-shore transect

    # coordinates, in km (x, y) and m (z); each field at its own staggered nodes
    xc_km = xnodes(T_ts) ./ 1e3
    yc_km = ynodes(T_ts) ./ 1e3
    xv_km = xnodes(v_ts) ./ 1e3
    yv_km = ynodes(v_ts) ./ 1e3
    xη_km = xnodes(η_ts) ./ 1e3
    yη_km = ynodes(η_ts) ./ 1e3
    z_m   = znodes(T_ts)

    # pre-collect the 2D slices we plot (single disk pass → stable color limits)
    @info "Reading $(Nt) frames"
    surf(fts, n)     = Array(interior(fts[n])[:, :, Nz])    # top center layer
    transect(fts, n) = Array(interior(fts[n])[:, jmid, :])  # x–z at mid-y
    eta_slice(n) = (a = interior(η_ts[n]); ndims(a) == 3 ? a[:, :, 1] : Array(a))

    v_surf = [surf(v_ts, n)     for n in 1:Nt]
    T_surf = [surf(T_ts, n)     for n in 1:Nt]
    v_xz   = [transect(v_ts, n) for n in 1:Nt]
    η_xy   = [eta_slice(n)      for n in 1:Nt]

    # color limits: symmetric for signed fields, min/max for temperature
    function finite_extrema(slices)
        lo, hi = Inf, -Inf
        for s in slices, val in s
            isfinite(val) || continue
            lo = min(lo, val); hi = max(hi, val)
        end
        return isfinite(lo) ? (lo, hi) : (0.0, 1.0)
    end
    sym_clims(slices) = (m = maximum(abs, finite_extrema(slices)); (-m, m))

    v_clims = sym_clims(v_surf)
    vxz_clims = sym_clims(v_xz)
    η_clims = sym_clims(η_xy)
    T_clims = finite_extrema(T_surf)

    # build the animation
    anim = @animate for n in 1:Nt
        day = times[n] / 86400

        p_v = heatmap(xv_km, yv_km, v_surf[n]'; title="surface v (m/s)",
                      xlabel="x (km)", ylabel="y (km)", c=:balance, clims=v_clims)
        p_T = heatmap(xc_km, yc_km, T_surf[n]'; title="surface T (°C)",
                      xlabel="x (km)", ylabel="y (km)", c=:thermal, clims=T_clims)
        p_xz = heatmap(xv_km, z_m, v_xz[n]'; title="v cross-shore (mid-y, m/s)",
                       xlabel="x (km)", ylabel="z (m)", c=:balance, clims=vxz_clims)
        p_η = heatmap(xη_km, yη_km, η_xy[n]'; title="free surface η (m)",
                      xlabel="x (km)", ylabel="y (km)", c=:balance, clims=η_clims)

        plot(p_v, p_T, p_xz, p_η; layout=(2, 2), size=(1200, 900),
             plot_title=@sprintf("flow over shoal — t = %.1f days", day))
    end

    gifname = first(splitext(basename(filename))) * ".gif"
    gif(anim, gifname; fps=10)
    @info "Saved animation to $(gifname)"
    return gifname
end

# ── resolve the output file and run ────────────────────────────────────────
# When `include`d at the end of the simulation, `run_tag` is in scope and names
# this run's output. Standalone, pass the JLD2 path as the first CLI argument.
let
    default_tag = @isdefined(run_tag) ? run_tag : "hydrostatic_standalone"
    filename = isempty(ARGS) ? "fields_$(default_tag).jld2" : ARGS[1]
    animate_hydrostatic_output(filename)
end
