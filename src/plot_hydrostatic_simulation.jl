# ═══════════════════════════════════════════════════════════════════════════
# plot_hydrostatic_simulation.jl
# ═══════════════════════════════════════════════════════════════════════════
# Animate the output of flow_over_shoals_hydrostatic.jl.
#
# That script writes a single JLD2 file, `fields_<run_tag>.jld2`, holding the
# state variables as FieldTimeSeries: u, v, w (velocities), T, S (tracers) and
# η (free-surface elevation), saved every 12 hours.
#
# This produces a GIF that evolves in time, laid out in three rows:
#   • top row    — surface (top-layer) x–y maps of u, v, w, T, S, and η
#   • middle row — one thin horizontal colorbar per column
#   • bottom row — cross-shore (mid-y) x–z transects of u, v, w, T, S
# Each column (field) shares a single color range and colorbar across its
# surface and transect panels.
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
using Statistics: quantile
using Plots

function animate_hydrostatic_output(filename)
    isfile(filename) || error("Output file not found: $(filename)\n" *
                              "Pass the JLD2 path as an argument, e.g.\n" *
                              "  julia --project src/plot_hydrostatic_simulation.jl fields_hydrostatic_standalone.jld2")
    @info "Animating simulation output from $(filename)"

    # load state-variable time series (OnDisk: read one frame at a time)
    load_fts(name) = FieldTimeSeries(filename, name; backend=OnDisk())
    u_ts = load_fts("u")
    v_ts = load_fts("v")
    w_ts = load_fts("w")
    T_ts = load_fts("T")
    S_ts = load_fts("S")
    η_ts = load_fts("η")

    times = T_ts.times
    Nt = length(times)
    Ny = size(T_ts.grid, 2)
    jmid = max(1, Ny ÷ 2)               # mid-y index for the cross-shore transect

    @info "Reading $(Nt) frames"
    surf(fts, n)     = Array(interior(fts[n])[:, :, end])    # topmost z-layer (x–y)
    transect(fts, n) = Array(interior(fts[n])[:, jmid, :])   # mid-y (x–z)

    # color limits from the 5th/95th percentiles (robust to outliers), shared per
    # column across all frames and both rows. `collect_finite` gathers the values;
    # signed fields are symmetrized about 0 for the diverging colormap.
    function collect_finite(slicesets...)
        vals = Float64[]
        for slices in slicesets, s in slices, val in s
            isfinite(val) && push!(vals, val)
        end
        return vals
    end
    function clims_symmetric(slicesets...)
        vals = collect_finite(slicesets...)
        isempty(vals) && return (-1.0, 1.0)
        lo, hi = quantile(vals, [0.05, 0.95])
        m = max(abs(lo), abs(hi))
        return m == 0 ? (-1.0, 1.0) : (-m, m)
    end
    function clims_percentile(slicesets...)
        vals = collect_finite(slicesets...)
        isempty(vals) && return (0.0, 1.0)
        lo, hi = quantile(vals, [0.05, 0.95])
        return hi > lo ? (lo, hi) : (lo, lo + 1)
    end

    # the five 3D state fields: (title, time series, colormap, signed?)
    fields = (("u (m/s)",  u_ts, :balance, true),
              ("v (m/s)",  v_ts, :balance, true),
              ("w (m/s)",  w_ts, :balance, true),
              ("T (°C)",   T_ts, :thermal, false),
              ("S (g/kg)", S_ts, :haline,  false))

    # ── precompute slices, nodes and (shared) color limits — single disk pass ──
    field_data = map(fields) do (title, fts, cmap, signed)
        s = [surf(fts, n)     for n in 1:Nt]
        t = [transect(fts, n) for n in 1:Nt]
        clims = signed ? clims_symmetric(s, t) : clims_percentile(s, t)
        (; title, cmap, clims, x=xnodes(fts) ./ 1e3, y=ynodes(fts) ./ 1e3, z=znodes(fts), surf=s, tran=t)
    end
    η_s = [(a = interior(η_ts[n]); Array(ndims(a) == 3 ? a[:, :, 1] : a)) for n in 1:Nt]
    η_data = (; title="η (m)", cmap=:balance, clims=clims_symmetric(η_s),
                x=xnodes(η_ts) ./ 1e3, y=ynodes(η_ts) ./ 1e3, surf=η_s)

    # thin horizontal colorbar: a 1×N gradient strip whose x-axis is the value scale
    function colorbar_strip(clims, cmap)
        g = collect(range(clims[1], clims[2], length=200))
        heatmap(g, [0.0], reshape(g, 1, :); c=cmap, clims=clims, colorbar=false,
                legend=false, framestyle=:box, yticks=false, ylims=(-0.5, 0.5),
                xlabel="", ylabel="", title="", titlefontsize=8, tickfontsize=6)
    end

    # 3 rows × 6 columns. Middle (colorbar) row is a thin strip; the bottom-right
    # cell (no η transect) is left empty (`_`).
    layout = @layout [a{0.47h} b c d e f
                      g{0.05h} h i j k l
                      m{0.47h} n o p q _]

    @info "Rendering animation"
    anim = @animate for n in 1:Nt
        ps = Plots.Plot[]

        # row 1 — surface maps (u, v, w, T, S, η)
        for (col, fd) in enumerate(field_data)
            push!(ps, heatmap(fd.x, fd.y, fd.surf[n]'; title=fd.title, c=fd.cmap, clims=fd.clims,
                              colorbar=false, xlabel="", ylabel=(col == 1 ? "y (km)" : ""),
                              titlefontsize=10))
        end
        push!(ps, heatmap(η_data.x, η_data.y, η_data.surf[n]'; title=η_data.title,
                          c=η_data.cmap, clims=η_data.clims, colorbar=false,
                          xlabel="", ylabel="", titlefontsize=10))

        # row 2 — one shared horizontal colorbar per column
        for fd in field_data
            push!(ps, colorbar_strip(fd.clims, fd.cmap))
        end
        push!(ps, colorbar_strip(η_data.clims, η_data.cmap))

        # row 3 — cross-shore transects (u, v, w, T, S); η has none
        for (col, fd) in enumerate(field_data)
            push!(ps, heatmap(fd.x, fd.z, fd.tran[n]'; c=fd.cmap, clims=fd.clims, colorbar=false,
                              xlabel="x (km)", ylabel=(col == 1 ? "z (m)" : ""), title=""))
        end

        plot(ps...; layout, size=(2100, 950),
             plot_title=@sprintf("flow over shoal — t = %.1f days  (top: surface x–y · bottom: cross-shore x–z)", times[n] / 86400))
    end

    gifname = first(splitext(basename(filename))) * ".mp4"
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
