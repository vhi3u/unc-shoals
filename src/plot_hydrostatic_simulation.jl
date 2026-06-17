# ═══════════════════════════════════════════════════════════════════════════
# plot_hydrostatic_simulation.jl
# ═══════════════════════════════════════════════════════════════════════════
# Animate the output of flow_over_shoals_hydrostatic.jl.
#
# That script writes a single JLD2 file, `fields_<run_tag>.jld2`, holding the
# state variables as FieldTimeSeries: u, v, w (velocities), T, S (tracers) and
# η (free-surface elevation), saved every 12 hours.
#
# This produces an MP4 that evolves in time, laid out in three rows:
#   • top row    — surface (top-layer) x–y maps of u, v, w, T, S, and η
#   • middle row — one thin horizontal colorbar per column
#   • bottom row — cross-shore (mid-y) x–z transects of u, v, w, T, S
# Each column (field) shares a single color range and colorbar across its
# surface and transect panels.
#
# Plotting uses CairoMakie via Oceananigans' Makie extension: `Field`s (and
# `view`-sliced `Field`s) are handed straight to `heatmap!`, which converts
# coordinates and masks immersed (bottom) cells with NaN automatically — there
# is no need to pull arrays out with `interior()` for plotting. Frames are read
# lazily through an `Observable` index, so only one frame is in memory at a time.
#
# Runs locally with CairoMakie (not on the GPU). Usage:
#   julia --project src/plot_hydrostatic_simulation.jl [path/to/fields_*.jld2]
#
# All work is wrapped in `animate_hydrostatic_output` so that, when this file is
# `include`d at the end of the simulation script, none of its locals collide
# with the simulation's globals (e.g. `free_surface`, `η`, `grid`, …).
# ═══════════════════════════════════════════════════════════════════════════

using Oceananigans
using Printf: @sprintf
using Statistics: quantile
using CairoMakie

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
    @info "Reading $(Nt) frames"

    # ── color limits ──────────────────────────────────────────────────────────
    # Shared per column across all frames and both rows; signed fields are
    # symmetrized about 0 for the diverging colormap. This is the only place that
    # touches raw values (statistics, not plotting): we take the 5th/95th
    # percentiles (robust to outliers) over the surface and transect slices.
    function collect_finite!(vals, A)
        for v in A
            isfinite(v) && push!(vals, Float64(v))
        end
        return vals
    end
    function clims_from(vals, signed)
        isempty(vals) && return signed ? (-1.0, 1.0) : (0.0, 1.0)
        lo, hi = quantile(vals, [0.05, 0.95])
        if signed
            m = max(abs(lo), abs(hi))
            return m == 0 ? (-1.0, 1.0) : (-m, m)
        else
            return hi > lo ? (lo, hi) : (lo, lo + 1)
        end
    end

    # the five 3D state fields: (title, time series, colormap, signed?)
    fields = (("u (m/s)",  u_ts, :balance, true),
              ("v (m/s)",  v_ts, :balance, true),
              ("w (m/s)",  w_ts, :balance, true),
              ("T (°C)",   T_ts, :thermal, false),
              ("S (g/kg)", S_ts, :haline,  false))

    # ── figure scaffold ─────────────────────────────────────────────────────
    # 4 layout rows: title · surface maps · thin colorbars · cross-shore transects
    # over 6 columns (u, v, w, T, S, η). η has no transect, so its bottom cell is
    # left empty. `n` indexes the frame; everything time-varying is `@lift`ed off it.
    n = Observable(1)
    title = @lift @sprintf("flow over shoal — t = %.1f days   (top: surface x–y · bottom: cross-shore x–z)",
                           times[$n] / 86400)

    fig = Figure(size=(2100, 1000))
    Label(fig[1, 1:6], title; fontsize=18, tellwidth=false)

    for (col, (ftitle, fts, cmap, signed)) in enumerate(fields)
        ktop = size(fts, 3)               # topmost z index (Nz for Centers, Nz+1 for w)
        jmid = max(1, size(fts, 2) ÷ 2)   # mid-y index for the cross-shore transect
        xkm = xnodes(fts) ./ 1e3
        ykm = ynodes(fts) ./ 1e3
        zm  = znodes(fts)

        # shared color range from finite values over both slices, all frames
        vals = Float64[]
        for m in 1:Nt
            f = fts[m]
            collect_finite!(vals, interior(f, :, :, ktop))
            collect_finite!(vals, interior(f, :, jmid, :))
        end
        crange = clims_from(vals, signed)

        # lazily slice the current frame into 2D Fields; Makie plots them directly
        fₙ    = @lift fts[$n]
        surfₙ = @lift view($fₙ, :, :, ktop)    # surface x–y (top z-layer)
        tranₙ = @lift view($fₙ, :, jmid, :)    # cross-shore x–z (mid-y)

        # row 1 — surface map
        ax_s = Axis(fig[2, col]; title=ftitle, titlesize=14,
                    ylabel = col == 1 ? "y (km)" : "", xticklabelsvisible=false)
        hm = heatmap!(ax_s, xkm, ykm, surfₙ; colormap=cmap, colorrange=crange, nan_color=:gray)

        # row 2 — thin shared horizontal colorbar
        Colorbar(fig[3, col], hm; vertical=false, flipaxis=false, height=12, ticklabelsize=9)

        # row 3 — cross-shore transect
        ax_t = Axis(fig[4, col]; xlabel="x (km)", ylabel = col == 1 ? "z (m)" : "")
        heatmap!(ax_t, xkm, zm, tranₙ; colormap=cmap, colorrange=crange, nan_color=:gray)
    end

    # η (free surface) — surface map only, column 6
    let xkm = xnodes(η_ts) ./ 1e3, ykm = ynodes(η_ts) ./ 1e3
        vals = Float64[]
        for m in 1:Nt
            collect_finite!(vals, interior(η_ts[m]))
        end
        crange = clims_from(vals, true)

        ηₙ = @lift η_ts[$n]
        ax = Axis(fig[2, 6]; title="η (m)", titlesize=14, xticklabelsvisible=false)
        hm = heatmap!(ax, xkm, ykm, ηₙ; colormap=:balance, colorrange=crange, nan_color=:gray)
        Colorbar(fig[3, 6], hm; vertical=false, flipaxis=false, height=12, ticklabelsize=9)
    end

    # tall surface/transect rows, thin title/colorbar rows (cf. original 0.47/0.05/0.47)
    rowsize!(fig.layout, 2, Relative(0.45))
    rowsize!(fig.layout, 4, Relative(0.45))
    colgap!(fig.layout, 12)
    rowgap!(fig.layout, 6)

    gifname = first(splitext(basename(filename))) * ".mp4"
    @info "Rendering animation"
    record(fig, gifname, 1:Nt; framerate=10) do i
        n[] = i
    end
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
