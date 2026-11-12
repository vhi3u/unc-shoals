# ═══════════════════════════════════════════════════════════════════════════
# plot_bathy_compare.jl — the Fig. 1 bathymetry panel, current vs proposed.
#
#   julia --project=. tools/plot_bathy_compare.jl
#
# Recreates the layout of fig3 in MATLAB/shoal_stuff/plot_shoals.m (3D surface
# on the left, X-Z transect on the right, blue shoal / red background) and
# stacks the two geometries so they can be read off against each other:
#
#   row 1  CURRENT        Zc fixed at -5 m
#   row 2  COAST TRACKS   Zc = Zs
#
# Why this is Julia and not MATLAB. plot_shoals.m carries its OWN copy of the
# bathymetry, written against the old Hs/shelf_length parameterization on a
# 100x300 km domain. The model has since moved to Zs/Zc/coastal_drop on
# 150x200 km with an offshore ramp at 63.5 km, so that copy no longer draws
# what the model runs -- the figure and the simulation had silently diverged.
# This script calls dshoal_param_bottom_coastal directly, so it cannot.
#
# Reference case (Zs = -5) must come out IDENTICAL in both rows. That is the
# built-in check that the proposed change leaves the published figure alone.
# ═══════════════════════════════════════════════════════════════════════════

using Plots
gr()

include(joinpath(@__DIR__, "..", "src", "dshoal_vn_param_coastal.jl"))

# ── Domain and shoal, matching flow_over_shoals_sweep.jl ──────────────────
const Lx, Ly, Lz = 150e3, 200e3, 50.0
const ZSH, SBE, LS, SIG = -25.0, 12.0e3, 40.0e3, 8.0e3
const ZC_FIXED = -5.0

const Zs_list = [-5.0, -10.0, -15.0]       # X-Z panel
const Zs_3d   = -15.0                      # 3D panel: the case that differs most
const COLORS  = [:dodgerblue, :seagreen, :mediumpurple]

bathy(Zs, Zc) = dshoal_param_bottom_coastal(Ly; Zs=Zs, Zc=Zc, shoal_length=LS,
                                            sigma=SIG, Zsh=ZSH, shelf_break_end=SBE)

const xs = range(0, Lx, length=301)
const ys = range(0, Ly, length=401)
const y_shoal = Ly / 2                     # shoal centreline
const y_bg    = Ly / 2 - 60e3              # 7.5 sigma away: background only

"""Flat-crest extent, so the label reports the L_S the model actually gets."""
function crest_extent(f, Zs)
    on = [i for (i, x) in enumerate(xs) if abs(f(x, y_shoal) - Zs) < 0.05]
    isempty(on) && return (NaN, NaN)
    (xs[first(on)], xs[last(on)])
end

function panel_3d(Zc_of, title_)
    f = bathy(Zs_3d, Zc_of(Zs_3d))
    Z = [f(x, y) for y in ys, x in xs]
    p = surface(xs ./ 1e3, ys ./ 1e3, Z;
        c=:deep, clims=(-Lz, 0), camera=(45, 30), colorbar=false,
        xlabel="x (km)", ylabel="y (km)", zlabel="z (m)",
        zlims=(-Lz, 0), title=title_, titlefontsize=11, guidefontsize=9,
        tickfontsize=7)
    plot!(p, xs ./ 1e3, fill(y_shoal / 1e3, length(xs)),
        [f(x, y_shoal) for x in xs]; lw=3, c=:dodgerblue, label="")
    plot!(p, xs ./ 1e3, fill(y_bg / 1e3, length(xs)),
        [f(x, y_bg) for x in xs]; lw=3, c=:firebrick, label="")
    p
end

function panel_xz(Zc_of, title_)
    p = plot(; xlabel="x (km)", ylabel="z (m)", xlims=(0, 100), ylims=(-Lz, 1),
        title=title_, titlefontsize=11, guidefontsize=9, tickfontsize=8,
        legend=:bottomleft, legendfontsize=7, grid=true, gridalpha=0.25)
    fbg = bathy(Zs_list[1], ZC_FIXED)
    plot!(p, xs ./ 1e3, [fbg(x, y_bg) for x in xs];
        lw=2.5, c=:firebrick, label="Background (y = $(round(Int, y_bg/1e3)) km)")
    for (k, Zs) in enumerate(Zs_list)
        f = bathy(Zs, Zc_of(Zs))
        z = [f(x, y_shoal) for x in xs]
        # No fillrange: three translucent fills stack into mud and, because
        # Plots registers the fill as its own series, they also shift the
        # legend swatches off the lines they label.
        plot!(p, xs ./ 1e3, z; lw=2.5, c=COLORS[k],
            label="Z_S = $(round(Int, -Zs)) m")
        xi, xo = crest_extent(f, Zs)
        if isfinite(xi)
            plot!(p, [xi, xo] ./ 1e3, [Zs + 1.3, Zs + 1.3];
                lw=1.4, c=COLORS[k], label="")
            annotate!(p, (xi + xo) / 2e3, Zs + 3.2,
                text("L_S = $(round((xo - xi)/1e3, digits=1)) km", 7, COLORS[k]))
        end
    end
    vline!(p, [63.5]; ls=:dot, c=:gray, lw=1, label="")
    p
end

rows = [("CURRENT  —  Zc fixed at -5 m", Zs -> ZC_FIXED),
        ("COAST TRACKS  —  Zc = Zs", Zs -> Zs)]

plots = Any[]
for (name, zc) in rows
    push!(plots, panel_3d(zc, "$name\n3D (Z_S = $(round(Int, -Zs_3d)) m)"))
    push!(plots, panel_xz(zc, "$name\nX-Z transects"))
end

fig = plot(plots...; layout=(2, 2), size=(1500, 1000),
    plot_title="Shoal bathymetry — shoal_length = $(round(Int, LS/1e3)) km, sigma = $(round(Int, SIG/1e3)) km",
    plot_titlefontsize=12, left_margin=6Plots.mm, bottom_margin=6Plots.mm)

outdir = joinpath(@__DIR__, "..", "figures")
mkpath(outdir)
out = joinpath(outdir, "bathy_current_vs_coast_tracking.png")
savefig(fig, out)
@info "Saved $out"

# ── Numbers behind the labels ────────────────────────────────────────────
println("\n  Zs    variant        crest_in   crest_out   L_S actual   max|dh/dx| 0-20km")
for Zs in Zs_list, (name, zc) in (("current", Zs -> ZC_FIXED), ("tracks", Zs -> Zs))
    f = bathy(Zs, zc(Zs))
    xi, xo = crest_extent(f, Zs)
    xin = collect(0:125.0:20e3)
    s = maximum(abs.(diff([f(x, y_shoal) for x in xin])) ./ 125.0)
    println("  ", lpad(Zs, 5), "   ", rpad(name, 12),
        lpad(round(xi / 1e3, digits=1), 8), "   ", lpad(round(xo / 1e3, digits=1), 9),
        "   ", lpad(round((xo - xi) / 1e3, digits=1), 10), "   ", lpad(round(s, digits=5), 17))
end
