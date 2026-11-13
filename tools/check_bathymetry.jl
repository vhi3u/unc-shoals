#!/usr/bin/env julia
# check_bathymetry.jl <params_file.toml>
#
# Builds the bottom for every run in a sweep table and measures the geometry
# the model will actually get -- crest length and inshore flank slope, by the
# same definitions tools/eke_geometry_controls.m applies to the output. Needs
# only TOML and src/dshoal_vn_param.jl: no Oceananigans, no GPU, so it runs on
# a login node in a second.
#
# Use it before submitting a geometry sweep. A control that was meant to hold
# the crest fixed and did not is very cheap to catch here and very expensive to
# catch in the EKE table afterwards.
#
#   julia tools/check_bathymetry.jl src/sweep/params_depthaxis.toml

using TOML
using Printf

const ROOT = dirname(@__DIR__)
include(joinpath(ROOT, "src", "dshoal_vn_param.jl"))

# Same window as eke_geometry_controls.m. GridFittedBottom quantises h to
# multiples of dz, so adjacent-cell slopes only ever return multiples of
# dz/dx; a window fixed in metres averages the staircase out and means the
# same thing at every dx.
const FLANK_WINDOW = 2500.0

function measure(bottom, Zs, Lx, dx, dz, y0)
    x = [dx * (i - 0.5) for i in 1:round(Int, Lx / dx)]
    # GridFittedBottom immerses a cell when its centre is below the bottom, so
    # the water column is a whole number of dz -- which is what <uu> > 0 counts.
    h = [-floor(abs(bottom(xi, y0)) / dz) * dz for xi in x]

    on = abs.(h .- Zs) .< dz * 1.5
    best, cur, s, bs, be = 0, 0, 0, 0, 0
    for (i, o) in enumerate(on)
        if o
            cur == 0 && (s = i)
            cur += 1
            cur > best && ((best, bs, be) = (cur, s, i))
        else
            cur = 0
        end
    end
    best == 0 && return (NaN, NaN, NaN, NaN)

    nw = max(2, round(Int, FLANK_WINDOW / dx))
    flank = 0.0
    for i in 1:(bs - nw - 1)
        flank = max(flank, abs(h[i + nw] - h[i]) / (x[i + nw] - x[i]))
    end
    return ((x[be] - x[bs]) / 1e3, flank, x[bs] / 1e3, x[be] / 1e3)
end

params_file = length(ARGS) >= 1 ? ARGS[1] : joinpath(ROOT, "src", "sweep", "params_depthaxis.toml")
runs = TOML.parsefile(params_file)["runs"]

println("\n", basename(params_file), " -- geometry as the model will build it")
println("  flank is the steepest slope over a $(round(Int, FLANK_WINDOW)) m window, shoreward of the crest\n")
@printf("  %-16s %7s %9s %9s %8s %10s   %s\n",
        "run_label", "Zs", "start", "w_in", "crest", "flank", "crest extent")
for r in runs
    Zs = haskey(r, "Zs") ? Float64(r["Zs"]) :
         Float64(get(r, "Zsh", -25.0)) + Float64(r["Hs"])
    dx = Float64(get(r, "dx", 500.0))
    Lx = 200e3
    Ly = 200e3
    bottom = dshoal_param_bottom(Ly;
        Zs = Zs,
        shoal_length = Float64(r["shoal_length"]),
        sigma = Float64(get(r, "sigma", 8000.0)),
        Zsh = Float64(get(r, "Zsh", get(r, "shelf_depth", -25.0))),
        shelf_break_end = Float64(get(r, "shelf_break_end", 12000.0)),
        rise_mid = Float64(get(r, "rise_mid", 63500.0)),
        rise_width = Float64(get(r, "rise_width", 5000.0)),
        shoal_start = Float64(get(r, "shoal_start", 0.0)),
        shoal_inner_width = Float64(get(r, "shoal_inner_width", 0.0)))

    crest, flank, x1, x2 = measure(bottom, Zs, Lx, dx, 1.0, Ly / 2)
    @printf("  %-16s %7.1f %9.0f %9.0f %7.1f km %10.5f   %.1f-%.1f km\n",
            r["run_label"], Zs, Float64(get(r, "shoal_start", 0.0)),
            Float64(get(r, "shoal_inner_width", 0.0)), crest, flank, x1, x2)
end
println()
