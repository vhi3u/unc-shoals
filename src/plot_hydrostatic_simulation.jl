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
# If no path is given it falls back to the `run_tag` in scope (when this file is
# `include`d at the end of the simulation) or to the standalone default.
# ═══════════════════════════════════════════════════════════════════════════

using Oceananigans
using Printf: @sprintf
using Plots

# ── locate the output file ─────────────────────────────────────────────────
default_tag = @isdefined(run_tag) ? run_tag : "hydrostatic_standalone"
filename = isempty(ARGS) ? "fields_$(default_tag).jld2" : ARGS[1]
isfile(filename) || error("Output file not found: $(filename)\n" *
                          "Pass the JLD2 path as an argument, e.g.\n" *
                          "  julia --project src/plot_hydrostatic_simulation.jl fields_hydrostatic_standalone.jld2")
@info "Animating simulation output from $(filename)"

# ── load state-variable time series (OnDisk: read one frame at a time) ─────
load(name) = FieldTimeSeries(filename, name; backend=OnDisk())
v_ts = load("v")
T_ts = load("T")
η_ts = load("η")

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

# ── pre-collect the 2D slices we plot (single disk pass → stable color limits)
@info "Reading $(Nt) frames"
surface_layer(fts, n) = @view interior(fts[n])[:, :, Nz]   # top center layer
transect(fts, n)      = @view interior(fts[n])[:, jmid, :] # x–z at mid-y
@inline function free_surface(n)
    a = interior(η_ts[n])
    return ndims(a) == 3 ? a[:, :, 1] : a
end

v_surf = [Array(surface_layer(v_ts, n)) for n in 1:Nt]
T_surf = [Array(surface_layer(T_ts, n)) for n in 1:Nt]
v_xz   = [Array(transect(v_ts, n))      for n in 1:Nt]
η_xy   = [Array(free_surface(n))         for n in 1:Nt]

# ── color limits: symmetric for signed fields, min/max for temperature ─────
function finite_extrema(slices)
    lo, hi = Inf, -Inf
    for s in slices, val in s
        isfinite(val) || continue
        lo = min(lo, val); hi = max(hi, val)
    end
    return isfinite(lo) ? (lo, hi) : (0.0, 1.0)
end
symmetric(slices) = (m = maximum(abs, finite_extrema(slices)); (-m, m))

v_clims = symmetric(v_surf)
vxz_clims = symmetric(v_xz)
η_clims = symmetric(η_xy)
T_clims = finite_extrema(T_surf)

# ── build the animation ────────────────────────────────────────────────────
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
