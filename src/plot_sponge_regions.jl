# ═══════════════════════════════════════════════════════════════════════════
# plot_sponge_regions.jl
# ═══════════════════════════════════════════════════════════════════════════
# Visualizes the 2D spatial distribution of the sponge masks used in
# flow_over_shoals.jl.
# ═══════════════════════════════════════════════════════════════════════════

using Plots

# Parameters from flow_over_shoals.jl (LES defaults)
Lx = 100e3
Ly = 200e3
Ls = 20e3  # North/South sponge width

# Mask Functions
function south_mask(x, y, z)
    return (0 <= y <= Ls) ? (1.0 - y / Ls) : 0.0
end

function north_mask(x, y, z)
    return (Ly - Ls <= y <= Ly) ? (1.0 - (Ly - y) / Ls) : 0.0
end

function sigmoidal_s2(x, Lx)
    xS = 65e3
    k2 = 40 / Lx
    return 1 / (1 + exp(k2 * (x - xS)))
end

function east_mask(x, y, z)
    # Using the updated logic: x + 20e3
    return 1.0 - sigmoidal_s2(x, Lx)
end

# The sponge logic in flow_over_shoals.jl
sponge_mask(x, y, z) = min(north_mask(x, y, z) + south_mask(x, y, z), 1.0)
sponge_mask_uvw(x, y, z) = min(north_mask(x, y, z) + south_mask(x, y, z) + east_mask(x, y, z), 1.0)

# Create a grid of points
xs = range(0, Lx, length=200)
ys = range(0, Ly, length=400)

# Evaluate masks
mask_vals_ts = [sponge_mask(x, y, 0.0) for y in ys, x in xs]
mask_vals_uvw = [sponge_mask_uvw(x, y, 0.0) for y in ys, x in xs]

# Create heatmaps
p_uvw = heatmap(xs ./ 1e3, ys ./ 1e3, mask_vals_uvw,
    title="Sponge Mask (u, v, w)",
    xlabel="x [km]",
    ylabel="y [km]",
    aspect_ratio=:equal,
    colorbar_title="Mask Value",
    cmap=:viridis,
    clim=(0, 1)
)
contour!(p_uvw, xs ./ 1e3, ys ./ 1e3, mask_vals_uvw, levels=[0.01, 0.5, 0.99], color=:white, alpha=0.5)

p_ts = heatmap(xs ./ 1e3, ys ./ 1e3, mask_vals_ts,
    title="Sponge Mask (T, S)",
    xlabel="x [km]",
    ylabel="y [km]",
    aspect_ratio=:equal,
    colorbar_title="Mask Value",
    cmap=:viridis,
    clim=(0, 1)
)
contour!(p_ts, xs ./ 1e3, ys ./ 1e3, mask_vals_ts, levels=[0.01, 0.5, 0.99], color=:white, alpha=0.5)

l = @layout [a b]
p = plot(p_uvw, p_ts, layout=l, size=(1000, 600))

savefig(p, "sponge_nudging_regions.png")
@info "Saved 2D sponge region plot to sponge_nudging_regions.png"
