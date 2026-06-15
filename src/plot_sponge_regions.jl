# ═══════════════════════════════════════════════════════════════════════════
# plot_sponge_regions.jl
# ═══════════════════════════════════════════════════════════════════════════
# Visualizes the 2D spatial distribution of the sponge masks used in
# flow_over_shoals_sweep.jl.
# ═══════════════════════════════════════════════════════════════════════════

using Plots

# Parameters from flow_over_shoals_sweep.jl (LES defaults)
Lx = 100e3
Ly = 200e3
Ls = 10e3  # North/South sponge width
Le = 40e3  # East sponge width

# Mask Functions
function south_mask(x, y, z)
    y0, y1 = 0.0, Ls
    return (0 <= y <= y1) ? (1 - y / y1) : 0.0
end

function north_mask(x, y, z)
    y0, y1 = Ly - Ls, Ly
    return (y0 <= y <= y1) ? ((y - y0) / (y1 - y0)) : 0.0
end

function east_mask(x, y, z)
    x0, x1 = Lx - Le, Lx
    return (x0 <= x <= x1) ? ((x - x0) / (x1 - x0)) : 0.0
end

# The sponge logic in flow_over_shoals_sweep.jl
sponge_mask(x, y, z) = min(north_mask(x, y, z) + south_mask(x, y, z) + east_mask(x, y, z), 1.0)

# Create a grid of points
xs = range(0, Lx, length=200)
ys = range(0, Ly, length=400)

# Evaluate masks
mask_vals = [sponge_mask(x, y, 0.0) for y in ys, x in xs]

# Create heatmap
p1 = heatmap(xs ./ 1e3, ys ./ 1e3, mask_vals, 
    title="Sponge Nudging Region Mask", 
    xlabel="x [km]", 
    ylabel="y [km]",
    aspect_ratio=:equal,
    colorbar_title="Mask Value",
    cmap=:viridis,
    clim=(0, 1)
)

# Overlay contours to show boundaries clearly
contour!(p1, xs ./ 1e3, ys ./ 1e3, mask_vals, levels=[0.01, 0.5, 0.99], color=:white, alpha=0.5)

savefig(p1, "sponge_nudging_regions.png")
@info "Saved 2D sponge region plot to sponge_nudging_regions.png"
