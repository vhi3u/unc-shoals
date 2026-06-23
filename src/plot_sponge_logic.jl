# ═══════════════════════════════════════════════════════════════════════════
# plot_sponge_comparison.jl
# ═══════════════════════════════════════════════════════════════════════════
# This script visualizes the unified sponge layer spatial profile across 
# different transects in the domain.
# ═══════════════════════════════════════════════════════════════════════════

using Plots

# Simulation Domain Parameters
Lx = 200e3
Ly = 200e3
Ls = 40e3  # North/South sponge width
xS = 65e3

function sigmoidal_s2(x, Lx)
    k2 = 80 / Lx
    return 1 / (1 + exp(k2 * (x - xS)))
end

function offshore_mask_uvw(x)
    return 1.0 - sigmoidal_s2(x, Lx)
end

function south_mask(y)
    y0, y1 = 0.0, Ls
    return (0 <= y <= y1) ? (1 - y / y1) : 0.0
end

function north_mask(y)
    y0, y1 = Ly - Ls, Ly
    return (y0 <= y <= y1) ? ((y - y0) / (y1 - y0)) : 0.0
end

# Range of coordinates
xs = range(0, Lx, length=200)
ys = range(0, Ly, length=200)

# Evaluate unified sponge mask
sponge(x, y) = min(south_mask(y) + north_mask(y) + offshore_mask_uvw(x), 1.0)

# Case 1: Transect along Y at the eastern boundary (x = Lx)
x_val = Lx
sponge_y = [sponge(x_val, y) for y in ys]

p1 = plot(ys / 1000, sponge_y, label="Unified Sponge", title="East Boundary (x=Lx)",
          ylabel="Sponge Mask", xlabel="y [km]", lw=2, color=:blue)

# Case 2: Transect along Y in the middle (x = Lx/2)
x_mid = Lx / 2
sponge_y_mid = [sponge(x_mid, y) for y in ys]

p2 = plot(ys / 1000, sponge_y_mid, label="Unified Sponge", title="Middle Domain (x=Lx/2)",
          ylabel="Sponge Mask", xlabel="y [km]", lw=2, color=:blue)

# Case 3: Cross-shore transect at Ly/2 (where s=0, n=0)
sponge_x = [sponge(x, Ly/2) for x in xs]

p3 = plot(xs / 1000, sponge_x, label="Unified Sponge", title="Cross-shore at center (y=Ly/2)",
          ylabel="Sponge Mask", xlabel="x [km]", lw=2, color=:blue)

l = @layout [p1 p2; p3]
plot(p1, p2, p3, layout=l, size=(1000, 700))

# Save output
savefig("sponge_logic_comparison.png")
@info "Plots saved to sponge_logic_comparison.png"
