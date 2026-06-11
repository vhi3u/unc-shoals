using Pkg
Pkg.activate(".")
using Oceananigans
using Plots

# Domain parameters
Lx = 100e3
Ly = 200e3

# Define masks identically to flow_over_windy_shoals.jl
north_mask = PiecewiseLinearMask{:y}(center=Ly, width=20e3)
south_mask = PiecewiseLinearMask{:y}(center=0, width=20e3)
east_mask = PiecewiseLinearMask{:x}(center=Lx, width=40e3)
east_mask_uvw(x, y, z) = clamp((x - 50e3) / 50e3, 0.0, 1.0)

# Combined mask functions
sponge_mask(x, y, z) = min(north_mask(x, y, z) + south_mask(x, y, z) + east_mask(x, y, z), 1.0)
sponge_mask_uvw(x, y, z) = min(north_mask(x, y, z) + south_mask(x, y, z) + east_mask_uvw(x, y, z), 1.0)

# Evaluate on a grid
xs = range(0, Lx, length=100)
ys = range(0, Ly, length=200)

mask_T_S = [sponge_mask(x, y, 0) for y in ys, x in xs]
mask_uvw = [sponge_mask_uvw(x, y, 0) for y in ys, x in xs]

# Create plots
p1 = heatmap(xs ./ 1e3, ys ./ 1e3, mask_T_S, 
             title="T, S Sponge Mask", 
             xlabel="x (km)", ylabel="y (km)", 
             color=:viridis, aspect_ratio=1, clims=(0, 1))

p2 = heatmap(xs ./ 1e3, ys ./ 1e3, mask_uvw, 
             title="u, v, w Sponge Mask", 
             xlabel="x (km)", ylabel="y (km)", 
             color=:viridis, aspect_ratio=1, clims=(0, 1))

p = plot(p1, p2, layout=(1, 2), size=(1000, 800))

# Save the plot
savefig(p, "nudging_regions.png")
println("Saved plot to nudging_regions.png")
