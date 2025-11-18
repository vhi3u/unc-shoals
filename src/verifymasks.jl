using Plots

# --- parameters ---
Lx = 100e3
Ly = 200e3
Lz = 50.0
Nx, Ny, Nz = 30, 30, 10
p = (; Lx, Ly, Ls = 20e3)

# --- masks ---
@inline function south_mask(x, y, z, p)
    y0, y1 = 0, p.Ls
    return (y0 <= y <= y1) ? 1 - y/y1 : 0.0
end

@inline function north_mask(x, y, z, p)
    y0, y1 = p.Ly - p.Ls, p.Ly
    return (y0 <= y <= y1) ? (y - y0)/(y1 - y0) : 0.0
end

@inline function east_mask(x, y, z, p)
    x0, x1 = p.Lx - p.Ls, p.Lx
    return (x0 <= x <= x1) ? (x - x0)/(x1 - x0) : 0.0
end

# --- east logistic nudge ---
@inline function east_nudge(x, y, z, p)
    xC, xS, Lw = 3e3, 60e3, p.Lx
    k1, k2 = 80/Lw, 40/Lw
    s1 = 1 / (1 + exp(-k1 * (x - xC)))
    s2 = 1 / (1 + exp(k2 * (x - xS)))
    s = (s1 - 1) + s2
    return clamp(s, 0.0, 1.0)
end

# --- grid ---
xs = range(0, Lx, length=Nx)
ys = range(0, Ly, length=Ny)

# --- evaluate masks ---
south = [south_mask(x, y, 0, p) for y in ys, x in xs]
north = [north_mask(x, y, 0, p) for y in ys, x in xs]
east  = [east_mask(x, y, 0, p)  for y in ys, x in xs]

# --- evaluate east_nudge (1D along x) ---
enudge_vals = [east_nudge(x, 0, 0, p) for x in xs]

# --- plot panels ---
plt1 = heatmap(xs ./ 1e3, ys ./ 1e3, south .+ north .+ east;
    title = "Combined Masks (south + north + east)",
    xlabel = "x (km)", ylabel = "y (km)",
    colorbar_title = "Mask value", c = :viridis)

plt2 = plot(xs ./ 1e3, enudge_vals;
    title = "east_nudge(x)",
    xlabel = "x (km)", ylabel = "nudge value",
    lw = 2, color = :darkorange)

plot(plt1, plt2; layout = (1, 2), size = (1000, 500))
