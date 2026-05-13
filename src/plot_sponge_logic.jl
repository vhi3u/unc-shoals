# ═══════════════════════════════════════════════════════════════════════════
# plot_sponge_comparison.jl
# ═══════════════════════════════════════════════════════════════════════════
# This script visualizes the difference between -min(...) and -(... + ...)
# sponge formulations to see their spatial impact on the simulation domain.
# ═══════════════════════════════════════════════════════════════════════════

using Plots

# Simulation Domain Parameters
Lx = 100e3
Ly = 300e3
Ls = 50e3  # North/South sponge width
Le = 60e3  # East sponge width
τ_s = τ_n = τ_e = 1.0  # Normalized timescales for visualization

# Mask Functions (consistent with flow_over_shoals_sweep.jl)
function south_mask(y)
    y0, y1 = 0.0, Ls
    return (0 <= y <= y1) ? (1 - y / y1) : 0.0
end

function north_mask(y)
    y0, y1 = Ly - Ls, Ly
    return (y0 <= y <= y1) ? ((y - y0) / (y1 - y0)) : 0.0
end

function east_mask(x)
    x0, x1 = Lx - Le, Lx
    return (x0 <= x <= x1) ? ((x - x0) / (x1 - x0)) : 0.0
end

# Range of coordinates
xs = range(0, Lx, length=200)
ys = range(0, Ly, length=200)

# Evaluate sponge terms at constant u = 1.0
u = 1.0

# Case 1: Transect along Y at the eastern boundary (x = Lx)
x_val = Lx
y_sum = [-(south_mask(y)/τ_s + north_mask(y)/τ_n + east_mask(x_val)/τ_e) for y in ys]
y_min = [-min(south_mask(y)/τ_s, north_mask(y)/τ_n, east_mask(x_val)/τ_e) for y in ys]

# Case 2: Transect along Y in the middle (x = Lx/2)
x_mid = Lx / 2
y_sum_mid = [-(south_mask(y)/τ_s + north_mask(y)/τ_n + east_mask(x_mid)/τ_e) for y in ys]
y_min_mid = [-min(south_mask(y)/τ_s, north_mask(y)/τ_n, east_mask(x_mid)/τ_e) for y in ys]

# Plotting
p1 = plot(ys / 1000, y_sum, label="Sum logic (-(s+n+e))", title="Sponge Comparison at East Boundary (x=Lx)",
          ylabel="Sponge Strength", xlabel="y [km]", lw=2)
plot!(p1, ys / 1000, y_min, label="Min logic (-min(s,n,e))", lw=2, linestyle=:dash)

p2 = plot(ys / 1000, y_sum_mid, label="Sum logic", title="Sponge Comparison in Middle (x=Lx/2)",
          ylabel="Sponge Strength", xlabel="y [km]", lw=2)
plot!(p2, ys / 1000, y_min_mid, label="Min logic", lw=2, linestyle=:dash)

# Cross-shore transect at Ly/2 (where s=0, n=0)
x_sum = [-(south_mask(Ly/2)/τ_s + north_mask(Ly/2)/τ_n + east_mask(x)/τ_e) for x in xs]
x_min = [-min(south_mask(Ly/2)/τ_s, north_mask(Ly/2)/τ_n, east_mask(x)/τ_e) for x in xs]

p3 = plot(xs / 1000, x_sum, label="Sum logic", title="Cross-shore at y=Ly/2",
          ylabel="Sponge Strength", xlabel="x [km]", lw=2)
plot!(p3, xs / 1000, x_min, label="Min logic", lw=2, linestyle=:dash)

l = @layout [p1 p2; p3]
plot(p1, p2, p3, layout=l, size=(1000, 700))

# Save output
savefig("sponge_logic_comparison.png")
@info "Plots saved to sponge_logic_comparison.png"
@info "Observation: Min logic is zero if any one of the masks is zero."
