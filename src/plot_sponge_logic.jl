# ═══════════════════════════════════════════════════════════════════════════
# plot_sponge_logic.jl
# ═══════════════════════════════════════════════════════════════════════════
# This script visualizes the spatial impact of the current sponge masks
# and evaluates the boundary nudging targets.
# ═══════════════════════════════════════════════════════════════════════════

using Plots

# Simulation Domain Parameters (Updated to match flow_over_shoals.jl)
Lx = 100e3
Ly = 200e3
Ls = 20e3  # North/South sponge width
τ = 1.0    # Normalized timescale

# Mask Functions
function south_mask(y)
    return (0 <= y <= Ls) ? (1.0 - y / Ls) : 0.0
end

function north_mask(y)
    return (Ly - Ls <= y <= Ly) ? (1.0 - (Ly - y) / Ls) : 0.0
end

function sigmoidal_s2(x, Lx)
    xS = 65e3
    k2 = 40 / Lx
    return 1 / (1 + exp(k2 * (x - xS)))
end

function east_mask(x)
    # Using the updated logic: x + 20e3
    return 1.0 - sigmoidal_s2(x + 20e3, Lx)
end

# Range of coordinates
xs = range(0, Lx, length=200)
ys = range(0, Ly, length=200)

# Evaluate sponge mask
# Current logic: min(n + s + e, 1.0)
mask_east_boundary = [min(south_mask(y) + north_mask(y) + east_mask(Lx), 1.0) for y in ys]
mask_mid_y = [min(south_mask(y) + north_mask(y) + east_mask(Lx / 2), 1.0) for y in ys]
mask_mid_x = [min(south_mask(Ly / 2) + north_mask(Ly / 2) + east_mask(x), 1.0) for x in xs]

p1 = plot(ys / 1000, mask_east_boundary, label="Sponge Mask (x=Lx)", title="Along Y at East Boundary",
    ylabel="Mask Value", xlabel="y [km]", lw=2, ylims=(-0.1, 1.1))

p2 = plot(ys / 1000, mask_mid_y, label="Sponge Mask (x=Lx/2)", title="Along Y in Middle",
    ylabel="Mask Value", xlabel="y [km]", lw=2, ylims=(-0.1, 1.1))

p3 = plot(xs / 1000, mask_mid_x, label="Sponge Mask (y=Ly/2)", title="Cross-shore at y=Ly/2",
    ylabel="Mask Value", xlabel="x [km]", lw=2, ylims=(-0.1, 1.1))

# T_target fixed visualization
T_south_val = 24.5
function T_target_user(x, y)
    n = north_mask(y)
    s = south_mask(y)
    tot = n + s
    # Fixed code from flow_over_shoals.jl:
    return tot > 0 ? (n * T_south_val + s * T_south_val) / tot : T_south_val
end

T_target_mid_x = [T_target_user(x, Ly / 2) for x in xs]

p4 = plot(xs / 1000, T_target_mid_x, label="T_target", title="T_target at y=Ly/2\n(Fixed: Target remains T_south)",
    ylabel="Temperature [°C]", xlabel="x [km]", lw=2, color=:green)

l = @layout [p1 p2; p3 p4]
plot(p1, p2, p3, p4, layout=l, size=(1000, 700))

# Save output
savefig("sponge_logic.png")
@info "Plots saved to sponge_logic.png"
