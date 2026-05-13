using Plots

# Current Parameters from flow_over_shoals.jl
Lx = 100e3
Ly = 300e3
Ls = 10e3  # North mask width
Le = 40e3  # East/Offshore mask width

# North Mask (Y-direction)
function north_mask(y)
    y0, y1 = Ly - Ls, Ly
    return (y0 <= y <= y1) ? ((y - y0) / (y1 - y0)) : 0.0
end

# East/Offshore Mask (X-direction)
# Mirroring your current linear ramp from 60km to 100km
function offshore_mask(x)
    x0, x1 = 60e3, 100e3
    if x <= x0; return 0.0
    elseif x >= x1; return 1.0
    else; return (x - x0) / (x1 - x1) # Wait, type: / (x1 - x0)
    end
end
# Corrected version for visualization
function correct_offshore_mask(x)
    x0, x1 = 60e3, 100e3
    if x <= x0; return 0.0
    elseif x >= x1; return 1.0
    else; return (x - x0) / (x1 - x0)
    end
end

# Range of coordinates
xs = range(0, Lx, length=200)
ys = range(0, Ly, length=200)

# 1. 2D Heatmap of the Masks
north_2d = [north_mask(y) for x in xs, y in ys]'
offshore_2d = [correct_offshore_mask(x) for x in xs, y in ys]'

p1 = heatmap(xs/1000, ys/1000, north_2d, title="North Mask (y-sponge)", xlabel="x (km)", ylabel="y (km)", color=:viridis)
p2 = heatmap(xs/1000, ys/1000, offshore_2d, title="Offshore/East Mask (x-sponge)", xlabel="x (km)", ylabel="y (km)", color=:viridis)

# Total T/S sponge logic (Sum)
total_mask = [north_mask(y) + correct_offshore_mask(x) for x in xs, y in ys]'
p3 = heatmap(xs/1000, ys/1000, total_mask, title="Combined Mask Strength (Sum)", xlabel="x (km)", ylabel="y (km)", color=:magma)

# 2. Vertical Profiles Comparison
# (Simplified manual reconstruction of your piecewise linear logic)
δ_smooth = 2.5
smooth_step(z, z0) = 0.5 * (1.0 - tanh((z - z0) / δ_smooth))

function T_south_pwl(z)
    z1, z2, z3 = -5.0, -15.0, -30.0
    v1, v2, v3 = 24.5378, 24.3073, 23.4116 # From your code
    m12 = (v2 - v1) / (z2 - z1); m23 = (v3 - v2) / (z3 - z2)
    val = (z >= z1) ? v1 : (z >= z2) ? (v1 + m12*(z-z1)) : (z >= z3) ? (v2 + m23*(z-z2)) : v3
    return val 
end

function T_east_pwl(z)
    z1, z2, z3 = -5.0, -25.0, -45.0
    v1, v2, v3 = 25.0, 23.0, 21.0 # Your RECENT reduced values
    m12 = (v2 - v1) / (z2 - z1); m23 = (v3 - v2) / (z3 - z2)
    val = (z >= z1) ? v1 : (z >= z2) ? (v1 + m12*(z-z1)) : (z >= z3) ? (v2 + m23*(z-z2)) : v3
    return val
end

zs = range(-50, 0, length=200)
T_s = [T_south_pwl(z) for z in zs]
T_e = [T_east_pwl(z) for z in zs]

p4 = plot(T_s, zs, label="Inflow/Inshore (South)", title="Target Temperature Profiles", xlabel="Temp (°C)", ylabel="z (m)", lw=2)
plot!(p4, T_e, zs, label="Offshore target (East)", lw=2, linestyle=:dash)

l = @layout [[p1 p2]; p3; p4]
plot(p1, p2, p3, p4, layout=l, size=(1200, 1000))
savefig("diagnostic_sponge_masks.png")
println("Plots saved to diagnostic_sponge_masks.png")
