using Printf

params = (; Lx=200e3, Ly=200e3, Lz=50, v₀=0.1, Le=100e3)

@inline function sigmoidal_s2(x, Lx)
    xS = 65e3
    k2 = 20 / Lx
    return 1 / (1 + exp(k2 * (x - xS)))
end

@inline function v∞(x, p)
    xC = 3e3
    k1 = 80 / p.Lx

    s1 = 1 / (1 + exp(-k1 * (x - xC)))
    s2 = sigmoidal_s2(x, p.Lx)
    s = (s1 - 1) + s2
    sc = clamp(s, 0.0, 1.0)
    return p.v₀ * sc
end

@inline offshore_mask_uvw(x) = 1.0 - sigmoidal_s2(x - 15e3, params.Lx)

# Mocking PiecewiseLinearMask as it behaves in Oceananigans
# width=Le (100km), center=Lx (200km) -> linearly ramps from 0 at x=100km to 1 at x=200km
@inline east_mask(x) = clamp((x - (params.Lx - params.Le)) / params.Le, 0.0, 1.0)

xs = range(0, params.Lx, length=21)

println("Comparing the Velocity profile, Velocity Sponge, and Tracer Sponge (east_mask)\n")
@printf("%-10s %-15s %-20s %-20s\n", "x (km)", "v∞ (m/s)", "Velocity Sponge", "Tracer Sponge")
println("-"^65)
for x in xs
    v_val = v∞(x, params)
    uvw_mask = offshore_mask_uvw(x)
    tr_mask = east_mask(x)
    @printf("%-10.1f %-15.4f %-20.4f %-20.4f\n", x/1000, v_val, uvw_mask, tr_mask)
end
