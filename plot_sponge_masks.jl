using Oceananigans
using Oceananigans.Forcings: PiecewiseLinearMask
using CairoMakie

Lx = 200e3
Ly = 200e3
Ls = 40e3 # Updated to your new value
Le = 100e3

north_mask = PiecewiseLinearMask{:y}(center=Ly, width=Ls)
south_mask = PiecewiseLinearMask{:y}(center=0, width=Ls)

@inline function sigmoidal_s2(x, Lx)
    xS = 65e3
    k2 = 20 / Lx
    return 1 / (1 + exp(k2 * (x - xS)))
end

@inline offshore_mask_uvw(x, y, z) = 1.0 - sigmoidal_s2(x - 80e3, Lx)

@inline sponge_mask(x, y, z) = north_mask(x, y, z)
@inline sponge_mask_uvw(x, y, z) = min(north_mask(x, y, z) + offshore_mask_uvw(x, y, z), 1.0)

Nx, Ny = 200, 200
x = range(0, Lx, length=Nx)
y = range(0, Ly, length=Ny)

# PiecewiseLinearMask might need to be called as a function or evaluated
# We wrap the call in case it expects time t as well. In recent Oceananigans, Forcing masks are callable as mask(x, y, z)
mask_T = zeros(Nx, Ny)
mask_uvw = zeros(Nx, Ny)

for i in 1:Nx, j in 1:Ny
    mask_T[i, j] = sponge_mask(x[i], y[j], 0.0)
    mask_uvw[i, j] = sponge_mask_uvw(x[i], y[j], 0.0)
end

fig = Figure(size=(600, 800))

ax1 = Axis(fig[1, 1], title="T/S Nudging Mask (north_mask only)", 
           xlabel="x (km)", ylabel="y (km)")
ax2 = Axis(fig[2, 1], title="u/v/w Nudging Mask", 
           xlabel="x (km)", ylabel="y (km)")

hm1 = heatmap!(ax1, x ./ 1e3, y ./ 1e3, mask_T, colormap=:viridis, colorrange=(0, 1))
Colorbar(fig[1, 2], hm1, label="Relaxation Weight")

hm2 = heatmap!(ax2, x ./ 1e3, y ./ 1e3, mask_uvw, colormap=:viridis, colorrange=(0, 1))
Colorbar(fig[2, 2], hm2, label="Relaxation Weight")

save("sponge_masks_plot.png", fig)
println("Plot successfully saved to sponge_masks_plot.png")
