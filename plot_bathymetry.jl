using Plots
using Oceananigans
include("src/dshoal_vn_gpu_NxNy.jl")
arch = CPU()
# Parameters from flow_over_shoals.jl
σ = 8.0
Hs = 15.0
Nx = 200
Ny = 600
Lx_km = 100e3 / 1e3
Ly_km = 300e3 / 1e3

x, y, h = dshoal_gpu(Lx_km, Ly_km, σ, Hs, Nx, Ny; arch=CPU)

p = surface(collect(x), collect(y), transpose(h),
    title="Bathymetry (σ=$σ km, Hs=$Hs m)",
    xlabel="x (km)", ylabel="y (km)", zlabel="Depth (m)",
    c=:viridis, camera=(30, 60), colorbar=true,
    zlims=(-60, 0), size=(900, 600))

savefig(p, "bathymetry.png")
display(p)
println("Saved bathymetry.png")
