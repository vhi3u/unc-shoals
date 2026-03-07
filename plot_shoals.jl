using Plots

include("src/dshoal_vn_param.jl")

# Setup base parameters matching your flow_over_shoals.jl
params = (; Lx=100000, Ly=300000, Nx=100, Ny=300)
σ = 8.0         # [km] Gaussian width for shoal cross-section
Hs = 15.0       # [m] shoal height

Lx_km = params.Lx / 1e3
Ly_km = params.Ly / 1e3

# --- 1. Shoal Length Variations ---
lengths = [20.0, 25.0, 35.0, 45.0]
length_plots = []

for len in lengths
    x, y, h = dshoal_param(Lx_km, Ly_km, σ, Hs, params.Nx, params.Ny, shoal_length=len, shelf_length=50.0)

    p_surf = surface(x, y, transpose(h),
        title="Length = $len km",
        xlabel="x (km)", ylabel="y (km)", zlabel="Depth (m)",
        c=:viridis, camera=(30, 60), colorbar=false,
        zlims=(-60, 0))

    mid_y_idx = div(params.Ny, 2)
    p_xz = plot(x, h[:, mid_y_idx],
        title="XZ Transect (y = $(round(y[mid_y_idx], digits=1)) km)",
        xlabel="x (km)", ylabel="Depth (m)",
        linewidth=3, legend=false,
        ylims=(-60, 0))

    push!(length_plots, p_surf)
    push!(length_plots, p_xz)
end

plt_len = plot(length_plots..., layout=(4, 2), size=(1400, 1600), margin=5Plots.mm)
savefig(plt_len, "shoal_length_variations.png")
println("Saved shoal_length_variations.png!")

# --- 2. Shoal Height Variations ---
heights = [10.0, 15.0, 20.0, 25.0]
height_plots = []

for height in heights
    # We change the 'Hs' amplitude here while keeping length constant at 30.0
    x, y, h = dshoal_param(Lx_km, Ly_km, σ, height, params.Nx, params.Ny, shoal_length=30.0, shelf_length=50.0)

    p_surf = surface(x, y, transpose(h),
        title="Height = $height m",
        xlabel="x (km)", ylabel="y (km)", zlabel="Depth (m)",
        c=:viridis, camera=(30, 60), colorbar=false,
        zlims=(-60, 0))

    mid_y_idx = div(params.Ny, 2)
    p_xz = plot(x, h[:, mid_y_idx],
        title="XZ Transect (y = $(round(y[mid_y_idx], digits=1)) km)",
        xlabel="x (km)", ylabel="Depth (m)",
        linewidth=3, legend=false,
        ylims=(-60, 0))

    push!(height_plots, p_surf)
    push!(height_plots, p_xz)
end

plt_ht = plot(height_plots..., layout=(4, 2), size=(1400, 1600), margin=5Plots.mm)
savefig(plt_ht, "shoal_height_variations.png")
println("Saved shoal_height_variations.png!")
