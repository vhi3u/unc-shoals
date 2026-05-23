# Animation script for Surface Along-shore Velocity (v)
using NCDatasets
using Plots
using Printf

local x, y, t, v_c, Nt

NCDataset("top_bounded_shoals_simple.nc") do ds
    # ── Coordinates ──
    global x = ds["x_caa"][:] ./ 1e3  # Convert meters to km
    global y = ds["y_aca"][:] ./ 1e3  # Convert meters to km
    global t = ds["time"][:] ./ 3600  # Convert seconds to hours

    # ── Velocity field v_c ──
    global v_c = ds["v_c"][:, :, 1, :]  # Shape: (Nx, Ny, Nt)
end

Nt = length(t)
@info "Found $(Nt) time steps."

# ── Dynamic scale limits ──
using Statistics: quantile
v_flat = filter(!isnan, v_c)
cmin = quantile(v_flat, 0.01)
cmax = quantile(v_flat, 0.99)
@info "Colorbar range: [$(cmin), $(cmax)] m/s"

# ── Generate Animation ──
anim = @animate for n in 1:Nt
    # We transpose the 2D array because Plots.heatmap expects columns as x and rows as y,
    # and in Oceananigans the first dimension is x and second is y.
    heatmap(x, y, transpose(v_c[:, :, n]),
        c = :viridis,
        clim = (cmin, cmax),
        title = @sprintf("Surface Along-shore Velocity v | t = %.1f hours", t[n]),
        xlabel = "Cross-shore distance x (km)",
        ylabel = "Along-shore distance y (km)",
        colorbar_title = "Velocity v (m/s)",
        aspect_ratio = :equal,
        xlims = (extrema(x)),
        ylims = (extrema(y)),
        dpi = 150,
        size = (600, 800),
        titlefont = font(12, "Helvetica"),
        guidefont = font(10, "Helvetica"),
        tickfont = font(8, "Helvetica")
    )
end

output_filename = "v_surface_animation.gif"
@info "Saving animation to $(output_filename)..."
gif(anim, output_filename, fps = 10)
@info "Animation saved successfully!"
