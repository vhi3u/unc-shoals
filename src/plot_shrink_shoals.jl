using Plots;
gr();

include("dshoal_vn_param.jl")

# Domain parameters
Ly = 2e3   # 2 km
Lx = 1e3   # 1 km

# Create shoal with parameterized bathymetry
bottom_fn = dshoal_param_bottom(Ly;
    shoal_length=200.0,
    sigma=80.0,
    Hs=5.0,
    scale=Lx/100e3)

# Sample the bathymetry
x_range = range(0, Lx, length=300)
y_range = range(0, Ly, length=600)
Z = [bottom_fn(xi, yi) for xi in x_range, yi in y_range]

# ── Top-down heatmap ──
p1 = heatmap(x_range, y_range, Z',
    xlabel="x [m]", ylabel="y [m]",
    title="Shrunken Shoal — Top View",
    color=:deep, aspect_ratio=:equal,
    clims=(minimum(Z), 0),
    colorbar_title="Depth [m]")
display(p1)
savefig(p1, "shrink_shoal_topview.png")
@info "Saved shrink_shoal_topview.png"

# ── Cross-section at y = center (through shoal) ──
y_center = Ly / 2
z_xsec = [bottom_fn(xi, y_center) for xi in x_range]
p2 = plot(x_range, z_xsec,
    xlabel="x [m]", ylabel="Depth [m]",
    title="Cross-section at y = $(y_center) m (through shoal)",
    lw=2, color=:dodgerblue, label="Seafloor",
    fill=(minimum(Z), 0.2, :dodgerblue),
    ylims=(minimum(Z), 0))
hline!(p2, [0.0], ls=:dash, color=:black, label="Sea surface")
display(p2)
savefig(p2, "shrink_shoal_xsection_center.png")
@info "Saved shrink_shoal_xsection_center.png"

# ── Cross-section away from shoal (background only) ──
y_off = Ly * 0.1
z_xsec_bg = [bottom_fn(xi, y_off) for xi in x_range]
p3 = plot(x_range, z_xsec_bg,
    xlabel="x [m]", ylabel="Depth [m]",
    title="Cross-section at y = $(y_off) m (background slope)",
    lw=2, color=:coral, label="Seafloor",
    fill=(minimum(Z), 0.2, :coral),
    ylims=(minimum(Z), 0))
hline!(p3, [0.0], ls=:dash, color=:black, label="Sea surface")
display(p3)
savefig(p3, "shrink_shoal_xsection_bg.png")
@info "Saved shrink_shoal_xsection_bg.png"

# ── 3D surface ──
p4 = surface(x_range, y_range, Z',
    xlabel="x [m]", ylabel="y [m]", zlabel="z [m]",
    title="Shrunken Shoal — 3D",
    color=:deep, camera=(45, 45),
    zlims=(minimum(Z), 0))
display(p4)
savefig(p4, "shrink_shoal_3d.png")
@info "Saved shrink_shoal_3d.png"

# ── Stacked y-z cross-sections at intervals of x ──
p5 = plot(
    xlabel="y [m]", ylabel="Depth [m]",
    title="y-z cross-sections along various x",
    legend=:outerright,
    ylims=(minimum(Z), 0),
    size=(700, 400)
)
x_intervals = range(0, Lx, length=11)
for x_val in x_intervals
    z_ysec = [bottom_fn(x_val, yi) for yi in y_range]
    plot!(p5, y_range, z_ysec, lw=2, label="x = $(round(x_val, digits=1)) m")
end
hline!(p5, [0.0], ls=:dash, color=:black, label="Sea surface")
display(p5)
savefig(p5, "shrink_shoal_stacked_yz.png")
@info "Saved shrink_shoal_stacked_yz.png"
