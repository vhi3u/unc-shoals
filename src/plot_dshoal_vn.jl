using Plots;
gr();

include("dshoal_vn_param.jl")

# Domain parameters
Ly = 100e3   # 300 km
Lx = 100e3   # 100 km

# Create shoal with parameterized bathymetry
bottom_fn = dshoal_param_bottom(Ly;
    shelf_length=50e3,
    shelf_depth=-30.0,
    shoal_length=40e3,
    deep_ocean_depth=-50.0,
    sigma=8e3,
    Hs=15.0,
    Ly_shoal=100e3)

# Sample the bathymetry
x_range = range(0, Lx, length=300)
y_range = range(0, Ly, length=600)
Z = [bottom_fn(xi, yi) for xi in x_range, yi in y_range]

# ── Top-down heatmap ──
p1 = heatmap(x_range ./ 1e3, y_range ./ 1e3, Z',
    xlabel="x [km]", ylabel="y [km]",
    title="Parameterized Shoal — Top View",
    color=:deep, aspect_ratio=:equal,
    clims=(minimum(Z), maximum(Z)),
    colorbar_title="Depth [m]")
display(p1)
savefig(p1, "dshoal_param_topview.png")
@info "Saved dshoal_param_topview.png"

# ── Cross-section at y = center (through shoal) ──
y_center = Ly / 2
z_xsec = [bottom_fn(xi, y_center) for xi in x_range]
p2 = plot(x_range ./ 1e3, z_xsec,
    xlabel="x [km]", ylabel="Depth [m]",
    title="Cross-section at y = $(y_center/1e3) km (through shoal)",
    lw=2, color=:dodgerblue, label="Seafloor",
    fill=(minimum(Z) - 2, 0.2, :dodgerblue),
    ylims=(minimum(Z) - 2, 2))
hline!(p2, [0.0], ls=:dash, color=:black, label="Sea surface")
display(p2)
savefig(p2, "dshoal_param_xsection_center.png")
@info "Saved dshoal_param_xsection_center.png"

# ── Cross-section away from shoal (background only) ──
y_off = Ly * 0.1
z_xsec_bg = [bottom_fn(xi, y_off) for xi in x_range]
p3 = plot(x_range ./ 1e3, z_xsec_bg,
    xlabel="x [km]", ylabel="Depth [m]",
    title="Cross-section at y = $(y_off/1e3) km (background slope)",
    lw=2, color=:coral, label="Seafloor",
    fill=(minimum(Z) - 2, 0.2, :coral),
    ylims=(minimum(Z) - 2, 2))
hline!(p3, [0.0], ls=:dash, color=:black, label="Sea surface")
display(p3)
savefig(p3, "dshoal_param_xsection_bg.png")
@info "Saved dshoal_param_xsection_bg.png"

# ── 3D surface ──
p4 = surface(x_range ./ 1e3, y_range ./ 1e3, Z',
    xlabel="x [km]", ylabel="y [km]", zlabel="z [m]",
    title="Parameterized Shoal — 3D",
    color=:deep, camera=(45, 45))
display(p4)
savefig(p4, "dshoal_param_3d.png")
@info "Saved dshoal_param_3d.png"

# ── Stacked y-z cross-sections at intervals of x ──
p5 = plot(
    xlabel="y [km]", ylabel="Depth [m]",
    title="y-z cross-sections along various x",
    legend=:outerright,
    ylims=(minimum(Z) - 2, 2),
    size=(700, 400)
)
x_intervals = range(0, Lx, length=11)
for x_val in x_intervals
    z_ysec = [bottom_fn(x_val, yi) for yi in y_range]
    plot!(p5, y_range ./ 1e3, z_ysec, lw=2, label="x = $(round(x_val/1e3, digits=1)) km")
end
hline!(p5, [0.0], ls=:dash, color=:black, label="Sea surface")
display(p5)
savefig(p5, "dshoal_param_stacked_yz.png")
@info "Saved dshoal_param_stacked_yz.png"
