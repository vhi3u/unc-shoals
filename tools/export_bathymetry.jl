# export_bathymetry.jl — write the bathymetry of every run in sweep tables to NetCDF
#
#   julia --project=. tools/export_bathymetry.jl                      # every src/sweep/params_*.toml
#   julia --project=. tools/export_bathymetry.jl a.toml b.toml        # just these tables
#
# One file per table: figures/bathymetry/<table>.nc, holding for each run
#   depth_grid(x, y, run)   the bottom the MODEL sees: GridFittedBottom snaps it
#                           to whole cells of the run's own dx and dz
#   depth(x, y, run)        the continuous dshoal_param_bottom it was built from
#   trench(run)             depth of any trench between coast and shoal (m, 0 = none)
# plus the run labels, one per line, in the global attribute `runs`.
#
# Uses src/dshoal_vn_param.jl itself, with the parameters mapped exactly as
# src/sweep/sweep_driver.jl maps them, so the plots show what was run.
# Plot with tools/plot_bathymetry_sweeps.m.

using TOML, NCDatasets

include(joinpath(@__DIR__, "..", "src", "dshoal_vn_param.jl"))

const Lx, Ly = 150e3, 200e3

# The trench check, without throwing (so tables that fail are still plotted).
function trench_depth(bottom, y0, shoal_length)
    xs = 0.0:100.0:shoal_length
    d = [bottom(x, y0) for x in xs]
    left = accumulate(max, d)
    right = reverse(accumulate(max, reverse(d)))
    return maximum(min.(left, right) .- d)
end

function run_params(r)
    Zsh = Float64(get(r, "Zsh", get(r, "shelf_depth", -25.0)))
    Zs = haskey(r, "Zs") ? Float64(r["Zs"]) : Zsh + Float64(r["Hs"])
    return (Zs=Zs, shoal_length=Float64(r["shoal_length"]), sigma=Float64(get(r, "sigma", 8000.0)), Zsh=Zsh,
            shelf_break_end=Float64(get(r, "shelf_break_end", 12000.0)),
            rise_mid=Float64(get(r, "rise_mid", 63500.0)), rise_width=Float64(get(r, "rise_width", 5000.0)),
            shoal_start=Float64(get(r, "shoal_start", 0.0)), shoal_inner_width=Float64(get(r, "shoal_inner_width", 0.0)),
            h_deep=Float64(get(r, "h_deep", -50.0)))
end

function export_table(path, outdir)
    runs = TOML.parsefile(path)["runs"]
    name = replace(basename(path), ".toml" => "")
    # One plotting grid per table: the finest dx in it, capped at 500 m so
    # 250 m tables stay a reasonable size. The grid-fitted depth uses each
    # run's own dz on this grid.
    dx = max(500.0, minimum(Float64(get(r, "dx", 500.0)) for r in runs))
    xc = collect(dx/2:dx:Lx); yc = collect(dx/2:dx:Ly)
    nr = length(runs)
    D = zeros(length(xc), length(yc), nr); G = similar(D); trench = zeros(nr)
    labels = String[]
    for (k, r) in enumerate(runs)
        p = run_params(r)
        # Build without the throwing check: call the internal function directly.
        y0 = Ly / 2; half = Ly / 2
        bottom(x, y) = _param_shoal_bottom(x, y, y0, p.sigma, p.Zs, half, p.shoal_length, p.Zsh,
                                           p.shelf_break_end, p.rise_mid, p.rise_width,
                                           p.shoal_start, p.shoal_inner_width, p.h_deep)
        Lz = Float64(get(r, "Lz", 50.0)); dz = Float64(get(r, "dz", 1.0))
        zc = collect(-Lz + dz/2:dz:-dz/2)
        for j in eachindex(yc), i in eachindex(xc)
            h = bottom(xc[i], yc[j])
            D[i, j, k] = h
            G[i, j, k] = -dz * count(z -> z > h, zc)      # wet cells above the bottom
        end
        trench[k] = trench_depth(bottom, y0, p.shoal_length)
        push!(labels, String(r["run_label"]))
    end
    out = joinpath(outdir, name * ".nc")
    isfile(out) && rm(out)
    NCDataset(out, "c") do ds
        defDim(ds, "x", length(xc)); defDim(ds, "y", length(yc)); defDim(ds, "run", nr)
        defVar(ds, "x", xc, ("x",)); defVar(ds, "y", yc, ("y",))
        defVar(ds, "depth", D, ("x", "y", "run"))
        defVar(ds, "depth_grid", G, ("x", "y", "run"))
        defVar(ds, "trench", trench, ("run",))
        ds.attrib["runs"] = join(labels, "\n")
        ds.attrib["table"] = name
        ds.attrib["source"] = path
    end
    bad = labels[trench .> 1.0]
    println(rpad(name, 32), lpad(nr, 3), " runs  -> ", out, isempty(bad) ? "" : "   TRENCH: " * join(bad, ", "))
end

outdir = joinpath(@__DIR__, "..", "figures", "bathymetry")
mkpath(outdir)
tables = isempty(ARGS) ? filter(f -> startswith(basename(f), "params") && endswith(f, ".toml"),
                                readdir(joinpath(@__DIR__, "..", "src", "sweep"); join=true)) : ARGS
for t in tables
    export_table(t, outdir)
end
