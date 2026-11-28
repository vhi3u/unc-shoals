# les_locate_max.jl -- where and when does an LES run blow up?
#
#   julia --project tools/les_locate_max.jl LES2_DS_base_S0_f2
#   julia --project tools/les_locate_max.jl /full/path/to/run_dir
#
# For every snapshot in top.nc and midshelf.nc (surface and mid-shelf slices):
# time in advective periods, max |u|, |v|, |w|, and where the largest speed
# is (x, y in m and as a fraction of the domain; which boundary it is near).
# Prints only snapshots where the max speed changed by > 20%, plus the first
# and last, so the onset stands out.
using NCDatasets, Printf

arg = length(ARGS) >= 1 ? ARGS[1] : error("give a run name or directory")
dir = isdir(arg) ? arg : joinpath("/glade/derecho/scratch/nguyen/unc-shoals/les", arg)
meta = NCDataset(joinpath(dir, "metadata.nc")) do ds; Dict(k => ds.attrib[k] for k in keys(ds.attrib)); end
Tadv, Lx, Ly = meta["T_adv"], meta["Lx"], meta["Ly"]
V = meta["V_m"]

function where(x, y)
    tags = String[]
    y < 0.1Ly && push!(tags, "inflow (south)")
    y > 0.9Ly && push!(tags, "OUTFLOW (north)")
    x > 0.9Lx && push!(tags, "offshore wall")
    x < 0.05Lx && push!(tags, "x = 0 wall")
    isempty(tags) ? "interior" : join(tags, ", ")
end

for file in ("top.nc", "midshelf.nc")
    path = joinpath(dir, file)
    isfile(path) || continue
    println("\n== $file   (V_inf = $V m/s; T_adv = $(round(Tadv)) s)")
    @printf("%8s %10s %10s %10s   %s\n", "t/T_adv", "max|u|", "max|v|", "max|w|", "largest speed at")
    NCDataset(path) do ds
        t = ds["time"][:]
        xu = ds["x_faa"][:]; yu = ds["y_aca"][:]
        prev = NaN
        for n in eachindex(t)
            u = Float64.(coalesce.(ds["u"][:, :, 1, n], 0.0))
            v = Float64.(coalesce.(ds["v"][:, :, 1, n], 0.0))
            w = Float64.(coalesce.(ds["w"][:, :, 1, n], 0.0))
            mu, iu = findmax(abs, u); mv, iv = findmax(abs, v); mw = maximum(abs, w)
            big = max(mu, mv)
            if n == 1 || n == length(t) || !(abs(big - prev) <= 0.2 * prev)
                i, j = Tuple(mu >= mv ? iu : iv)
                x = xu[min(i, end)]; y = yu[min(j, end)]
                @printf("%8.2f %10.3g %10.3g %10.3g   x = %5.0f m (%.2f Lx), y = %5.0f m (%.2f Ly): %s\n",
                        t[n] / Tadv, mu, mv, mw, x, x / Lx, y, y / Ly, where(x, y))
            end
            prev = big
        end
    end
end
