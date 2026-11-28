# ═══════════════════════════════════════════════════════════════════════════
# make_les_bathymetry.jl -- real Diamond Shoals bathymetry in the shoal's frame,
# at real scale, for src/les_diamond_real.jl
#
#   julia --project tools/make_les_bathymetry.jl
#
# Writes data/diamond_shoals_frame.nc (committed, so Casper needs no GEBCO).
#
# FRAME (the same as the idealized runs and diamond_shoals_model_domain.m):
#   X  cross-shore: distance along the shoal axis (132.5 deg, measured by
#      MATLAB/shoal_stuff/diamond_shoals_geometry.m on GEBCO 2026) from the
#      cape point, shifted so the cape is at X = 25 km
#   Y  along-shore: across the axis, the NE side (upstream in the usual
#      southward flow) at SMALL Y, the cape at Y = 60 km
# Domain 80 x 160 km, sampled every 200 m (GEBCO is 15 arc-seconds, ~0.4 km).
#
# DATA: GEBCO 2026 (35.0-36.0 N) where it exists, else the GEBCO 2024 subset,
# blended linearly over the 0.05 deg just north of 2026's southern edge.
#
# WATER AND LAND. Only water connected to the open ocean is kept:
#   1. "ocean": cells deeper than 5 m connected to the offshore edge (X = 80 km)
#      through cells deeper than 5 m. This cuts off Pamlico Sound, whose
#      inlets are shallower than that.
#   2. plus any shallower water (0 to 5 m) within 2 km of the ocean, reached
#      through water: the shoal's crest (1-3 m near the cape) and the surf
#      zone. Only a 2 km pocket can leak through an inlet.
#   Everything else is land (depth 0).
# Depths are clamped to [h_min, h_max] = [3, 50] m: the model needs a few
# grid cells over the crest, and the real slope drops to hundreds of metres
# where the model floor is (scaled) 50 m.
# ═══════════════════════════════════════════════════════════════════════════

using NCDatasets

const ROOT = dirname(@__DIR__)
const F2026 = joinpath(ROOT, "GEBCO_27_Nov_2026_00de4bdb7f16", "gebco_2026_n36.0095_s35.007_w-76.0_e-75.0.nc")
const F2024 = "/Users/victornguyen/Documents/MATLAB/nc-bathy/nc-bathy/GEBCO_26_Jun_2025_4b48b3ff1762/gebco_2024_n36.5345_s33.9037_w-76.6066_e-73.5451.nc"
const OUT = joinpath(ROOT, "data", "diamond_shoals_frame.nc")

const cape = (-75.525078, 35.221512)
const az = 132.5
const X_cape, Y_cape = 25.0, 60.0          # km
const Lx, Ly, dxy = 80.0, 160.0, 0.2       # km
const h_min, h_max = 3.0, 50.0             # m
const shallow_reach = 2.0                  # km

function read_gebco(f)
    NCDataset(f) do ds
        lon = Float64.(ds["lon"][:]); lat = Float64.(ds["lat"][:])
        z = Float64.(ds["elevation"][:, :])           # (lon, lat)
        size(z) == (length(lon), length(lat)) || (z = permutedims(z))
        (; lon, lat, z)
    end
end

# bilinear interpolation on a regular lon/lat grid; NaN outside
function bilinear(g, lo, la)
    (lo < g.lon[1] || lo > g.lon[end] || la < g.lat[1] || la > g.lat[end]) && return NaN
    dlo = g.lon[2] - g.lon[1]; dla = g.lat[2] - g.lat[1]
    i = min(floor(Int, (lo - g.lon[1]) / dlo) + 1, length(g.lon) - 1)
    j = min(floor(Int, (la - g.lat[1]) / dla) + 1, length(g.lat) - 1)
    a = (lo - g.lon[i]) / dlo; b = (la - g.lat[j]) / dla
    return (1 - a) * (1 - b) * g.z[i, j] + a * (1 - b) * g.z[i+1, j] + (1 - a) * b * g.z[i, j+1] + a * b * g.z[i+1, j+1]
end

g26 = read_gebco(F2026)
g24 = read_gebco(F2024)

X = collect(dxy/2:dxy:Lx); Y = collect(dxy/2:dxy:Ly)
nx, ny = length(X), length(Y)
kx = 6371 * cosd(cape[2]) * π / 180; ky = 6371 * π / 180
ue, un = sind(az), cosd(az); ve, vn = -un, ue               # axis; NE-pointing normal
elev = fill(NaN, nx, ny); src = zeros(Int8, nx, ny)
for j in 1:ny, i in 1:nx
    s = X[i] - X_cape; n = Y_cape - Y[j]                    # NE side at small Y
    E = s * ue + n * ve; N = s * un + n * vn
    lo, la = cape[1] + E / kx, cape[2] + N / ky
    z26 = bilinear(g26, lo, la); z24 = bilinear(g24, lo, la)
    # blend across GEBCO 2026's southern edge (35.007 N) over 0.05 deg, so the
    # two products (which differ by a few metres) leave no seam
    w = isnan(z26) ? 0.0 : clamp((la - (g26.lat[1] + 0.003)) / 0.05, 0.0, 1.0)
    elev[i, j] = w > 0 ? w * z26 + (1 - w) * z24 : z24
    src[i, j] = w >= 0.5 ? 26 : 24
end
any(isnan, elev) && error("the frame reaches outside both GEBCO files")

# 1. ocean: deeper than h_min-ish (5 m), connected to the offshore edge
deep = elev .< -5.0
ocean = falses(nx, ny)
queue = [(nx, j) for j in 1:ny if deep[nx, j]]
foreach(c -> ocean[c...] = true, queue)
while !isempty(queue)
    i, j = popfirst!(queue)
    for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
        a, b = i + di, j + dj
        (1 <= a <= nx && 1 <= b <= ny) || continue
        if deep[a, b] && !ocean[a, b]; ocean[a, b] = true; push!(queue, (a, b)); end
    end
end
# 2. shallow water within shallow_reach of the ocean, through water
water = copy(ocean)
steps = round(Int, shallow_reach / dxy)
frontier = [(i, j) for j in 1:ny, i in 1:nx if ocean[i, j]]
for _ in 1:steps
    next = Tuple{Int,Int}[]
    for (i, j) in frontier, (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
        a, b = i + di, j + dj
        (1 <= a <= nx && 1 <= b <= ny) || continue
        if !water[a, b] && elev[a, b] < 0; water[a, b] = true; push!(next, (a, b)); end
    end
    global frontier = next
end

depth = [water[i, j] ? clamp(elev[i, j], -h_max, -h_min) : 0.0 for i in 1:nx, j in 1:ny]

mkpath(dirname(OUT))
isfile(OUT) && rm(OUT)
NCDataset(OUT, "c") do ds
    defDim(ds, "X", nx); defDim(ds, "Y", ny)
    defVar(ds, "X", X, ("X",), attrib = ["units" => "km", "long_name" => "along the shoal axis; cape at $(X_cape) km"])
    defVar(ds, "Y", Y, ("Y",), attrib = ["units" => "km", "long_name" => "across the axis, NE side at small Y; cape at $(Y_cape) km"])
    defVar(ds, "depth", Float32.(depth), ("X", "Y"), attrib = ["units" => "m", "long_name" => "bottom height (negative); 0 = land",
        "clamped" => "water depth in [$(h_min), $(h_max)] m"])
    defVar(ds, "elevation_raw", Float32.(elev), ("X", "Y"), attrib = ["units" => "m", "long_name" => "GEBCO elevation, unmasked"])
    defVar(ds, "source", src, ("X", "Y"), attrib = ["long_name" => "GEBCO year (26 or 24)"])
    ds.attrib["cape_lon"] = cape[1]; ds.attrib["cape_lat"] = cape[2]; ds.attrib["axis_azimuth"] = az
    ds.attrib["X_cape_km"] = X_cape; ds.attrib["Y_cape_km"] = Y_cape
    ds.attrib["h_min"] = h_min; ds.attrib["h_max"] = h_max
    ds.attrib["source_files"] = "$(basename(F2026)); $(basename(F2024))"
end
@info "wrote $OUT: $(nx) x $(ny), $(round(100 * count(water) / length(water), digits=1))% water, $(round(100 * count(==(26), src) / length(src), digits=1))% from GEBCO 2026"
