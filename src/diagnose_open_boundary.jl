#!/usr/bin/env julia
# ═══════════════════════════════════════════════════════════════════════════
# diagnose_open_boundary.jl
# ═══════════════════════════════════════════════════════════════════════════
# Two questions about an open-east run, and a sanity block.
#
#   A. Is the open boundary holding mass balance?
#      Expect a TWO-LAYER u profile at x = Lx (offshore at the surface,
#      onshore below) integrating to ~0. A flat profile with net ~0 means the
#      boundary is behaving like a wall and nothing was gained.
#
#   B. Which way is the offshore streak moving?
#      offshore (+x) -> outgoing adjustment transient, benign
#      onshore  (-x) -> reflection off the open boundary, needs tuning
#      stationary    -> neither; something standing
#
# Usage:  julia --project=. src/diagnose_open_boundary.jl [run_tag]
# ═══════════════════════════════════════════════════════════════════════════

using NCDatasets, Printf, Statistics

const DIR = length(ARGS) ≥ 2 ? ARGS[2] : get(ENV, "NC_DIR", "/glade/work/nguyen/unc-shoals/nc")
const TAG = length(ARGS) ≥ 1 ? ARGS[1] : "periodic_shoals59_hydro"
const f₀ = 8.4168e-5     # FPlane(latitude=35.2480)
const ρ₀ = 1024.0
const τ_wind = 0.15      # N m⁻², set to the run's wind_stress

nanmax(a) = maximum(x -> isfinite(x) ? abs(x) : -Inf, a)
finite(a) = filter(isfinite, vec(a))
safemean(a) = (v = finite(a); isempty(v) ? 0.0 : mean(v))

"Read `var` as (x, y, z, time) plus its coordinate vectors."
function load(path, var)
    ds = NCDataset(path)
    haskey(ds, var) || (close(ds); error("$(basename(path)) has no variable $var"))
    dn = NCDatasets.dimnames(ds[var])
    a = Array(ds[var][:, :, :, :])
    x = Array(ds[dn[1]][:])
    y = Array(ds[dn[2]][:])
    z = Array(ds[dn[3]][:])
    t = Array(ds["time"][:])
    close(ds)
    return a, x, y, z, t
end

println("="^70)
println("run tag: ", TAG)
println("="^70)

# ───────────────────────────────────────────────────────────────────────────
# A. MASS BALANCE AT THE OPEN BOUNDARY
#    Uses the 3-D time average so the y-average is a real one.
# ───────────────────────────────────────────────────────────────────────────
avg_file = joinpath(DIR, "time_avg_3d_$(TAG).nc")
if isfile(avg_file)
    # Read ONLY the easternmost column — the full 3-D field is ~160 MB here.
    ds = NCDataset(avg_file)
    dn = NCDatasets.dimnames(ds["u_c"])
    x = Array(ds[dn[1]][:]); z = Array(ds[dn[3]][:]); t = Array(ds["time"][:])
    iE = length(x)                       # easternmost centre
    uE = Array(ds["u_c"][iE, :, :, length(t)])   # (y, z), last averaging window
    close(ds)
    Δz = length(z) > 1 ? abs(z[2] - z[1]) : 1.0

    # y-averaged vertical profile, ignoring immersed/missing cells
    prof = [safemean(uE[:, k]) for k in 1:length(z)]
    prof = map(v -> isnan(v) ? 0.0 : v, prof)

    net = sum(prof) * Δz                 # m² s⁻¹ per unit y
    M_ek = τ_wind / (ρ₀ * f₀)

    println("\nA. OPEN BOUNDARY MASS BALANCE   (x = $(round(x[iE]/1e3, digits=1)) km, ",
        "window ending t = $(round(t[end]/86400, digits=1)) d)")
    println("   Ekman transport for reference: M = τ/(ρf) = ", @sprintf("%.2f", M_ek), " m²/s\n")
    println("        z (m)     y-avg u (m/s)")
    for k in length(z):-1:1
        bar = repeat(prof[k] > 0 ? "+" : "-", min(40, round(Int, abs(prof[k]) / 0.005)))
        @printf("   %8.1f %12.4f  %s\n", z[k], prof[k], bar)
    end
    @printf("\n   NET transport = %+.4f m²/s   (|net|/M = %.3f)\n", net, abs(net) / M_ek)

    pos = sum(p for p in prof if p > 0; init=0.0) * Δz
    neg = sum(p for p in prof if p < 0; init=0.0) * Δz
    @printf("   outflow (u>0) = %+.3f m²/s ; inflow (u<0) = %+.3f m²/s\n", pos, neg)

    if abs(net) / M_ek < 0.1 && pos > 0.3 * M_ek
        println("   => TWO-LAYER exchange with net ~ 0. The open BC is working.")
    elseif abs(net) / M_ek < 0.1
        println("   => net ~ 0 but exchange is WEAK: behaving like a wall.")
    else
        println("   => NET IS NOT CLOSING. Mass is leaving uncompensated.")
    end
else
    println("\nA. skipped — no $(basename(avg_file))")
end

# ───────────────────────────────────────────────────────────────────────────
# B. STREAK DIRECTION  (surface u, y-averaged, Hovmöller in x and t)
# ───────────────────────────────────────────────────────────────────────────
top_file = joinpath(DIR, "top_$(TAG).nc")
if isfile(top_file)
    u, x, y, z, t = load(top_file, "u_c")
    nt = length(t)
    ux = [safemean(u[i, :, 1, n]) for i in 1:length(x), n in 1:nt]   # (x, t)
    ux = map(v -> isnan(v) ? 0.0 : v, ux)

    off = findall(xi -> xi > 100e3, x)      # offshore of all topography
    println("\nB. OFFSHORE STREAK TRACKING   (surface u, y-averaged, x > 100 km)\n")
    println("      day     x of max|u|      max|u| (m/s)")
    xs = Float64[]
    days = Float64[]
    for n in 1:nt
        seg = ux[off, n]
        j = argmax(abs.(seg))
        push!(xs, x[off[j]])
        push!(days, t[n] / 86400)
        @printf("   %6.2f %12.1f km %14.5f\n", t[n] / 86400, x[off[j]] / 1e3, seg[j])
    end

    # Is there a localized peak to track at all? On a near-uniform field argmax
    # wanders between frames and any "drift" it reports is noise, not propagation.
    lastseg = abs.(ux[off, end])
    peakiness = maximum(lastseg) / (mean(lastseg) + eps())
    mean_u = mean(ux[off, end])
    @printf("\n   offshore band (last frame): mean u = %+.4f m/s, max|u| = %.4f, peak/mean = %.2f\n",
        mean_u, maximum(lastseg), peakiness)
    @printf("   if that fills a 17 m Ekman layer: %.2f m²/s  (M = %.2f m²/s)\n",
        mean_u * 17.0, τ_wind / (ρ₀ * f₀))
    localized = peakiness > 1.8
    localized || println("   => NO localized streak: the offshore band is a broad PLATEAU.\n" *
                         "      argmax tracking is meaningless here — ignore the drift below.")

    if localized && nt ≥ 3
        # least-squares slope of x(t) over the second half of the record
        h = max(1, nt ÷ 2):nt
        t̄, x̄ = mean(days[h]), mean(xs[h])
        num = sum((days[i] - t̄) * (xs[i] - x̄) for i in h)
        den = sum((days[i] - t̄)^2 for i in h)
        if den > 0
            slope = num / den                    # m per day
            @printf("\n   drift over the last half of the record: %+.1f km/day (%+.4f m/s)\n",
                slope / 1e3, slope / 86400)
            if slope / 1e3 > 1
                println("   => moving OFFSHORE: outgoing adjustment transient. Benign.")
            elseif slope / 1e3 < -1
                println("   => moving ONSHORE: REFLECTION off the open boundary.")
                println("      Try a longer outflow_timescale in NormalRadiation.")
            else
                println("   => roughly STATIONARY: not a propagating wave front.")
            end
            println("   (mode-1 = 12.4 km/day, mode-2 = 6.2, mode-3 = 4.1)")
        end
    end

    # compact ASCII Hovmöller
    println("\n   Hovmöller of y-averaged surface u  (rows = time, cols = x from 100 km out)")
    cols = range(first(off), length(x), length=min(60, length(x) - first(off) + 1))
    scale = nanmax(ux[off, :])
    println("        day |", repeat("-", length(cols)), "| x = ",
        round(x[end] / 1e3), " km   (scale ±", @sprintf("%.3f", scale), ")")
    for n in 1:nt
        row = map(collect(cols)) do c
            v = ux[round(Int, c), n] / (scale + eps())
            v > 0.5 ? '#' : v > 0.15 ? '+' : v < -0.5 ? '@' : v < -0.15 ? 'o' : '.'
        end
        @printf("   %8.2f |%s|\n", t[n] / 86400, String(row))
    end
    println("   ('+#' = offshore, 'o@' = onshore)  — a diagonal band is a propagating front.")
else
    println("\nB. skipped — no $(basename(top_file))")
end

# ───────────────────────────────────────────────────────────────────────────
# C. SANITY: field ranges, excluding the two contaminated wall columns
# ───────────────────────────────────────────────────────────────────────────
if isfile(top_file)
    println("\nC. SURFACE FIELD RANGES  (i = 3 : Nx-1, wall columns excluded)")
    ds = NCDataset(top_file)
    xc = Array(ds[NCDatasets.dimnames(ds["Ro"])[1]][:])
    for v in ("u_c", "v_c", "T", "S", "Ro", "KE")
        haskey(ds, v) || continue
        a = Array(ds[v][:, :, :, :])
        b = a[3:end-1, :, :, end]
        fv = finite(b)
        isempty(fv) && continue
        @printf("   %-5s  min %10.4f   max %10.4f\n", v, minimum(fv), maximum(fv))
    end

    # Where does the extreme actually live? A domain-wide max can be set by a
    # boundary artifact rather than by the eddy field, and then "max Ro fell"
    # means the boundary got better, not that the eddies went away.
    println("\n   max |Ro| by cross-shore band (last frame):")
    Ro = Array(ds["Ro"][:, :, :, :])[:, :, 1, end]
    for (lo, hi) in ((0, 50), (50, 100), (100, 150), (150, 200), (200, 250))
        idx = findall(q -> lo*1e3 ≤ q < hi*1e3, xc)
        idx = filter(i -> 3 ≤ i ≤ length(xc) - 1, idx)
        isempty(idx) && continue
        fv = finite(Ro[idx, :])
        isempty(fv) && continue
        @printf("      x = %3d-%3d km :  %8.3f\n", lo, hi, maximum(abs, fv))
    end
    close(ds)
end

# ───────────────────────────────────────────────────────────────────────────
# D. EDDY vs MEAN KINETIC ENERGY
#    This is the metric to compare ACROSS WIND CASES. max|Ro| is a single
#    extreme and is easily set by a boundary artifact; EKE is an energy and
#    lives at the resolved scales.
#       EKE = ½[(⟨uu⟩-⟨u⟩²) + (⟨vv⟩-⟨v⟩²)]
#       MKE = ½[⟨u⟩² + ⟨v⟩²]
# ───────────────────────────────────────────────────────────────────────────
if isfile(avg_file)
    ds = NCDataset(avg_file)
    need = ("u_c", "v_c", "uu", "vv")
    if all(v -> haskey(ds, v), need)
        xc = Array(ds[NCDatasets.dimnames(ds["u_c"])[1]][:])
        na = length(ds["time"])
        g(v) = Array(ds[v][:, :, :, na])
        U, V, UU, VV = g("u_c"), g("v_c"), g("uu"), g("vv")
        EKE = 0.5 .* ((UU .- U .^ 2) .+ (VV .- V .^ 2))
        MKE = 0.5 .* (U .^ 2 .+ V .^ 2)

        println("\nD. KINETIC ENERGY  (last averaging window)")
        ek, mk = finite(EKE), finite(MKE)
        if !isempty(ek)
            @printf("   domain mean EKE = %.6e m²/s²\n", mean(ek))
            @printf("   domain mean MKE = %.6e m²/s²\n", mean(mk))
            @printf("   EKE / MKE       = %.3f\n\n", mean(ek) / (mean(mk) + eps()))
            println("   by cross-shore band:")
            println("        x (km)        EKE          MKE      EKE/MKE")
            for (lo, hi) in ((0, 50), (50, 100), (100, 150), (150, 200), (200, 250))
                idx = findall(q -> lo * 1e3 ≤ q < hi * 1e3, xc)
                isempty(idx) && continue
                e = finite(EKE[idx, :, :])
                m = finite(MKE[idx, :, :])
                isempty(e) && continue
                @printf("      %3d-%3d  %11.4e  %11.4e  %8.3f\n",
                    lo, hi, mean(e), mean(m), mean(e) / (mean(m) + eps()))
            end
            println("\n   Compare domain-mean EKE across wind cases on an IDENTICAL grid,")
            println("   domain and boundary condition. Rising EKE with wind = the")
            println("   APE -> EKE pathway is active; falling = APE reservoir depleted.")
        end
    else
        miss = filter(v -> !haskey(ds, v), need)
        println("\nD. skipped — $(basename(avg_file)) lacks: ", join(miss, ", "))
    end
    close(ds)
end

println("\n", "="^70)
