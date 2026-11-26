"""
GPU-compatible parameterized shoal bathymetry using smooth sigmoidal transitions.
Optimized for Conjugate Gradient pressure solver convergence by ensuring C1 continuity.

Background Geometry (from Schematic):
Background follows the specific piecewise definition:
- 0 to 5 km:   coastal ramp (-3 m to -5 m)
- 5 to shelf_break_end:  shelf break (-5 m to Zsh)
- shelf_break_end to 62 km: shelf slope (Zsh to -30 m)
- 62 to 65 km: offshore ramp (-30 m to -50 m)
- beyond 65 km: flat offshore (-50 m)

Shoal Geometry:
- Zs: Absolute depth of the peak of the shoal.
- Ls: Length of the elevated section, starting at x = 8km.
- Taper: Smooth cosine ramp at the offshore end.
"""

# ── Helper functions for smoothness ─────────────────────────────────────

# Sigmoid step function for solver-friendly transitions
@inline smooth_step(x, x0, w) = 0.5 * (1.0 + tanh((x - x0) / w))

# C1 continuous max function for solver-friendly corners
@inline smooth_max(a, b, k=0.1) = 0.5 * (a + b + sqrt((a - b)^2 + k^4))

# ── Background bathymetry ───────────────────────────────────────────────

@inline function param_background_depth(x, Zsh, shelf_break_end, rise_mid, rise_width, h_deep)
    # h_deep was hardcoded at -50.0, which is also the domain floor. That made
    # Zsh = -50 erase the shelf break entirely -- the offshore ramp term below
    # is (h_deep - Zsh)*smooth_step, identically zero when they are equal -- so
    # the shelf merged into the deep ocean and every "coast to shelf break"
    # diagnostic silently fell back to a fixed 50 km box. It also capped the
    # usable shelf-depth range at about -40 m, which is the axis the APE
    # scaling most needs. Now a parameter.
    h0, h1 = -5.0, -7.0
    h4 = h_deep
    h2 = Zsh

    # 1. Coastal ramp (0-5km)
    h = h0 + (h1 - h0) * smooth_step(x, 2.5e3, 1.5e3)

    # 2. Shelf break (5km to shelf_break_end)
    sb_mid = 5.0e3 + (shelf_break_end - 5.0e3) / 2.0
    sb_width = (shelf_break_end - 5.0e3) / 2.8
    h += (h2 - h1) * smooth_step(x, sb_mid, sb_width)

    # 3. Offshore ramp. `rise_mid` is the offshore end of the flat shelf, so
    # SHELF LENGTH is rise_mid - shelf_break_end. It was hardcoded at 63.5e3,
    # which made shelf length the one geometric quantity the sweep could not
    # vary independently: shelf_break_end moves the inshore end AND the break
    # steepness at the same time.
    h += (h4 - h2) * smooth_step(x, rise_mid, rise_width)

    return h
end

# ── Along-shore compact window ──────────────────────────────────────────

@inline function param_shoal_window(y, y0, half_extent)
    # If shoal spans the full domain (Ly_shoal >= Ly), no taper needed
    if half_extent >= y0
        return 1.0
    end
    dy = abs(y - y0)
    if dy >= half_extent
        return 0.0
    else
        taper_start = 0.5 * half_extent  # taper in outer 50%
        if dy <= taper_start
            return 1.0
        else
            return 0.5 * (1.0 + cos(π * (dy - taper_start) / (half_extent - taper_start)))
        end
    end
end

# ── Combined bottom function ────────────────────────────────────────────

@inline function _param_shoal_bottom(x, y, y0, sigma, Zs, half_extent, shoal_length, Zsh,
                                     shelf_break_end, rise_mid, rise_width,
                                     shoal_start, shoal_inner_width, h_deep)
    # 1. Background depth
    hw = param_background_depth(x, Zsh, shelf_break_end, rise_mid, rise_width, h_deep)

    # 2. Along-shore window (compact support)
    window = param_shoal_window(y, y0, half_extent)

    # 3a. Offshore edge: fully elevated out to `shoal_length`, then a 15 km
    # cosine ramp back down to the background.
    x_taper_start = shoal_length
    x_taper_end = shoal_length + 15.0e3

    if x <= x_taper_start
        taper_off = 1.0
    elseif x >= x_taper_end
        taper_off = 0.0
    else
        taper_off = 0.5 * (1.0 + cos(π * (x - x_taper_start) / (x_taper_end - x_taper_start)))
    end

    # 3b. SHOREWARD edge. Without one the shoal is welded to the coast at full
    # amplitude from x = 0, and then Zs sets three geometric properties at
    # once, because the shoal is "fill the background up to Zs" (step 5):
    #
    #   Zs = -5 m   the fill reaches the coast, burying the coastal ramp AND
    #               the shelf break -- a flat slab, inshore |dh/dx| = 0
    #   Zs = -15 m  the fill is identically zero inshore of x ~ 7.8 km, so the
    #               background break survives AS the shoal's inshore flank,
    #               |dh/dx| ~ 0.0035
    #
    # and the flat crest runs from wherever the background crosses Zs out to
    # `shoal_length`, so crest length is a function of Zs too. No choice of
    # the other parameters can separate them: the inshore flank was never a
    # property of the shoal, it was the background break, buried or exposed.
    #
    # With `shoal_start` the shoal sits ON the shelf instead of against the
    # coast. The background break is then identical in every run, the crest is
    # exactly shoal_length - shoal_start, and the inshore flank is set by
    # `shoal_inner_width` -- which can be chosen per run to hold the slope
    # fixed while Zs varies, the control this sweep could not express before.
    #
    # shoal_start = 0 (the default) takes the first branch and reproduces the
    # welded-to-the-coast geometry exactly, so every existing sweep is
    # unchanged.
    x_in_end = shoal_start
    x_in_start = shoal_start - shoal_inner_width

    if shoal_start <= 0.0
        taper_in = 1.0
    elseif x >= x_in_end
        taper_in = 1.0
    elseif x <= x_in_start
        taper_in = 0.0
    else
        taper_in = 0.5 * (1.0 - cos(π * (x - x_in_start) / (x_in_end - x_in_start)))
    end

    taper = taper_in * taper_off

    # 4. Along-shore Gaussian and Windowing
    gauss_y = exp(-((y - y0)^2) / (2 * sigma^2))
    factor = taper * gauss_y * window

    # 5. Connection logic: 
    # Zs is the absolute depth of the top of the shoal.
    elevation_target = Zs

    # Take the exact max to perfectly match the coastal ramp without leakage
    potential_height = max(0.0, elevation_target - hw)

    # Apply spatial factors and add to background
    return min(-5.0, hw + potential_height * factor)
end

# ── Public constructor ──────────────────────────────────────────────────

"""
    dshoal_param_bottom(Ly; kwargs...)

Returns a function `bottom(x, y)` for `GridFittedBottom`.
Zs is the absolute depth of the peak of the shoal.
Zsh is the absolute depth of the shelf.
shoal_length (Ls) is the OFFSHORE end of the flat crest; the shoal then tapers
over a further 15 km.
shoal_start is the SHOREWARD end of the flat crest, and shoal_inner_width the
cosine ramp leading up to it from the background. Both default to 0, which
puts the shoal against the coast at full amplitude -- the original geometry.
Set them to put the shoal on the shelf, so that the crest length
(shoal_length - shoal_start) and the inshore flank slope stop being functions
of Zs. See the comment in `_param_shoal_bottom` for why that matters.
h_deep is the flat offshore depth (default -50.0, the old hardcoded value).
It must be DEEPER than Zsh or there is no shelf break at all. Note that the
break's steepness is (h_deep - Zsh)/(2*rise_width), so deepening the basin
without widening the ramp makes the break steeper in proportion -- and the
break is exactly where the offshore flux is measured.
"""
function dshoal_param_bottom(Ly;
    sigma=8e3,
    Zs=-5.0,
    shoal_length=40e3,
    Ly_shoal=Ly,
    Zsh=-25.0,
    shelf_break_end=12.0e3,
    rise_mid=63.5e3,
    rise_width=5.0e3,
    shoal_start=0.0,
    shoal_inner_width=0.0,
    h_deep=-50.0,
    crest_length=nothing,
    shoal_y0=nothing)

    # Alongshore centre of the shoal: the middle of the domain unless placed.
    y0 = isnothing(shoal_y0) ? Ly / 2.0 : Float64(shoal_y0)
    half_extent = Ly_shoal / 2.0

    # crest_length: the crest is THIS long. An attached shoal's flat crest
    # starts where the coastal slope first reaches Zs, which moves with Zs, so
    # shoal_length (where the crest ENDS) is derived from it. Without this, a
    # sweep "holding Ls = 40 km" had crests of 40 / 33.8 / 32.1 km at
    # Zs = -5 / -10 / -15. Mirrors tools/shoal_bathymetry.m.
    if !isnothing(crest_length)
        shoal_start > 0 && error("crest_length is for shoals attached to the coast (shoal_start = 0)")
        shoal_length = shoal_crest_start(Zs, Zsh, shelf_break_end, rise_mid, rise_width, h_deep) + crest_length
    end

    bottom(x, y) = _param_shoal_bottom(x, y, y0, sigma, Zs, half_extent, shoal_length, Zsh,
                                       shelf_break_end, rise_mid, rise_width,
                                       shoal_start, shoal_inner_width, h_deep)
    check_no_trench(bottom, y0, shoal_length)
    check_shelf_gap(Zsh, shoal_length, shelf_break_end, rise_mid, rise_width, h_deep)
    return bottom
end

"""
    shoal_crest_start(Zs, Zsh, shelf_break_end, rise_mid, rise_width, h_deep)

Where an attached shoal's flat crest begins: the first x at which the
background is as deep as Zs (0 if the coast already is). Bisection on the
background, which deepens monotonically from the coast to the shelf.
"""
function shoal_crest_start(Zs, Zsh, shelf_break_end, rise_mid, rise_width, h_deep)
    f(x) = param_background_depth(x, Zsh, shelf_break_end, rise_mid, rise_width, h_deep) - Zs
    f(0.0) <= 0 && return 0.0
    a, b = 0.0, rise_mid
    for _ in 1:60
        m = (a + b) / 2
        f(m) > 0 ? (a = m) : (b = m)
    end
    return (a + b) / 2
end

"""
    check_shelf_gap(Zsh, shoal_length, ...; min_gap=10e3)

WARN (not error) if there is less than `min_gap` of flat shelf between the end
of the shoal's 15 km offshore taper and the start of the shelf break (the
background 1 m below Zsh). The reference shoal (Ls = 40 km, break at
63.5 km) has 0.6 km and is accepted as it is; the warning records it.
"""
function check_shelf_gap(Zsh, shoal_length, shelf_break_end, rise_mid, rise_width, h_deep; min_gap=10e3)
    drop = Zsh - h_deep
    drop <= 1 && return nothing                       # no shelf break at all
    break_start = rise_mid + rise_width * atanh(2 / drop - 1)
    gap = break_start - (shoal_length + 15e3)
    gap < min_gap && @warn "only $(round(gap / 1e3, digits=1)) km of flat shelf between the shoal's taper and the shelf break (minimum $(min_gap / 1e3) km)"
    return nothing
end

"""
    check_no_trench(bottom, y0, shoal_length; tolerance=1.0)

Error if the cross-shore profile through the shoal's centre (y = y0) has a
trench between the coast and the end of the crest: a point deeper, by more
than `tolerance` metres, than the shallowest points on BOTH sides of it.
A shoal placed on the shelf with `shoal_start > 0` makes exactly that: the
background drops to the shelf depth by `shelf_break_end` and the shoal's
inshore ramp then climbs back up, leaving a channel (25 m deep and ~10 km
wide for the Zs05_edge geometry of params_depthaxis.toml). There should be
no channels between the coast and the shoal in any run.
"""
function check_no_trench(bottom, y0, shoal_length; tolerance=1.0)
    xs = 0.0:100.0:shoal_length
    d = [bottom(x, y0) for x in xs]                       # negative: depth below 0
    left = accumulate(max, d)                             # shallowest from the coast to here
    right = reverse(accumulate(max, reverse(d)))          # shallowest from here to the crest end
    dip = min.(left, right) .- d                          # how far below both sides
    k = argmax(dip)
    if dip[k] > tolerance
        error("bathymetry has a $(round(dip[k], digits=1)) m deep trench at x = $(round(xs[k] / 1e3, digits=1)) km " *
              "between the coast and the shoal (depth $(round(d[k], digits=1)) m, with " *
              "$(round(left[k], digits=1)) m inshore and $(round(right[k], digits=1)) m offshore of it). " *
              "Channels between the coast and the shoal are not allowed: attach the shoal to the coast (shoal_start = 0).")
    end
    return nothing
end
