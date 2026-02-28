"""
GPU-compatible parameterized shoal bathymetry using @inline functions.

All bathymetry parameters are configurable via keyword arguments.
Returns a `bottom(x, y)` function for use with `GridFittedBottom`.

Usage:
    bottom = dshoal_param_bottom(Ly;
        shelf_length=30e3, shelf_depth=-20.0,
        shoal_length=30e3, shoal_crest_depth=-5.0,
        deep_ocean_depth=-50.0, sigma=8e3, Hs=15.0,
        Ly_shoal=100e3)
    ib_grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom))
"""

# ── Background bathymetry: parameterized piecewise linear slope in x ────

@inline function param_background_depth(x, shelf_length, shelf_depth, deep_ocean_depth)
    # Breakpoints scaled from shelf_length
    aw = shelf_length * 0.5
    h_aw = shelf_depth
    bw = shelf_length
    h_bw = shelf_depth - 5.0
    cw = shelf_length + 20e3
    h_cw = deep_ocean_depth

    m1 = h_aw / aw
    m2 = (h_bw - h_aw) / (bw - aw)
    m3 = (h_cw - h_bw) / (cw - bw)

    if x < aw
        return m1 * x
    elseif x < bw
        return h_aw + m2 * (x - aw)
    elseif x < cw
        return h_bw + m3 * (x - bw)
    else
        return h_cw
    end
end

# ── Shoal x-profile: parameterized gentle slope ─────────────────────────

@inline function param_shoal_xprofile(x, shoal_length, shoal_crest_depth)
    # Near-shore break at ~8% of shoal length
    as = shoal_length * (2.5 / 30.0)
    h_as = shoal_crest_depth

    # Offshore end: stays shallow (above background)
    ds = shoal_length * (36.0 / 30.0)
    h_ds = shoal_crest_depth - 10.0  # only 10m deeper than crest

    m1 = h_as / as
    m2 = (h_ds - h_as) / (ds - as)

    if x < as
        return m1 * x
    elseif x < ds
        return h_as + m2 * (x - as)
    else
        return h_ds
    end
end

# ── Shoal taper: smooth cosine ramp ─────────────────────────────────────

@inline function param_shoal_taper(x, shoal_length)
    x_start = shoal_length * (20.0 / 30.0)  # begin tapering at ~67% of shoal
    x_end = shoal_length * (36.0 / 30.0)  # fully zero at ~120% of shoal
    if x >= x_end
        return 0.0
    elseif x <= x_start
        return 1.0
    else
        return 0.5 * (1.0 + cos(π * (x - x_start) / (x_end - x_start)))
    end
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

@inline function _param_shoal_bottom(x, y, y0, sigma, Hs, half_extent,
    shelf_length, shelf_depth,
    shoal_length, shoal_crest_depth,
    deep_ocean_depth)
    # Compact-support window (shared by background slope and shoal)
    window = param_shoal_window(y, y0, half_extent)

    # Background: blend between slope (inside window) and flat deep ocean (outside)
    hw_slope = param_background_depth(x, shelf_length, shelf_depth, deep_ocean_depth)
    hw = deep_ocean_depth + (hw_slope - deep_ocean_depth) * window

    # Shoal centerline profile (without taper)
    hs_center = param_shoal_xprofile(x, shoal_length, shoal_crest_depth)

    # Bump above background (only positive)
    bump = max(0.0, hs_center - hw_slope)

    # Taper kills the bump offshore → no protrusion
    taper = param_shoal_taper(x, shoal_length)

    # Gaussian cross-section
    gauss_norm = exp(-((y - y0)^2) / (2 * sigma^2))

    return hw + bump * taper * gauss_norm * window
end

# ── Public constructor ──────────────────────────────────────────────────

"""
    dshoal_param_bottom(Ly; kwargs...)

Returns a function `bottom(x, y)` (x, y in meters) for `GridFittedBottom`.

Keyword arguments:
- `shelf_length`:      length of the shelf [m], default 30e3
- `shelf_depth`:       depth at inner shelf break [m], default -20.0
- `shoal_length`:      along-shoal extent in x [m], default 30e3
- `shoal_crest_depth`: shallowest depth of the shoal [m], default -5.0
- `deep_ocean_depth`:  deep ocean floor depth [m], default -50.0
- `sigma`:             Gaussian width of shoal cross-section [m], default 8e3
- `Hs`:                shoal Gaussian amplitude [m], default 15.0
- `Ly_shoal`:          along-shore extent of the shoal [m], default Ly
"""
function dshoal_param_bottom(Ly;
    shelf_length=30e3,
    shelf_depth=-20.0,
    shoal_length=30e3,
    shoal_crest_depth=-5.0,
    deep_ocean_depth=-50.0,
    sigma=8e3,
    Hs=15.0,
    Ly_shoal=Ly)

    y0 = Ly / 2.0
    half_extent = Ly_shoal / 2.0

    bottom(x, y) = _param_shoal_bottom(x, y, y0, sigma, Hs, half_extent,
        shelf_length, shelf_depth,
        shoal_length, shoal_crest_depth,
        deep_ocean_depth)
    return bottom
end
