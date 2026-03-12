"""
GPU-compatible parameterized shoal bathymetry using smooth sigmoidal transitions.

All bathymetry parameters are configurable via keyword arguments.
Returns a `bottom(x, y)` function for use with `GridFittedBottom`.

Optimization Note:
This version replaces piecewise linear segments with smooth sigmoids (tanh) and 
ensures a minimum background depth to avoid the H -> 0 singularity at the shore,
which significantly improves the convergence of Conjugate Gradient pressure solvers.

Usage:
    bottom = dshoal_param_bottom(Ly;
        shelf_length=30e3, shelf_depth=-20.0,
        shoal_length=30e3, shoal_crest_depth=-5.0,
        deep_ocean_depth=-50.0, sigma=8e3, Hs=15.0,
        Ly_shoal=100e3, min_depth=-5.0)
    ib_grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom))
"""

# ── Helper functions for smoothness ─────────────────────────────────────

@inline smooth_step(x, x0, w) = 0.5 * (1.0 + tanh((x - x0) / w))
@inline smooth_max(a, b, k=0.5) = 0.5 * (a + b + sqrt((a - b)^2 + k^2))

# ── Background bathymetry: smooth parameterized slope in x ───────────────

@inline function param_background_depth(x, shelf_length, shelf_depth, deep_ocean_depth, min_depth)
    # Background levels derived from the visual profile
    h_shore = min_depth
    h_shelf_top = shelf_depth
    h_shelf_bottom = deep_ocean_depth + 12.0
    h_deep = deep_ocean_depth

    # Transition points scaled by shelf_length
    # 1. Shore drop
    t1 = shelf_length * 0.25
    w1 = shelf_length * 0.10

    # 2. Middle shelf slope (spread out to match the black curve)
    t2 = shelf_length * 0.85
    w2 = shelf_length * 0.40

    # 3. Final drop to deep ocean
    t3 = shelf_length * 1.30
    w3 = shelf_length * 0.10

    # Smooth transition sum
    h = h_shore
    h += (h_shelf_top - h_shore) * smooth_step(x, t1, w1)
    h += (h_shelf_bottom - h_shelf_top) * smooth_step(x, t2, w2)
    h += (h_deep - h_shelf_bottom) * smooth_step(x, t3, w3)

    return h
end

# ── Shoal x-profile: smooth parameterized slope ─────────────────────────

@inline function param_shoal_xprofile(x, shoal_length, shoal_crest_depth, min_depth)
    # Levels
    h1 = min_depth
    h2 = shoal_crest_depth
    h3 = shoal_crest_depth - 10.0

    # Transitions
    t12 = shoal_length * (2.5 / 30.0) * 0.5
    w12 = shoal_length * 0.05

    t23 = shoal_length * (20.0 / 30.0)
    w23 = shoal_length * 0.1

    h = h1
    h += (h2 - h1) * smooth_step(x, t12, w12)
    h += (h3 - h2) * smooth_step(x, t23, w23)

    return h
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
    deep_ocean_depth, min_depth)

    # Compact-support window (shared by background slope and shoal)
    window = param_shoal_window(y, y0, half_extent)

    # Background: blend between slope (inside window) and flat deep ocean (outside)
    hw_slope = param_background_depth(x, shelf_length, shelf_depth, deep_ocean_depth, min_depth)
    hw = deep_ocean_depth + (hw_slope - deep_ocean_depth) * window

    # Shoal centerline profile (without taper)
    hs_center = param_shoal_xprofile(x, shoal_length, shoal_crest_depth, min_depth)

    # Bump above background (smoothed max to avoid sharp corners)
    # We want max(0.0, hs_center - hw_slope)
    bump = smooth_max(0.0, hs_center - hw_slope, 0.1)

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
- `min_depth`:         minimum depth at the shore (x=0) [m], default -5.0
"""
function dshoal_param_bottom(Ly;
    shelf_length=30e3,
    shelf_depth=-20.0,
    shoal_length=30e3,
    shoal_crest_depth=-5.0,
    deep_ocean_depth=-50.0,
    sigma=8e3,
    Hs=15.0,
    Ly_shoal=Ly,
    min_depth=-5.0)

    y0 = Ly / 2.0
    half_extent = Ly_shoal / 2.0

    bottom(x, y) = _param_shoal_bottom(x, y, y0, sigma, Hs, half_extent,
        shelf_length, shelf_depth,
        shoal_length, shoal_crest_depth,
        deep_ocean_depth, min_depth)
    return bottom
end
