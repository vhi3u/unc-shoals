"""
GPU-compatible parameterized shoal bathymetry using smooth sigmoidal transitions.
Optimized for Conjugate Gradient pressure solver convergence by ensuring C1 continuity.

Background Geometry (from Schematic):
Background follows the specific piecewise definition:
- 0 to 5 km:   coastal ramp (-3 m to -5 m)
- 5 to 12 km:  shelf break (-5 m to -25 m)
- 12 to 62 km: shelf slope (-25 m to -30 m)
- 62 to 65 km: offshore ramp (-30 m to -50 m)
- beyond 65 km: flat offshore (-50 m)

Shoal Geometry:
- Hs: Peak height elevated above the -25m shelf.
- Ls: Length of the elevated section, starting at x = 8km.
- Taper: Smooth cosine ramp at the offshore end.
"""

# ── Helper functions for smoothness ─────────────────────────────────────

# Sigmoid step function for solver-friendly transitions
@inline smooth_step(x, x0, w) = 0.5 * (1.0 + tanh((x - x0) / w))

# C1 continuous max function for solver-friendly corners
@inline smooth_max(a, b, k=0.1) = 0.5 * (a + b + sqrt((a - b)^2 + k^4))

# ── Background bathymetry ───────────────────────────────────────────────

@inline function param_background_depth(x)
    h0, h1, h2, h3, h4 = -3.0, -5.0, -25.0, -30.0, -50.0

    # Transitions midpoint and width
    # 1. Coastal ramp (0-5km)
    h = h0 + (h1 - h0) * smooth_step(x, 2.5e3, 1.5e3)

    # 2. Shelf break (5-12km)
    h += (h2 - h1) * smooth_step(x, 8.5e3, 2.5e3)

    # 3. Shelf slope (12-62km)
    h += (h3 - h2) * smooth_step(x, 37.0e3, 15.0e3)

    # 4. Offshore ramp (62-65km)
    h += (h4 - h3) * smooth_step(x, 63.5e3, 5.0e3)

    return h
end

# ── Shoal x-profile: elevated slope with cosine taper ───────────────────

@inline function param_shoal_xprofile(x, x_start, shoal_length, Hs)
    # Rising slope (sigmoid) - sharpened to fill the nearshore gap
    rise_width = 0.5e3
    rise = smooth_step(x, x_start + rise_width, rise_width)

    # Offshore taper (smooth cosine ramp) as indicated in provided snippet
    # Using the specific ratios (20/30 and 36/30)
    x_taper_start = x_start + shoal_length * (20.0 / 30.0)
    x_taper_end = x_start + shoal_length * (36.0 / 30.0)

    if x <= x_taper_start
        taper = 1.0
    elseif x >= x_taper_end
        taper = 0.0
    else
        taper = 0.5 * (1.0 + cos(π * (x - x_taper_start) / (x_taper_end - x_taper_start)))
    end

    return Hs * rise * taper
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

@inline function _param_shoal_bottom(x, y, y0, sigma, Hs, half_extent, shoal_length)
    # 1. Background depth
    hw = param_background_depth(x)

    # 2. Along-shore window (compact support)
    window = param_shoal_window(y, y0, half_extent)

    # 3. Shoal geometry
    # Taper (smooth cosine ramp) defines the offshore end.
    # Widened the transition from 36/30 to 45/30 for a gentler offshore slope.
    x_ref = 8.0e3
    x_taper_start = x_ref + shoal_length * (20.0 / 30.0)
    x_taper_end = x_ref + shoal_length * (45.0 / 30.0)

    if x <= x_taper_start
        taper = 1.0
    elseif x >= x_taper_end
        taper = 0.0
    else
        taper = 0.5 * (1.0 + cos(π * (x - x_taper_start) / (x_taper_end - x_taper_start)))
    end

    # 4. Along-shore Gaussian and Windowing
    gauss_y = exp(-((y - y0)^2) / (2 * sigma^2))
    factor = taper * gauss_y * window

    # 5. Connection logic: 
    # Hs is the height above the -25m reference (target elevation = -10m if Hs=15).
    elevation_target = -25.0 + Hs

    # Increased k from 0.2 to 2.0 to significantly round the connection to the shelf break.
    potential_height = smooth_max(0.0, elevation_target - hw, 2.0)

    # Apply spatial factors and add to background
    return min(-1.0, hw + potential_height * factor)
end

# ── Public constructor ──────────────────────────────────────────────────

"""
    dshoal_param_bottom(Ly; kwargs...)

Returns a function `bottom(x, y)` for `GridFittedBottom`.
Hs is peak height above the background shelf.
shoal_length (Ls) is the length scale for the x-profile ramp and taper.
"""
function dshoal_param_bottom(Ly;
    sigma=8e3,
    Hs=15.0,
    shoal_length=40e3,
    Ly_shoal=Ly)

    y0 = Ly / 2.0
    half_extent = Ly_shoal / 2.0

    bottom(x, y) = _param_shoal_bottom(x, y, y0, sigma, Hs, half_extent, shoal_length)
    return bottom
end
