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

@inline function param_background_depth(x, Zsh, shelf_break_end)
    h0, h1, h4 = -5.0, -7.0, -50.0
    h2 = Zsh

    # 1. Coastal ramp (0-5km)
    h = h0 + (h1 - h0) * smooth_step(x, 2.5e3, 1.5e3)

    # 2. Shelf break (5km to shelf_break_end)
    sb_mid = 5.0e3 + (shelf_break_end - 5.0e3) / 2.0
    sb_width = (shelf_break_end - 5.0e3) / 2.8
    h += (h2 - h1) * smooth_step(x, sb_mid, sb_width)

    # 3. Offshore ramp (62-65km)
    # The shelf stays flat at `Zsh` until it reaches the offshore ramp.
    h += (h4 - h2) * smooth_step(x, 63.5e3, 5.0e3)

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

@inline function _param_shoal_bottom(x, y, y0, sigma, Zs, half_extent, shoal_length, Zsh, shelf_break_end)
    # 1. Background depth
    hw = param_background_depth(x, Zsh, shelf_break_end)

    # 2. Along-shore window (compact support)
    window = param_shoal_window(y, y0, half_extent)

    # 3. Shoal geometry
    # The shoal remains fully elevated up to `shoal_length`.
    # It then tapers down over 15 km
    x_taper_start = shoal_length
    x_taper_end = shoal_length + 15.0e3

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
shoal_length (Ls) is the length scale for the x-profile ramp and taper.
"""
function dshoal_param_bottom(Ly;
    sigma=8e3,
    Zs=-5.0,
    shoal_length=40e3,
    Ly_shoal=Ly,
    Zsh=-25.0,
    shelf_break_end=12.0e3)

    y0 = Ly / 2.0
    half_extent = Ly_shoal / 2.0

    bottom(x, y) = _param_shoal_bottom(x, y, y0, sigma, Zs, half_extent, shoal_length, Zsh, shelf_break_end)
    return bottom
end
