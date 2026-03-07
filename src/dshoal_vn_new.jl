"""
GPU-compatible shoal bathymetry using @inline functions.

Defines the bathymetry as composable inline functions rather than pre-computed arrays.
Returns a bottom(x, y) function for use with GridFittedBottom — architecture-agnostic
(works on both CPU and GPU, no separate versions needed).

Usage:
    bottom = dshoal_bottom(Ly, σ, Hs)
    ib_grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom))
"""

# ── Background bathymetry: piecewise linear slope in x ──────────────────
# x_km is the cross-shore coordinate in km
# Returns depth in meters (negative below sea level)

@inline function background_depth(x_km)
    # Breakpoints (km, m)
    aw, h_aw = 15.0, -20.0
    bw, h_bw = 30.0, -25.0
    cw, h_cw = 50.0, -50.0

    # Slopes
    m1 = h_aw / aw
    m2 = (h_bw - h_aw) / (bw - aw)
    m3 = (h_cw - h_bw) / (cw - bw)

    if x_km < aw
        return m1 * x_km
    elseif x_km < bw
        return h_aw + m2 * (x_km - aw)
    elseif x_km < cw
        return h_bw + m3 * (x_km - bw)
    else
        return h_cw
    end
end

# ── Shoal bathymetry: piecewise linear x-profile + Gaussian y-section ───

# Along-shore shoal profile (x in km → depth in m)
@inline function shoal_xprofile(x_km)
    as, h_as = 2.5, -5.0
    bs, h_bs = 27.0, -15.0
    cs, h_cs = 30.0, -27.0
    ds, h_ds = 36.0, -50.0

    m1 = h_as / as
    m2 = (h_bs - h_as) / (bs - as)
    m3 = (h_cs - h_bs) / (cs - bs)
    m4 = (h_ds - h_cs) / (ds - cs)

    if x_km < as
        return m1 * x_km
    elseif x_km < bs
        return h_as + m2 * (x_km - as)
    elseif x_km < cs
        return h_bs + m3 * (x_km - bs)
    elseif x_km < ds
        return h_cs + m4 * (x_km - ds)
    else
        return h_ds
    end
end

# X-direction taper: smooth cutoff at cs = 30 km
@inline function shoal_taper(x_km)
    cs = 30.0
    if x_km > cs
        return 0.0
    elseif x_km >= cs - 1
        return 0.5 * (1 + cos(π * (x_km - (cs - 1))))
    else
        return 1.0
    end
end

# Cross-shore Gaussian bump (y in km → height in m)
@inline shoal_gaussian(y_km, y0_km, σ, Hs) = Hs * exp(-((y_km - y0_km)^2) / (2 * σ^2))

# ── Combined bottom height function ─────────────────────────────────────

@inline function _shoal_bottom(x_km, y_km, y0_km, σ, Hs)
    # Background: piecewise linear slope, max = 0 at x = 0
    hw = background_depth(x_km)

    # Shoal: Gaussian × taper + x-profile, normalized so max = 0
    # (max of raw shoal = Hs at x=0, y=y0, so subtract Hs)
    hs = shoal_gaussian(y_km, y0_km, σ, Hs) * shoal_taper(x_km) + shoal_xprofile(x_km) - Hs

    return max(hw, hs)
end

# ── Public constructor ───────────────────────────────────────────────────

"""
    dshoal_bottom(Ly, σ, Hs)

Returns a function `bottom(x, y)` (x, y in meters) defining the shoal bathymetry.
Pass directly to `GridFittedBottom`:

    bottom = dshoal_bottom(params.Ly, σ, Hs)
    ib_grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom))

Parameters:
- `Ly`: domain length in y (meters)
- `σ`:  Gaussian width of shoal cross-section (km)
- `Hs`: shoal height (m)
"""
function dshoal_bottom(Ly, σ, Hs)
    y0_km = Ly / 2e3  # center of domain in km
    bottom(x, y) = _shoal_bottom(x / 1e3, y / 1e3, y0_km, σ, Hs)
    return bottom
end
