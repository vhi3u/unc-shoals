"""
GPU-compatible parameterised shoal bathymetry with an ADJUSTABLE COASTAL RAMP.

Copy of dshoal_vn_param.jl with the nearshore made movable in the vertical.
In the original, two separate things pinned the coast at 5 m depth:

  1. `h0 = -5.0` hard-coded as the depth at x = 0, and
  2. the `min(-5.0, ...)` clamp at the end of the bottom function.

Changing Zs or Zsh moved the shoal and the shelf but left the 0-5 km ramp
exactly where it was, leaving a very shallow nearshore strip that funnels the
alongshore flow into a coastal jet.

Both are now parameters, and they move together:

  Zc            depth at the coast, x = 0            (was fixed at -5 m)
  coastal_drop  drop across the 0-5 km ramp, metres  (was fixed at 2 m)

The ramp SHAPE is preserved: the drop and its horizontal extent are unchanged
unless you ask otherwise, so lowering Zc translates the ramp downward at
constant slope rather than steepening or stretching it.

Background geometry:
- 0 to ~5 km:                  coastal ramp (Zc to Zc - coastal_drop)
- 5 km to shelf_break_end:     shelf break (Zc - coastal_drop to Zsh)
- shelf_break_end to ~62 km:   flat shelf at Zsh
- 62 to 65 km:                 offshore ramp (Zsh to -50 m)
- beyond 65 km:                flat offshore (-50 m)

Shoal geometry:
- Zs: absolute depth of the peak of the shoal
- shoal_length: length of the elevated section
- taper: smooth cosine ramp over 15 km at the offshore end

NOTE: the public constructor is `dshoal_param_bottom_coastal`, deliberately a
different name from `dshoal_param_bottom` in dshoal_vn_param.jl, so including
both files cannot silently give you the wrong geometry.
"""

# ── Helper functions for smoothness ─────────────────────────────────────

@inline smooth_step_c(x, x0, w) = 0.5 * (1.0 + tanh((x - x0) / w))

# ── Background bathymetry ───────────────────────────────────────────────

@inline function param_background_depth_c(x, Zsh, shelf_break_end,
    Zc, coastal_drop, coastal_mid, coastal_width)
    h_deep = -50.0

    h0 = Zc                      # depth at the coast
    h1 = Zc - coastal_drop       # depth at the foot of the coastal ramp

    # 1. Coastal ramp. Translating Zc moves h0 and h1 together, so (h1 - h0)
    #    and the ramp's horizontal extent are unchanged: same slope, new depth.
    h = h0 + (h1 - h0) * smooth_step_c(x, coastal_mid, coastal_width)

    # 2. Shelf break (5 km to shelf_break_end)
    sb_mid = 5.0e3 + (shelf_break_end - 5.0e3) / 2.0
    sb_width = (shelf_break_end - 5.0e3) / 2.8
    h += (Zsh - h1) * smooth_step_c(x, sb_mid, sb_width)

    # 3. Offshore ramp (62-65 km)
    h += (h_deep - Zsh) * smooth_step_c(x, 63.5e3, 5.0e3)

    return h
end

# ── Along-shore compact window ──────────────────────────────────────────

@inline function param_shoal_window_c(y, y0, half_extent)
    if half_extent >= y0
        return 1.0
    end
    dy = abs(y - y0)
    if dy >= half_extent
        return 0.0
    else
        taper_start = 0.5 * half_extent
        if dy <= taper_start
            return 1.0
        else
            return 0.5 * (1.0 + cos(π * (dy - taper_start) / (half_extent - taper_start)))
        end
    end
end

# ── Combined bottom function ────────────────────────────────────────────

@inline function _param_shoal_bottom_c(x, y, y0, sigma, Zs, half_extent, shoal_length,
    Zsh, shelf_break_end, Zc, coastal_drop, coastal_mid, coastal_width, min_depth)

    hw = param_background_depth_c(x, Zsh, shelf_break_end,
        Zc, coastal_drop, coastal_mid, coastal_width)

    window = param_shoal_window_c(y, y0, half_extent)

    x_taper_start = shoal_length
    x_taper_end = shoal_length + 15.0e3

    if x <= x_taper_start
        taper = 1.0
    elseif x >= x_taper_end
        taper = 0.0
    else
        taper = 0.5 * (1.0 + cos(π * (x - x_taper_start) / (x_taper_end - x_taper_start)))
    end

    gauss_y = exp(-((y - y0)^2) / (2 * sigma^2))
    factor = taper * gauss_y * window

    potential_height = max(0.0, Zs - hw)

    # The shallow-water clamp. In the original this was hard-coded to -5.0,
    # which silently re-pinned the coast to 5 m however Zc was set. It now
    # tracks the shallowest feature actually requested.
    return min(min_depth, hw + potential_height * factor)
end

# ── Public constructor ──────────────────────────────────────────────────

"""
    dshoal_param_bottom_coastal(Ly; kwargs...)

Return a function `bottom(x, y)` for `GridFittedBottom`, with the coastal ramp
free to move in the vertical.

Keyword arguments
=================
- `Zs`: absolute depth of the shoal peak. Default `-5.0`.
- `Zsh`: absolute depth of the shelf. Default `-25.0`.
- `shoal_length`: length of the elevated section before the 15 km taper.
- `sigma`: along-shore Gaussian width of the shoal.
- `shelf_break_end`: offshore end of the shelf break.
- `Zc`: **depth at the coast (x = 0)**. Default `-5.0`, which reproduces
  dshoal_vn_param.jl exactly. Make it more negative to drop the nearshore.
- `coastal_drop`: drop in metres across the coastal ramp (positive deepens
  offshore). Default `2.0`, i.e. the original -5 m to -7 m.
- `coastal_mid`, `coastal_width`: centre and width of the coastal ramp's
  tanh. Defaults `2.5e3`, `1.5e3` reproduce the original. Leave them alone to
  preserve the slope while changing `Zc`.
- `min_depth`: shallowest bottom allowed. Defaults to `max(Zc, Zs)`, i.e. the
  shallower of the coast and the shoal crest, which is almost always what you
  want. Pass a value only to override.

Notes
=====
Setting `Zc` deeper than `Zsh` inverts the shelf break (the bottom rises
offshore of the coast). That is permitted but is probably not what you mean.
"""
function dshoal_param_bottom_coastal(Ly;
    sigma=8e3,
    Zs=-5.0,
    shoal_length=40e3,
    Ly_shoal=Ly,
    Zsh=-25.0,
    shelf_break_end=12.0e3,
    Zc=-5.0,
    coastal_drop=2.0,
    coastal_mid=2.5e3,
    coastal_width=1.5e3,
    min_depth=nothing)

    y0 = Ly / 2.0
    half_extent = Ly_shoal / 2.0

    # Shallowest requested feature: the coast or the shoal crest, whichever is
    # closer to the surface. Depths are negative, so that is a `max`.
    md = isnothing(min_depth) ? max(Zc, Zs) : min_depth

    if Zc < Zsh
        @warn "dshoal_param_bottom_coastal: Zc ($Zc) is deeper than Zsh ($Zsh); the shelf break now rises offshore."
    end

    bottom(x, y) = _param_shoal_bottom_c(x, y, y0, sigma, Zs, half_extent, shoal_length,
        Zsh, shelf_break_end, Zc, coastal_drop, coastal_mid, coastal_width, md)
    return bottom
end
