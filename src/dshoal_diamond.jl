"""
Diamond Shoals-shaped bathymetry, optionally with a land cape.

A separate bathymetry from dshoal_vn_param.jl, built from measurements of the
real shoal (MATLAB/shoal_stuff/diamond_shoals_geometry.m on GEBCO 2026; see
diamond_shoals_model_domain.m for the comparison figures). Selected in
flow_over_shoals_sweep.jl with SWEEP_BATHY = "diamond" (`bathy = "diamond"`
in a sweep table).

COORDINATES. x is cross-shore (the coast, or the cape's tip, is where the
shoal starts) and y along-shore, as in every run. The shoal runs out along x
from the cape tip at x = cape_length, centred on y = y0.

THE COAST.
  cape_length = 0   a straight coast at x = 0 (the domain wall), as in
                    dshoal_vn_param.jl.
  cape_length > 0   a land CAPE: the coast is x = 0 far from the shoal, and
                    within cape_length / tan(cape_angle) of y0 it comes out
                    at cape_angle to the y axis to a point at
                    (x, y) = (cape_length, y0). The shoal starts at that
                    point. At Cape Hatteras the shoal axis bisects a 90 degree
                    corner of the coast, so each coast makes 45 degrees with
                    it: cape_angle = 45.
                    Land is bottom = 0 (the whole column immersed).

THE BACKGROUND follows the distance d to the NEAREST coast, not x: a
shoreface ramp from -5 m at the coast to Zsh over ~shoreface_width, then the
flat shelf, then the shelf break at x = cape_length + break_distance (an
absolute x, the same everywhere along the coast). With a cape, the shelf is
therefore narrow at the tip (break_distance) and wide far from it
(cape_length + break_distance): Diamond Shoals' shelf is ~22 km wide along
the shoal and ~45 km wide north of Hatteras.

THE SHOAL, with s = x - cape_length the distance along it from the tip:
  crest target    Zs at s = 0, sloping linearly to tip_depth at
                  s = crest_length (Diamond Shoals: -1 to -3 m at the cape,
                  -13 m at the end; the model keeps every water column at
                  least 5 m deep)
  offshore end    a cosine ramp over taper_length (3 km: the real shoal stops
                  at the shelf break; dshoal_vn_param.jl's fixed 15 km cannot)
  inshore of tip  (cape only) the same ramp, so the shoal does not fill the
                  corners of the bay beside the cape
  along-shore     Gaussian, sigma (measured 4.6 km)
The shoal fills the background up to the crest target, as in
dshoal_vn_param.jl, so it never makes the bottom deeper.

Every water column is at least |h_coast| deep (default 5 m, as in
dshoal_vn_param.jl); h_coast is also the depth at the shoreline. The LES
(src/les_shoal.jl) scales it with the rest of the vertical.
"""

# ── helpers (GPU-safe scalar code) ──────────────────────────────────────

@inline ds_step(x, x0, w) = 0.5 * (1.0 + tanh((x - x0) / w))

# distance from (x, y) to the segment (ax, ay)-(bx, by)
@inline function ds_segment_distance(x, y, ax, ay, bx, by)
    dx, dy = bx - ax, by - ay
    t = clamp(((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy), 0.0, 1.0)
    return hypot(x - (ax + t * dx), y - (ay + t * dy))
end

# x of the coast at y (the land is x < coast_x)
@inline ds_coast_x(y, y0, cape_length, half_base) =
    cape_length <= 0 ? 0.0 : max(0.0, cape_length * (1.0 - abs(y - y0) / half_base))

# distance to the nearest coast, for a point in the water
@inline function ds_coast_distance(x, y, y0, cape_length, half_base)
    cape_length <= 0 && return x
    d_left  = ds_segment_distance(x, y, 0.0, y0 - half_base, cape_length, y0)
    d_right = ds_segment_distance(x, y, 0.0, y0 + half_base, cape_length, y0)
    # the straight coast x = 0 beyond the cape's base
    d_wall = abs(y - y0) >= half_base ? x : hypot(x, abs(y - y0) - half_base)
    return min(d_left, d_right, d_wall)
end

# shoreface: h_coast at the coast (exactly, at d = 0), Zsh by ~shoreface_width
@inline function ds_shoreface(d, Zsh, shoreface_width, h_coast)
    mid, w = 0.5 * shoreface_width, shoreface_width / 5.6
    s0 = ds_step(0.0, mid, w)
    s = (ds_step(d, mid, w) - s0) / (1.0 - s0)
    return h_coast + (Zsh - h_coast) * s
end

@inline function ds_background(x, y, y0, cape_length, half_base, Zsh, shoreface_width,
                               x_break, rise_width, h_deep, h_coast)
    d = ds_coast_distance(x, y, y0, cape_length, half_base)
    return ds_shoreface(d, Zsh, shoreface_width, h_coast) + (h_deep - Zsh) * ds_step(x, x_break, rise_width)
end

@inline function _ds_bottom(x, y, y0, cape_length, half_base, Zs, tip_depth, crest_length,
                            taper_length, sigma, Zsh, shoreface_width, x_break, rise_width, h_deep, h_coast)
    # land
    x < ds_coast_x(y, y0, cape_length, half_base) && return 0.0

    hw = ds_background(x, y, y0, cape_length, half_base, Zsh, shoreface_width, x_break, rise_width, h_deep, h_coast)

    s = x - cape_length                                  # along the shoal from the tip
    # crest target: Zs at the tip, tip_depth at the end, held under the taper
    frac = clamp(s / crest_length, 0.0, 1.0)
    target = Zs + (tip_depth - Zs) * frac

    # offshore taper beyond the crest; with a cape, the same taper inshore of
    # the tip (beside the cape, where the water is the bay, not the shoal)
    if s <= crest_length
        taper_off = 1.0
    elseif s >= crest_length + taper_length
        taper_off = 0.0
    else
        taper_off = 0.5 * (1.0 + cos(π * (s - crest_length) / taper_length))
    end
    if cape_length <= 0 || s >= 0.0
        taper_in = 1.0
    elseif s <= -taper_length
        taper_in = 0.0
    else
        taper_in = 0.5 * (1.0 + cos(π * s / taper_length))
    end

    gauss_y = exp(-((y - y0)^2) / (2 * sigma^2))
    potential = max(0.0, target - hw)
    return min(h_coast, hw + potential * taper_off * taper_in * gauss_y)
end

"""
    dshoal_diamond_bottom(Ly; kwargs...)

`bottom(x, y)` for GridFittedBottom. Keywords (defaults = GEBCO 2026
measurements of Diamond Shoals):

  Zs = -5.0               crest depth at the cape tip (model floor -5 m)
  tip_depth = -13.0       crest depth at the end of the crest
  crest_length = 17e3     length of the crest from the tip
  taper_length = 3e3      cosine ramp from the crest's end down to the shelf
  sigma = 4.6e3           along-shore Gaussian width
  Zsh = -24.0             shelf depth
  shoreface_width = 10e3  the coast-to-shelf ramp (-5 m to Zsh)
  break_distance = 22e3   shelf break, measured from the cape tip along x
  rise_width = 2e3        width of the break (tanh scale)
  h_deep = -50.0          depth offshore of the break (the domain floor)
  cape_length = 0.0       0: straight coast. > 0: a land cape this far out
  cape_angle = 45.0       angle of the cape's coasts to the y axis (degrees)
  h_coast = -5.0          depth at the shoreline, and the shallowest water anywhere
  shoal_y0 = Ly / 2       along-shore centre of the shoal (and the cape)
"""
function dshoal_diamond_bottom(Ly;
    Zs=-5.0, tip_depth=-13.0, crest_length=17e3, taper_length=3e3, sigma=4.6e3,
    Zsh=-24.0, shoreface_width=10e3, break_distance=22e3, rise_width=2e3, h_deep=-50.0,
    cape_length=0.0, cape_angle=45.0, h_coast=-5.0, shoal_y0=nothing)

    y0 = isnothing(shoal_y0) ? Ly / 2 : Float64(shoal_y0)
    cape_length = Float64(cape_length)
    half_base = cape_length > 0 ? cape_length / tand(cape_angle) : 1.0
    x_break = cape_length + break_distance

    cape_length > 0 && (y0 - half_base < 0 || y0 + half_base > Ly) &&
        error("the cape's base ($(2 * half_base / 1e3) km wide at y0 = $(y0 / 1e3) km) does not fit in Ly = $(Ly / 1e3) km")
    crest_length + taper_length > break_distance + 2 * rise_width &&
        @warn "the shoal's taper ends past the shelf break"

    bottom(x, y) = _ds_bottom(x, y, y0, cape_length, half_base, Float64(Zs), Float64(tip_depth),
                              Float64(crest_length), Float64(taper_length), Float64(sigma),
                              Float64(Zsh), Float64(shoreface_width), x_break, Float64(rise_width),
                              Float64(h_deep), Float64(h_coast))
    return bottom
end

"""
    diamond_shelf_break_x(; break_distance=22e3, cape_length=0.0)

Absolute x of the shelf break, for placing the inflow jet's offshore edge.
"""
diamond_shelf_break_x(; break_distance=22e3, cape_length=0.0) = cape_length + break_distance
