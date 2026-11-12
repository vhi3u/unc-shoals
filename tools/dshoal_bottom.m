function z = dshoal_bottom(x, y, varargin)
% DSHOAL_BOTTOM  Shoal-shelf bathymetry: MATLAB twin of the model's Julia.
%
%   z = dshoal_bottom(x, y)                       % reference case
%   z = dshoal_bottom(x, y, 'Zs', -15, 'Ls', 30e3)
%   [X,Y] = meshgrid(xv, yv); Z = dshoal_bottom(X, Y, ...);
%
% x and y may be arrays of the same size, or y may be scalar. Returns the
% bottom depth [m], negative down.
%
% Options (name/value), with the Julia defaults:
%   'Ly'        domain length in y [m]                       200e3
%   'Zs'        absolute depth of the shoal crest [m]        -5
%   'Ls'        shoal length before the taper starts [m]     40e3
%   'sigma'     shoal half-width in y (Gaussian) [m]         8e3
%   'Zsh'       shelf depth [m]                              -25
%   'shelf_break_end'  offshore end of the shelf break [m]   12e3
%   'Ly_shoal'  along-shore extent of the shoal [m]          Ly
%   'Zc'        depth at the coast, x = 0 [m]                -5
%   'coastal_drop'     drop across the coastal ramp [m]      2
%   'min_depth' shallowest bottom allowed [m]          max(Zc, Zs)
%
% Zc / coastal_drop / min_depth exist only in the _coastal Julia variant.
% src/dshoal_vn_param.jl -- the one flow_over_shoals_sweep.jl actually calls
% -- hardcodes them to -5, 2 and -5. The two are bit-identical at those
% values (verified across Zs, Zsh and shelf_break_end), so leaving them alone
% reproduces the model exactly; changing Zc explores a geometry the model
% cannot currently run without adding that argument on the Julia side.
%
% See also DSHOAL_SELFTEST, BATHY_EXPLORER, PLOT_BATHY_COMPARE.

p = inputParser;
addParameter(p, 'Ly', 200e3);
addParameter(p, 'Zs', -5.0);
addParameter(p, 'Ls', 40e3);
addParameter(p, 'sigma', 8e3);
addParameter(p, 'Zsh', -25.0);
addParameter(p, 'shelf_break_end', 12.0e3);
addParameter(p, 'Ly_shoal', []);
addParameter(p, 'Zc', -5.0);
addParameter(p, 'coastal_drop', 2.0);
addParameter(p, 'coastal_mid', 2.5e3);
addParameter(p, 'coastal_width', 1.5e3);
addParameter(p, 'min_depth', []);
addParameter(p, 'h_deep', -50.0);
parse(p, varargin{:});
o = p.Results;

if isempty(o.Ly_shoal),  o.Ly_shoal  = o.Ly;                 end
if isempty(o.min_depth), o.min_depth = max(o.Zc, o.Zs);      end
if isscalar(y), y = repmat(y, size(x)); end

y0 = o.Ly / 2.0;
half_extent = o.Ly_shoal / 2.0;

% ── Background: coastal ramp, shelf break, offshore ramp ────────────────
h0 = o.Zc;
h1 = o.Zc - o.coastal_drop;
hw = h0 + (h1 - h0) .* smooth_step(x, o.coastal_mid, o.coastal_width);
sb_mid   = 5.0e3 + (o.shelf_break_end - 5.0e3) / 2.0;
sb_width = (o.shelf_break_end - 5.0e3) / 2.8;
hw = hw + (o.Zsh - h1)       .* smooth_step(x, sb_mid, sb_width);
hw = hw + (o.h_deep - o.Zsh) .* smooth_step(x, 63.5e3, 5.0e3);

% ── Along-shore compact window ──────────────────────────────────────────
window = ones(size(y));
if half_extent < y0
    dy = abs(y - y0);
    taper_start = 0.5 * half_extent;
    window(dy >= half_extent) = 0.0;
    m = (dy > taper_start) & (dy < half_extent);
    window(m) = 0.5 * (1.0 + cos(pi * (dy(m) - taper_start) ./ ...
                                (half_extent - taper_start)));
end

% ── Offshore taper of the shoal ─────────────────────────────────────────
x_taper_start = o.Ls;
x_taper_end   = o.Ls + 15.0e3;
taper = ones(size(x));
taper(x >= x_taper_end) = 0.0;
m = (x > x_taper_start) & (x < x_taper_end);
taper(m) = 0.5 * (1.0 + cos(pi * (x(m) - x_taper_start) ./ ...
                            (x_taper_end - x_taper_start)));

% ── Fill the background up to the crest level, then clamp ───────────────
gauss_y = exp(-((y - y0).^2) ./ (2 * o.sigma^2));
factor  = taper .* gauss_y .* window;
z = min(o.min_depth, hw + max(0.0, o.Zs - hw) .* factor);
end

function s = smooth_step(x, x0, w)
s = 0.5 * (1.0 + tanh((x - x0) / w));
end
