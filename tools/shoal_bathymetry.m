function h = shoal_bathymetry(x, y, p)
% h = shoal_bathymetry(x, y, p)
%
% Bottom depth [m, negative down] of the model's shoal bathymetry: a line-by-
% line MATLAB port of src/dshoal_vn_param.jl (_param_shoal_bottom,
% param_background_depth, param_shoal_window), the function
% flow_over_shoals_sweep.jl builds its grid from. x, y in metres, any
% matching sizes (or y scalar).
%
% p is a struct of run parameters; missing fields take the sweep defaults:
%   Zs (-5)  shoal_length (40e3)  sigma (8e3)  Zsh (-25)  shelf_break_end (12e3)
%   rise_mid (63.5e3)  rise_width (5e3)  shoal_start (0)  shoal_inner_width (0)
%   h_deep (-50)  Ly (200e3)
% Hs (shoal height above the shelf) is accepted instead of Zs, as in
% sweep_driver.jl; so is shelf_depth instead of Zsh.
%
% Keep in step with src/dshoal_vn_param.jl if that file changes.

p = defaults_(p);
y0 = p.Ly / 2;  half_extent = p.Ly / 2;

% 1. background: coastal ramp, shelf break, offshore ramp
ss = @(x, x0, w) 0.5 * (1 + tanh((x - x0) ./ w));
h0 = -5.0;  h1 = -7.0;
hw = h0 + (h1 - h0) * ss(x, 2.5e3, 1.5e3);
sb_mid = 5.0e3 + (p.shelf_break_end - 5.0e3) / 2;
sb_width = (p.shelf_break_end - 5.0e3) / 2.8;
hw = hw + (p.Zsh - h1) * ss(x, sb_mid, sb_width);
hw = hw + (p.h_deep - p.Zsh) * ss(x, p.rise_mid, p.rise_width);

% 2. alongshore window (1 everywhere when the shoal spans the domain)
if half_extent >= y0
    window = ones(size(y));
else
    dy = abs(y - y0);  ts = 0.5 * half_extent;
    window = (dy <= ts) + (dy > ts & dy < half_extent) .* 0.5 .* (1 + cos(pi * (dy - ts) / (half_extent - ts)));
end

% 3a. offshore taper: crest out to shoal_length, then 15 km cosine ramp
xt0 = p.shoal_length;  xt1 = p.shoal_length + 15e3;
taper_off = (x <= xt0) + (x > xt0 & x < xt1) .* 0.5 .* (1 + cos(pi * (x - xt0) / (xt1 - xt0)));

% 3b. shoreward edge (shoal_start = 0: attached to the coast)
if p.shoal_start <= 0
    taper_in = ones(size(x));
else
    xi1 = p.shoal_start;  xi0 = p.shoal_start - p.shoal_inner_width;
    taper_in = (x >= xi1) + (x > xi0 & x < xi1) .* 0.5 .* (1 - cos(pi * (x - xi0) / (xi1 - xi0)));
end

% 4. alongshore Gaussian; 5. fill up to Zs, never shallower than -5 m
gauss_y = exp(-((y - y0).^2) / (2 * p.sigma^2));
factor = taper_in .* taper_off .* gauss_y .* window;
potential_height = max(0, p.Zs - hw);
h = min(-5.0, hw + potential_height .* factor);
end

function p = defaults_(p)
d = struct('Zs', NaN, 'shoal_length', 40e3, 'sigma', 8e3, 'Zsh', -25, 'shelf_break_end', 12e3, ...
           'rise_mid', 63.5e3, 'rise_width', 5e3, 'shoal_start', 0, 'shoal_inner_width', 0, ...
           'h_deep', -50, 'Ly', 200e3);
if isfield(p, 'shelf_depth') && ~isfield(p, 'Zsh'), p.Zsh = p.shelf_depth; end
for f = fieldnames(d)'
    if ~isfield(p, f{1}), p.(f{1}) = d.(f{1}); end
end
if isnan(p.Zs)
    if isfield(p, 'Hs'), p.Zs = p.Zsh + p.Hs; else, p.Zs = -5; end
end
end
