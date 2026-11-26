function [h, geom] = shoal_bathymetry(x, y, p)
% [h, geom] = shoal_bathymetry(x, y, p)
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
%   h_deep (-50)  Ly (200e3)  shoal_y0 (Ly/2): the shoal's alongshore centre
% Hs (shoal height above the shelf) is accepted instead of Zs, as in
% sweep_driver.jl; so is shelf_depth instead of Zsh.
%
% ── SHOAL DEPTH AND LENGTH: what the parameters actually mean ────────────
% PROPOSED, not yet in the Julia model: approve before it is ported.
%
% The model's `shoal_length` is NOT the shoal's length. It is where the flat
% crest ENDS, measured from the coast. An attached shoal's crest starts where
% the coastal slope first reaches Zs, so its real length is
% shoal_length - x_start(Zs), and x_start moves with Zs (0 km at Zs = -5,
% 6.5 km at -10, 8.2 km at -15 on the standard background). A sweep that
% "holds Ls = 40 km" therefore had crests of 40, 33.5 and 31.8 km.
%
%   p.crest_length   (new, optional) the length of the flat crest [m]. When
%                    given, the crest is placed from x_start(Zs) -- solved
%                    exactly on the background -- to x_start + crest_length,
%                    i.e. shoal_length is DERIVED, not chosen. The shoal stays
%                    attached to the coast, so no trench is possible. Use this
%                    for any sweep that varies Zs and claims a fixed length.
%
%   p.min_shelf_gap  (new, optional, default 10e3) a CHECK, not a control:
%                    the least flat shelf there must be between the end of
%                    the shoal's 15 km offshore taper and the start of the
%                    shelf break [m]. geom.shelf_gap is measured and flagged
%                    if smaller. The break itself stays where rise_mid puts
%                    it: one fixed break per sweep, because the inflow jet's
%                    offshore edge (65 km) and the analysis box are fixed in
%                    x, so moving the break per run would change how the jet
%                    sits over the topography. Pick that one rise_mid with
%                    tools/fixed_break_rise_mid.m.
%
%   p.tip_depth      (new, optional) a crest that DEEPENS SEAWARD instead of a
%                    flat plateau: Zs at the start of the crest, sloping
%                    linearly to tip_depth at its end (shoal_length), then
%                    the usual 15 km taper to the shelf. Closer to a real
%                    cape shoal (shallow near the cape, a deeper platform
%                    further out) than a long flat top, which ends in one
%                    abrupt corner. Attached shoals only. Omit for a flat
%                    crest (as before).
%
% Second output, `geom`, MEASURES the shoal from the bottom itself and checks
% it against what was declared (computed only when asked for):
%   crest_depth      shallowest depth on the shoal centreline, y = Ly/2   [m]
%   crest_start/end  where the centreline is within tol of crest_depth  [m]
%   crest_length     crest_end - crest_start                              [m]
%   height           crest_depth - Zsh: height above the shelf            [m]
%   width_y          alongshore full width at half height, mid-crest      [m]
%   trench           dip below both sides between coast and crest end     [m]
%   taper_end        where the 15 km offshore taper reaches the shelf      [m]
%   break_start      where the shelf break begins (background 1 m below Zsh) [m]
%   shelf_gap        break_start - taper_end: flat shelf left offshore of
%                    the shoal. Negative = the taper runs into the break,
%                    so the shoal's offshore flank and the break merge.    [m]
%   declared_depth / declared_length   what the parameters claim
%   depth_ok / length_ok / trench_ok   pass/fail (tolerances below)
%   note             why a check failed, in words
% The flat crest is "within tol = 0.01 m of crest_depth"; the 15 km cosine
% taper is flat at its start, so a measured crest runs ~0.2 km past the
% declared end. length_ok allows 0.5 km.
%
% Keep in step with src/dshoal_vn_param.jl if that file changes.

p = defaults_(p);
y0 = p.shoal_y0;  half_extent = p.Ly / 2;

% Crest length declared directly: derive where the crest has to end.
if isfield(p, 'crest_length') && ~isempty(p.crest_length)
    if p.shoal_start > 0
        error('shoal_bathymetry:crestLength', ...
              'crest_length is for shoals attached to the coast (shoal_start = 0).');
    end
    p.shoal_length = crest_start_(p) + p.crest_length;
end

hw = background_(x, p);

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
potential_height = max(0, target_(x, p) - hw);
h = min(-5.0, hw + potential_height .* factor);

if nargout > 1
    geom = measure_(p);
end
end

% ═══════════════════════════════════════════════════════════════════════════

function hw = background_(x, p)
% Shelf without the shoal: coastal ramp, shelf break, offshore ramp.
ss = @(x, x0, w) 0.5 * (1 + tanh((x - x0) ./ w));
h0 = -5.0;  h1 = -7.0;
hw = h0 + (h1 - h0) * ss(x, 2.5e3, 1.5e3);
sb_mid = 5.0e3 + (p.shelf_break_end - 5.0e3) / 2;
sb_width = (p.shelf_break_end - 5.0e3) / 2.8;
hw = hw + (p.Zsh - h1) * ss(x, sb_mid, sb_width);
hw = hw + (p.h_deep - p.Zsh) * ss(x, p.rise_mid, p.rise_width);
end

function z = target_(x, p)
% The depth the shoal fills up to: Zs everywhere (flat crest), or, with
% tip_depth, Zs at the crest start sloping linearly to tip_depth at
% shoal_length and held there under the taper.
if ~isfield(p, 'tip_depth') || isempty(p.tip_depth)
    z = p.Zs;
    return;
end
if p.shoal_start > 0
    error('shoal_bathymetry:tipDepth', 'tip_depth is for shoals attached to the coast (shoal_start = 0).');
end
x0 = crest_start_(p);
s = min(max((x - x0) / max(p.shoal_length - x0, 1), 0), 1);
z = p.Zs + (p.tip_depth - p.Zs) * s;
end

function xs = crest_start_(p)
% Where an attached shoal's flat crest begins: the first x at which the
% background is as deep as Zs. Zero if the coast is already that deep.
f = @(x) background_(x, p) - p.Zs;
if f(0) <= 0, xs = 0; return; end
xs = fzero(f, [0, p.rise_mid]);
end

function g = measure_(p)
% Measure the shoal from the bottom itself, along the centreline y = Ly/2.
tol = 0.01;  y0 = p.shoal_y0;
pc = rmfield_(p, 'crest_length');                % p.shoal_length is already final
x = 0:10:(p.shoal_length + 25e3);
hc = shoal_bathymetry(x, y0, pc);
% Anchor on the SHOAL, not on the shallowest point: an attached shoal deeper
% than the coast has the coast as its shallowest point. Find where the shoal
% stands highest above the background shelf; the bottom there is the crest,
% and the crest extends either way while the bottom stays within tol of it.
eta = hc - background_(x, p);
[eta_max, imax] = max(eta);
if eta_max < 0.5                                 % Zs at or below the shelf: nothing built
    g = struct('crest_depth', NaN, 'crest_start', NaN, 'crest_end', NaN, 'crest_length', 0, ...
               'height', 0, 'width_y', 0, 'trench', 0, 'taper_end', NaN, 'break_start', NaN, ...
               'shelf_gap', NaN, 'gap_ok', true, 'declared_depth', p.Zs, ...
               'declared_length', p.shoal_length, 'depth_ok', true, 'length_ok', true, ...
               'trench_ok', true, 'note', '(no shoal: Zs is at or below the shelf)');
    return;
end
if isfield(p, 'tip_depth') && ~isempty(p.tip_depth)
    % Sloping crest: the crest is where the bottom is at or ABOVE the sloping
    % line from Zs (crest start) to tip_depth (shoal_length). Near the coast
    % the coastal ramp can be shallower than that line, and the bottom follows
    % the ramp there (shallower, so no trench); that still counts as crest.
    on = hc >= target_(x, p) - tol & x <= p.shoal_length;
    i0 = find(on & x >= crest_start_(p) - 10, 1);
    i1 = i0;  while i1 < numel(x) && on(i1+1), i1 = i1 + 1; end
    g.crest_depth = hc(i0);
    g.tip_depth = hc(i1);
else
    g.crest_depth = hc(imax);
    on = abs(hc - g.crest_depth) <= tol;
    i0 = imax;  while i0 > 1 && on(i0-1), i0 = i0 - 1; end
    i1 = imax;  while i1 < numel(x) && on(i1+1), i1 = i1 + 1; end
    g.tip_depth = g.crest_depth;
end
g.crest_start = x(i0);  g.crest_end = x(i1);
g.crest_length = g.crest_end - g.crest_start;
g.height = g.crest_depth - p.Zsh;

% alongshore full width at half height, at the middle of the crest
xm = 0.5 * (g.crest_start + g.crest_end);
yy = 0:50:p.Ly;
eta = shoal_bathymetry(xm * ones(size(yy)), yy, pc) - background_(xm, p);
half = eta >= 0.5 * max(eta);
g.width_y = (find(half, 1, 'last') - find(half, 1, 'first')) * 50;

% trench between the coast and the end of the crest
zt = hc(x <= g.crest_end);
g.trench = max(min(cummax(zt), flip(cummax(flip(zt)))) - zt);

% room between the shoal and the shelf break
g.taper_end = p.shoal_length + 15e3;
xb = g.crest_end:50:(p.rise_mid + 4 * p.rise_width);
ib = find(background_(xb, p) < p.Zsh - 1, 1);
if isempty(ib), g.break_start = NaN; else, g.break_start = xb(ib); end
g.shelf_gap = g.break_start - g.taper_end;

% declared vs measured
g.declared_depth = p.Zs;
if isfield(p, 'crest_length') && ~isempty(p.crest_length)
    g.declared_length = p.crest_length;
else
    g.declared_length = p.shoal_length;          % what the old parameter implies
end
g.depth_ok  = abs(g.crest_depth - g.declared_depth) <= tol;
g.length_ok = abs(g.crest_length - g.declared_length) <= 500;
g.trench_ok = g.trench <= 1;
notes = {};
if ~g.depth_ok
    if p.Zs > -5
        notes{end+1} = sprintf('Zs = %g is shallower than the -5 m floor the model clamps to', p.Zs);
    else
        notes{end+1} = sprintf('crest is %.2f m, declared %.2f m', g.crest_depth, g.declared_depth);
    end
end
if ~g.length_ok
    notes{end+1} = sprintf('crest is %.1f km long, declared %.1f km (crest starts at %.1f km)', ...
                           g.crest_length / 1e3, g.declared_length / 1e3, g.crest_start / 1e3);
end
if ~g.trench_ok, notes{end+1} = sprintf('%.1f m trench between coast and shoal', g.trench); end
if isnan(g.break_start)
    notes{end+1} = 'no shelf break (h_deep = Zsh)';
elseif g.shelf_gap < 0
    notes{end+1} = sprintf('offshore taper runs %.1f km into the shelf break', -g.shelf_gap / 1e3);
elseif g.shelf_gap < p.min_shelf_gap - 100          % 100 m: the break is located on a 50 m grid
    notes{end+1} = sprintf('only %.1f km of flat shelf before the break (minimum %.1f km)', ...
                           g.shelf_gap / 1e3, p.min_shelf_gap / 1e3);
end
g.gap_ok = ~isnan(g.shelf_gap) && g.shelf_gap >= p.min_shelf_gap - 100;
g.note = strjoin(notes, '; ');
end

function p = rmfield_(p, f)
if isfield(p, f), p = rmfield(p, f); end
end

function p = defaults_(p)
d = struct('Zs', NaN, 'shoal_length', 40e3, 'sigma', 8e3, 'Zsh', -25, 'shelf_break_end', 12e3, ...
           'rise_mid', 63.5e3, 'rise_width', 5e3, 'shoal_start', 0, 'shoal_inner_width', 0, ...
           'h_deep', -50, 'Ly', 200e3, 'min_shelf_gap', 10e3);
if isfield(p, 'shelf_depth') && ~isfield(p, 'Zsh'), p.Zsh = p.shelf_depth; end
for f = fieldnames(d)'
    if ~isfield(p, f{1}), p.(f{1}) = d.(f{1}); end
end
if ~isfield(p, 'shoal_y0') || isempty(p.shoal_y0), p.shoal_y0 = p.Ly / 2; end
if isnan(p.Zs)
    if isfield(p, 'Hs'), p.Zs = p.Zsh + p.Hs; else, p.Zs = -5; end
end
end
