function inflow_diagnostics(runs, varargin)
% inflow_diagnostics(runs, ...)
%
% What flow does the shoal actually see, and how disturbed is it? Two
% questions about periodic vs open-y runs:
%
%   1. SPEED. Open runs pin the alongshore transport at the target Q_in (the
%      full v∞ jet). Periodic runs only nudge toward it in the 20 km south
%      strip, so bottom and shoal drag can leave the flow reaching the shoal
%      well below v0. Measured as the transport through a section upstream
%      of the shoal (from time_avg_3d, 10-day means) divided by Q_in, and the
%      daily surface speed over the shelf on that section (from top_*.nc).
%
%   2. SEED. A periodic run's wake re-enters through the south strip, which
%      damps it to a few percent but not to zero; an open run's inflow is
%      exactly v∞. The shoal's shear layer grows from whatever disturbance
%      arrives, so its size sets how soon eddies appear. Measured as the RMS
%      over the shelf of the FLUCTUATING surface cross-shore velocity on the
%      upstream section (u minus its running 5-day mean, so steady
%      cross-shore flow does not count), daily, and along y (u minus its
%      time mean over the last days).
%
% runs: cell array of run folder names or labels, searched for under the
% sweep_output root (any depth, so cluster folders work), or full paths.
%   inflow_diagnostics({'B_narrow_open_500', 'B_narrow_periodic_500'})
%   inflow_diagnostics({'cluster_Fnnie_1_19/sweep_005_Zs5_Ls40k', 'B_narrow_open_500'})
%
% Options:
%   'root'     sweep_output root (default: derecho scratch, or ../sweep_output)
%   'y_up'     upstream section, km from the south boundary (default 30:
%              10 km past the 20 km south strip)
%   'shelf'    cross-shore range for shelf averages, km (default [5 60])
%   'last'     days at the end averaged for the table (default 30)
%   'name'     figure name (default: the run labels joined)
%
% Writes animations/inflow_<name>.png and prints a table.

ip = inputParser;
scratch = '/glade/derecho/scratch/nguyen/unc-shoals/sweep_output';
here = fileparts(mfilename('fullpath'));
if isfolder(scratch), root0 = scratch; else, root0 = fullfile(here, '..', 'sweep_output'); end
addParameter(ip, 'root', root0);
addParameter(ip, 'y_up', 30);
addParameter(ip, 'shelf', [5 60]);
addParameter(ip, 'last', 30);
addParameter(ip, 'name', '');
parse(ip, varargin{:});
o = ip.Results;
if ischar(runs) || isstring(runs), runs = cellstr(runs); end

nr = numel(runs);
R = cell(1, nr);
for k = 1:nr
    R{k} = one_run_(find_run_(runs{k}, o.root), o);
end

% ── table ────────────────────────────────────────────────────────────────
fprintf('\n%-36s %5s %6s %7s %9s %9s %8s %11s %11s %11s\n', 'run', 'open', 'noise', 'v0', 'Q/Q_in', 'v_up', 'shelf%', 'u''_up', 'u''(y0+20)', 'u''(y0+50)');
fprintf('%s\n', repmat('-', 1, 121));
for k = 1:nr
    r = R{k};
    fprintf('%-36s %5d %6.3f %7.3f %9.2f %9.3f %7.0f%% %11.2e %11.2e %11.2e\n', r.label, r.open, r.noise, r.v0, ...
            r.Qratio_last, r.vup_last, 100 * r.shelf_frac, r.seed_last, r.prof_at(20), r.prof_at(50));
end
fprintf(['\nQ/Q_in   transport through y = %g km (10-day means, last %g days) over the pinned target\n' ...
         'v_up     surface v averaged over x = %g-%g km at y = %g km (m/s), last %g days\n' ...
         'shelf%%   share of that transport inshore of the jet edge - 5 km (60 km for param bathymetry; v_inf profile: %.0f%%)\n' ...
         'u''_up    RMS over the shelf of fluctuating surface u at y = %g km (m/s): the disturbance reaching the shoal\n' ...
         'u''(y0+d) the same, d km downstream of the shoal centre: how far it has grown\n\n'], ...
        o.y_up, o.last, o.shelf, o.y_up, o.last, 100 * R{1}.shelf_frac_inf, o.y_up);

% ── figure ───────────────────────────────────────────────────────────────
f = figure('Color', 'w', 'Position', [50 50 2100 900], 'Visible', 'off');
tl = tiledlayout(f, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
cc = lines(nr);

ax = nexttile(tl); hold(ax, 'on'); grid(ax, 'on');
for k = 1:nr
    r = R{k};
    plot(ax, r.tavg_days, r.Qratio, 'o-', 'Color', cc(k,:), 'LineWidth', 1.5, 'DisplayName', r.label);
end
yline(ax, 1, 'k--', 'HandleVisibility', 'off');
xlabel(ax, 'time (days)'); ylabel(ax, 'Q / Q_{in}');
title(ax, sprintf('alongshore transport through y = %g km (10-day means)', o.y_up));
legend(ax, 'Location', 'best', 'Interpreter', 'none');

ax = nexttile(tl); hold(ax, 'on'); grid(ax, 'on');
for k = 1:nr
    r = R{k};
    plot(ax, r.t_days, r.vup, '-', 'Color', cc(k,:), 'LineWidth', 1.2, 'DisplayName', r.label);
    yline(ax, r.v0, ':', 'Color', cc(k,:), 'HandleVisibility', 'off');
end
xlabel(ax, 'time (days)'); ylabel(ax, 'v (m s^{-1})');
title(ax, sprintf('surface v, mean over x = %g-%g km, at y = %g km (dotted: v_0)', o.shelf, o.y_up));

ax = nexttile(tl); hold(ax, 'on'); grid(ax, 'on');
for k = 1:nr
    r = R{k};
    semilogy(ax, r.t_days, max(r.seed, 1e-12), '-', 'Color', cc(k,:), 'LineWidth', 1.2, 'DisplayName', r.label);
end
set(ax, 'YScale', 'log');
xlabel(ax, 'time (days)'); ylabel(ax, 'RMS u (m s^{-1})');
title(ax, sprintf('disturbance arriving: RMS over the shelf of surface u minus its 5-day mean, y = %g km', o.y_up));

ax = nexttile(tl); hold(ax, 'on'); grid(ax, 'on');
for k = 1:nr
    r = R{k};
    semilogy(ax, r.prof_y, max(r.prof, 1e-12), '-', 'Color', cc(k,:), 'LineWidth', 1.5, 'DisplayName', r.label);
end
set(ax, 'YScale', 'log');
xline(ax, 0, 'k-', 'shoal centre', 'HandleVisibility', 'off');
xlabel(ax, 'y - y_{shoal} (km)'); ylabel(ax, 'RMS u (m s^{-1})');
title(ax, sprintf('fluctuating surface u along y: temporal std, RMS over the shelf, last %g days', o.last));

ax = nexttile(tl); hold(ax, 'on'); grid(ax, 'on');
for k = 1:nr
    r = R{k};
    plot(ax, r.xprof / 1e3, r.vbar, '-', 'Color', cc(k,:), 'LineWidth', 1.5, 'DisplayName', [r.label ' (depth mean)']);
    plot(ax, r.xprof / 1e3, r.vinf, ':', 'Color', cc(k,:), 'LineWidth', 1.2, 'HandleVisibility', 'off');
end
xlabel(ax, 'x (km from coast)'); ylabel(ax, 'v (m s^{-1})');
title(ax, sprintf('depth-mean v across the shelf at y = %g km, last %g days (dotted: v_\\infty)', o.y_up, o.last));

ax = nexttile(tl); hold(ax, 'on'); grid(ax, 'on');
for k = 1:nr
    r = R{k};
    plot(ax, r.xprof / 1e3, cumsum(r.qx) / r.Q_in, '-', 'Color', cc(k,:), 'LineWidth', 1.5);
    plot(ax, r.xprof / 1e3, cumsum(r.qx_inf) / r.Q_in, ':', 'Color', cc(k,:), 'LineWidth', 1.2);
end
yline(ax, 1, 'k--');
xlabel(ax, 'x (km from coast)'); ylabel(ax, 'cumulative transport / Q_{in}');
title(ax, 'where the transport goes: cumulative from the coast (dotted: v_\infty)');

if isempty(o.name)
    labs = cellfun(@(r) r.label, R, 'UniformOutput', false);
    o.name = strjoin(labs, '__');
    if numel(o.name) > 120, o.name = o.name(1:120); end
end
title(tl, 'What reaches the shoal: flow speed and incoming disturbance', 'FontWeight', 'bold');
outdir = fullfile(here, '..', 'animations');
if ~isfolder(outdir), mkdir(outdir); end
png = fullfile(outdir, ['inflow_' o.name '.png']);
exportgraphics(f, png, 'Resolution', 130);
fprintf('Wrote %s\n', png);
end

% ═════════════════════════════════════════════════════════════════════════

function d = find_run_(name, root)
% A full path, a path under root, or a label matched against run folders at
% any depth under root (sweep_NNN_<label>, or any folder name containing it).
if isfolder(name), d = name; return; end
if isfolder(fullfile(root, name)), d = fullfile(root, name); return; end
% cluster/label: the run folder sweep_NNN_<label> inside that cluster
[cl, lab] = fileparts(name);
if ~isempty(cl) && isfolder(fullfile(root, cl))
    c = dir(fullfile(root, cl, ['sweep_*_' lab]));
    c = c([c.isdir]);
    if numel(c) == 1, d = fullfile(c.folder, c.name); return; end
end
c = dir(fullfile(root, '**', ['*' name]));
c = c([c.isdir]);
c = c(~ismember({c.name}, {'.', '..'}));
% prefer exact label matches: sweep_NNN_<name>
exact = ~cellfun(@isempty, regexp({c.name}, ['^sweep_\d+_' regexptranslate('escape', name) '$'], 'once'));
if any(exact), c = c(exact); end
if isempty(c)
    error('inflow_diagnostics: no run folder matching "%s" under %s', name, root);
elseif numel(c) > 1
    fprintf('Several folders match "%s":\n', name);
    for k = 1:numel(c), fprintf('   %s\n', fullfile(c(k).folder, c(k).name)); end
    error('inflow_diagnostics: "%s" is ambiguous; pass cluster/run or a full path', name);
end
d = fullfile(c.folder, c.name);
end

function r = one_run_(d, o)
[~, folder] = fileparts(d);
ftop = dir(fullfile(d, 'top_*.nc'));
favg = dir(fullfile(d, 'time_avg_3d_*.nc'));
if isempty(ftop) || isempty(favg)
    error('inflow_diagnostics: %s has no top_*.nc or time_avg_3d_*.nc', d);
end
ftop = fullfile(d, ftop(1).name);
favg = fullfile(d, favg(1).name);
fmeta = dir(fullfile(d, 'sweep_metadata_*.nc'));

r.label = regexprep(folder, '^sweep_\d+_', '');
r.v0 = 0.2; r.open = 0; r.noise = 0; Ly = NaN; y0 = NaN; r.jet_xs = 65e3;
if ~isempty(fmeta)
    fm = fullfile(d, fmeta(1).name);
    r.v0    = gatt_(fm, 'v0', r.v0);
    r.open  = gatt_(fm, 'open_y', 0);
    r.noise = gatt_(fm, 'inflow_noise', 0);
    r.jet_xs = gatt_(fm, 'jet_xs', 65e3);          % the jet's offshore edge (65 km before bathy = diamond)
    Ly      = gatt_(fm, 'Ly', NaN);
    y0      = gatt_(fm, 'shoal_y0', NaN);
end

x = double(rd_(ftop, {'x_caa', 'xC', 'xc'}));
y = double(rd_(ftop, {'y_aca', 'yC', 'yc'}));
t = double(ncread(ftop, 'time'));
if isnan(Ly), Ly = max(y) + (y(2) - y(1)) / 2; end
if isnan(y0), y0 = Ly / 2; end
[~, jup] = min(abs(y - o.y_up * 1e3));
% shelf: the requested range, but never past 5 km inshore of the jet's edge
shelf = x >= o.shelf(1) * 1e3 & x <= min(o.shelf(2) * 1e3, r.jet_xs - 5e3);
nx = numel(x); ny = numel(y); nt = numel(t);

% ── daily surface: speed and disturbance on the upstream row ─────────────
uname = pick_(ftop, {'u_c', 'u'}); vname = pick_(ftop, {'v_c', 'v'});
u_row = squeeze(double(ncread(ftop, uname, [1 jup 1 1], [nx 1 1 nt])));   % nx x nt
v_row = squeeze(double(ncread(ftop, vname, [1 jup 1 1], [nx 1 1 nt])));
r.t_days = t / 86400;
r.vup  = mean(v_row(shelf, :), 1);
% fluctuation: u minus its running 5-day mean (steady cross-shore flow, e.g.
% the bottom Ekman layer's, is not a disturbance)
W = max(3, 2 * round(2.5 * 86400 / median(diff(t))) + 1);
u_hp = u_row - movmean(u_row, W, 2);
r.seed = sqrt(mean(u_hp(shelf, :).^2, 1));
last = r.t_days >= r.t_days(end) - o.last;
r.vup_last  = mean(r.vup(last));
r.seed_last = sqrt(mean(r.seed(last).^2));

% ── profile along y of the shelf RMS u, last o.last days ─────────────────
% (temporal standard deviation of u at each point, RMS over the shelf)
it = find(last);
s1 = zeros(sum(shelf), ny); s2 = s1;
for n = it(:)'
    u = double(ncread(ftop, uname, [1 1 1 n], [nx ny 1 1]));
    s1 = s1 + u(shelf, :);  s2 = s2 + u(shelf, :).^2;
end
m = numel(it);
r.prof = sqrt(max(mean(s2 / m - (s1 / m).^2, 1), 0));
r.prof_y = (y' - y0) / 1e3;
r.prof_at = @(dkm) interp1(r.prof_y, r.prof, dkm, 'linear', NaN);

% ── transport through the upstream section, 10-day means ─────────────────
z  = double(rd_(favg, {'z_aac', 'zC', 'zc'}));
dz = double(rd_(favg, {'Δz_aac'}, []));
if isempty(dz), dz = gradient(z); end
dz = dz(:)';
xa = double(rd_(favg, {'x_caa', 'xC', 'xc'}));
dx = xa(2) - xa(1);
ta = double(ncread(favg, 'time'));
nza = numel(z); nta = numel(ta);
keep = ta > 0;          % the writer also stores an empty record at t = 0
vname_a = pick_(favg, {'v_c', 'v'});
va = squeeze(double(ncread(favg, vname_a, [1 jup 1 1], [numel(xa) 1 nza nta])));   % nx x nz x nt
% wet cells: centre above the stored bottom; else nonzero time-mean T
wet = [];
try
    hb = double(ncread(favg, 'bottom_height', [1 jup], [numel(xa) 1]));
    wet = z(:)' > hb(:);                                              % nx x nz
catch
end
if isempty(wet)
    Tn = pick_(favg, {'T'});
    Ta = squeeze(double(ncread(favg, Tn, [1 jup 1 nta], [numel(xa) 1 nza 1])));
    wet = Ta > 1;
end
vinf = v_inf_(xa, r.v0, r.jet_xs);
Q_in = sum(sum(vinf(:) .* wet .* dz)) * dx;
Q = zeros(1, nta);
for n = 1:nta
    Q(n) = sum(sum(va(:, :, n) .* wet .* dz)) * dx;
end
r.tavg_days = ta(keep) / 86400;
r.Qratio = Q(keep) / Q_in;
% cross-shore structure, averaged over the 10-day means in the last o.last days
la = find(keep(:)' & (ta(:)' / 86400 >= ta(end) / 86400 - o.last));
if isempty(la), la = nta; end
vm = mean(va(:, :, la), 3);                                       % nx x nz
H = sum(wet .* dz, 2);                                            % column depth
r.xprof = xa(:);
r.qx = sum(vm .* wet .* dz, 2) * dx;                              % transport per column
r.vbar = r.qx ./ max(H * dx, eps);  r.vbar(H == 0) = NaN;
r.qx_inf = vinf(:) .* H * dx;
r.vinf = vinf(:);  r.vinf(H == 0) = NaN;
r.Q_in = Q_in;
% share of the transport over the shelf (inshore of 5 km short of the jet's
% edge: 60 km for the param bathymetry) and offshore of it
sh = r.xprof < r.jet_xs - 5e3;
r.shelf_frac = sum(r.qx(sh)) / sum(r.qx);
r.shelf_frac_inf = sum(r.qx_inf(sh)) / sum(r.qx_inf);
lasta = r.tavg_days >= r.tavg_days(end) - o.last;
r.Qratio_last = mean(r.Qratio(lasta));
end

function v = v_inf_(x, v0, xs)
% The model's inflow profile v∞(x) (flow_over_shoals_sweep.jl, sigmoid_v_bc):
% coastal sigmoid at 3 km, offshore sigmoid at xs (JET_XS: 65 km, or the
% shelf break for the diamond bathymetry), k pinned to 150 km.
k1 = 80 / 150e3; k2 = 40 / 150e3;
s1 = 1 ./ (1 + exp(-k1 * (x - 3e3)));
s2 = 1 ./ (1 + exp(k2 * (x - xs)));
v = v0 * min(max((s1 - 1) + s2, 0), 1);
end

function val = gatt_(f, name, default)
try
    val = double(ncreadatt(f, '/', name));
catch
    val = default;
end
end

function n = pick_(f, names)
info = ncinfo(f);
have = {info.Variables.Name};
for k = 1:numel(names)
    if any(strcmp(have, names{k})), n = names{k}; return; end
end
error('inflow_diagnostics: none of {%s} in %s', strjoin(names, ', '), f);
end

function d = rd_(f, names, default)
for k = 1:numel(names)
    try
        d = ncread(f, names{k}); return;
    catch
    end
end
if nargin > 2, d = default; return; end
error('inflow_diagnostics: none of {%s} in %s', strjoin(names, ', '), f);
end
