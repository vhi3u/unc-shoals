function locate_extremes(run, varargin)
% locate_extremes(run, ...)
%
% Where do T and S leave the range of the initial profile? Finds every cell,
% in the surface slice (top_*.nc) and the shoal-axis section (midy_*.nc), with
% T or S outside [initial min - tol, initial max + tol], and reports how many
% there are and where: near the north (outflow) or south (inflow) boundary,
% the east boundary, the coast (land), or the interior. Writes a map of how
% often each surface cell is out of range, and the worst cells per snapshot.
%
%   locate_extremes('DSN_cape_v020_N2x6')
%   locate_extremes('cluster_dsstrat/DSN_cape_v020_N2x6', 'tolT', 0.1)
%
% The run is found as in inflow_diagnostics.m (label, cluster/label, folder,
% or path). The initial range is taken from the first snapshot of the section.
% Writes animations/extremes_<label>.png.

ip = inputParser;
scratch = '/glade/derecho/scratch/nguyen/unc-shoals/sweep_output';
here = fileparts(mfilename('fullpath'));
if isfolder(scratch), root0 = scratch; else, root0 = fullfile(here, '..', 'sweep_output'); end
addParameter(ip, 'root', root0);
addParameter(ip, 'tolT', 0.05);     % degC beyond the initial range that counts
addParameter(ip, 'tolS', 0.02);     % psu
addParameter(ip, 'edge_km', 20);    % "near a boundary" = within this distance
parse(ip, varargin{:});
o = ip.Results;

d = find_run_(run, o.root);
[~, folder] = fileparts(d);
label = regexprep(folder, '^sweep_\d+_', '');
ftop = dir(fullfile(d, 'top_*.nc'));   ftop = fullfile(d, ftop(1).name);
fmid = dir(fullfile(d, 'midy_*.nc'));  fmid = fullfile(d, fmid(1).name);

x = double(rd_(ftop, {'x_caa', 'xC'})) / 1e3;  y = double(rd_(ftop, {'y_aca', 'yC'})) / 1e3;
t = double(ncread(ftop, 'time')) / 86400;
hb = double(ncread(ftop, 'bottom_height'));                     % nx x ny
land = hb >= 0;
nx = numel(x);  ny = numel(y);  nt = numel(t);

% initial range from the first section snapshot (water cells)
T0 = squeeze(double(ncread(fmid, 'T', [1 1 1 1], [Inf 1 Inf 1])));
S0 = squeeze(double(ncread(fmid, 'S', [1 1 1 1], [Inf 1 Inf 1])));
w0 = T0 ~= 0 | S0 ~= 0;                                         % immersed cells are 0
Tlo = min(T0(w0)) - o.tolT;  Thi = max(T0(w0)) + o.tolT;
Slo = min(S0(w0)) - o.tolS;  Shi = max(S0(w0)) + o.tolS;
fprintf('\n%s: initial T %.2f-%.2f C, S %.3f-%.3f psu (tolerance %.2f C, %.2f psu)\n', ...
        label, Tlo + o.tolT, Thi - o.tolT, Slo + o.tolS, Shi - o.tolS, o.tolT, o.tolS);

% where is a surface cell? (coast = water within 2 cells of land)
[X, Y] = ndgrid(x, y);
nearland = conv2(double(land), ones(5), 'same') > 0 & ~land;
Ly = y(end) + (y(2) - y(1)) / 2;  Lx = x(end) + (x(2) - x(1)) / 2;
zone = repmat("interior", nx, ny);
zone(X > Lx - o.edge_km) = "east";
zone(Y < o.edge_km) = "south (inflow)";
zone(Y > Ly - o.edge_km) = "north (outflow)";
zone(nearland) = "coast";
zones = ["north (outflow)", "south (inflow)", "east", "coast", "interior"];

count = zeros(nx, ny);  worst = zeros(nt, 4);  byzone = zeros(numel(zones), 1);
for n = 1:nt
    T = double(ncread(ftop, 'T', [1 1 1 n], [Inf Inf 1 1]));
    S = double(ncread(ftop, 'S', [1 1 1 n], [Inf Inf 1 1]));
    bad = ~land & (T < Tlo | T > Thi | S < Slo | S > Shi);
    count = count + bad;
    for k = 1:numel(zones), byzone(k) = byzone(k) + nnz(bad & zone == zones(k)); end
    Tw = T;  Tw(land) = NaN;
    worst(n, :) = [min(Tw(:)), max(Tw(:)), min(S(~land)), max(S(~land))];
end

fprintf('\nsurface cells out of range, summed over %d snapshots, by where they are:\n', nt);
for k = 1:numel(zones)
    fprintf('  %-18s %10d  (%.1f%%)\n', zones(k), byzone(k), 100 * byzone(k) / max(sum(byzone), 1));
end
[~, n1] = max(worst(:, 2));  [~, n2] = min(worst(:, 1));
fprintf('\nworst surface T: max %.2f C (day %.1f), min %.2f C (day %.1f)\n', worst(n1, 2), t(n1), worst(n2, 1), t(n2));
fprintf('first snapshot with any out-of-range surface cell: day %.1f\n', t(find(any([worst(:,1) < Tlo, worst(:,2) > Thi], 2), 1)));

% section: where in depth?
nm = numel(double(ncread(fmid, 'time')));
Tm = squeeze(double(ncread(fmid, 'T', [1 1 1 nm], [Inf 1 Inf 1])));
zc = double(rd_(fmid, {'z_aac', 'zC'}));
wm = Tm ~= 0;
[ib, kb] = find(wm & (Tm < Tlo | Tm > Thi));
if ~isempty(ib)
    fprintf('last section snapshot: %d cells out of range, x = %.0f-%.0f km, z = %.0f to %.0f m\n', ...
            numel(ib), min(x(ib)), max(x(ib)), min(zc(kb)), max(zc(kb)));
end

f = figure('Color', 'w', 'Position', [60 60 1300 650], 'Visible', 'off');
tl = tiledlayout(f, 1, 2, 'TileSpacing', 'compact');
ax = nexttile(tl);  C = count;  C(land) = NaN;
pcolor(ax, x, y, C' / nt * 100);  shading(ax, 'flat');  axis(ax, 'equal', 'tight');  hold(ax, 'on');
contour(ax, x, y, hb', [-0.01 -0.01], 'k');  cb = colorbar(ax);  cb.Label.String = '% of snapshots out of range';
xlabel(ax, 'x (km)');  ylabel(ax, 'y (km)');  title(ax, 'surface T or S outside the initial range');
ax = nexttile(tl);  hold(ax, 'on');  grid(ax, 'on');
plot(ax, t, worst(:, 2), 'r', t, worst(:, 1), 'b', 'LineWidth', 1.5);
yline(ax, [Tlo + o.tolT, Thi - o.tolT], 'k--');
xlabel(ax, 'day');  ylabel(ax, 'T (C)');  title(ax, 'surface T max (red) and min (blue); dashed: initial range');
title(tl, label, 'Interpreter', 'none', 'FontWeight', 'bold');
outdir = fullfile(here, '..', 'animations');  if ~isfolder(outdir), mkdir(outdir); end
png = fullfile(outdir, ['extremes_' label '.png']);
exportgraphics(f, png, 'Resolution', 120);
fprintf('Wrote %s\n', png);
end

function d = find_run_(name, root)
if isfolder(name), d = name; return; end
if isfolder(fullfile(root, name)), d = fullfile(root, name); return; end
[cl, lab] = fileparts(name);
if ~isempty(cl) && isfolder(fullfile(root, cl))
    c = dir(fullfile(root, cl, ['sweep_*_' lab]));  c = c([c.isdir]);
    if numel(c) == 1, d = fullfile(c.folder, c.name); return; end
end
c = dir(fullfile(root, '**', ['sweep_*_' name]));  c = c([c.isdir]);
if numel(c) ~= 1, error('locate_extremes: %d folders match "%s"', numel(c), name); end
d = fullfile(c.folder, c.name);
end

function v = rd_(f, names)
for k = 1:numel(names)
    try, v = ncread(f, names{k}); return; catch, end
end
error('locate_extremes: none of {%s} in %s', strjoin(names, ', '), f);
end
