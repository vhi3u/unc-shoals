function plot_bathymetry_sweeps(varargin)
% plot_bathymetry_sweeps(...)
%
% Every run's bathymetry in every sweep table, as interactive MATLAB figures
% (rotate the 3-D views to check them). Pure MATLAB: reads the tables in
% src/sweep/params_*.toml and builds each bottom with shoal_bathymetry.m, a
% port of the model's src/dshoal_vn_param.jl.
%
% Two figures per table:
%   3-D        one surf per run, the GRID-FITTED bottom (snapped to whole cells
%              of the run's own dz, as GridFittedBottom does in the model)
%   profiles   the cross-shore profile through the shoal centre (y = Ly/2)
%              for every run: continuous (solid) and grid-fitted (dotted)
% A run with a trench between the coast and the shoal is titled in red with
% the trench depth (the check src/dshoal_vn_param.jl now enforces).
%
%   plot_bathymetry_sweeps                                  % every table
%   plot_bathymetry_sweeps('tables', {'params_deep_geom', 'params_deep_wind'})
%   plot_bathymetry_sweeps('save', true)                    % also write PNGs
%
% Run from the repository root, or anywhere with tools/ on the path.

here = fileparts(mfilename('fullpath'));
ip = inputParser;
addParameter(ip, 'sweep_dir', fullfile(here, '..', 'src', 'sweep'));
addParameter(ip, 'tables', {});
addParameter(ip, 'save', false);
addParameter(ip, 'outdir', fullfile(here, '..', 'figures', 'bathymetry'));
addParameter(ip, 'dx', 1000);              % plotting resolution (m)
parse(ip, varargin{:});
o = ip.Results;

if isempty(o.tables)
    f = dir(fullfile(o.sweep_dir, 'params*.toml'));
    tables = fullfile({f.folder}, {f.name});
else
    tables = fullfile(o.sweep_dir, strcat(strrep(o.tables, '.toml', ''), '.toml'));
end
if o.save && ~exist(o.outdir, 'dir'), mkdir(o.outdir); end

Lx = 150e3;  Ly = 200e3;
x = (o.dx/2 : o.dx : Lx)';  y = (o.dx/2 : o.dx : Ly)';
[X, Y] = ndgrid(x, y);
[~, jc] = min(abs(y - Ly/2));

for t = 1:numel(tables)
    [~, name] = fileparts(tables{t});
    runs = read_sweep_table(tables{t});
    nr = numel(runs);
    G = zeros(numel(x), numel(y), nr);  D = G;  trench = zeros(1, nr);  labels = cell(1, nr);
    for k = 1:nr
        p = run_params_(runs(k));
        D(:,:,k) = shoal_bathymetry(X, Y, p);
        G(:,:,k) = grid_fit_(D(:,:,k), p);
        trench(k) = trench_depth_(p);
        labels{k} = runs(k).run_label;
    end
    zmin = min(G(:));

    % ── 3-D figure ───────────────────────────────────────────────────────
    nc = min(4, nr);  nrow = ceil(nr / nc);
    f3 = figure('Color', 'w', 'Name', [name ' (3-D)'], 'NumberTitle', 'off', ...
                'Position', [40 40 min(1800, 430*nc) min(1000, 340*nrow)]);
    tl = tiledlayout(f3, nrow, nc, 'TileSpacing', 'compact', 'Padding', 'compact');
    for k = 1:nr
        ax = nexttile(tl);
        surf(ax, X/1e3, Y/1e3, G(:,:,k), 'EdgeColor', 'none');
        colormap(ax, flipud(parula));  clim(ax, [zmin 0]);
        view(ax, -35, 35);  axis(ax, 'tight');  zlim(ax, [zmin 0]);
        daspect(ax, [1 1 abs(zmin)/40]);
        lighting(ax, 'gouraud');  camlight(ax, 'headlight');  material(ax, 'dull');
        xlabel(ax, 'x (km)');  ylabel(ax, 'y (km)');  zlabel(ax, 'z (m)');
        ttl = strrep(labels{k}, '_', '\_');
        if trench(k) > 1
            title(ax, sprintf('%s -- TRENCH %.1f m', ttl, trench(k)), 'Color', [0.8 0 0]);
        else
            title(ax, ttl);
        end
        set(ax, 'FontSize', 9);
    end
    title(tl, sprintf('%s: grid-fitted bottom (coast at x = 0, shoal centred at y = %g km)', ...
          strrep(name, '_', '\_'), Ly/2e3), 'FontWeight', 'bold');

    % ── profile figure ───────────────────────────────────────────────────
    fp = figure('Color', 'w', 'Name', [name ' (profiles)'], 'NumberTitle', 'off', ...
                'Position', [80 80 1150 480]);
    ax = axes(fp);  hold(ax, 'on');  grid(ax, 'on');
    cc = lines(nr);
    for k = 1:nr
        lbl = strrep(labels{k}, '_', '\_');
        if trench(k) > 1, sty = '--'; lbl = sprintf('%s  (trench %.1f m)', lbl, trench(k)); else, sty = '-'; end
        plot(ax, x/1e3, D(:, jc, k), sty, 'Color', cc(k,:), 'LineWidth', 1.5, 'DisplayName', lbl);
        stairs(ax, (x - o.dx/2)/1e3, G(:, jc, k), ':', 'Color', cc(k,:), 'HandleVisibility', 'off');
    end
    xlim(ax, [0 110]);  ylim(ax, [zmin - 2, 0]);
    xlabel(ax, 'x (km from coast)');  ylabel(ax, 'depth (m)');
    title(ax, sprintf('%s: profile through the shoal centre (solid continuous, dotted grid-fitted, dashed = trench)', ...
          strrep(name, '_', '\_')));
    legend(ax, 'Location', 'eastoutside', 'FontSize', 8);

    if o.save
        exportgraphics(f3, fullfile(o.outdir, [name '_3d.png']), 'Resolution', 150);
        exportgraphics(fp, fullfile(o.outdir, [name '_profiles.png']), 'Resolution', 150);
    end
    bad = labels(trench > 1);
    if isempty(bad), note = ''; else, note = ['   TRENCH: ' strjoin(bad, ', ')]; end
    fprintf('%-28s %2d runs%s\n', name, nr, note);
end
end

function p = run_params_(r)
% The table entry as shoal_bathymetry parameters, dropping empty fields
% (keys other runs in the table set but this one does not).
p = struct();
for f = fieldnames(r)'
    v = r.(f{1});
    if ~isempty(v) && (isnumeric(v) || islogical(v)), p.(f{1}) = double(v); end
end
end

function G = grid_fit_(D, p)
% GridFittedBottom: a cell is water if its centre lies above the bottom.
Lz = 50;  dz = 1;
if isfield(p, 'Lz'), Lz = p.Lz; end
if isfield(p, 'dz'), dz = p.dz; end
zc = reshape(-Lz + dz/2 : dz : -dz/2, 1, 1, []);
G = -dz * sum(zc > D, 3);
end

function d = trench_depth_(p)
% How far the centre-line profile dips below the shallowest points on both
% sides of it, between the coast and the end of the crest.
if ~isfield(p, 'shoal_length'), p.shoal_length = 40e3; end
Ly = 200e3;  if isfield(p, 'Ly'), Ly = p.Ly; end
xs = 0:100:p.shoal_length;
z = shoal_bathymetry(xs, Ly/2, p);
left = cummax(z);  right = flip(cummax(flip(z)));
d = max(min(left, right) - z);
end
