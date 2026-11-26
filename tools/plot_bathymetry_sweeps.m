function plot_bathymetry_sweeps(bathy_dir)
% plot_bathymetry_sweeps(bathy_dir)
%
% 3-D view of the bathymetry of every run in every sweep table, to check them
% by eye. Reads the NetCDF files written by tools/export_bathymetry.jl, which
% builds each run's bottom with the model's own src/dshoal_vn_param.jl.
%
% For each table (bathy_dir/<table>.nc) it saves
%   <table>_3d.png        one surf per run: the GRID-FITTED bottom, i.e. what
%                         the model sees after snapping to whole dz cells
%   <table>_profiles.png  the cross-shore profile through the shoal centre
%                         (y = Ly/2) for every run, smooth and grid-fitted
% A run whose profile has a trench between the coast and the shoal is titled
% in red with the trench depth.
%
%   plot_bathymetry_sweeps                       % figures/bathymetry
%   plot_bathymetry_sweeps('figures/bathymetry')

if nargin < 1 || isempty(bathy_dir)
    bathy_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'figures', 'bathymetry');
end
files = dir(fullfile(bathy_dir, '*.nc'));
if isempty(files)
    error('No .nc files in %s. Run: julia --project=. tools/export_bathymetry.jl', bathy_dir);
end

for f = files'
    fn = fullfile(f.folder, f.name);
    [~, table] = fileparts(fn);
    x  = ncread(fn, 'x') / 1e3;  y = ncread(fn, 'y') / 1e3;
    D  = ncread(fn, 'depth');  G = ncread(fn, 'depth_grid');
    tr = ncread(fn, 'trench');
    labels = strsplit(ncreadatt(fn, '/', 'runs'), newline);
    nr = numel(labels);
    zmin = min(G(:));

    % ── 3-D surfaces ─────────────────────────────────────────────────────
    nc = min(4, nr);  nrow = ceil(nr / nc);
    fig = figure('Color', 'w', 'Position', [40 40 430 * nc, 360 * nrow], 'Visible', 'off');
    tl = tiledlayout(fig, nrow, nc, 'TileSpacing', 'compact', 'Padding', 'compact');
    st = max(1, round(numel(x) / 150));            % thin the mesh for drawing speed
    [X, Y] = ndgrid(x(1:st:end), y(1:st:end));
    for k = 1:nr
        ax = nexttile(tl);
        Z = G(1:st:end, 1:st:end, k);
        surf(ax, X, Y, Z, 'EdgeColor', 'none');
        colormap(ax, flipud(parula));  clim(ax, [zmin 0]);
        view(ax, -35, 35);  axis(ax, 'tight');  zlim(ax, [zmin 0]);
        daspect(ax, [1 1 abs(zmin) / 40]);       % ~ 40x vertical exaggeration per km
        lighting(ax, 'gouraud');  camlight(ax, 'headlight');  material(ax, 'dull');
        xlabel(ax, 'x (km)');  ylabel(ax, 'y (km)');  zlabel(ax, 'z (m)');
        ttl = strrep(labels{k}, '_', '\_');
        if tr(k) > 1
            title(ax, sprintf('%s  --  TRENCH %.1f m', ttl, tr(k)), 'Color', [0.8 0 0]);
        else
            title(ax, ttl);
        end
        set(ax, 'FontSize', 9);
    end
    title(tl, sprintf('%s: grid-fitted bottom, every run (coast at x = 0, shoal centred at y = %g km)', ...
          strrep(table, '_', '\_'), (min(y) + max(y)) / 2), 'FontWeight', 'bold');
    exportgraphics(fig, fullfile(bathy_dir, [table '_3d.png']), 'Resolution', 150);
    close(fig);

    % ── centre-line profiles ─────────────────────────────────────────────
    [~, jc] = min(abs(y - (min(y) + max(y)) / 2));
    fig = figure('Color', 'w', 'Position', [40 40 1100 480], 'Visible', 'off');
    ax = axes(fig);  hold(ax, 'on');  grid(ax, 'on');
    cc = lines(nr);
    for k = 1:nr
        sty = '-';  if tr(k) > 1, sty = '--'; end
        plot(ax, x, D(:, jc, k), sty, 'Color', cc(k,:), 'LineWidth', 1.5, ...
             'DisplayName', sprintf('%s%s', strrep(labels{k}, '_', '\_'), ternary_(tr(k) > 1, sprintf('  (trench %.1f m)', tr(k)), '')));
        stairs(ax, x - (x(2) - x(1)) / 2, G(:, jc, k), ':', 'Color', cc(k,:), 'LineWidth', 0.8, 'HandleVisibility', 'off');
    end
    xlim(ax, [0 min(110, max(x))]);  ylim(ax, [zmin - 2, 0]);
    xlabel(ax, 'x (km from coast)');  ylabel(ax, 'depth (m)');
    title(ax, sprintf('%s: cross-shore profile through the shoal centre (solid = continuous, dotted = grid-fitted; dashed = has a trench)', ...
          strrep(table, '_', '\_')));
    legend(ax, 'Location', 'eastoutside', 'FontSize', 8);
    exportgraphics(fig, fullfile(bathy_dir, [table '_profiles.png']), 'Resolution', 150);
    close(fig);
    fprintf('%-28s %2d runs -> %s_3d.png, %s_profiles.png%s\n', table, nr, table, table, ...
            ternary_(any(tr > 1), sprintf('   (%d with a trench)', sum(tr > 1)), ''));
end
end

function v = ternary_(c, a, b)
if c, v = a; else, v = b; end
end
