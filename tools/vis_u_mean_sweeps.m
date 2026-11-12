function vis_u_mean_sweeps(cluster_name, varargin)
% vis_u_mean_sweeps(cluster_name, 'outdir', OUTDIR, 'clipped', true/false, ...
%   'x_line', X_KM, 'isopycnals', true/false, 'rho_levels', ..., ...
%   'climu', [lo hi], 'climU_height', [lo hi], 'climU_length', [lo hi])
%
% 'climu' sets the color limits for both rows; 'climU_height' / 'climU_length'
% override it for one row. If none are given, each row is auto-scaled.
%
% Time-mean, depth-averaged cross-shelf velocity (u) from the LAST time-
% averaging window of time_avg_3d_sweep_<run>.nc, for the shoal height and
% shoal length sweeps.
%
% Layout (2 rows):
%   Row 1 — shoal height sweep (Ls = 40 km):  Zs = -15, -10, -5 m  | line plot
%   Row 2 — shoal length sweep (Zs = -5 m):   Ls = 20, 30, 40 km   | line plot
%
%   Maps:  depth-averaged u_bar(x, y), with surface σ0 isopycnals as in
%          vishoals_still.m. Dashed line marks x = x_line.
%   Lines: depth-integrated cross-shelf transport ∫ u_bar dz (m^2 s^-1)
%          at x = x_line, versus y.
%
% Each row shares one color scale, shown on the rightmost map of that row.
%
% Requires: cmocean, GSW Toolbox (for isopycnals)
%
% Example:
%   vis_u_mean_sweeps('cluster_Fnnie_1_19')

if nargin < 1 || isempty(cluster_name)
    cluster_name = 'cluster_Fnnie_1_19';
end

% ---------- options ----------
p = inputParser;
addParameter(p, 'sweep_root', '/glade/work/nguyen/unc-shoals/sweep_output', @ischar);
addParameter(p, 'outdir', '/glade/work/nguyen/unc-shoals/animations', @ischar);
addParameter(p, 'clipped', true, @(x)islogical(x)&&isscalar(x));
addParameter(p, 'x_line', 40, @(x)isnumeric(x)&&isscalar(x));   % km, same as plot_full_scatter_sweeps x_target_km
% Isopycnal options
addParameter(p, 'isopycnals', true, @(x)islogical(x)&&isscalar(x));
addParameter(p, 'rho_levels', [23:0.025:24.0, 24.1:0.4:26.4, 26.6:0.8:28], @(v)isnumeric(v)&&isvector(v));
addParameter(p, 'iso_color', 'k', @ischar);
addParameter(p, 'iso_linewidth', 0.8, @(x)isnumeric(x)&&isscalar(x)&&x>0);
% Manual color limits (optional): climu = both rows, or one per row
addParameter(p, 'climu', [], @(v)isempty(v) || (isnumeric(v) && numel(v)==2));
addParameter(p, 'climU_height', [], @(v)isempty(v) || (isnumeric(v) && numel(v)==2));
addParameter(p, 'climU_length', [], @(v)isempty(v) || (isnumeric(v) && numel(v)==2));
addParameter(p, 'no_outlier', false, @(x)islogical(x)&&isscalar(x));

parse(p, varargin{:});
clipped      = p.Results.clipped;
x_line_km    = p.Results.x_line;
want_iso     = p.Results.isopycnals;
rho_levels   = p.Results.rho_levels;
iso_color    = p.Results.iso_color;
iso_lw       = p.Results.iso_linewidth;
outdir       = p.Results.outdir;
cluster_dir  = fullfile(p.Results.sweep_root, cluster_name);

% ---------- sweeps ----------
rows(1).runs   = {'Zs15_Ls40k', 'Zs10_Ls40k', 'Zs5_Ls40k'};
rows(1).titles = {'Z_s = -15 m', 'Z_s = -10 m', 'Z_s = -5 m'};
rows(1).legend = {'Z_s = -15 m', 'Z_s = -10 m', 'Z_s = -5 m'};
rows(1).name   = 'Shoal Height';
rows(1).clim   = p.Results.climU_height;
if isempty(rows(1).clim), rows(1).clim = p.Results.climu; end

rows(2).runs   = {'Zs5_Ls20k', 'Zs5_Ls30k', 'Zs5_Ls40k'};
rows(2).titles = {'L_s = 20 km', 'L_s = 30 km', 'L_s = 40 km'};
rows(2).legend = {'L_s = 20 km', 'L_s = 30 km', 'L_s = 40 km'};
rows(2).name   = 'Shoal Length';
rows(2).clim   = p.Results.climU_length;
if isempty(rows(2).clim), rows(2).clim = p.Results.climu; end

% Line colors: same blue / red / green used in plot_full_scatter_sweeps
line_cols = [0.0, 0.0, 0.7;
             0.7, 0.0, 0.0;
             0.0, 0.5, 0.2];

% ---------- grid (from first run; grids assumed identical) ----------
fname0 = find_time_avg(cluster_dir, rows(1).runs{1});
xc = try_ncread_(fname0, {'xC','x_caa','xc','x'});
yc = try_ncread_(fname0, {'yC','y_aca','yc','y'});
zc = try_ncread_(fname0, {'zC','z_aac','zc','z'}); zc = zc(:);

xc_km = xc / 1000;
yc_km = yc / 1000;
ix = 1:numel(xc);
iy = 1:numel(yc);
clip_x = [0 100];   % km, same as vishoals_still.m
clip_y = [20 200];  % km
if clipped
    ix = find(xc_km >= clip_x(1) & xc_km <= clip_x(2));
    iy = find(yc_km >= clip_y(1) & yc_km <= clip_y(2));
end
xc_km = xc_km(ix);
yc_km = yc_km(iy);
Nx = numel(ix); Ny = numel(iy);
[Xc, Yc] = meshgrid(xc_km, yc_km);

[~, ix_line] = min(abs(xc_km - x_line_km));
x_line_actual = xc_km(ix_line);

% Cell thicknesses (handles stretched vertical grids)
dz = cell_thickness(fname0, zc);
DZ = reshape(dz, 1, 1, []);
[~, k_top] = max(zc);

% ---------- TEOS-10 prep for isopycnals ----------
lat = 35.2480;
lon = 0;
P_xy = gsw_p_from_z(0, lat) * ones(Ny, Nx);

% ---------- load data ----------
for r = 1:2
    nr = numel(rows(r).runs);
    rows(r).ubar  = cell(1, nr);   % depth-averaged u (Ny x Nx)
    rows(r).uint  = cell(1, nr);   % ∫ u dz at x_line (Ny x 1)
    rows(r).sig0  = cell(1, nr);   % surface σ0 (Ny x Nx)
    for k = 1:nr
        fname = find_time_avg(cluster_dir, rows(r).runs{k});
        fprintf('%-12s -> %s\n', rows(r).runs{k}, fname);

        u = read_last(fname, {'u_c','u'}, ix, iy);   % (Nx, Ny, Nz)
        u(u == 0) = NaN;                             % mask land/immersed cells
        wet = ~isnan(u);
        uint_xy = sum(u .* DZ, 3, 'omitnan');        % ∫ u dz  (Nx, Ny)
        h_wet   = sum(DZ .* wet, 3);                 % local water depth
        ubar = uint_xy ./ h_wet;
        ubar(h_wet == 0) = NaN;
        uint_xy(h_wet == 0) = NaN;

        rows(r).ubar{k} = ubar';
        rows(r).uint{k} = uint_xy(ix_line, :)';

        if want_iso
            T = read_last(fname, {'T'}, ix, iy);  T = T(:, :, k_top)';
            S = read_last(fname, {'S'}, ix, iy);  S = S(:, :, k_top)';
            bad = (T == 0) | (S == 0) | ~isfinite(T) | ~isfinite(S);
            T(bad) = NaN; S(bad) = NaN;
            SA = gsw_SA_from_SP(S, P_xy, lon, lat);
            CT = gsw_CT_from_t(SA, T, P_xy);
            rows(r).sig0{k} = gsw_sigma0(SA, CT);
        end
    end
end

% ---------- color limits (shared within each row) ----------
% Trim boundary cells to avoid numerical artifacts in color limits
trim = @(x) x(2:end-1, 2:end-1);
for r = 1:2
    if isempty(rows(r).clim)
        um = 0;
        for k = 1:numel(rows(r).runs)
            v = abs(reshape(trim(rows(r).ubar{k}), [], 1));
            if p.Results.no_outlier
                um = max(um, prctile(v, 99));
            else
                um = max(um, max(v, [], 'omitnan'));
            end
        end
        rows(r).clim = [-um, um];
    end
    fprintf('Color limits (row %d, u_bar): [%.4g, %.4g]\n', r, rows(r).clim(1), rows(r).clim(2));
end

% ---------- figure / panels ----------
% Same per-panel width and compact spacing as vishoals_still.m full_panel
% (6 columns in 1350 px), scaled to 4 columns: 1350 * 4/6 = 900 px.
% 2 rows x 4 columns — columns 1-3: maps; column 4: line plot.
fig_w = 1350 * 4/6; fig_h = 700;
fig = figure('Color','w','Position',[80 50 fig_w fig_h]);
tl  = tiledlayout(fig, 2, 4, 'TileSpacing','compact','Padding','compact');
cmap = cmocean('balance');

map_axs = gobjects(2, 3);
map_cbs = gobjects(2, 3);
line_axs = gobjects(1, 2);
line_ph  = gobjects(1, 2);
for r = 1:2
    for k = 1:3
        ax = nexttile(tl, (r-1)*4 + k);
        pcolor(ax, Xc, Yc, rows(r).ubar{k});
        shading(ax,'flat'); axis(ax,'tight'); set(ax,'YDir','normal');
        set(ax, 'Layer', 'top'); box(ax, 'on'); set(ax, 'LineWidth', 0.8, 'XColor', 'k', 'YColor', 'k');
        colormap(ax, cmap); caxis(ax, rows(r).clim);
        set(ax, 'NextPlot', 'add');
        map_cbs(r, k) = colorbar(ax);
        map_cbs(r, k).Title.String = '(m s^{-1})';
        title(ax, sprintf('(%s) %s', char('a' + (r-1)*4 + k - 1), rows(r).titles{k}), ...
            'FontSize', 18, 'FontWeight', 'bold');
        if k == 1
            ylabel(ax, 'y (km)', 'FontSize', 16, 'FontWeight', 'bold');
        else
            yticklabels(ax, []);
        end
        if r == 2
            xlabel(ax, 'x (km)', 'FontSize', 16, 'FontWeight', 'bold');
        else
            xticklabels(ax, []);
        end
        set(ax, 'FontSize', 14);
        xlim(ax, [min(xc_km), max(xc_km)]);
        ylim(ax, [min(yc_km), max(yc_km)]);

        % --- surface σ0 isopycnals ---
        if want_iso
            s0 = rows(r).sig0{k};
            fin = s0(isfinite(s0));
            if ~isempty(fin) && range(fin) > 0
                [C, hc] = contour(ax, Xc, Yc, s0, rho_levels, ...
                    'Color', iso_color, 'LineWidth', iso_lw);
                if ~isempty(C)
                    clabel(C, hc, 'FontSize', 7, 'Color', iso_color, 'LabelSpacing', 400);
                end
            end
        end

        % Mark where the line plot is taken
        xline(ax, x_line_actual, '--', 'Color', [0.3 0.3 0.3], 'LineWidth', 1);
        map_axs(r, k) = ax;
    end

    % --- placeholder tile for the line plot (real axes drawn after layout) ---
    % Styled exactly like the line plot (labels, title, square aspect) but drawn
    % in white, so the tiled layout reserves the same space as a real line plot.
    ph = nexttile(tl, (r-1)*4 + 4);
    yall = [rows(r).uint{:}];
    set(ph, 'FontSize', 14, 'LineWidth', 1, 'TickDir', 'out', 'YAxisLocation', 'right', ...
        'Color', 'none', 'XColor', 'w', 'YColor', 'w', 'Box', 'off');
    xlim(ph, [min(yc_km), max(yc_km)]);
    ylim(ph, [min(yall(:)), max(yall(:))] + [-eps, eps]);
    pbaspect(ph, [1 1 1]);
    ylabel(ph, 'Cross-shelf transport (m^2 s^{-1})', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'w');
    xlabel(ph, 'y (km)', 'FontSize', 16, 'FontWeight', 'bold', 'Color', 'w');
    title(ph, sprintf('(%s) %s', char('a' + (r-1)*4 + 3), rows(r).name), ...
        'FontSize', 14, 'FontWeight', 'bold', 'Color', 'w');
    line_ph(r) = ph;
end

drawnow;
% --- line plots: ∫ u dz vs y at x = x_line ---
% Square axes placed manually so their top lines up with the top of the maps
% in the same row (tiled axes with a square aspect would be vertically centered).
for r = 1:2
    mapP = map_axs(r, 3).Position;        % rightmost map in this row
    phP  = line_ph(r).Position;           % placeholder tile area (normalized)
    % =========================================================================
    % 🟢 EDIT LINE-PLOT SIZE / LEGEND POSITION HERE
    % size_scale = multiplies the square size (1 = size the tiled layout gives it)
    % lgd_gap    = space (px) between the bottom of the plot and the top of the
    %              legend; it must clear the tick labels and the x-label.
    % =========================================================================
    size_scale = 1.0;
    lgd_gap    = 50;

    % Square plot box as the tiled layout would draw it (same size and left/right
    % position as the placeholder), but moved up so its top lines up with the maps.
    wpx = phP(3) * fig_w;  hpx = phP(4) * fig_h;
    side_px = min(wpx, hpx) * size_scale;
    left = phP(1) + (wpx - side_px) / 2 / fig_w;
    top  = mapP(2) + mapP(4);
    ax = axes(fig, 'Position', [left, top - side_px / fig_h, side_px / fig_w, side_px / fig_h]);
    hold(ax, 'on');
    yline(ax, 0, '-', 'Color', [0.6 0.6 0.6], 'LineWidth', 0.8, 'HandleVisibility', 'off');
    hl = gobjects(1, 3);
    for k = 1:3
        hl(k) = plot(ax, yc_km, rows(r).uint{k}, '-', 'LineWidth', 2, 'Color', line_cols(k, :));
    end
    set(ax, 'FontSize', 14, 'Box', 'on', 'XGrid', 'on', 'YGrid', 'on', ...
        'LineWidth', 1, 'TickDir', 'out', 'XColor', 'k', 'YColor', 'k', ...
        'YAxisLocation', 'right');
    xlim(ax, [min(yc_km), max(yc_km)]);
    ylabel(ax, 'Cross-shelf transport (m^2 s^{-1})', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel(ax, 'y (km)', 'FontSize', 16, 'FontWeight', 'bold');
    title(ax, sprintf('(%s) %s', char('a' + (r-1)*4 + 3), rows(r).name), ...
        'FontSize', 14, 'FontWeight', 'bold');
    line_axs(r) = ax;

    % Legend centered underneath the x-label
    lg = legend(ax, hl, rows(r).legend, 'Location', 'none', 'FontSize', 11, 'Box', 'on');
    lg.Units = 'normalized';
    axP = ax.Position;
    lp  = lg.Position;
    lg.Position = [axP(1) + axP(3) / 2 - lp(3) / 2, ...
                   axP(2) - lgd_gap / fig_h - lp(4), lp(3), lp(4)];
end

% Shared colorbar per row, inside the bottom-right corner of the rightmost map.
% (Every map keeps a hidden colorbar so the panel sizes stay identical.)
for r = 1:2
    for k = 1:3
        cb = map_cbs(r, k);
        if k ~= 3
            cb.Visible = 'off';
            continue;
        end
        axP = map_axs(r, k).Position;
        % =========================================================================
        % 🟢 EDIT COLORBAR POSITIONS HERE
        %
        % axP(1) = left edge of panel     axP(3) = width of panel
        % axP(2) = bottom edge of panel   axP(4) = height of panel
        % =========================================================================
        cw = 0.02 * 1350 / fig_w;    % Colorbar width (same pixel width as vishoals_still)
        ch = axP(4) * 0.25;          % Colorbar height
        xpos = axP(1) + axP(3) - cw - 0.015 * 1350 / fig_w;   % larger shift = further left
        ypos = axP(2) + 0.02 * 700 / fig_h;     % larger = further up
        cb.Position = [xpos, ypos, cw, ch];
        cb.YAxisLocation = 'left'; % tick labels inside to the left
        cb.FontSize = 11;
        cb.Title.FontSize = 11;

        % Calculate exactly 3 ticks.
        % If range is very small, use 2-3 decimal places to avoid collapsing to 0.0.
        L1 = cb.Limits(1); L2 = cb.Limits(2);
        if (L2 - L1) < 0.021
            cmin = ceil(L1 * 1000) / 1000;
            cmax = floor(L2 * 1000) / 1000;
            mid = round((cmin + cmax) / 2, 3);
            fmt = '%.3f';
        elseif (L2 - L1) < 0.21
            cmin = ceil(L1 * 100) / 100;
            cmax = floor(L2 * 100) / 100;
            mid = round((cmin + cmax) / 2, 2);
            fmt = '%.2f';
        else
            cmin = ceil(L1 * 10) / 10;
            cmax = floor(L2 * 10) / 10;
            mid = round((cmin + cmax) / 2, 1);
            fmt = '%.1f';
        end
        ticks = unique([cmin, mid, cmax]);
        cb.Ticks = ticks;
        cb.TickLabels = sprintfc(fmt, ticks);
    end
end
drawnow;

% ---------- output ----------
if ~exist(outdir,'dir'), mkdir(outdir); end
clip_str = '';
if clipped
    clip_str = '_clipped';
end
outfile = fullfile(outdir, sprintf('%s_u_mean_sweeps%s.png', cluster_name, clip_str));
exportgraphics(fig, outfile, 'Resolution', 300);
fprintf('Saved still image: %s \n', outfile);
close(fig);

end

% ================== helpers ==================

function fname = find_time_avg(cluster_dir, sw_core)
% Locate time_avg_3d file inside the sweep_NNN_<sw_core> folder
d = dir(fullfile(cluster_dir, ['*' sw_core '*']));
d = d([d.isdir]);
d = d(~ismember({d.name}, {'.', '..'}));
if isempty(d)
    error('No folder found matching *%s* in %s', sw_core, cluster_dir);
end
datadir = fullfile(cluster_dir, d(1).name);
candidates = { fullfile(datadir, ['time_avg_3d_sweep_' sw_core '.nc']), ...
               fullfile(datadir, ['time_avg_3d_' sw_core '.nc']) };
for c = 1:numel(candidates)
    if isfile(candidates{c}), fname = candidates{c}; return; end
end
error('No time_avg_3d file for %s in %s', sw_core, datadir);
end

function data = read_last(fname, names, ix, iy)
% Read (Nx, Ny, Nz) block over [ix, iy] from the last time-averaging window
for i = 1:numel(names)
    try
        vinfo = ncinfo(fname, names{i});
        vsize = vinfo.Size;
        if numel(vsize) == 4
            data = ncread(fname, names{i}, [ix(1) iy(1) 1 vsize(4)], [numel(ix) numel(iy) vsize(3) 1]);
        else
            data = ncread(fname, names{i}, [ix(1) iy(1) 1], [numel(ix) numel(iy) vsize(3)]);
        end
        return;
    catch
        % try next name
    end
end
error('Could not read any of {%s} from %s', strjoin(names, ', '), fname);
end

function dz = cell_thickness(fname, zc)
% Cell thicknesses from face coordinates if present, else from center midpoints
dz = [];
for nm = {'zF', 'z_aaf', 'zf'}
    try
        zf = ncread(fname, nm{1});
        if numel(zf) == numel(zc) + 1
            dz = abs(diff(zf(:)));
            return;
        end
    catch
    end
end
if numel(zc) == 1
    dz = 1;
    return;
end
mid   = (zc(1:end-1) + zc(2:end)) / 2;
edges = [zc(1) - (mid(1) - zc(1)); mid; zc(end) + (zc(end) - mid(end))];
dz    = abs(diff(edges));
end

function data = try_ncread_(fname, names)
% Try reading coordinate from file using a list of possible variable names
for i = 1:numel(names)
    try
        data = ncread(fname, names{i});
        return;
    catch
        % try next name
    end
end
info = ncinfo(fname);
avail = strjoin({info.Variables.Name}, ', ');
error('Could not find coordinate variable. Tried: {%s}. Available: %s', ...
    strjoin(names, ', '), avail);
end
