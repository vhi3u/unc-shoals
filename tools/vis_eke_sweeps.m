function vis_eke_sweeps(cluster_name, varargin)
% vis_eke_sweeps(cluster_name, 'level', 'surface'/'depth_avg'/'depth_int', ...
%   'scale', 'log'/'linear', 'outdir', OUTDIR, ...
%   'clipped', true/false, 'x_avg', [x0 x1], 'climEKE', [lo hi], ...
%   'log_decades', N, 'bathy_levels', [...])
%
% Surface eddy kinetic energy (EKE) for the shoal height and shoal length
% sweeps, from the LAST time-averaging window of time_avg_3d_sweep_<run>.nc.
%
%   EKE = 1/2 [ (<u^2> - ubar^2) + (<v^2> - vbar^2) ]
%   MKE = 1/2 ( ubar^2 + vbar^2 )
% (<u^2> = uu, ubar = u_c from the time-averaged output; horizontal
%  components only; small negative round-off -> 0), evaluated as
%   'level' = 'surface'   : at the top grid level             [m^2 s^-2]
%   'level' = 'depth_avg' : averaged over the water column,
%                           (1/H) int EKE dz  (H = local depth) [m^2 s^-2]
%   'level' = 'depth_int' : integrated over the water column,
%                           int EKE dz (cell thickness, wet cells) [m^3 s^-2]
%
% Layout (2 rows x 4 columns, as vis_u_mean_sweeps.m):
%   Row 1 — shoal height (Ls = 40 km): Zs = -15, -10, -5 m | eddy fraction
%   Row 2 — shoal length (Zs = -5 m):  Ls = 20, 30, 40 km  | eddy fraction
%
%   Maps: surface EKE (log10 or linear colour scale), with bathymetry
%         contours. Dashed line marks the offshore edge of x_avg.
%   Lines: alongshore profile of the eddy fraction (same 'level')
%            EF(y) = <EKE>_x / ( <EKE>_x + <MKE>_x ),
%          with <.>_x the average over x_avg (coast to x_avg(2)).
%
% Each row shares one colour scale, shown on the rightmost map of that row.
%
% Requires: pmkmp (KE colormap, as in vishoals.m)
%
% Example:
%   vis_eke_sweeps('cluster_Fnnie_1_19')                     % log scale
%   vis_eke_sweeps('cluster_Fnnie_1_19', 'scale', 'linear')
%   vis_eke_sweeps('cluster_Fnnie_1_19', 'level', 'depth_avg')
%   vis_eke_sweeps('cluster_Fnnie_1_19', 'level', 'depth_int')

if nargin < 1 || isempty(cluster_name)
    cluster_name = 'cluster_Fnnie_1_19';
end

% ---------- options ----------
p = inputParser;
addParameter(p, 'sweep_root', '/glade/work/nguyen/unc-shoals/sweep_output', @ischar);
addParameter(p, 'outdir', '/glade/work/nguyen/unc-shoals/animations', @ischar);
addParameter(p, 'scale', 'log', @(x)ischar(x) && ismember(x, {'log','linear'}));
addParameter(p, 'level', 'surface', @(x)ischar(x) && ismember(x, {'surface','depth_avg','depth_int'}));
addParameter(p, 'clipped', true, @(x)islogical(x)&&isscalar(x));
addParameter(p, 'x_avg', [0 50], @(v)isnumeric(v) && numel(v)==2);           % km, cross-shelf averaging range for EF
addParameter(p, 'climEKE', [], @(v)isempty(v) || (isnumeric(v) && numel(v)==2)); % m^2/s^2 or m^3/s^2 (both rows; linear values even for log)
addParameter(p, 'log_decades', 3, @(x)isnumeric(x)&&isscalar(x)&&x>0);       % log scale: decades shown below the max
addParameter(p, 'bathy_levels', 5:5:50, @(v)isnumeric(v)&&isvector(v));     % depths (m) of the bathymetry contours
addParameter(p, 'bathy_color', [0.25 0.25 0.25], @(x)ischar(x) || (isnumeric(x) && numel(x)==3));
addParameter(p, 'bathy_linewidth', 0.6, @(x)isnumeric(x)&&isscalar(x)&&x>0);

parse(p, varargin{:});
use_log     = strcmp(p.Results.scale, 'log');
level       = p.Results.level;
if strcmp(level, 'depth_int')
    eke_units = '(m^3 s^{-2})';
else
    eke_units = '(m^2 s^{-2})';
end
clipped     = p.Results.clipped;
x_avg       = p.Results.x_avg;
outdir      = p.Results.outdir;
cluster_dir = fullfile(p.Results.sweep_root, cluster_name);

% ---------- sweeps ----------
rows(1).runs   = {'Zs15_Ls40k', 'Zs10_Ls40k', 'Zs5_Ls40k'};
rows(1).titles = {'Z_s = -15 m', 'Z_s = -10 m', 'Z_s = -5 m'};
rows(1).legend = {'Z_s = -15 m', 'Z_s = -10 m', 'Z_s = -5 m'};
rows(1).name   = 'Shoal Height';

rows(2).runs   = {'Zs5_Ls20k', 'Zs5_Ls30k', 'Zs5_Ls40k'};
rows(2).titles = {'L_s = 20 km', 'L_s = 30 km', 'L_s = 40 km'};
rows(2).legend = {'L_s = 20 km', 'L_s = 30 km', 'L_s = 40 km'};
rows(2).name   = 'Shoal Length';

% Line colors: same blue / red / green used in the other sweep plots
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
clip_y = [20 200];  % km (excludes the 20 km southern nudging zone)
if clipped
    ix = find(xc_km >= clip_x(1) & xc_km <= clip_x(2));
    iy = find(yc_km >= clip_y(1) & yc_km <= clip_y(2));
end
xc_km = xc_km(ix);
yc_km = yc_km(iy);
[Xc, Yc] = meshgrid(xc_km, yc_km);

dz = cell_thickness(fname0, zc);
[~, k_top] = max(zc);
ix_avg = find(xc_km >= x_avg(1) & xc_km <= x_avg(2));
x_edge = xc_km(ix_avg(end));

% ---------- load data ----------
for r = 1:2
    nr = numel(rows(r).runs);
    rows(r).eke   = cell(1, nr);   % surface EKE (Ny x Nx)
    rows(r).depth = cell(1, nr);   % water depth (Ny x Nx)
    rows(r).ef    = cell(1, nr);   % eddy fraction vs y (Ny x 1)
    for k = 1:nr
        fname = find_time_avg(cluster_dir, rows(r).runs{k});
        fprintf('%-12s -> %s\n', rows(r).runs{k}, fname);

        if strcmp(level, 'surface')
            % Surface level, last averaging window
            u  = read_level(fname, {'u_c','u'}, ix, iy, k_top);
            v  = read_level(fname, {'v_c','v'}, ix, iy, k_top);
            uu = read_level(fname, {'uu'},      ix, iy, k_top);
            vv = read_level(fname, {'vv'},      ix, iy, k_top);
            Ts = read_level(fname, {'T'},       ix, iy, k_top);
            land = (Ts == 0) | ~isfinite(Ts);

            eke = 0.5 * ((uu - u.^2) + (vv - v.^2));
            eke(eke < 0) = 0;                          % round-off
            mke = 0.5 * (u.^2 + v.^2);
        else
            % Full water column, last averaging window; integrate (or average) with cell thickness
            u  = read_block(fname, {'u_c','u'}, ix, iy);
            v  = read_block(fname, {'v_c','v'}, ix, iy);
            uu = read_block(fname, {'uu'},      ix, iy);
            vv = read_block(fname, {'vv'},      ix, iy);
            T3 = read_block(fname, {'T'},       ix, iy);
            wet = (T3 ~= 0) & isfinite(T3);
            DZ  = reshape(dz, 1, 1, []) .* wet;        % thickness of wet cells only

            e3 = 0.5 * ((uu - u.^2) + (vv - v.^2));
            e3(e3 < 0 | ~isfinite(e3)) = 0;
            m3 = 0.5 * (u.^2 + v.^2);
            m3(~isfinite(m3)) = 0;
            eke  = sum(e3 .* DZ, 3);                   % int EKE dz  [m^3/s^2]
            mke  = sum(m3 .* DZ, 3);
            land = ~any(wet, 3);
            if strcmp(level, 'depth_avg')
                H = sum(DZ, 3);                        % local water depth
                eke = eke ./ H;                        % (1/H) int EKE dz  [m^2/s^2]
                mke = mke ./ H;
            end
        end
        eke(land) = NaN;  mke(land) = NaN;

        % Eddy fraction along y: ratio of cross-shelf averages over x_avg
        eke_x = mean(eke(ix_avg, :), 1, 'omitnan');   % (1 x Ny)
        mke_x = mean(mke(ix_avg, :), 1, 'omitnan');
        rows(r).ef{k}  = (eke_x ./ (eke_x + mke_x))';
        rows(r).eke{k} = eke';

        % Water depth (bathymetry) for contours
        rows(r).depth{k} = water_depth(fname, ix, iy, dz)';
    end
end

% ---------- colour limits (shared within each row) ----------
trim = @(x) x(2:end-1, 2:end-1);
for r = 1:2
    if ~isempty(p.Results.climEKE)
        clim_lin = p.Results.climEKE;
    else
        emax = 0;
        for k = 1:3
            emax = max(emax, max(reshape(trim(rows(r).eke{k}), [], 1), [], 'omitnan'));
        end
        if use_log
            clim_lin = [emax * 10^(-p.Results.log_decades), emax];
        else
            clim_lin = [0, emax];
        end
    end
    rows(r).clim_lin = clim_lin;
    if use_log
        rows(r).clim = log10(clim_lin);
    else
        rows(r).clim = clim_lin;
    end
    fprintf('EKE colour limits (row %d): [%.3g, %.3g] %s (%s, %s)\n', r, clim_lin(1), clim_lin(2), ...
        strrep(strrep(eke_units, '{', ''), '}', ''), level, p.Results.scale);
end

% ---------- figure / panels ----------
% Same layout as vis_u_mean_sweeps.m: 2 rows x 4 columns, 900 px wide
% (per-panel width / compact spacing of the 6-column 1350 px figures).
fig_w = 1350 * 4/6; fig_h = 700;
fig = figure('Color','w','Position',[80 50 fig_w fig_h]);
tl  = tiledlayout(fig, 2, 4, 'TileSpacing','compact','Padding','compact');
cmap = pmkmp(100, 'swtth');      % KE colormap used in vishoals.m

map_axs = gobjects(2, 3);
map_cbs = gobjects(2, 3);
line_axs = gobjects(1, 2);
line_ph  = gobjects(1, 2);
for r = 1:2
    for k = 1:3
        ax = nexttile(tl, (r-1)*4 + k);
        E = rows(r).eke{k};
        if use_log
            E = log10(max(E, rows(r).clim_lin(1)));     % floor at the lower limit
        end
        pcolor(ax, Xc, Yc, E);
        shading(ax,'flat'); axis(ax,'tight'); set(ax,'YDir','normal');
        set(ax, 'Layer', 'top', 'Color', [0.9 0.9 0.9]);   % gray = land
        box(ax, 'on'); set(ax, 'LineWidth', 0.8, 'XColor', 'k', 'YColor', 'k');
        colormap(ax, cmap); caxis(ax, rows(r).clim);
        set(ax, 'NextPlot', 'add');
        map_cbs(r, k) = colorbar(ax);
        map_cbs(r, k).Title.String = eke_units;
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

        % --- bathymetry contours ---
        D = rows(r).depth{k};
        lev = p.Results.bathy_levels(p.Results.bathy_levels < max(D(:), [], 'omitnan'));
        if ~isempty(lev)
            contour(ax, Xc, Yc, D, as_levels(lev), 'Color', p.Results.bathy_color, ...
                'LineWidth', p.Results.bathy_linewidth);
        end

        % Offshore edge of the eddy-fraction averaging range
        xline(ax, x_edge, '--', 'Color', [0.3 0.3 0.3], 'LineWidth', 1);
        map_axs(r, k) = ax;
    end

    % --- placeholder tile for the line plot (real axes drawn after layout) ---
    % Styled like the line plot but drawn in white, so the tiled layout
    % reserves the same space as a real line plot.
    ph = nexttile(tl, (r-1)*4 + 4);
    set(ph, 'FontSize', 14, 'LineWidth', 1, 'TickDir', 'out', 'YAxisLocation', 'right', ...
        'Color', 'none', 'XColor', 'w', 'YColor', 'w', 'Box', 'off');
    xlim(ph, [min(yc_km), max(yc_km)]);
    ylim(ph, [0 1]);
    pbaspect(ph, [1 1 1]);
    ylabel(ph, 'Eddy fraction', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'w');
    xlabel(ph, 'y (km)', 'FontSize', 16, 'FontWeight', 'bold', 'Color', 'w');
    title(ph, sprintf('(%s) %s', char('a' + (r-1)*4 + 3), rows(r).name), ...
        'FontSize', 14, 'FontWeight', 'bold', 'Color', 'w');
    line_ph(r) = ph;
end

drawnow;
% --- line plots: eddy fraction vs y ---
% Square axes, top-aligned with the maps of the same row.
for r = 1:2
    mapP = map_axs(r, 3).Position;
    phP  = line_ph(r).Position;
    % =========================================================================
    % 🟢 EDIT LINE-PLOT SIZE / LEGEND POSITION HERE
    % size_scale = multiplies the square size (1 = size the tiled layout gives it)
    % lgd_gap    = space (px) between the bottom of the plot and the top of the legend
    % =========================================================================
    size_scale = 1.0;
    lgd_gap    = 50;

    wpx = phP(3) * fig_w;  hpx = phP(4) * fig_h;
    side_px = min(wpx, hpx) * size_scale;
    left = phP(1) + (wpx - side_px) / 2 / fig_w;
    top  = mapP(2) + mapP(4);
    ax = axes(fig, 'Position', [left, top - side_px / fig_h, side_px / fig_w, side_px / fig_h]);
    hold(ax, 'on');
    hl = gobjects(1, 3);
    for k = 1:3
        hl(k) = plot(ax, yc_km, rows(r).ef{k}, '-', 'LineWidth', 2, 'Color', line_cols(k, :));
    end
    efall = [rows(r).ef{:}];
    set(ax, 'FontSize', 14, 'Box', 'on', 'XGrid', 'on', 'YGrid', 'on', ...
        'LineWidth', 1, 'TickDir', 'out', 'XColor', 'k', 'YColor', 'k', ...
        'YAxisLocation', 'right');
    xlim(ax, [min(yc_km), max(yc_km)]);
    ylim(ax, [0, min(1, max(efall(:), [], 'omitnan') * 1.1 + eps)]);
    ylabel(ax, 'EKE / (EKE + MKE)', 'FontSize', 12, 'FontWeight', 'bold');
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
        % =========================================================================
        cw = 0.02 * 1350 / fig_w;
        ch = axP(4) * 0.25;
        xpos = axP(1) + axP(3) - cw - 0.015 * 1350 / fig_w;
        ypos = axP(2) + 0.02 * 700 / fig_h;
        cb.Position = [xpos, ypos, cw, ch];
        cb.YAxisLocation = 'left';
        cb.FontSize = 11;
        cb.Title.FontSize = 11;

        if use_log
            % Ticks at whole powers of ten, labelled 10^n
            tk = ceil(rows(r).clim(1)):floor(rows(r).clim(2));
            if isempty(tk), tk = rows(r).clim; end
            if numel(tk) > 4, tk = tk(round(linspace(1, numel(tk), 3))); end
            cb.Ticks = tk;
            cb.TickLabels = arrayfun(@(e) sprintf('10^{%d}', round(e)), tk, 'UniformOutput', false);
        else
            tk = linspace(rows(r).clim(1), rows(r).clim(2), 3);
            cb.Ticks = tk;
            cb.TickLabels = sprintfc('%.1e', tk);
        end
    end
end
drawnow;

% ---------- output ----------
if ~exist(outdir,'dir'), mkdir(outdir); end
clip_str = '';
if clipped, clip_str = '_clipped'; end
outfile = fullfile(outdir, sprintf('%s_eke_sweeps_%s_%s%s.png', cluster_name, level, p.Results.scale, clip_str));
exportgraphics(fig, outfile, 'Resolution', 300);
fprintf('Saved still image: %s \n', outfile);
close(fig);

end

% ================== helpers ==================

function lev = as_levels(lev)
% contour() treats a scalar as a NUMBER of levels; repeat it to mean one level
if isscalar(lev), lev = [lev lev]; end
end

function D = water_depth(fname, ix, iy, dz)
% Water depth (m, positive) from bottom_height if present, else from wet cells
D = [];
try
    hb = ncread(fname, 'bottom_height');
    hb = squeeze(hb(:, :, 1, 1));
    D  = abs(hb(ix, iy));
    return;
catch
end
vinfo = ncinfo(fname, 'T');  vsize = vinfo.Size;
if numel(vsize) == 4
    T = ncread(fname, 'T', [ix(1) iy(1) 1 vsize(4)], [numel(ix) numel(iy) vsize(3) 1]);
else
    T = ncread(fname, 'T', [ix(1) iy(1) 1], [numel(ix) numel(iy) vsize(3)]);
end
wet = (T ~= 0) & isfinite(T);
D = sum(wet .* reshape(dz, 1, 1, []), 3);
end

function A = read_block(fname, names, ix, iy)
% (Nx, Ny, Nz) block over [ix, iy], last time-averaging window
for i = 1:numel(names)
    try
        vinfo = ncinfo(fname, names{i});
        vsize = vinfo.Size;
        if numel(vsize) == 4
            A = ncread(fname, names{i}, [ix(1) iy(1) 1 vsize(4)], [numel(ix) numel(iy) vsize(3) 1]);
        else
            A = ncread(fname, names{i}, [ix(1) iy(1) 1], [numel(ix) numel(iy) vsize(3)]);
        end
        return;
    catch
        % try next name
    end
end
error('Could not read any of {%s} from %s', strjoin(names, ', '), fname);
end

function A = read_level(fname, names, ix, iy, k)
% (Nx, Ny) slice at vertical level k, last time-averaging window
for i = 1:numel(names)
    try
        vinfo = ncinfo(fname, names{i});
        vsize = vinfo.Size;
        if numel(vsize) == 4
            A = ncread(fname, names{i}, [ix(1) iy(1) k vsize(4)], [numel(ix) numel(iy) 1 1]);
        else
            A = ncread(fname, names{i}, [ix(1) iy(1) k], [numel(ix) numel(iy) 1]);
        end
        A = squeeze(A);
        return;
    catch
        % try next name
    end
end
error('Could not read any of {%s} from %s', strjoin(names, ', '), fname);
end

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

function dz = cell_thickness(fname, zc)
% Cell thicknesses from face coordinates if present, else from center midpoints
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
if numel(zc) == 1, dz = 1; return; end
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
