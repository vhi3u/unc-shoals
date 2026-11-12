function vishoals_vort_wind(cluster_name, varargin)
% vishoals_vort_wind(cluster_name, 'wind_sign', 'positive'/'negative', ...
%   'outdir', OUTDIR, 'clipped', true/false, ...
%   'isopycnals', true/false, 'rho_levels', ..., 'climRo', [lo hi])
%
% 6-panel still image of surface (XY) vorticity (Ro = zeta/f) across the
% wind-stress sweep, using the LAST saved frame of each run. The leftmost
% panel is always the no-wind reference (baseline Zs5_Ls40k):
%   'positive' (default): tau = 0,  0.03,  0.05,  0.08,  0.10,  0.15 N m^-2
%   'negative':           tau = 0, -0.03, -0.05, -0.08, -0.10, -0.15 N m^-2
%
% Reads top_<run>.nc from each run folder in
%   <sweep_root>/<cluster_name>/sweep_NNN_<run>/
%
% Figure width and panel spacing match vishoals_still.m 'full_panel'
% (top row only). All panels share one color scale.
%
% Requires: cmocean, GSW Toolbox (for isopycnals)
%
% Example:
%   vishoals_vort_wind('cluster_Fnnie_1_19', 'clipped', true)
%   vishoals_vort_wind('cluster_Fnnie_1_19', 'clipped', true, 'wind_sign', 'negative')

if nargin < 1 || isempty(cluster_name)
    cluster_name = 'cluster_Fnnie_1_19';
end

% ---------- options ----------
p = inputParser;
addParameter(p, 'sweep_root', '/glade/work/nguyen/unc-shoals/sweep_output', @ischar);
addParameter(p, 'outdir', '/glade/work/nguyen/unc-shoals/animations', @ischar);
addParameter(p, 'wind_sign', 'positive', @(x)ischar(x) && ismember(x, {'positive','negative'}));
addParameter(p, 'clipped', false, @(x)islogical(x)&&isscalar(x));
% Isopycnal options
addParameter(p, 'isopycnals', true, @(x)islogical(x)&&isscalar(x));
addParameter(p, 'rho_levels', [23:0.025:24.0, 24.1:0.4:26.4, 26.6:0.8:28], @(v)isnumeric(v)&&isvector(v));
addParameter(p, 'iso_color', 'k', @ischar);
addParameter(p, 'iso_linewidth', 0.8, @(x)isnumeric(x)&&isscalar(x)&&x>0);
% Manual color limits (optional)
addParameter(p, 'climRo', [], @(v)isempty(v) || (isnumeric(v) && numel(v)==2));
addParameter(p, 'no_outlier', false, @(x)islogical(x)&&isscalar(x));

parse(p, varargin{:});
wind_sign    = p.Results.wind_sign;
clipped      = p.Results.clipped;
want_iso     = p.Results.isopycnals;
rho_levels   = p.Results.rho_levels;
iso_color    = p.Results.iso_color;
iso_lw       = p.Results.iso_linewidth;
climRo_user  = p.Results.climRo;
no_outlier   = p.Results.no_outlier;
outdir       = p.Results.outdir;

% ---------- runs: no-wind reference first, then the chosen wind sweep ----------
tau_mag = [0.03, 0.05, 0.08, 0.10, 0.15];
if strcmp(wind_sign, 'negative')
    prefix = 'sweep_wind_m';
    tau_vals = [0, -tau_mag];
else
    prefix = 'sweep_wind_';
    tau_vals = [0, tau_mag];
end
runs = [{'sweep_Zs5_Ls40k'}, arrayfun(@(t) sprintf('%s%sp%02d', prefix, '0', round(t*100)), ...
                                      tau_mag, 'UniformOutput', false)];
n_runs = numel(runs);

% ---------- locate files ----------
cluster_dir = fullfile(p.Results.sweep_root, cluster_name);
fname_top = cell(1, n_runs);
for k = 1:n_runs
    sw_core = strrep(runs{k}, 'sweep_', '');
    d = dir(fullfile(cluster_dir, ['*' sw_core '*']));
    d = d([d.isdir]);
    d = d(~ismember({d.name}, {'.', '..'}));
    if isempty(d)
        error('No folder found matching *%s* in %s', sw_core, cluster_dir);
    end
    fname_top{k} = fullfile(cluster_dir, d(1).name, ['top_' runs{k} '.nc']);
    if ~isfile(fname_top{k})
        error('Surface file not found: %s', fname_top{k});
    end
end

% ---------- coordinates (from first run; grids assumed identical) ----------
% Oceananigans NetCDFWriter uses xC/yC/zC for Center-located fields.
xc_top = try_ncread_(fname_top{1}, {'xC','x_caa','xc'});
yc_top = try_ncread_(fname_top{1}, {'yC','y_aca','yc'});

xc_km_top = xc_top / 1000;
yc_km_top = yc_top / 1000;
ix_top = 1:numel(xc_top);
iy_top = 1:numel(yc_top);
clip_x = [0 100];   % km, same as vishoals_still.m
clip_y = [20 200];  % km
if clipped
    ix_top = find(xc_km_top >= clip_x(1) & xc_km_top <= clip_x(2));
    iy_top = find(yc_km_top >= clip_y(1) & yc_km_top <= clip_y(2));
    xc_km_top = xc_km_top(ix_top);
    yc_km_top = yc_km_top(iy_top);
end
Nx_top = numel(ix_top);
Ny_top = numel(iy_top);

% XY grids in km
[Xc, Yc] = meshgrid(xc_km_top, yc_km_top);

% ---------- TEOS-10 prep for isopycnals ----------
lat = 35.2480;
lon = 0;
P_xy = gsw_p_from_z(0, lat) * ones(Ny_top, Nx_top);

% ---------- read last frame of each run ----------
% Surface file: data is (Nx, Ny, 1, time) — z is singleton
read_surf = @(fname, varname, t) squeeze(ncread(fname, varname, ...
    [ix_top(1) iy_top(1) 1 t], [Nx_top Ny_top 1 1]));

    function A = read_surf_safe(fname, varname, t)
        try, A = read_surf(fname, varname, t)'; catch, A = nan(Ny_top, Nx_top); end
    end

Ro_all  = cell(1, n_runs);
T_all   = cell(1, n_runs);
S_all   = cell(1, n_runs);
day_end = zeros(1, n_runs);
for k = 1:n_runs
    time = ncread(fname_top{k}, 'time');
    t = numel(time);
    day_end(k) = (time(t) - time(1)) / 86400;
    fprintf('tau = %.2f  ->  %s  (last frame: day %.1f)\n', tau_vals(k), fname_top{k}, day_end(k));

    % Surface XY — transposed for pcolor (Ny×Nx), land/immersed cells masked
    T = read_surf_safe(fname_top{k}, 'T', t);  T(T == 0) = NaN;
    S = read_surf_safe(fname_top{k}, 'S', t);  S(S == 0) = NaN;
    R = read_surf_safe(fname_top{k}, 'Ro', t); R(isnan(T)) = NaN;
    Ro_all{k} = R; T_all{k} = T; S_all{k} = S;
end
if range(day_end) > 0.5
    warning('Runs end on different days (%.1f to %.1f).', min(day_end), max(day_end));
end

% ---------- shared color limits ----------
% Trim boundary cells (i=1, i=end, j=1, j=end) to avoid numerical artifacts in color limits
trim = @(x) x(2:end-1, 2:end-1);
if no_outlier
    getmin = @(x) prctile(reshape(trim(x), [], 1), 1);
    getmax = @(x) prctile(reshape(trim(x), [], 1), 99);
else
    getmin = @(x) min(reshape(trim(x), [], 1), [], 'omitnan');
    getmax = @(x) max(reshape(trim(x), [], 1), [], 'omitnan');
end

if isempty(climRo_user)
    ro_min = inf; ro_max = -inf;
    for k = 1:n_runs
        ro_min = min(ro_min, getmin(Ro_all{k}));
        ro_max = max(ro_max, getmax(Ro_all{k}));
    end
    rm = max(abs([ro_min, ro_max]));
    clim_ro = [-rm, rm];
else
    clim_ro = climRo_user;
end
fprintf('\nColor limits (Ro): [%.4g, %.4g]\n\n', clim_ro(1), clim_ro(2));

% ---------- figure / panels ----------
% Same width / spacing as vishoals_still.m full_panel (1350 px, 6 columns,
% compact), with the height reduced since the XZ row is dropped.
fig_w = 1350; fig_h = 520;
fig = figure('Color','w','Position',[80 50 fig_w fig_h]);
tl  = tiledlayout(fig, 1, 6, 'TileSpacing','compact','Padding','compact');

axs = gobjects(1, n_runs);
cbs = gobjects(1, n_runs);
cb_panel = n_runs;  % which panel shows the (shared) colorbar; 1 = leftmost, n_runs = rightmost
cmap = cmocean('balance');
for k = 1:n_runs
    ax = nexttile(tl, k);
    pcolor(ax, Xc, Yc, Ro_all{k});
    shading(ax,'flat'); axis(ax,'tight'); set(ax,'YDir','normal');
    set(ax, 'Layer', 'top'); box(ax, 'on'); set(ax, 'LineWidth', 0.8, 'XColor', 'k', 'YColor', 'k');
    colormap(ax, cmap); caxis(ax, clim_ro);
    cbs(k) = colorbar(ax);
    cbs(k).Title.String = '(Ro)';
    if tau_vals(k) == 0
        title_str = '\tau = 0';
    else
        title_str = sprintf('\\tau = %.2f N m^{-2}', tau_vals(k));
    end
    title(ax, sprintf('(%s) %s', char('a' + k - 1), title_str), 'FontSize', 18, 'FontWeight', 'bold');
    xlabel(ax, 'x (km)', 'FontSize', 16, 'FontWeight', 'bold');
    if k == 1
        ylabel(ax, 'y (km)', 'FontSize', 16, 'FontWeight', 'bold');
    else
        yticklabels(ax, []);
    end
    set(ax, 'FontSize', 14);
    xlim(ax, [min(xc_km_top), max(xc_km_top)]);
    ylim(ax, [min(yc_km_top), max(yc_km_top)]);
    set(ax, 'NextPlot', 'add');
    axs(k) = ax;
end

drawnow;
for k = 1:n_runs
    axP = axs(k).Position;
    % =========================================================================
    % 🟢 EDIT COLORBAR POSITIONS HERE
    %
    % axP(1) = left edge of panel     axP(3) = width of panel
    % axP(2) = bottom edge of panel   axP(4) = height of panel
    % Vertical offset is scaled so it matches the 700 px tall vishoals_still figure.
    % =========================================================================
    cw = 0.02; % Colorbar width
    ch = axP(4) * 0.18; % Colorbar height
    xpos = axP(1) + axP(3) - cw - 0.015;  % last term: shift left (normalized figure units); more negative = further left
    ypos = axP(2) + 0.02 * 700 / fig_h;   % bottom-right corner; larger = further up
    cbs(k).Position = [xpos, ypos, cw, ch];
    cbs(k).YAxisLocation = 'left'; % tick labels inside to the left
    cbs(k).FontSize = 11;
    cbs(k).Title.FontSize = 11;

    % Calculate exactly 3 ticks.
    % If range is very small, use 2 decimal places to avoid collapsing to 0.0.
    L1 = cbs(k).Limits(1); L2 = cbs(k).Limits(2);
    if (L2 - L1) < 0.21
        cmin = ceil(L1 * 100) / 100;
        cmax = floor(L2 * 100) / 100;
        mid = round((cmin + cmax) / 2, 2);
        ticks = unique([cmin, mid, cmax]);
        cbs(k).Ticks = ticks;
        cbs(k).TickLabels = sprintfc('%.2f', ticks);
    else
        cmin = ceil(L1 * 10) / 10;
        cmax = floor(L2 * 10) / 10;
        mid = round((cmin + cmax) / 2, 1);
        ticks = unique([cmin, mid, cmax]);
        cbs(k).Ticks = ticks;
        cbs(k).TickLabels = sprintfc('%.1f', ticks);
    end

    % All panels share one color scale, so only show the colorbar on one panel.
    % (Every panel keeps a hidden colorbar so the panel sizes stay identical.)
    if k ~= cb_panel
        cbs(k).Visible = 'off';
    end
end

% --- TEOS-10 σ0 surface isopycnals ---
if want_iso
    for k = 1:n_runs
        T = T_all{k}; S = S_all{k};
        bad = ~isfinite(T) | ~isfinite(S);
        T(bad) = NaN; S(bad) = NaN;
        SA = gsw_SA_from_SP(S, P_xy, lon, lat);
        CT = gsw_CT_from_t(SA, T, P_xy);
        sigma0 = gsw_sigma0(SA, CT);

        fin = sigma0(isfinite(sigma0));
        if ~isempty(fin) && range(fin) > 0
            [C, hc] = contour(axs(k), Xc, Yc, sigma0, rho_levels, ...
                'Color', iso_color, 'LineWidth', iso_lw);
            if ~isempty(C)
                clabel(C, hc, 'FontSize', 7, 'Color', iso_color, 'LabelSpacing', 400);
            end
        end
    end
end
drawnow;

% ---------- output ----------
if ~exist(outdir,'dir'), mkdir(outdir); end
clip_str = '';
if clipped
    clip_str = '_clipped';
end
outfile = fullfile(outdir, sprintf('%s_wind_vort_6panel_%s_lastframe%s.png', cluster_name, wind_sign, clip_str));
exportgraphics(fig, outfile, 'Resolution', 300);
fprintf('Saved still image: %s \n', outfile);
close(fig);

end

% ================== helpers ==================

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
% If nothing worked, list what's in the file
info = ncinfo(fname);
avail = strjoin({info.Variables.Name}, ', ');
error('Could not find coordinate variable. Tried: {%s}. Available: %s', ...
    strjoin(names, ', '), avail);
end
