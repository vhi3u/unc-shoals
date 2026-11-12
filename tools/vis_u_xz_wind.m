function vis_u_xz_wind(cluster_name, varargin)
% vis_u_xz_wind(cluster_name, 'outdir', OUTDIR, 'clipped', true/false, ...
%   'y_slice', Y_KM, 'climu', [lo hi], 'rho_step', DRHO, 'isopycnals', true/false, ...
%   'eos_alpha', A, 'eos_beta', B, 'rho0', R0)
%
% 6-panel still of the time-mean cross-shore velocity u on an x-z slice, from
% the LAST time-averaging window of time_avg_3d_sweep_<run>.nc, for the
% no-wind reference and the upwelling-favourable wind sweep:
%   tau = 0 (Zs5_Ls40k), 0.03, 0.05, 0.08, 0.10, 0.15 N m^-2
%
% Slice: y = y_slice (default: the y of the run's midy_ file, i.e. the same
% transect as vishoals.m; domain centre if that file is missing).
%
% Isopycnals: density from the model's LINEAR equation of state (the runs do
% not use TEOS-10),
%   rho = rho0 [ 1 - alpha (T - T_ref) + beta (S - S_ref) ],
% with T_ref, S_ref = section means of the reference run (so the same density
% surfaces are contoured in every panel), every rho_step kg m^-3 (default 0.1,
% the same spacing as the vishoals_still.m contours).
%
% Panels are small squares, about the size of the x-z panels of vishoals.m
% full_panel (1350 px wide, 6 columns, compact). One shared colour scale,
% shown on the rightmost panel.
%
% Requires: cmocean
%
% Example:
%   vis_u_xz_wind('cluster_Fnnie_1_19')

if nargin < 1 || isempty(cluster_name)
    cluster_name = 'cluster_Fnnie_1_19';
end

% ---------- options ----------
p = inputParser;
addParameter(p, 'sweep_root', '/glade/work/nguyen/unc-shoals/sweep_output', @ischar);
addParameter(p, 'outdir', '/glade/work/nguyen/unc-shoals/animations', @ischar);
addParameter(p, 'clipped', true, @(x)islogical(x)&&isscalar(x));
addParameter(p, 'y_slice', [], @(x)isempty(x) || (isnumeric(x)&&isscalar(x)));     % km
addParameter(p, 'climu', [], @(v)isempty(v) || (isnumeric(v) && numel(v)==2));
addParameter(p, 'isopycnals', true, @(x)islogical(x)&&isscalar(x));
addParameter(p, 'rho_step', 0.1, @(x)isnumeric(x)&&isscalar(x)&&x>0);              % kg/m^3
addParameter(p, 'iso_color', 'k', @(x)ischar(x) || (isnumeric(x) && numel(x)==3));
addParameter(p, 'iso_linewidth', 0.8, @(x)isnumeric(x)&&isscalar(x)&&x>0);
addParameter(p, 'eos_alpha', 1.67e-4, @(x)isnumeric(x)&&isscalar(x));   % thermal expansion  [1/K]
addParameter(p, 'eos_beta',  7.80e-4, @(x)isnumeric(x)&&isscalar(x));   % haline contraction [1/psu]
addParameter(p, 'rho0',      1024,    @(x)isnumeric(x)&&isscalar(x));   % reference density  [kg/m^3]

parse(p, varargin{:});
clipped     = p.Results.clipped;
want_iso    = p.Results.isopycnals;
outdir      = p.Results.outdir;
cluster_dir = fullfile(p.Results.sweep_root, cluster_name);

% ---------- runs: no-wind reference first, then the upwelling sweep ----------
runs     = {'Zs5_Ls40k', 'wind_0p03', 'wind_0p05', 'wind_0p08', 'wind_0p10', 'wind_0p15'};

% Order here is MEANINGFUL (wind-stress axis / panel order), so this is NOT
% auto-discovered -- discover_sweeps returns directory order and would
% scramble the axis. Runs absent from this cluster are dropped and reported.
runs = keep_existing(cluster_dir, runs);

tau_vals = [0, 0.03, 0.05, 0.08, 0.10, 0.15];
n_runs   = numel(runs);

% ---------- grid (from the reference run; grids assumed identical) ----------
[fname0, fmidy0] = find_files(cluster_dir, runs{1});
xc = try_ncread_(fname0, {'xC','x_caa','xc','x'});  xc = xc(:);
yc = try_ncread_(fname0, {'yC','y_aca','yc','y'});  yc = yc(:);
zc = try_ncread_(fname0, {'zC','z_aac','zc','z'});  zc = zc(:);
xc_km = xc / 1000;  yc_km = yc / 1000;

% Slice position: same transect as the midy_ file used by vishoals.m
y_slice = p.Results.y_slice;
if isempty(y_slice)
    y_slice = mean(yc_km([1 end]));
    if ~isempty(fmidy0)
        try
            ym = try_ncread_(fmidy0, {'yC','y_aca','yc','y'});
            y_slice = ym(1) / 1000;
        catch
        end
    end
end
[~, iy] = min(abs(yc_km - y_slice));
fprintf('x-z slice at y = %.1f km\n', yc_km(iy));

ix = 1:numel(xc);
if clipped
    ix = find(xc_km >= 0 & xc_km <= 100);           % same x-clip as vishoals.m
end
xs = xc_km(ix);
[Xz, Zz] = meshgrid(xs, zc);

% ---------- read the last averaging window of each run ----------
U = cell(1, n_runs);  T = cell(1, n_runs);  S = cell(1, n_runs);
for k = 1:n_runs
    fname = find_files(cluster_dir, runs{k});
    fprintf('tau = %.2f  ->  %s\n', tau_vals(k), fname);
    u  = read_slice(fname, {'u_c','u'}, ix, iy);     % (Nx, Nz)
    Tk = read_slice(fname, {'T'}, ix, iy);
    Sk = read_slice(fname, {'S'}, ix, iy);
    land = (Tk == 0) | ~isfinite(Tk);
    u(land) = NaN;  Tk(land) = NaN;  Sk(land) = NaN;
    U{k} = u';  T{k} = Tk';  S{k} = Sk';               % (Nz, Nx) for pcolor
end

% ---------- shared colour limits ----------
if isempty(p.Results.climu)
    um = 0;
    for k = 1:n_runs, um = max(um, max(abs(U{k}(:)), [], 'omitnan')); end
    clim_u = [-um, um];
else
    clim_u = p.Results.climu;
end
fprintf('Colour limits (u): [%.4g, %.4g] m/s\n', clim_u(1), clim_u(2));

% ---------- linear-EOS density, same reference for every panel ----------
T_ref = mean(T{1}(:), 'omitnan');  S_ref = mean(S{1}(:), 'omitnan');
lin_rho = @(Tk, Sk) p.Results.rho0 * (1 - p.Results.eos_alpha * (Tk - T_ref) + p.Results.eos_beta * (Sk - S_ref));
RHO = cellfun(lin_rho, T, S, 'UniformOutput', false);
rmin = min(cellfun(@(r) min(r(:), [], 'omitnan'), RHO));
rmax = max(cellfun(@(r) max(r(:), [], 'omitnan'), RHO));
rho_levels = (ceil(rmin / p.Results.rho_step):floor(rmax / p.Results.rho_step)) * p.Results.rho_step;
fprintf('Isopycnals every %.3g kg/m^3 (%d levels; rho0 = %g at T_ref = %.2f, S_ref = %.2f)\n', ...
    p.Results.rho_step, numel(rho_levels), p.Results.rho0, T_ref, S_ref);

% ---------- figure ----------
% 1350 px wide, 6 columns, compact: each panel is about the size of an x-z
% panel of vishoals.m full_panel (~200 px), and close to square.
% =========================================================================
% 🟢 EDIT FIGURE SIZE HERE (fig_h sets how square the panels are)
% =========================================================================
fig_w = 1350; fig_h = 230;
fig = figure('Color','w','Position',[80 50 fig_w fig_h]);
tl  = tiledlayout(fig, 1, 6, 'TileSpacing','compact','Padding','compact');
cmap = cmocean('balance');

axs = gobjects(1, n_runs);
cbs = gobjects(1, n_runs);
cb_panel = n_runs;      % which panel shows the shared colorbar
for k = 1:n_runs
    ax = nexttile(tl, k);
    pcolor(ax, Xz, Zz, U{k});
    shading(ax, 'flat'); set(ax, 'YDir', 'normal');
    set(ax, 'Layer', 'top', 'Color', [0.9 0.9 0.9]);   % gray = land / seafloor
    box(ax, 'on'); set(ax, 'LineWidth', 0.8, 'XColor', 'k', 'YColor', 'k');
    colormap(ax, cmap); caxis(ax, clim_u);
    hold(ax, 'on');
    if want_iso && ~isempty(rho_levels)
        contour(ax, Xz, Zz, RHO{k}, as_levels(rho_levels), 'Color', p.Results.iso_color, ...
            'LineWidth', p.Results.iso_linewidth);
    end
    cbs(k) = colorbar(ax);
    cbs(k).Title.String = '(m s^{-1})';
    if tau_vals(k) == 0
        title_str = '\tau = 0';
    else
        title_str = sprintf('\\tau = %.2f N m^{-2}', tau_vals(k));
    end
    title(ax, sprintf('(%s) %s', char('a' + k - 1), title_str), 'FontSize', 16, 'FontWeight', 'bold');
    xlabel(ax, 'x (km)', 'FontSize', 16, 'FontWeight', 'bold');
    if k == 1
        ylabel(ax, 'z (m)', 'FontSize', 16, 'FontWeight', 'bold');
    else
        yticklabels(ax, []);
    end
    set(ax, 'FontSize', 14);
    xlim(ax, [min(xs), max(xs)]);
    ylim(ax, [min(zc) - abs(zc(2) - zc(1)) / 2, 0]);
    axs(k) = ax;
end

drawnow;
% Colorbar inside the bottom-left corner (over land), as the x-z panels of vishoals.m.
% (Every panel keeps a hidden colorbar so the panel sizes stay identical.)
for k = 1:n_runs
    if k ~= cb_panel
        cbs(k).Visible = 'off';
        continue;
    end
    axP = axs(k).Position;
    % =========================================================================
    % 🟢 EDIT COLORBAR POSITION HERE
    % =========================================================================
    cw = 0.015;                              % same as vishoals.m x-z colorbars
    ch = axP(4) * 0.35;
    cbs(k).Position = [axP(1) + 0.01, axP(2) + 0.02 * 700 / fig_h, cw, ch];
    cbs(k).YAxisLocation = 'right';
    cbs(k).FontSize = 11;
    cbs(k).Title.FontSize = 11;
    % exactly 3 ticks; 2 decimals for narrow ranges
    L1 = cbs(k).Limits(1); L2 = cbs(k).Limits(2);
    if (L2 - L1) < 0.21
        cmin = ceil(L1 * 100) / 100;  cmax = floor(L2 * 100) / 100;
        mid = round((cmin + cmax) / 2, 2);  fmt = '%.2f';
    else
        cmin = ceil(L1 * 10) / 10;  cmax = floor(L2 * 10) / 10;
        mid = round((cmin + cmax) / 2, 1);  fmt = '%.1f';
    end
    ticks = unique([cmin, mid, cmax]);
    cbs(k).Ticks = ticks;
    cbs(k).TickLabels = sprintfc(fmt, ticks);
end
drawnow;

% ---------- output ----------
if ~exist(outdir, 'dir'), mkdir(outdir); end
clip_str = '';
if clipped, clip_str = '_clipped'; end
outfile = fullfile(outdir, sprintf('%s_u_xz_wind_6panel%s.png', cluster_name, clip_str));
exportgraphics(fig, outfile, 'Resolution', 300);
fprintf('Saved still image: %s\n', outfile);
close(fig);
end

% ================== helpers ==================

function lev = as_levels(lev)
% contour() treats a scalar as a NUMBER of levels; repeat it to mean one level
if isscalar(lev), lev = [lev lev]; end
end

function A = read_slice(fname, names, ix, iy)
% (Nx, Nz) x-z slice at alongshore index iy, last time-averaging window
for i = 1:numel(names)
    try
        vinfo = ncinfo(fname, names{i});
        vsize = vinfo.Size;
        if numel(vsize) == 4
            A = ncread(fname, names{i}, [ix(1) iy 1 vsize(4)], [numel(ix) 1 vsize(3) 1]);
        else
            A = ncread(fname, names{i}, [ix(1) iy 1], [numel(ix) 1 vsize(3)]);
        end
        A = squeeze(A);
        return;
    catch
        % try next name
    end
end
error('Could not read any of {%s} from %s', strjoin(names, ', '), fname);
end

function [fname, fmidy] = find_files(cluster_dir, sw_core)
% time_avg_3d file (and the midy_ file, if present) in the sweep_NNN_<sw_core> folder
d = dir(fullfile(cluster_dir, ['*' sw_core '*']));
d = d([d.isdir]);
d = d(~ismember({d.name}, {'.', '..'}));
if isempty(d)
    error('No folder found matching *%s* in %s', sw_core, cluster_dir);
end
datadir = fullfile(cluster_dir, d(1).name);
fname = '';
for c = {['time_avg_3d_sweep_' sw_core '.nc'], ['time_avg_3d_' sw_core '.nc']}
    if isfile(fullfile(datadir, c{1})), fname = fullfile(datadir, c{1}); break; end
end
if isempty(fname)
    error('No time_avg_3d file for %s in %s', sw_core, datadir);
end
fmidy = '';
for c = {['midy_sweep_' sw_core '.nc'], ['midy_' sw_core '.nc']}
    if isfile(fullfile(datadir, c{1})), fmidy = fullfile(datadir, c{1}); break; end
end
end

function data = try_ncread_(fname, names)
for i = 1:numel(names)
    try
        data = ncread(fname, names{i});
        return;
    catch
    end
end
info = ncinfo(fname);
avail = strjoin({info.Variables.Name}, ', ');
error('Could not find coordinate variable. Tried: {%s}. Available: %s', ...
    strjoin(names, ', '), avail);
end
