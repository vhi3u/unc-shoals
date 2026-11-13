function vishoals_vort_wind(cluster_name, varargin)
% vishoals_vort_wind(cluster_name, 'wind_sign', 'positive'/'negative'/'both', ...
%   'runs', {...}, 'ref_run', LABEL, 'outdir', OUTDIR, 'clipped', true/false, ...
%   'isopycnals', true/false, 'rho_levels', ..., 'climRo', [lo hi])
%
% Still image of surface (XY) vorticity (Ro = zeta/f) across the wind-stress
% sweep, using the LAST saved frame of each run -- one panel per run, sharing
% one color scale.
%
% WHICH RUNS: discovered from the cluster, not hardcoded. Every wind_* run
% present is used, tau is parsed back out of each label, and the panels are
% ordered by tau. So this works on any wind cluster, including ones that did
% not exist when the script was written:
%
%   'positive' (default)  tau > 0, unforced reference first, then rising |tau|
%   'negative'            tau < 0, same ordering
%   'both'                every wind run, signed tau order, tau = 0 in the middle
%
% Up to v1 this built the names arithmetically from a hardcoded
% tau_mag = [0.03 0.05 0.08 0.10 0.15] with a '%02d' hundredths format, so
% params_weakwind.toml (+/-0.005, 0.010, 0.020) could not be plotted at all --
% 0.005 formatted to 'wind_0p01', which is not a run.
%
% 'runs' overrides discovery with an explicit list, in the order given, taking
% either run tags or bare labels. 'ref_run' names the unforced run (default
% 'Zs5_Ls40k'); it is a geometry label and carries no tau, so it is pinned to
% tau = 0. A cluster without it (cluster_weakwinds) warns and plots the forced
% runs only -- pass 'ref_run', '' to silence that.
%
% Missing runs are dropped with a warning rather than aborting, so a
% part-finished sweep still plots.
%
% Reads top_sweep_<label>.nc from each run folder in
%   <sweep_root>/<cluster_name>/sweep_NNN_<label>/
% (the older un-indexed sweep_<label>/ layout is handled too)
%
% Panel spacing matches vishoals_still.m 'full_panel' (top row only), and the
% figure width scales at the original 225 px per panel.
%
% Requires: cmocean, GSW Toolbox (for isopycnals), discover_sweeps
%
% Example:
%   vishoals_vort_wind('cluster_Fnnie_1_19', 'clipped', true)
%   vishoals_vort_wind('cluster_Fnnie_1_19', 'clipped', true, 'wind_sign', 'negative')
%   vishoals_vort_wind('cluster_weakwinds', 'wind_sign', 'both', 'ref_run', '')

if nargin < 1 || isempty(cluster_name)
    cluster_name = 'cluster_Fnnie_1_19';
end

% ---------- options ----------
p = inputParser;
addParameter(p, 'sweep_root', '/glade/work/nguyen/unc-shoals/sweep_output', @ischar);
addParameter(p, 'outdir', '/glade/work/nguyen/unc-shoals/animations', @ischar);
addParameter(p, 'wind_sign', 'positive', @(x)ischar(x) && ismember(x, {'positive','negative','both'}));
addParameter(p, 'runs', {}, @(c)iscellstr(c) || ischar(c));
addParameter(p, 'ref_run', 'Zs5_Ls40k', @ischar);
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
want_runs    = p.Results.runs;
if ischar(want_runs), want_runs = {want_runs}; end
ref_run      = p.Results.ref_run;
clipped      = p.Results.clipped;
want_iso     = p.Results.isopycnals;
rho_levels   = p.Results.rho_levels;
iso_color    = p.Results.iso_color;
iso_lw       = p.Results.iso_linewidth;
climRo_user  = p.Results.climRo;
no_outlier   = p.Results.no_outlier;
outdir       = p.Results.outdir;

% ---------- runs: discovered from the cluster, then sorted by tau ----------
% This used to build the run names arithmetically from
%   tau_mag = [0.03 0.05 0.08 0.10 0.15]
% with sprintf('%s%sp%02d', ...), which hardcoded both WHICH runs exist and
% the assumption that tau is a whole number of hundredths. The weak-wind
% sweep (params_weakwind.toml: +/-0.005, 0.010, 0.020) breaks the second
% assumption -- %02d turns 0.005 into 'wind_0p01', a run that does not
% exist -- and any new wind run breaks the first. Labels are now read off
% the cluster and tau is parsed back out of them, so this plots whatever
% wind runs are present, at whatever tau, in tau order.
cluster_dir = fullfile(p.Results.sweep_root, cluster_name);
[avail_labels, ~] = discover_sweeps(cluster_dir);

if ~isempty(want_runs)
    % Explicit list wins. Accept run tags or bare labels, keep caller order.
    labels   = regexprep(want_runs, '^sweep_', '');
    tau_vals = cellfun(@tau_from_label_, labels);
    unknown  = labels(isnan(tau_vals) & ~strcmp(labels, ref_run));
    if ~isempty(unknown)
        warning('vishoals_vort_wind:unparsedTau', ...
                ['tau could not be parsed from: %s\n' ...
                 'These panels are plotted in the order given, titled tau = NaN.'], ...
                strjoin(unknown, ', '));
    end
    tau_vals(strcmp(labels, ref_run)) = 0;
else
    % Every wind_* run in the cluster, filtered by sign, plus the unforced
    % reference. ref_run is a geometry run (Zs5_Ls40k), not a wind_* name,
    % so it carries no tau in its label and is pinned to 0 here.
    tau_all = cellfun(@tau_from_label_, avail_labels);
    switch wind_sign
        case 'positive', keep = tau_all > 0;
        case 'negative', keep = tau_all < 0;
        case 'both',     keep = ~isnan(tau_all);
    end
    labels   = avail_labels(keep);
    tau_vals = tau_all(keep);

    if ~isempty(ref_run)
        if any(strcmp(ref_run, avail_labels))
            labels   = [{ref_run}, labels];
            tau_vals = [0, tau_vals];
        else
            % A weak-wind-only cluster has no unforced run. Not fatal: the
            % figure just starts at the weakest forced case.
            warning('vishoals_vort_wind:noRefRun', ...
                    ['Reference run %s is not in %s -- plotting the forced runs only.\n' ...
                     'Pass ''ref_run'', '''' to silence this, or merge the ' ...
                     'reference run into the cluster.'], ref_run, cluster_name);
        end
    end

    % Sort key: |tau| for a single-signed row, so the unforced panel leads and
    % the sweep marches away from it (matching the original layout). Signed tau
    % for 'both', which puts tau = 0 in the middle where it belongs.
    if strcmp(wind_sign, 'both')
        [tau_vals, ord] = sort(tau_vals);
    else
        [~, ord] = sort(abs(tau_vals));
        tau_vals = tau_vals(ord);
    end
    labels = labels(ord);
end

if isempty(labels)
    error('vishoals_vort_wind:noRuns', ...
          ['No %s-wind runs found in %s.\nAvailable: %s'], ...
          wind_sign, cluster_dir, strjoin(avail_labels, ', '));
end

% ---------- locate files ----------
% A missing run is dropped rather than fatal, so a part-finished sweep still
% plots. tau_vals is PARALLEL to labels and is filtered by the same mask --
% filtering one alone would shift every panel title by the number of missing
% runs and mislabel the figure silently.
n_want    = numel(labels);
fname_top = cell(1, n_want);
ok        = false(1, n_want);
for k = 1:n_want
    dname = find_run_dir_(cluster_dir, labels{k});
    if isempty(dname)
        warning('vishoals_vort_wind:missingRun', ...
                'No run directory for %s in %s -- skipped.', labels{k}, cluster_dir);
        continue;
    end
    fname_top{k} = fullfile(cluster_dir, dname, ['top_sweep_' labels{k} '.nc']);
    if ~isfile(fname_top{k})
        warning('vishoals_vort_wind:missingFile', ...
                'Surface file not found, skipped: %s', fname_top{k});
        continue;
    end
    ok(k) = true;
end
fname_top = fname_top(ok);
labels    = labels(ok);
tau_vals  = tau_vals(ok);
runs      = strcat('sweep_', labels);
n_runs    = numel(runs);
if n_runs == 0
    error('vishoals_vort_wind:noFiles', ...
          'None of the %d selected runs had a readable top_*.nc in %s', n_want, cluster_dir);
end

% Every panel is read with the index ranges computed from run 1 below, so a
% run on a different grid would be sliced out of bounds. read_surf_safe
% swallows that in a try/catch and returns all-NaN, i.e. a blank panel and no
% error -- check up front instead. Mixed-resolution clusters are real
% (cluster_res250_testing holds R500_* and R250_* side by side).
grid_nx = nan(1, n_runs); grid_ny = nan(1, n_runs);
for k = 1:n_runs
    grid_nx(k) = numel(try_ncread_(fname_top{k}, {'xC','x_caa','xc'}));
    grid_ny(k) = numel(try_ncread_(fname_top{k}, {'yC','y_aca','yc'}));
end
same = grid_nx == grid_nx(1) & grid_ny == grid_ny(1);
if ~all(same)
    warning('vishoals_vort_wind:mixedGrids', ...
            ['Dropping %d run(s) whose grid differs from %s (%dx%d): %s\n' ...
             'Panels share one set of index ranges, so grids cannot be mixed. ' ...
             'Plot each resolution as its own cluster.'], ...
            sum(~same), labels{1}, grid_nx(1), grid_ny(1), ...
            strjoin(labels(~same), ', '));
    fname_top = fname_top(same);
    labels    = labels(same);
    tau_vals  = tau_vals(same);
    runs      = runs(same);
    n_runs    = numel(runs);
end

fprintf('vishoals_vort_wind: %d panel(s), tau = %s N m^-2\n', ...
        n_runs, strjoin(arrayfun(@(t) sprintf('%g', t), tau_vals, 'UniformOutput', false), ', '));

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
    % %g not %.2f: at %.2f both wind_0p005 and wind_0p010 log as "0.01",
    % so the line could not be used to confirm which run was read.
    fprintf('tau = %-7g ->  %s  (last frame: day %.1f)\n', tau_vals(k), fname_top{k}, day_end(k));

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
% 225 px per panel is the original 1350 px / 6 panels, so a 6-run cluster
% comes out pixel-identical to before and a 9-run one just gets wider rather
% than squeezing 9 panels into 6 tiles (nexttile would have errored at k = 7).
fig_w = max(600, 225 * n_runs); fig_h = 520;
fig = figure('Color','w','Position',[80 50 fig_w fig_h]);
tl  = tiledlayout(fig, 1, n_runs, 'TileSpacing','compact','Padding','compact');

axs = gobjects(1, n_runs);
cbs = gobjects(1, n_runs);
cb_panel = n_runs;  % which panel shows the (shared) colorbar; 1 = leftmost, n_runs = rightmost
cmap = cmocean('balance');
% %.2f renders both wind_0p005 and wind_0p010 as "0.01" -- two panels, one
% label. Widen to 3 decimals whenever any run is weaker than 0.01 N/m^2.
tau_nz  = tau_vals(tau_vals ~= 0 & ~isnan(tau_vals));
tau_fmt = '%.2f';
if ~isempty(tau_nz) && min(abs(tau_nz)) < 0.01, tau_fmt = '%.3f'; end
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
        title_str = sprintf(['\\tau = ' tau_fmt ' N m^{-2}'], tau_vals(k));
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
outfile = fullfile(outdir, sprintf('%s_wind_vort_%dpanel_%s_lastframe%s.png', ...
                                   cluster_name, n_runs, wind_sign, clip_str));
exportgraphics(fig, outfile, 'Resolution', 300);
fprintf('Saved still image: %s \n', outfile);
close(fig);

end

% ================== helpers ==================

function tau = tau_from_label_(lbl)
% Wind stress [N m^-2] parsed back out of a run label; NaN if it is not a
% wind run. Mirrors params_from_name in plot_theory_vs_model.m.
%
% The fractional digits are taken VERBATIM rather than assumed to be
% hundredths, which is what lets the two naming schemes coexist:
%   wind_0p03  -> 0.03    (params_HsLs.toml, two decimals)
%   wind_0p010 -> 0.010   (params_weakwind.toml, three decimals)
%   wind_0p10  -> 0.10
% wind_0p010 and wind_0p10 differ by one character and by a factor of 10, so
% they must not be conflated.
tok = regexp(lbl, '^wind_(m?)(\d+)p(\d+)$', 'tokens', 'once');
if isempty(tok)
    tau = NaN;
    return;
end
tau = str2double([tok{2} '.' tok{3}]);
if ~isempty(tok{1}), tau = -tau; end
end

function dname = find_run_dir_(cluster_dir, label)
% Run directory for a label, matched EXACTLY. The old code globbed
% ['*' label '*'] and silently took d(1) of however many matched. No current
% label is a substring of another (checked across params_HsLs.toml,
% params_weakwind.toml and params_geomcontrol.toml), so this fixes no live
% bug -- but the two wind naming schemes are one character apart at
% wind_0p010 / wind_0p10, and a future label like 'wind_0p10_hires' would
% make d(1) a coin flip. Match exactly instead.
% Handles both layouts: sweep_NNN_<label>/ and the older sweep_<label>/.
d = dir(fullfile(cluster_dir, 'sweep_*'));
d = d([d.isdir]);
names = {d.name};
names = names(~ismember(names, {'.', '..'}));

hit = strcmp(names, ['sweep_' label]);
if ~any(hit)
    pat = ['^sweep_\d+_' regexptranslate('escape', label) '$'];
    hit = ~cellfun(@isempty, regexp(names, pat, 'once'));
end
if ~any(hit)
    dname = '';
else
    dname = names{find(hit, 1)};
end
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
% If nothing worked, list what's in the file
info = ncinfo(fname);
avail = strjoin({info.Variables.Name}, ', ');
error('Could not find coordinate variable. Tried: {%s}. Available: %s', ...
    strjoin(names, ', '), avail);
end
