function vis_froude(run_tag, varargin)
% vis_froude(run_tag, 'datadir', DATA_DIR, 'outdir', OUT_DIR, ...)
%
% REDONE: Two-layer Internal Froude Number Analysis
% Divider: Max stratification depth (|drho/dz|)
% Formulas: Fr1 = V1/sqrt(g'h1), Fr2 = V2/sqrt(g'h2), g' = g(rho2-rho1)/rho2
%
% Now with more permissive thresholding and diagnostic prints.

%% ---------- Options ----------
p = inputParser;
addParameter(p, 'datadir',  '', @ischar);
addParameter(p, 'outdir',   '', @ischar);
addParameter(p, 'g',        9.81, @isnumeric);
addParameter(p, 'time_idx', [], @(x)isnumeric(x) || isempty(x)); 
addParameter(p, 'clim',     [0, 1.5], @(v)isnumeric(v) && numel(v)==2);
addParameter(p, 'depth',    50.0, @isnumeric); 
parse(p, varargin{:});

datadir = p.Results.datadir;
if isempty(datadir), datadir = pwd; end
outdir = p.Results.outdir;
if isempty(outdir)
    % Default to /glade/work/nguyen/unc-shoals/animations
    script_dir = fileparts(mfilename('fullpath'));
    outdir = fullfile(fileparts(script_dir), 'animations');
end
if ~exist(outdir,'dir'), mkdir(outdir); end

g = p.Results.g;
clim_user = p.Results.clim;
Lz = p.Results.depth;

% ---------- Spatial Clip Bounds ----------
x_clip_km = [0,   60];
% y_clip_km is now calculated dynamically below based on Ly

%% ---------- Filenames ----------
fname = fullfile(datadir, ['time_avg_3d_' run_tag '.nc']);
if ~isfile(fname), fname = fullfile(datadir, ['vnshoals_time_avg_3d_' run_tag '.nc']); end
if ~isfile(fname), error('Could not find time-average NetCDF: %s', fname); end

fprintf('Two-Layer Fr Analysis for: %s\n', run_tag);

%% ---------- Load Coordinates & Bathymetry ----------
xc = try_ncread_(fname, {'xC','x_caa','xc','x'});
yc = try_ncread_(fname, {'yC','y_aca','yc','y'});
zc = try_ncread_(fname, {'zC','z_aac','zc','z'});
time = ncread(fname, 'time');

xc_km = xc / 1e3;
yc_km = yc / 1e3;

% Dynamic y-clipping (Centered at Ly/2)
Ly_km     = yc_km(end);
y_clip_km = [Ly_km/2 - 50, Ly_km/2 + 50];

ix = find(xc_km >= x_clip_km(1) & xc_km <= x_clip_km(2));
iy = find(yc_km >= y_clip_km(1) & yc_km <= y_clip_km(2));

xc_km_c = xc_km(ix);
yc_km_c = yc_km(iy);
zc_c = zc(:);

try
    bot_z = ncread(fname, 'bottom_height', [ix(1) iy(1)], [numel(ix) numel(iy)]);
catch
    v_test = squeeze(ncread(fname, 'v_c', [ix(1) iy(1) 1 numel(time)], [numel(ix) numel(iy) Inf 1]));
    mask = ~isnan(v_test) & (v_test ~= 0);
    bot_z = -squeeze(sum(mask, 3)) * abs(zc(2)-zc(1));
end

%% ---------- Calculate Two-Layer Dynamics ----------
t_idx = numel(time);
u_3d = squeeze(ncread(fname, 'u_c', [ix(1) iy(1) 1 t_idx], [numel(ix) numel(iy) Inf 1]));
v_3d = squeeze(ncread(fname, 'v_c', [ix(1) iy(1) 1 t_idx], [numel(ix) numel(iy) Inf 1]));
T_3d = squeeze(ncread(fname, 'T',   [ix(1) iy(1) 1 t_idx], [numel(ix) numel(iy) Inf 1]));
S_3d = squeeze(ncread(fname, 'S',   [ix(1) iy(1) 1 t_idx], [numel(ix) numel(iy) Inf 1]));

speed_3d = sqrt(u_3d.^2 + v_3d.^2);

try
    lat = 35.2480; lon = 0;
    P = repmat(gsw_p_from_z(zc_c(1:size(T_3d,3)), lat), [1, numel(ix), numel(iy)]);
    P = permute(P, [2 3 1]);
    SA = gsw_SA_from_SP(S_3d, P, lon, lat);
    CT = gsw_CT_from_t(SA, T_3d, P);
    rho = gsw_rho(SA, CT, 0);
catch
    rho = 1025 * (1 - 2e-4 * (T_3d - 20) + 7e-4 * (S_3d - 35));
end

% MASK LAND: Remove unphysical values inside bathymetry (where T=0, S=0)
rho(rho < 1010) = NaN;

[Nx_c, Ny_c] = size(bot_z);
[Fr1, Fr2, h1, h2] = deal(zeros(Nx_c, Ny_c));
all_dr = [];

fprintf('  Processing grid: identifying layers and calculating Fr...\n');
for i = 1:Nx_c
    for j = 1:Ny_c
        r_col = squeeze(rho(i,j,:));
        s_col = squeeze(speed_3d(i,j,:));
        
        valid = ~isnan(r_col) & (r_col ~= 0);
        if ~any(valid), continue; end
        
        z_valid = zc_c(valid);
        rho_valid = r_col(valid);
        speed_valid = s_col(valid);
        
        if numel(rho_valid) < 5, continue; end
        
        drho_dz = abs(diff(rho_valid) ./ diff(z_valid));
        
        N_v = numel(drho_dz);
        skip = max(1, round(N_v * 0.15));
        search_range = (1+skip):(N_v-skip);
        if isempty(search_range), search_range = 1:N_v; end
        
        [~, max_rel] = max(drho_dz(search_range));
        max_idx = search_range(max_rel);
        z_split = (z_valid(max_idx) + z_valid(max_idx+1)) / 2;
        
        h1(i,j) = abs(z_valid(end) - z_split);
        h2(i,j) = abs(z_split - z_valid(1));
        
        idx1 = z_valid >= z_split;
        idx2 = z_valid < z_split;
        V1 = mean(speed_valid(idx1), 'omitnan');
        V2 = mean(speed_valid(idx2), 'omitnan');
        rho1 = mean(rho_valid(idx1), 'omitnan');
        rho2 = mean(rho_valid(idx2), 'omitnan');
        
        dr = rho2 - rho1;
        all_dr(end+1) = dr;
        
        if dr > 1e-4 % MUCH MORE PERMISSIVE
            g_prime = g * (dr / rho2);
            Fr1(i,j) = V1 / sqrt(g_prime * h1(i,j));
            Fr2(i,j) = V2 / sqrt(g_prime * h2(i,j));
        else
            Fr1(i,j) = NaN; Fr2(i,j) = NaN;
        end
    end
end

if ~isempty(all_dr)
    fprintf('  Diagnostic: drho range [%.2e, %.2e], Mean: %.2e kg/m^3\n', ...
        min(all_dr), max(all_dr), mean(all_dr));
end

Fr1(Fr1 == 0) = NaN; Fr2(Fr2 == 0) = NaN;

%% ---------- Visualization ----------
fig = figure('Color','w', 'Position', [100 100 1200 600]);
tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'tight');

if isempty(clim_user)
    max_fr = max([Fr1(:); Fr2(:)], [], 'omitnan');
    if isempty(max_fr) || isnan(max_fr) || max_fr == 0, max_fr = 1.0; end
    clim_eff = [0, min(2.0, max_fr)];
else
    clim_eff = clim_user;
end

[ax1, ~] = nested_plot_fr(1, Fr1, 'Top Layer Fr_1', clim_eff);
[ax2, ~] = nested_plot_fr(2, Fr2, 'Bottom Layer Fr_2', clim_eff);

sgtitle(tl, sprintf('Two-Layer Internal Froude Analysis: %s (day %.1f)', ...
    run_tag, (time(t_idx)-time(1))/86400), 'Interpreter', 'none', 'FontSize', 18, 'FontWeight', 'bold');

drawnow;

%% ---------- Save ----------
outfile = fullfile(outdir, [run_tag '_two_layer_froude_clipped.png']);
print(fig, outfile, '-dpng', '-r200');
fprintf('Saved two-layer Froude plot: %s\n', outfile);
close(fig);

    function [ax, hp] = nested_plot_fr(col, data, titleStr, clim)
        ax = nexttile(tl, col);
        hp = pcolor(ax, xc_km_c, yc_km_c, data');
        shading flat; axis tight equal; hold on;
        set(ax, 'YDir', 'normal');
        colormap(ax, cmocean('amp'));
        caxis(ax, clim);
        cb = colorbar(ax, 'Location', 'eastoutside');
        ylabel(cb, 'Internal Froude Number', 'FontSize', 14);
        title(ax, titleStr, 'FontSize', 16, 'FontWeight', 'bold');
        xlabel(ax, 'x (km)', 'FontSize', 14); ylabel(ax, 'y (km)', 'FontSize', 14);
        xlim(ax, x_clip_km); ylim(ax, y_clip_km); grid on;
        try
            contour(ax, xc_km_c, yc_km_c, round(abs(bot_z))', 5, 'Color', 'k', 'LineStyle', '--');
        end
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
error('Coordinate variable not found.');
end
