function vis_streamfunction_xz(run_tag, varargin)
% vis_streamfunction_xz(run_tag, 'y_range', [100 140], 'datadir', '/path')
%
% Calculates and plots the XZ streamfunction by integrating the mean 
% cross-shore velocity (u) over depth, overlaid on density.
%
% Output: PDF/PNG saved to '/glade/work/nguyen/unc-shoals/animations/'

% ---------- options ----------
p = inputParser;
addParameter(p, 'datadir', '', @ischar);
addParameter(p, 'outdir', '/glade/work/nguyen/unc-shoals/animations', @ischar);
addParameter(p, 'y_range', [155 200], @(x)isnumeric(x) && numel(x)==2); % in km
addParameter(p, 'background', 'density', @ischar);
parse(p, varargin{:});

datadir = p.Results.datadir;
if isempty(datadir), datadir = pwd; end
outdir = p.Results.outdir;
if ~exist(outdir,'dir'), mkdir(outdir); end

y_lims = p.Results.y_range;
bg_var = p.Results.background;

% ---------- filename ----------
fname = fullfile(datadir, ['time_avg_3d_' run_tag '.nc']);
if ~isfile(fname), error('Time-avg file not found: %s', fname); end

% ---------- coordinates ----------
xc = try_ncread_(fname, {'xC','x_caa','xc','x'});
zc = try_ncread_(fname, {'zC','z_aac','zc','z'});
yc = try_ncread_(fname, {'yC','y_aca','yc','y'});
xc_km = xc / 1e3;
yc_km = yc / 1e3;
Dz = diff([zc(1)-1; zc]); % Approximate dz (negative values)

% ---------- find y indices ----------
y_idx = find(yc_km >= y_lims(1) & yc_km <= y_lims(2));
if isempty(y_idx), error('Requested y_range [%.1f %.1f] is outside domain.', y_lims(1), y_lims(2)); end

% ---------- load and average u ----------
time = ncread(fname, 'time');
Nt = numel(time);

fprintf('Loading and y-averaging u field (y-range: %.1f-%.1f km)...\n', y_lims(1), y_lims(2));
% u is (Nx, Ny, Nz, Nt). Read Nx, y_slice, Nz, last_t
u_3d = squeeze(ncread(fname, 'u_c', [1 y_idx(1) 1 Nt], [Inf numel(y_idx) Inf 1]));
u_mean_y = squeeze(mean(u_3d, 2, 'omitnan')); % (Nx, Nz)

% Load background for plotting
if strcmpi(bg_var, 'density') || strcmpi(bg_var, 'rho')
    T_3d = squeeze(ncread(fname, 'T', [1 y_idx(1) 1 Nt], [Inf numel(y_idx) Inf 1]));
    S_3d = squeeze(ncread(fname, 'S', [1 y_idx(1) 1 Nt], [Inf numel(y_idx) Inf 1]));
    
    T_3d(T_3d == 0) = NaN;
    S_3d(S_3d == 0) = NaN;
    
    T_mean = squeeze(mean(T_3d, 2, 'omitnan'));
    S_mean = squeeze(mean(S_3d, 2, 'omitnan'));
    
    p_dbar = -zc(:) * 1025 * 9.81 / 1e4;
    P = repmat(p_dbar', size(T_mean,1), 1);
    
    SA = gsw_SA_from_SP(S_mean, P, 0, 35.2480);
    CT = gsw_CT_from_t(SA, T_mean, P);
    bg_mean_y = gsw_rho(SA, CT, P);
    bg_mean_y(bg_mean_y < 1010) = NaN; % mask land
    
    cmap = cmocean('dense');
    clabel = 'Density (kg/m^3)';
else
    bg_3d = squeeze(ncread(fname, bg_var, [1 y_idx(1) 1 Nt], [Inf numel(y_idx) Inf 1]));
    bg_3d(bg_3d == 0) = NaN;
    bg_mean_y = squeeze(mean(bg_3d, 2, 'omitnan'));
    cmap = cmocean('thermal');
    clabel = bg_var;
end

% ---------- Calculate Streamfunction psi ----------
% psi(x, z) = -integral from bottom up of u(x, z') dz'
% Note: Oceananigans coordinates go -50 to 0. 
% We need to integrate from z(1) (bottom) upwards.
u_mean_y(isnan(bg_mean_y)) = NaN;
psi = zeros(size(u_mean_y));

for i = 1:size(u_mean_y, 1)
    u_col = u_mean_y(i, :);
    valid_z = ~isnan(u_col);
    if ~any(valid_z)
        psi(i, :) = NaN;
        continue;
    end
    
    u_col_clean = u_col;
    u_col_clean(~valid_z) = 0; % Treat bathymetry as zero velocity
    
    psi_col = -cumsum(u_col_clean .* Dz');
    psi_col(~valid_z) = NaN; % Mask inside bathymetry
    psi(i, :) = psi_col;
end

% Mask out the immersed boundary so contours don't draw inside rock
psi(isnan(bg_mean_y)) = NaN;

% ---------- figure ----------
fig = figure('Color','w', 'Position', [100 100 800 500]);
ax = axes(fig);
set(ax, 'Color', [0.8 0.8 0.8]);
hold on;

% --- Plot Background (Density) ---
[X, Z] = meshgrid(xc_km, zc / 1e3);
p = pcolor(ax, X, Z, bg_mean_y');
p.EdgeColor = 'none';
shading interp;
colormap(ax, cmap);

% Explicitly freeze color limits to background data range
caxis(ax, [min(bg_mean_y(:)) max(bg_mean_y(:))]);

% Colorbar
cb = colorbar('Location', 'northoutside'); 
cb.Label.String = clabel; 
cb.Label.FontWeight = 'bold';

% --- Plot psi contour lines ---
psi_min = min(psi(:), [], 'omitnan');
psi_max = max(psi(:), [], 'omitnan');
max_abs = max(abs(psi_min), abs(psi_max));

if max_abs == 0 || isnan(max_abs)
    levels = 15; % fallback
else
    % Exclude near-zero noise levels (< 2% of max_abs) to prevent spiky bottom-boundary artifacts
    norm_pos = linspace(0.02, 1.0, 15).^1.5; 
    levels_pos = norm_pos * max_abs;
    levels_neg = -norm_pos * max_abs;
    levels = sort(unique([levels_neg, levels_pos]));
    
    % Keep only levels within actual range
    levels = levels(levels >= psi_min & levels <= psi_max);
end

[Cx, Ch] = contour(ax, xc_km, zc / 1e3, psi', levels, 'k', 'LineWidth', 1.2);

xlabel('x (km)'); ylabel('z (km)');
title(sprintf('XZ Density & Streamfunction: %s\nTime & Y-Avg (y = %.1f-%.1f km)', ...
    run_tag, y_lims(1), y_lims(2)), 'Interpreter', 'none');

set(ax, 'Layer', 'top', 'Box', 'on');

% ---------- save ----------
outfile = fullfile(outdir, [run_tag '_streamfunction_xz_y' num2str(round(y_lims(1))) '-' num2str(round(y_lims(2))) '.png']);
print(fig, outfile, '-dpng', '-r200');
fprintf('Saved streamfunction: %s\n', outfile);
end

function data = try_ncread_(fname, names)
for i = 1:numel(names)
    try, data = ncread(fname, names{i}); return; catch, end
end
error('Coordinate variable not found.');
end
