function vis_density_diagnostics(run_tag, varargin)
% vis_density_diagnostics(run_tag, 'datadir', DATA_DIR, 'outdir', OUT_DIR)
%
% Generates diagnostic plots for density to debug stratification issues.
% Shows:
%   1. Surface Density (XY)
%   2. Bottom Density (XY)
%   3. XZ Transect of Density at mid-channel (y=150km)

%% ---------- Options ----------
p = inputParser;
addParameter(p, 'datadir',  '', @ischar);
addParameter(p, 'outdir',   '', @ischar);
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

%% ---------- Filenames & Coordinates ----------
fname = fullfile(datadir, ['time_avg_3d_' run_tag '.nc']);
if ~isfile(fname), fname = fullfile(datadir, ['vnshoals_time_avg_3d_' run_tag '.nc']); end
if ~isfile(fname), error('Could not find time-average NetCDF: %s', fname); end

fprintf('Density Diagnostics for: %s\n', run_tag);

% Use robust coordinate reader
xc = try_ncread_(fname, {'xC','x_caa','xc','x'});
yc = try_ncread_(fname, {'yC','y_aca','yc','y'});
zc = try_ncread_(fname, {'zC','z_aac','zc','z'});
time = ncread(fname, 'time');
t_idx = numel(time);

xc_km = xc / 1e3;
yc_km = yc / 1e3;

% Mid-channel index
mid_y = find(yc_km >= 150, 1, 'first');

%% ---------- Load T, S and Calculate Density ----------
T = squeeze(ncread(fname, 'T', [1 1 1 t_idx], [Inf Inf Inf 1]));
S = squeeze(ncread(fname, 'S', [1 1 1 t_idx], [Inf Inf Inf 1]));

% Calculate Pressure (dbar) from depth zc
% p (dbar) approx -z (m) / 10
p_dbar = reshape(-zc(:) * 1025 * 9.81 / 1e4, 1, 1, []);
P = repmat(p_dbar, size(T,1), size(T,2), 1);

% Gibbs SeaWater (GSW) / TEOS-10 Density
% 1. Convert Practical Salinity (S) to Absolute Salinity (SA)
lat = 35.2480; lon = 0;
SA = gsw_SA_from_SP(S, P, lon, lat);

% 2. Convert In-situ Temperature (T) to Conservative Temperature (CT)
CT = gsw_CT_from_t(SA, T, P);

% 3. Calculate Density
rho = gsw_rho(SA, CT, P);

% MASK LAND: Remove unphysical values inside bathymetry (where T=0, S=0)
rho(rho < 1010) = NaN;

% Calculate Vertical Density Gradient (drho/dz)
% zc is vertical centers. diff(zc) gives dZ between centers.
% We use diff(rho) / diff(zc)
dz = diff(zc);
drho_dz = diff(rho, 1, 3) ./ reshape(dz, 1, 1, []);
max_drho_dz = squeeze(max(abs(drho_dz), [], 3, 'omitnan'));

% Calculate Reduced Gravity (g') for two-layer approximation
% g' = g * delta_rho / rho0
rho_surf = squeeze(rho(:,:,end));
% rho_bot is calculated below

%% ---------- Visualization ----------
fig = figure('Color','w', 'Position', [50 50 1400 800]);
tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'tight');

% Panel 1: Surface Density
ax1 = nexttile(tl, 1);
pcolor(ax1, xc_km, yc_km, rho(:,:,end)');
shading flat; axis tight equal; colorbar;
title('Surface Density \rho(z=0)');
xlabel('x (km)'); ylabel('y (km)');

% Panel 2: Bottom Density
ax2 = nexttile(tl, 2);
[Nx, Ny, Nz] = size(rho);
rho_bot = zeros(Nx, Ny);
for i=1:Nx
    for j=1:Ny
        col = squeeze(rho(i,j,:));
        idx = find(~isnan(col) & col ~= 0, 1, 'first');
        if ~isempty(idx), rho_bot(i,j) = col(idx); else, rho_bot(i,j) = NaN; end
    end
end
pcolor(ax2, xc_km, yc_km, rho_bot');
shading flat; axis tight equal; colorbar;
title('Bottom Density \rho_{bot}');
xlabel('x (km)'); ylabel('y (km)');

% Panel 3: Max Density Gradient Map
ax3 = nexttile(tl, 3);
% Limit colorbar for clarity if needed, or use dynamic
pcolor(ax3, xc_km, yc_km, max_drho_dz');
shading flat; axis tight equal; colorbar;
title('Max Vertical Density Gradient |d\rho/dz|');
xlabel('x (km)'); ylabel('y (km)');

% Panel 4: XZ Transect (Mid-channel)
ax4 = nexttile(tl, 4);
[Xz, Zz] = meshgrid(xc_km, zc);
pcolor(ax4, Xz, Zz, squeeze(rho(:,mid_y,:))');
shading flat; axis tight; colorbar;
title(sprintf('XZ Density Transect at y = %.1f km', yc_km(mid_y)));
xlabel('x (km)'); ylabel('z (m)');

sgtitle(tl, sprintf('Density Diagnostics: %s', run_tag), 'Interpreter', 'none');

outfile = fullfile(outdir, [run_tag '_density_diagnostic.png']);
print(fig, outfile, '-dpng', '-r200');
fprintf('Saved diagnostic plot: %s\n', outfile);

close(fig);

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
