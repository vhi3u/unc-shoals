function vis_density_gradients(run_tag, varargin)
% vis_density_gradients(run_tag, 'datadir', DATADIR, 'outdir', OUTDIR)
%
% Creates XY maps of:
%   1. Maximum Vertical Density Gradient |drho/dz|_max
%   2. Reduced Gravity g' = g * (rho_bot - rho_surf) / rho0
%
% This is useful for identifying pycnocline sharpening and stratification strength.

% ---------- options ----------
p = inputParser;
addParameter(p, 'datadir',  '', @ischar);
addParameter(p, 'outdir',   '', @ischar);
addParameter(p, 'climGrad', [], @(v)isempty(v) || (isnumeric(v) && numel(v)==2));
addParameter(p, 'climGP',   [0, 0.05], @(v)isnumeric(v) && numel(v)==2);
parse(p, varargin{:});

datadir = p.Results.datadir;
if isempty(datadir), datadir = pwd; end
outdir = p.Results.outdir;

%% ---------- Filenames & Coordinates ----------
fname = fullfile(datadir, ['time_avg_3d_' run_tag '.nc']);
if ~isfile(fname), fname = fullfile(datadir, ['vnshoals_time_avg_3d_' run_tag '.nc']); end
if ~isfile(fname), error('Could not find time-average NetCDF: %s', fname); end

fprintf('Density Gradient Diagnostics for: %s\n', run_tag);

% ---------- coordinates ----------
xc = try_ncread_(fname, {'xC','x_caa','xc','x'});
yc = try_ncread_(fname, {'yC','y_aca','yc','y'});
zc = try_ncread_(fname, {'zC','z_aac','zc','z'});

xc_km = xc / 1000;
yc_km = yc / 1000;

% ---------- load T, S (last time step) ----------
time = ncread(fname, 'time');
t_idx = numel(time);

% Read 3D fields
T = squeeze(ncread(fname, 'T', [1 1 1 t_idx], [Inf Inf Inf 1]));
S = squeeze(ncread(fname, 'S', [1 1 1 t_idx], [Inf Inf Inf 1]));

% Calculate Pressure (dbar) from depth zc
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
rho(rho < 1010) = NaN; % Mask land

% ---------- calculations ----------
% 1. Vertical Gradient |drho/dz|
dz = diff(zc);
drho_dz = diff(rho, 1, 3) ./ reshape(dz, 1, 1, []);
max_grad = squeeze(max(abs(drho_dz), [], 3, 'omitnan'));

% 2. Reduced Gravity g'
% Surface density
rho_surf = squeeze(rho(:,:,end));
% Bottom density (find first non-NaN from bottom)
[Nx, Ny, Nz] = size(rho);
rho_bot = NaN(Nx, Ny);
for i=1:Nx
    for j=1:Ny
        col = squeeze(rho(i,j,:));
        valid_idx = find(~isnan(col) & col ~= 0, 1, 'first');
        if ~isempty(valid_idx), rho_bot(i,j) = col(valid_idx); end
    end
end
gp = 9.81 * (rho_bot - rho_surf) / 1025;

% ---------- plotting ----------
[Xc, Yc] = meshgrid(xc_km, yc_km);

fig = figure('Color','w','Position',[100 100 1200 500], 'Visible', 'off');
tl = tiledlayout(fig, 1, 2, 'TileSpacing','compact','Padding','compact');

% Panel 1: Max Density Gradient
ax1 = nexttile(tl);
pcolor(ax1, Xc, Yc, max_grad');
shading flat; axis tight equal; colorbar;
colormap(ax1, parula);
if ~isempty(p.Results.climGrad), caxis(ax1, p.Results.climGrad); end
title('Max Vertical Density Gradient |d\rho/dz|_{max} [kg/m^4]');
xlabel('x (km)'); ylabel('y (km)');

% Panel 2: Reduced Gravity g'
ax2 = nexttile(tl);
pcolor(ax2, Xc, Yc, gp');
shading flat; axis tight equal; colorbar;
colormap(ax2, turbo);
caxis(ax2, p.Results.climGP);
title('Reduced Gravity g'' [m/s^2]');
xlabel('x (km)'); ylabel('y (km)');

sgtitle(tl, sprintf('%s: Density Gradient Diagnostics', run_tag), 'Interpreter', 'none');

% ---------- save ----------
if isempty(outdir), outdir = '/glade/work/nguyen/unc-shoals/animations/'; end
if ~exist(outdir,'dir'), mkdir(outdir); end

outname = fullfile(outdir, [run_tag '_density_gradients.png']);
print(fig, outname, '-dpng', '-r200');
fprintf('Saved map to: %s\n', outname);

end

function data = try_ncread_(fname, names)
for i = 1:numel(names)
    try
        data = ncread(fname, names{i});
        return;
    catch
    end
end
error('Coordinate variable not found in %s', fname);
end
