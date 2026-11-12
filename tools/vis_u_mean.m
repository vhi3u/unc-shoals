function vis_u_mean(run_tag, varargin)
% vis_u_mean(run_tag, 'datadir', DATA_DIR, 'outdir', OUT_DIR, 'clim', CLIM, 'z_target', Z)
%
% Visualizes time-averaged U-velocity on the XY plane, spatially clipped
% to the shoal region.
%
% Features:
%   - Spatially clips to x=[0, 60] km, y=[100, 200] km.
%   - Shows XY slice at target depth (default z=0).
%
% Requires: cmocean toolbox

%% ---------- Options ----------
p = inputParser;
addParameter(p, 'datadir', '', @ischar);
addParameter(p, 'outdir',  '', @ischar);
addParameter(p, 'clim',    [-0.15, 0.15], @(v)isnumeric(v) && numel(v)==2);
addParameter(p, 'z_target', 0, @isnumeric);
parse(p, varargin{:});

datadir = p.Results.datadir;
if isempty(datadir), datadir = pwd; end
outdir = p.Results.outdir;
if isempty(outdir)
    % Default to /glade/work/nguyen/unc-shoals/animations
    script_dir = fileparts(mfilename('fullpath'));
    outdir = fullfile(fileparts(script_dir), 'animations');
end
if ~exist(outdir, 'dir'), mkdir(outdir); end

clim_u = p.Results.clim;
z_target = p.Results.z_target;

% ---------- Spatial Clip Bounds ----------
x_clip_km = [0,   60];
% y_clip_km is now calculated dynamically below based on Ly

%% ---------- Filenames ----------
fname = fullfile(datadir, ['time_avg_3d_' run_tag '.nc']);
if ~isfile(fname)
    error('Could not find time-average NetCDF file: %s', fname);
end

%% ---------- Load Coordinates ----------
xc = try_ncread_(fname, {'xC','x_caa','xc','x'});
yc = try_ncread_(fname, {'yC','y_aca','yc','y'});
zc = try_ncread_(fname, {'zC','z_aac','zc','z'});
time = ncread(fname, 'time');

xc_km = xc / 1e3;
yc_km = yc / 1e3;

% Dynamic y-clipping (Centered at Ly/2)
Ly_km     = yc_km(end);
y_clip_km = [Ly_km/2 - 50, Ly_km/2 + 50];

% Find nearest z index
[~, z_idx] = min(abs(zc - z_target));
z_actual = zc(z_idx);

% Spatial index masks
ix = find(xc_km >= x_clip_km(1) & xc_km <= x_clip_km(2));
iy = find(yc_km >= y_clip_km(1) & yc_km <= y_clip_km(2));

xc_km_c = xc_km(ix);
yc_km_c = yc_km(iy);

[Xc, Yc] = meshgrid(xc_km_c, yc_km_c);

%% ---------- Load U Mean (Last Window) ----------
Nt = numel(time);

% Read clipped XY slice at target depth
% Note: ncread indices are [x y z t]
u_slice = squeeze(ncread(fname, 'u_c', [ix(1) iy(1) z_idx Nt], [numel(ix) numel(iy) 1 1]));

% Mask land/immersed boundaries
u_slice(u_slice == 0) = NaN;

%% ---------- Visualization ----------
fig = figure('Color','w','Position', [100 100 800 600]);
ax = axes('Parent', fig);

% XY slice plot — transpose because pcolor expects (Ny, Nx)
p1 = pcolor(ax, Xc, Yc, u_slice');
shading flat; axis tight equal; hold on;
set(ax, 'YDir', 'normal');
colormap(ax, cmocean('balance')); 

% Adaptive color limits: Symmetric around 0
if isempty(p.Results.clim) || all(p.Results.clim == [-0.15, 0.15])
    um = max(abs(u_slice(:)), [], 'omitnan');
    if isempty(um) || um == 0
        % No hard floor, just ensure it's not strictly 0 to avoid caxis error
        um = 1e-3; 
    end
    clim_eff = [-um, um];
else
    clim_eff = clim_u;
end
caxis(ax, clim_eff);
cb = colorbar(ax);
ylabel(cb, 'U (m/s)', 'FontSize', 14);

title(sprintf('Time-Averaged U Velocity (z = %.1f m)\nRun: %s', z_actual, run_tag), ...
    'FontSize', 16, 'FontWeight', 'bold', 'Interpreter', 'none');
xlabel('x (km)', 'FontSize', 14);
ylabel('y (km)', 'FontSize', 14);
set(ax, 'FontSize', 13);
grid on;

% Constrain axis to the clip window
xlim(ax, x_clip_km);
ylim(ax, y_clip_km);

drawnow;

%% ---------- Save ----------
outfile = fullfile(outdir, [run_tag '_u_mean_clipped.png']);
print(fig, outfile, '-dpng', '-r200');
fprintf('Saved clipped mean U plot: %s\n', outfile);

close(fig);

end

%% ================== Helpers ==================

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
