function offshore_flux_clipped(run_tag, varargin)
% offshore_flux_clipped(run_tag, 'datadir', DATADIR, 'outdir', OUTDIR, ...)
%
% Visualizes depth-averaged XY offshore heat and salt flux, spatially clipped
% to the shoal region.
%
% Features:
%   - Spatially clips to x=[0, 60] km, y=[100, 200] km.
%   - Sign switch: Offshore transport is POSITIVE (+), Onshore is NEGATIVE (-).
%   - Formula: uT_offshore = -uT  (Offshore Positive)
%   - Panels and labels consistent with vishoals_clipped.m.
%
% Output: PNG saved to '/glade/work/nguyen/unc-shoals/animations/'

% ---------- options ----------
p = inputParser;
addParameter(p, 'datadir', '', @ischar);
addParameter(p, 'outdir',  '', @ischar);
addParameter(p, 'climuT',  [], @(v)isnumeric(v) && numel(v)==2);
addParameter(p, 'climuS',  [], @(v)isnumeric(v) && numel(v)==2);
parse(p, varargin{:});

datadir = p.Results.datadir;
if isempty(datadir), datadir = pwd; end
outdir = p.Results.outdir;
if isempty(outdir), outdir = '/glade/work/nguyen/unc-shoals/animations'; end
if ~exist(outdir,'dir'), mkdir(outdir); end

climuT_user = p.Results.climuT;
climuS_user = p.Results.climuS;

% ---------- spatial clip bounds ----------
x_clip_km = [0,   60];
% y_clip_km is now calculated dynamically below based on Ly

% ---------- filename ----------
% Prefer time-averaged 3D fields
fname = fullfile(datadir, ['time_avg_3d_' run_tag '.nc']);
if ~isfile(fname)
    fname = fullfile(datadir, ['zavg_xy_' run_tag '.nc']);
end
if ~isfile(fname), error('Flux data not found for: %s', run_tag); end

fprintf('Reading flux data from: %s\n', fname);

% ---------- coordinates ----------
xc = try_ncread_(fname, {'xC','x_caa','xc'});
yc = try_ncread_(fname, {'yC','y_aca','yc'});
xc_km = xc / 1000;
yc_km = yc / 1000;

% Dynamic y-clipping (Centered at Ly/2)
Ly_km     = yc_km(end);
y_clip_km = [Ly_km/2 - 50, Ly_km/2 + 50];

% Spatial index masks
ix = find(xc_km >= x_clip_km(1) & xc_km <= x_clip_km(2));
iy = find(yc_km >= y_clip_km(1) & yc_km <= y_clip_km(2));

xc_km_c = xc_km(ix);
yc_km_c = yc_km(iy);

[Xc, Yc] = meshgrid(xc_km_c, yc_km_c);

% ---------- load and process (clipped) ----------
% Load last time step of 3D fields
u  = load_last_3d_clip(fname, 'u_c', ix, iy); 
if isempty(u), u = load_last_3d_clip(fname, 'u', ix, iy); end
T  = load_last_3d_clip(fname, 'T', ix, iy);
S  = load_last_3d_clip(fname, 'S', ix, iy);
uT_raw = load_last_3d_clip(fname, 'uT', ix, iy);
uS_raw = load_last_3d_clip(fname, 'uS', ix, iy);

% Offshore Flux Calculation
uT_offshore = uT_raw;
uS_offshore = uS_raw;

% Depth Averaging
uT_zavg = squeeze(mean(uT_offshore, 3, 'omitnan'))'; % (Ny_c, Nx_c)
uS_zavg = squeeze(mean(uS_offshore, 3, 'omitnan'))'; % (Ny_c, Nx_c)

% Mask zeros
uT_zavg(uT_zavg == 0) = NaN;
uS_zavg(uS_zavg == 0) = NaN;

% ---------- color limits (clipped) ----------
sym_clim = @(A) [-1, 1] * max(abs(A(:)), [], 'omitnan');
if isempty(climuT_user), climuT = sym_clim(uT_zavg); else, climuT = climuT_user; end
if isempty(climuS_user), climuS = sym_clim(uS_zavg); else, climuS = climuS_user; end

% ---------- figure ----------
% Position matching vishoals_clipped.m exactly: [80 80 1000 600]
fig = figure('Color','w','Position',[80 80 1000 600]);
tl  = tiledlayout(fig, 1, 2, 'TileSpacing', 'tight', 'Padding', 'tight');

    function [ax, h] = panel_flux(col, titleStr, data, clim, cmap, cbLabel)
        ax = nexttile(tl, col);
        h = pcolor(ax, Xc, Yc, data);
        shading(ax, 'flat'); axis(ax, 'equal', 'tight'); set(ax, 'YDir', 'normal');
        colormap(ax, cmap); caxis(ax, clim);
        cb = colorbar(ax, 'Location', 'eastoutside');
        cb.FontSize = 16; cb.Label.String = cbLabel; cb.Label.FontSize = 18;
        title(ax, titleStr, 'FontSize', 18, 'FontWeight', 'bold');
        xlabel(ax, 'x (km)', 'FontSize', 18);
        ylabel(ax, 'y (km)', 'FontSize', 18);
        set(ax, 'FontSize', 16);
        xlim(ax, x_clip_km); ylim(ax, y_clip_km);
    end

[ax1, ~] = panel_flux(1, 'Heat Flux', uT_zavg, climuT, cmocean('balance'), '\langle uT \rangle_z');
[ax2, ~] = panel_flux(2, 'Salt Flux', uS_zavg, climuS, cmocean('balance'), '\langle uS \rangle_z');

sgtitle(tl, sprintf('%s: Offshore Heat & Salt Flux (Clipped Window)', run_tag), ...
    'FontSize', 16, 'FontWeight', 'bold', 'Interpreter', 'none');

drawnow;

% ---------- save ----------
outfile = fullfile(outdir, [run_tag '_offshore_flux_clipped.png']);
print(fig, outfile, '-dpng', '-r200');
fprintf('Saved figure: %s\n', outfile);
close(fig);

end

% ================== helpers ==================

function data = load_last_3d_clip(fn, varname, ix, iy)
    try
        vinfo = ncinfo(fn, varname);
        vsize = vinfo.Size;
        % (Nx, Ny, Nz, Nt)
        data = ncread(fn, varname, [ix(1) iy(1) 1 vsize(4)], [numel(ix) numel(iy) vsize(3) 1]);
        data = squeeze(data);
    catch
        data = [];
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
