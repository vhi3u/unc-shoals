function eke_clipped(run_tag, varargin)
% eke_clipped(run_tag, 'datadir', DATADIR, 'outdir', OUTDIR, ...)
%
% Identical computation to vis_ke_eke.m, but clips the figure to
% x=[0, 60] km, y=[100, 200] km for presentation.
%
% Output: PNG saved to '/glade/work/nguyen/unc-shoals/animations/'

% ---------- options ----------
p = inputParser;
addParameter(p, 'datadir',  '', @ischar);
addParameter(p, 'outdir',   '', @ischar);
addParameter(p, 'climKE',   [], @(v)isempty(v) || (isnumeric(v) && numel(v)==2));
addParameter(p, 'climEKE',  [], @(v)isempty(v) || (isnumeric(v) && numel(v)==2));
parse(p, varargin{:});

datadir = p.Results.datadir;
if isempty(datadir), datadir = pwd; end
outdir = p.Results.outdir;
if isempty(outdir), outdir = '/glade/work/nguyen/unc-shoals/animations'; end
if ~exist(outdir,'dir'), mkdir(outdir); end

climKE_user  = p.Results.climKE;
climEKE_user = p.Results.climEKE;

% ---------- spatial clip bounds (display only) ----------
x_clip_km = [0,   60];
% y_clip_km is now calculated dynamically below based on Ly

% ---------- filename ----------
fname = fullfile(datadir, ['time_avg_3d_' run_tag '.nc']);
if ~isfile(fname), error('Average file not found: %s', fname); end

fprintf('Reading time-averaged fields from: %s\n', fname);

% ---------- coordinates ----------
xc = try_ncread_(fname, {'xC','x_caa','xc'});
yc = try_ncread_(fname, {'yC','y_aca','yc'});
zc = try_ncread_(fname, {'zC','z_aac','zc'});

xc_km = xc / 1000;
yc_km = yc / 1000;

% Dynamic y-clipping (Centered at Ly/2)
Ly_km     = yc_km(end);
y_clip_km = [Ly_km/2 - 50, Ly_km/2 + 50];

% ---------- load full 3D fields (last time step) ----------
u  = load_last_3d(fname, 'u_c');  
v  = load_last_3d(fname, 'v_c'); 
w  = load_last_3d(fname, 'w_c');  
uu = load_last_3d(fname, 'uu');
vv = load_last_3d(fname, 'vv');
ww = load_last_3d(fname, 'ww');

if isempty(u) || isempty(uu)
    error('Required variables (u, uu) not found in %s', fname);
end

% Mask land/immersed cells (matching vis_ke_eke.m)
u(u == 0) = NaN; v(v == 0) = NaN; w(w == 0) = NaN;
uu(uu == 0) = NaN; vv(vv == 0) = NaN; ww(ww == 0) = NaN;

% ---------- calculations (identical to vis_ke_eke.m) ----------
% Total KE: 0.5 * (<uu> + <vv> + <ww>)
ke_3d = 0.5 * (uu + vv + ww);

% EKE: 0.5 * (<uu> - <u>^2 + <vv> - <v>^2 + <ww> - <w>^2)
eke_3d = 0.5 * (uu - u.^2 + vv - v.^2 + ww - w.^2);

% Depth Averaging (Z is dimension 3)
ke_zavg  = squeeze(mean(ke_3d, 3, 'omitnan'));
eke_zavg = squeeze(mean(eke_3d, 3, 'omitnan'));

% ---------- color limits ----------
% Compute clims from the clipped window only
ix = find(xc_km >= x_clip_km(1) & xc_km <= x_clip_km(2));
iy = find(yc_km >= y_clip_km(1) & yc_km <= y_clip_km(2));

if isempty(climKE_user)
    ke_sub = ke_zavg(ix, iy);
    climKE = [0, max(ke_sub(:), [], 'omitnan')];
    if climKE(2) == 0 || isnan(climKE(2)), climKE = [0, 1]; end
else
    climKE = climKE_user;
end

if isempty(climEKE_user)
    eke_sub = eke_zavg(ix, iy);
    climEKE = [0, max(eke_sub(:), [], 'omitnan')];
    if climEKE(2) == 0 || isnan(climEKE(2)), climEKE = [0, 1]; end
else
    climEKE = climEKE_user;
end

% ---------- plotting ----------
[Xc, Yc] = meshgrid(xc_km, yc_km);

fig = figure('Color','w','Position',[80 80 1000 600]);
tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Panel 1: Total KE
ax1 = nexttile(tl);
pcolor(ax1, Xc, Yc, ke_zavg');
shading(ax1, 'flat'); axis(ax1, 'equal', 'tight'); set(ax1, 'YDir', 'normal');
colormap(ax1, turbo); caxis(ax1, climKE);
cb1 = colorbar(ax1, 'Location', 'eastoutside');
cb1.FontSize = 16; cb1.Label.String = 'KE  [m^2 s^{-2}]'; cb1.Label.FontSize = 18;
title(ax1, 'KE', 'FontSize', 18, 'FontWeight', 'bold');
xlabel(ax1, 'x (km)', 'FontSize', 18); ylabel(ax1, 'y (km)', 'FontSize', 18);
set(ax1, 'FontSize', 16);
xlim(ax1, x_clip_km); ylim(ax1, y_clip_km);

% Panel 2: EKE
ax2 = nexttile(tl);
pcolor(ax2, Xc, Yc, eke_zavg');
shading(ax2, 'flat'); axis(ax2, 'equal', 'tight'); set(ax2, 'YDir', 'normal');
colormap(ax2, turbo); caxis(ax2, climEKE);
cb2 = colorbar(ax2, 'Location', 'eastoutside');
cb2.FontSize = 16; cb2.Label.String = 'EKE  [m^2 s^{-2}]'; cb2.Label.FontSize = 18;
title(ax2, 'EKE', 'FontSize', 18, 'FontWeight', 'bold');
xlabel(ax2, 'x (km)', 'FontSize', 18); ylabel(ax2, 'y (km)', 'FontSize', 18);
set(ax2, 'FontSize', 16);
xlim(ax2, x_clip_km); ylim(ax2, y_clip_km);

% Title
sgtitle(tl, sprintf('%s: KE and EKE (Clipped Window)', run_tag), ...
    'FontSize', 16, 'FontWeight', 'bold', 'Interpreter', 'none');
drawnow;

% ---------- save ----------
outfile = fullfile(outdir, [run_tag '_energy_maps.png']);
print(fig, outfile, '-dpng', '-r200');
fprintf('Saved figure: %s\n', outfile);
close(fig);

end

% ================== helpers ==================

function data = load_last_3d(fn, varname)
    try
        vinfo = ncinfo(fn, varname);
        vsize = vinfo.Size;
        nd = numel(vsize);
        
        if nd == 4
            % (Nx, Ny, Nz, Nt) -> last t
            data = ncread(fn, varname, [1 1 1 vsize(4)], [vsize(1) vsize(2) vsize(3) 1]);
        elseif nd == 3
            if vsize(3) > 10 % Likely Nz
                data = ncread(fn, varname);
            else % Likely Nt
                data = ncread(fn, varname, [1 1 vsize(3)], [vsize(1) vsize(2) 1]);
            end
        else
            data = ncread(fn, varname);
        end
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
info = ncinfo(fname);
avail = strjoin({info.Variables.Name}, ', ');
error('Could not find variable. Tried: {%s}. Available: %s', ...
    strjoin(names, ', '), avail);
end
