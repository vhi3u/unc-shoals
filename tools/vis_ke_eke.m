function vis_ke_eke(run_tag, varargin)
% vis_ke_eke(run_tag, 'datadir', DATADIR, 'outdir', OUTDIR)
%
% Creates depth-averaged XY maps of Kinetic Energy (KE) and 
% Eddy Kinetic Energy (EKE) from time-averaged 3D Oceananigans output.
%
% Formula used for EKE: 
%   EKE = 0.5 * (<uu> - <u>^2 + <vv> - <v>^2 + <ww> - <w>^2)
%
% Style modeled after vishoals.m

% ---------- options ----------
p = inputParser;
addParameter(p, 'datadir', '', @ischar);
addParameter(p, 'outdir', '', @ischar);
addParameter(p, 'climKE',  [0, 3e-3], @(v)isnumeric(v) && numel(v)==2);
addParameter(p, 'climEKE', [],       @(v)isempty(v) || (isnumeric(v) && numel(v)==2));
parse(p, varargin{:});

datadir = p.Results.datadir;
if isempty(datadir), datadir = pwd; end
outdir = p.Results.outdir;

climKE_user  = p.Results.climKE;
climEKE_user = p.Results.climEKE;

% Build filename
fname = fullfile(datadir, ['time_avg_3d_' run_tag '.nc']);
if ~isfile(fname), error('Average file not found: %s', fname); end

fprintf('Reading time-averaged fields from: %s\n', fname);

% ---------- detect coordinate names ----------
xc = try_ncread_(fname, {'xC','x_caa','xc'});
yc = try_ncread_(fname, {'yC','y_aca','yc'});
zc = try_ncread_(fname, {'zC','z_aac','zc'});
time = ncread(fname, 'time');
Nt = numel(time);

% ---------- load and process ----------
% Use robust loaders to get the last time window
u  = load_last_3d(fname, 'u_c');  if isempty(u),  u  = load_last_3d(fname, 'u'); end
v  = load_last_3d(fname, 'v_c');  if isempty(v),  v  = load_last_3d(fname, 'v'); end
w  = load_last_3d(fname, 'w_c');  if isempty(w),  w  = load_last_3d(fname, 'w'); end
uu = load_last_3d(fname, 'uu');
vv = load_last_3d(fname, 'vv');
ww = load_last_3d(fname, 'ww');

if isempty(u) || isempty(uu)
    error('Required variables (u, uu) not found in %s', fname);
end

% Mask land/immersed cells
u(u == 0) = NaN; v(v == 0) = NaN; w(w == 0) = NaN;
uu(uu == 0) = NaN; vv(vv == 0) = NaN; ww(ww == 0) = NaN;

% ---------- calculations ----------
% Total Kinetic Energy: 0.5 * (<u^2> + <v^2> + <w^2>)
ke_3d = 0.5 * (uu + vv + ww);

% Eddy Kinetic Energy: 0.5 * (<uu> - <u>^2 + <vv> - <v>^2 + <ww> - <w>^2)
eke_3d = 0.5 * (uu - u.^2 + vv - v.^2 + ww - w.^2);

% Depth Averaging (Z is dimension 3)
ke_zavg  = squeeze(mean(ke_3d, 3, 'omitnan'));
eke_zavg = squeeze(mean(eke_3d, 3, 'omitnan'));

% Dynamic color limits for EKE if not provided
if isempty(climEKE_user)
    climEKE = [0, max(eke_zavg(:), [], 'omitnan')];
    if climEKE(2) == 0 || isnan(climEKE(2)), climEKE = [0, 1e-4]; end
else
    climEKE = climEKE_user;
end

% ---------- plotting ----------
xc_km = xc / 1000;
yc_km = yc / 1000;
[Xc, Yc] = meshgrid(xc_km, yc_km);

fig = figure('Color','w','Position', [80 80 1200 600]);
tl = tiledlayout(fig, 1, 2, 'TileSpacing','compact','Padding','compact');

% Panel 1: Kinetic Energy
ax1 = nexttile(tl);
pcolor(ax1, Xc, Yc, ke_zavg');
shading(ax1, 'flat'); axis(ax1, 'equal', 'tight'); set(ax1, 'YDir', 'normal');
colormap(ax1, turbo); caxis(ax1, climKE_user);
cb1 = colorbar(ax1, 'Location', 'eastoutside');
cb1.FontSize = 16; cb1.Label.String = 'KE (m^2 s^{-2})'; cb1.Label.FontSize = 18;
title(ax1, 'KE', 'FontSize', 18, 'FontWeight', 'bold');
xlabel(ax1, 'x (km)', 'FontSize', 18); ylabel(ax1, 'y (km)', 'FontSize', 18);
set(ax1, 'FontSize', 16);

% Panel 2: Eddy Kinetic Energy
ax2 = nexttile(tl);
pcolor(ax2, Xc, Yc, eke_zavg');
shading(ax2, 'flat'); axis(ax2, 'equal', 'tight'); set(ax2, 'YDir', 'normal');
colormap(ax2, turbo); caxis(ax2, climEKE);
cb2 = colorbar(ax2, 'Location', 'eastoutside');
cb2.FontSize = 16; cb2.Label.String = 'EKE (m^2 s^{-2})'; cb2.Label.FontSize = 18;
title(ax2, 'EKE', 'FontSize', 18, 'FontWeight', 'bold');
xlabel(ax2, 'x (km)', 'FontSize', 18); ylabel(ax2, 'y (km)', 'FontSize', 18);
set(ax2, 'FontSize', 16);

% Global Title
avg_label = 'Last Time-Averaged Window';
sgtitle(tl, sprintf('%s — Depth-Averaged Energy\n%s', ...
    run_tag, avg_label), 'Interpreter', 'none', 'FontWeight', 'bold', 'FontSize', 16);

% ---------- saving ----------
outdir = p.Results.outdir;
if isempty(outdir), outdir = fullfile(fileparts(mfilename('fullpath')), 'animations'); end
if ~exist(outdir,'dir'), mkdir(outdir); end

outname = fullfile(outdir, [run_tag '_energy_maps.png']);
print(fig, outname, '-dpng', '-r200');
fprintf('Saved map to: %s\n', outname);

end

% ================== helpers ==================

function data = load_last_3d(fn, varname)
    % Loads the last time step of a variable, keeping 3D [Nx, Ny, Nz]
    try
        vinfo = ncinfo(fn, varname);
        vsize = vinfo.Size;
        nd = numel(vsize);
        
        if nd == 4
            % (Nx, Ny, Nz, Nt) -> last t
            data = ncread(fn, varname, [1 1 1 vsize(4)], [vsize(1) vsize(2) vsize(3) 1]);
        elseif nd == 3
            % (Nx, Ny, Nz) or (Nx, Ny, Nt)
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

% ================== helper ==================
function data = try_ncread_(fname, names)
% Tries reading a variable from a list of possible names
for i = 1:numel(names)
    try
        data = ncread(fname, names{i});
        return;
    catch
    end
end
% Error reporting if not found
info = ncinfo(fname);
avail = strjoin({info.Variables.Name}, ', ');
error('Could not find variable. Tried: {%s}. Available: %s', ...
    strjoin(names, ', '), avail);
end
