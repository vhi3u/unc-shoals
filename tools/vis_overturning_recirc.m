function vis_overturning_recirc(cluster_name, varargin)
% vis_overturning_recirc(cluster_name, 'run', 'Zs5_Ls40k', 'outdir', OUTDIR, ...
%   'clipped', true/false, 'y_range', [y0 y1], 'n_psi', N, ...
%   'climRho', [lo hi], 'climPsi', [lo hi], 'eos_alpha', A, 'eos_beta', B, 'rho0', R0)
%
% 2-panel still of the time-mean circulation for one run (default: the
% reference case Zs5_Ls40k), from the LAST averaging window of
% time_avg_3d_sweep_<run>.nc:
%
%   (a) Overturning (x–z): alongshore-averaged density (color) from the
%       model's LINEAR equation of state,
%           rho = rho0 [ 1 - alpha (T - T_ref) + beta (S - S_ref) ]
%       (T_ref, S_ref = section means, so the mean density is rho0), with the
%       alongshore-integrated overturning streamfunction (contours)
%           Ψ(x,z) = -∫_y ∫_{-H}^{z} u dz' dy          [m^3 s^-1]
%       so that u = -(1/L_y) ∂Ψ/∂z. Positive Ψ (solid) = clockwise in the
%       x–z plane (offshore at the top, onshore below); negative (dashed) =
%       counter-clockwise.
%
%   (b) Recirculation (x–y): depth-integrated transport streamfunction
%           ψ(x,y) = ∫_0^x ∫_{-H}^{0} v dz dx'          [m^3 s^-1]
%       with ψ = 0 at the coast (x = 0). Positive (red) = clockwise,
%       negative (blue) = counter-clockwise.
%
% Layout: square x–z panel (a) beside a tall x–y panel (b).
%
% Requires: cmocean
%
% Example:
%   vis_overturning_recirc('cluster_Fnnie_1_19', 'clipped', true)

if nargin < 1 || isempty(cluster_name)
    cluster_name = 'cluster_Fnnie_1_19';
end

% ---------- options ----------
p = inputParser;
addParameter(p, 'sweep_root', '/glade/work/nguyen/unc-shoals/sweep_output', @ischar);
addParameter(p, 'outdir', '/glade/work/nguyen/unc-shoals/animations', @ischar);
addParameter(p, 'run', 'Zs5_Ls40k', @ischar);
addParameter(p, 'clipped', true, @(x)islogical(x)&&isscalar(x));
addParameter(p, 'y_range', [], @(v)isempty(v) || (isnumeric(v) && numel(v)==2));  % km; [] = whole domain
addParameter(p, 'n_psi', 8, @(x)isnumeric(x)&&isscalar(x)&&x>=1);               % contour levels per sign
addParameter(p, 'climRho', [], @(v)isempty(v) || (isnumeric(v) && numel(v)==2));
addParameter(p, 'climPsi', [], @(v)isempty(v) || (isnumeric(v) && numel(v)==2));
% Linear equation of state of the model (Oceananigans LinearEquationOfState defaults)
addParameter(p, 'eos_alpha', 1.67e-4, @(x)isnumeric(x)&&isscalar(x));   % thermal expansion  [1/K]
addParameter(p, 'eos_beta',  7.80e-4, @(x)isnumeric(x)&&isscalar(x));   % haline contraction [1/psu]
addParameter(p, 'rho0',      1024,    @(x)isnumeric(x)&&isscalar(x));   % reference density  [kg/m^3]

parse(p, varargin{:});
run_core    = p.Results.run;
clipped     = p.Results.clipped;
y_range     = p.Results.y_range;
n_psi       = p.Results.n_psi;
outdir      = p.Results.outdir;
cluster_dir = fullfile(p.Results.sweep_root, cluster_name);

% ---------- file / grid ----------
fname = find_time_avg(cluster_dir, run_core);
fprintf('Reading time-mean fields from: %s\n', fname);

xc = try_ncread_(fname, {'xC','x_caa','xc','x'}); xc = xc(:);
yc = try_ncread_(fname, {'yC','y_aca','yc','y'}); yc = yc(:);
zc = try_ncread_(fname, {'zC','z_aac','zc','z'}); zc = zc(:);
dx = cell_widths(fname, xc, {'xF','x_faa','xf'});
dy = cell_widths(fname, yc, {'yF','y_afa','yf'});
dz = cell_widths(fname, zc, {'zF','z_aaf','zf'});
xc_km = xc / 1000;
yc_km = yc / 1000;

% Alongshore range used for the overturning integral / density average
if isempty(y_range)
    iy_ot = 1:numel(yc);
else
    iy_ot = find(yc_km >= y_range(1) & yc_km <= y_range(2));
end

% Display clipping (same as the other figures)
ix_show = 1:numel(xc);
iy_show = 1:numel(yc);
clip_x = [0 100];   % km
clip_y = [20 200];  % km
if clipped
    ix_show = find(xc_km >= clip_x(1) & xc_km <= clip_x(2));
    iy_show = find(yc_km >= clip_y(1) & yc_km <= clip_y(2));
end

% ---------- load (Nx, Ny, Nz) fields, last window ----------
u = read_last(fname, {'u_c','u'});
v = read_last(fname, {'v_c','v'});
T = read_last(fname, {'T'});
S = read_last(fname, {'S'});
wet = (T ~= 0) & isfinite(T);
u(~wet) = 0; v(~wet) = 0;            % zero flux through land / immersed cells

DX = reshape(dx, [], 1, 1);
DY = reshape(dy, 1, [], 1);
DZ = reshape(dz, 1, 1, []);

% ---------- (a) overturning streamfunction Ψ(x,z) ----------
U_y = squeeze(sum(u(:, iy_ot, :) .* DY(1, iy_ot, 1), 2));    % ∫ u dy   (Nx, Nz)  [m^2/s]
[zc_sorted, kz] = sort(zc);                                  % integrate bottom -> top
Psi = zeros(size(U_y));
Psi(:, kz) = -cumsum(U_y(:, kz) .* dz(kz)', 2);              % [m^3/s]

% Alongshore-averaged density over wet cells (linear equation of state)
Tm = T; Sm = S; Tm(~wet) = NaN; Sm(~wet) = NaN;
T_xz = squeeze(mean(Tm(:, iy_ot, :), 2, 'omitnan'));          % (Nx, Nz)
S_xz = squeeze(mean(Sm(:, iy_ot, :), 2, 'omitnan'));
T_ref = mean(T_xz(:), 'omitnan');  S_ref = mean(S_xz(:), 'omitnan');
sig_xz = p.Results.rho0 * (1 - p.Results.eos_alpha * (T_xz - T_ref) + p.Results.eos_beta * (S_xz - S_ref));
land_xz = all(~wet(:, iy_ot, :), 2);
land_xz = squeeze(land_xz);
sig_xz(land_xz) = NaN;
Psi(land_xz) = NaN;

% ---------- (b) recirculation streamfunction ψ(x,y) ----------
V_z = sum(v .* DZ, 3);                                       % ∫ v dz   (Nx, Ny)  [m^2/s]
psi_xy = cumsum(V_z .* DX, 1);                               % ∫_0^x V dx' [m^3/s]
land_xy = all(~wet, 3);
psi_xy(land_xy) = NaN;

% ---------- clip for display ----------
xs = xc_km(ix_show);  ys = yc_km(iy_show);
sig_show = sig_xz(ix_show, :)';                              % (Nz, Nx)
Psi_show = Psi(ix_show, :)';
psi_show = psi_xy(ix_show, iy_show)';                        % (Ny, Nx)
[Xz, Zz] = meshgrid(xs, zc);
[Xc, Yc] = meshgrid(xs, ys);

% ---------- color / contour limits ----------
if isempty(p.Results.climRho)
    clim_rho = [min(sig_show(:), [], 'omitnan'), max(sig_show(:), [], 'omitnan')];
else
    clim_rho = p.Results.climRho;
end
if isempty(p.Results.climPsi)
    pm = max(abs(psi_show(:)), [], 'omitnan');
    clim_psi = [-pm, pm];
else
    clim_psi = p.Results.climPsi;
end
Pm = max(abs(Psi_show(:)), [], 'omitnan');
psi_lev = linspace(0, Pm, n_psi + 1);  psi_lev = psi_lev(2:end);   % exclude 0
fprintf('Density range: [%.3f, %.3f] kg m^-3  (rho0 = %g at T_ref = %.2f, S_ref = %.2f)\n', clim_rho(1), clim_rho(2), p.Results.rho0, T_ref, S_ref);
fprintf('Overturning |Ψ| max: %.4g m^3/s,  contour spacing: %.3g m^3/s\n', Pm, Pm / n_psi);
fprintf('Recirculation color limits: [%.4g, %.4g] m^3/s\n', clim_psi(1), clim_psi(2));

% ---------- figure ----------
% Manual layout (pixels) so both panels share exactly the same top and bottom:
%   (a) square x–z panel | colorbar | (b) tall x–y panel | colorbar
% Colorbars sit outside the panels so their labels stay legible.
% =========================================================================
% 🟢 EDIT LAYOUT HERE (pixels)
% =========================================================================
fig_scale = 0.5;        % shrinks the panels (fonts and label margins stay the same)
H       = round(405 * fig_scale);   % panel height = side of the square panel (a)
Wb      = round(200 * fig_scale);   % width of panel (b) (≈ half its height, like the XY surface maps)
pad_l   = 75;           % left margin: (a) y tick labels + y-label
pad_b   = 70;           % bottom margin: x tick labels + x-label
pad_t   = 45;           % top margin: titles
cb_gap  = 10;           % gap between a panel and its colorbar
cb_w    = 16;           % colorbar width
cb_lab  = 85;           % room right of a colorbar for its tick labels + label
ylab_b  = 70;           % room left of (b) for its y tick labels + y-label

pos_a  = [pad_l, pad_b, H, H];
pos_ca = [pos_a(1) + H + cb_gap, pad_b, cb_w, H];
pos_b  = [pos_ca(1) + cb_w + cb_lab + ylab_b, pad_b, Wb, H];
pos_cb = [pos_b(1) + Wb + cb_gap, pad_b, cb_w, H];
fig_w  = pos_cb(1) + cb_w + cb_lab;
fig_h  = pad_b + H + pad_t;
fig = figure('Color','w','Units','pixels','Position',[80 50 fig_w fig_h]);

% --- (a) overturning on density ---
ax1 = axes(fig, 'Units', 'pixels', 'Position', pos_a);
pcolor(ax1, Xz, Zz, sig_show);
shading(ax1, 'flat'); axis(ax1, 'tight'); set(ax1, 'YDir', 'normal');
set(ax1, 'Layer', 'top', 'Color', [0.9 0.9 0.9]);   % gray = land / immersed
box(ax1, 'on'); set(ax1, 'LineWidth', 0.8, 'XColor', 'k', 'YColor', 'k');
colormap(ax1, cmocean('dense')); caxis(ax1, clim_rho);
hold(ax1, 'on');
contour(ax1, Xz, Zz, Psi_show, as_levels(psi_lev),          '-',  'Color', 'k', 'LineWidth', 1.2);  % clockwise
contour(ax1, Xz, Zz, Psi_show, as_levels(-fliplr(psi_lev)), '--', 'Color', 'k', 'LineWidth', 1.2);  % counter-clockwise
cb1 = colorbar(ax1);

title(ax1, '(a) Overturning', 'FontSize', 18, 'FontWeight', 'bold');
xlabel(ax1, 'x (km)', 'FontSize', 16, 'FontWeight', 'bold');
ylabel(ax1, 'z (m)', 'FontSize', 16, 'FontWeight', 'bold');
set(ax1, 'FontSize', 14);
xlim(ax1, [min(xs), max(xs)]);
ylim(ax1, [min(zc_sorted) - dz(kz(1))/2, 0]);

% --- (b) recirculation (plotted in units of 10^e m^3/s for readable ticks) ---
e_psi = floor(log10(max(abs(clim_psi))));
sc    = 10^e_psi;
ax2 = axes(fig, 'Units', 'pixels', 'Position', pos_b);
pcolor(ax2, Xc, Yc, psi_show / sc);
shading(ax2, 'flat'); axis(ax2, 'tight'); set(ax2, 'YDir', 'normal');
set(ax2, 'Layer', 'top', 'Color', [0.9 0.9 0.9]);
box(ax2, 'on'); set(ax2, 'LineWidth', 0.8, 'XColor', 'k', 'YColor', 'k');
colormap(ax2, cmocean('balance')); caxis(ax2, clim_psi / sc);
hold(ax2, 'on');
rec_lev = linspace(0, clim_psi(2) / sc, n_psi + 1);  rec_lev = rec_lev(2:end);
contour(ax2, Xc, Yc, psi_show / sc, as_levels(rec_lev),          '-',  'Color', 'k', 'LineWidth', 0.8);
contour(ax2, Xc, Yc, psi_show / sc, as_levels(-fliplr(rec_lev)), '--', 'Color', 'k', 'LineWidth', 0.8);
cb2 = colorbar(ax2);

title(ax2, '(b) Recirculation', 'FontSize', 18, 'FontWeight', 'bold');
xlabel(ax2, 'x (km)', 'FontSize', 16, 'FontWeight', 'bold');
ylabel(ax2, 'y (km)', 'FontSize', 16, 'FontWeight', 'bold');
set(ax2, 'FontSize', 14);
xlim(ax2, [min(xs), max(xs)]);
ylim(ax2, [min(ys), max(ys)]);

% Mark the alongshore range used in (a), if restricted
if ~isempty(y_range)
    yline(ax2, y_range(1), ':', 'Color', [0.2 0.2 0.2], 'LineWidth', 1.2);
    yline(ax2, y_range(2), ':', 'Color', [0.2 0.2 0.2], 'LineWidth', 1.2);
end

% ---------- colorbars (outside, full panel height) ----------
cb1.Units = 'pixels';  cb1.Position = pos_ca;
cb2.Units = 'pixels';  cb2.Position = pos_cb;
cb1.Label.String = '\rho (kg m^{-3})';
cb2.Label.String = sprintf('\\psi (10^{%d} m^3 s^{-1})', e_psi);
for cb = [cb1, cb2]
    cb.FontSize = 13;
    cb.Label.FontSize = 15;
    cb.Label.FontWeight = 'bold';
    cb.TickDirection = 'out';
    cb.LineWidth = 0.8;
end
% Adding a colorbar shrinks its axes; restore the exact panel positions
ax1.Position = pos_a;
ax2.Position = pos_b;
drawnow;

% ---------- output ----------
if ~exist(outdir,'dir'), mkdir(outdir); end
clip_str = '';
if clipped, clip_str = '_clipped'; end
outfile = fullfile(outdir, sprintf('%s_%s_overturning_recirc%s.png', cluster_name, run_core, clip_str));
exportgraphics(fig, outfile, 'Resolution', 300);
fprintf('Saved still image: %s \n', outfile);
close(fig);

end

% ================== helpers ==================

function lev = as_levels(lev)
% contour() treats a scalar as a NUMBER of levels; repeat it to mean one level
if isscalar(lev), lev = [lev lev]; end
end

function fname = find_time_avg(cluster_dir, sw_core)
% Locate time_avg_3d file inside the sweep_NNN_<sw_core> folder
d = dir(fullfile(cluster_dir, ['*' sw_core '*']));
d = d([d.isdir]);
d = d(~ismember({d.name}, {'.', '..'}));
if isempty(d)
    error('No folder found matching *%s* in %s', sw_core, cluster_dir);
end
datadir = fullfile(cluster_dir, d(1).name);
candidates = { fullfile(datadir, ['time_avg_3d_sweep_' sw_core '.nc']), ...
               fullfile(datadir, ['time_avg_3d_' sw_core '.nc']) };
for c = 1:numel(candidates)
    if isfile(candidates{c}), fname = candidates{c}; return; end
end
error('No time_avg_3d file for %s in %s', sw_core, datadir);
end

function data = read_last(fname, names)
% Read full (Nx, Ny, Nz) field from the last time-averaging window
for i = 1:numel(names)
    try
        vinfo = ncinfo(fname, names{i});
        vsize = vinfo.Size;
        if numel(vsize) == 4
            data = ncread(fname, names{i}, [1 1 1 vsize(4)], [vsize(1) vsize(2) vsize(3) 1]);
        else
            data = ncread(fname, names{i});
        end
        return;
    catch
        % try next name
    end
end
error('Could not read any of {%s} from %s', strjoin(names, ', '), fname);
end

function d = cell_widths(fname, c, face_names)
% Cell widths from face coordinates if present, else from center midpoints
for nm = face_names
    try
        f = ncread(fname, nm{1});
        if numel(f) == numel(c) + 1
            d = abs(diff(f(:)));
            return;
        end
    catch
    end
end
if numel(c) == 1
    d = 1;
    return;
end
mid   = (c(1:end-1) + c(2:end)) / 2;
edges = [c(1) - (mid(1) - c(1)); mid; c(end) + (c(end) - mid(end))];
d     = abs(diff(edges));
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
info = ncinfo(fname);
avail = strjoin({info.Variables.Name}, ', ');
error('Could not find coordinate variable. Tried: {%s}. Available: %s', ...
    strjoin(names, ', '), avail);
end
