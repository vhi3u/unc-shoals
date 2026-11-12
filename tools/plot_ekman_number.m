function plot_ekman_number(cluster_name)
% plot_ekman_number(cluster_name)
%
% Ekman number  Ek = nu / (f H0^2)  for the reference run (tau = 0) and the
% wind-stress runs, from the LAST time-averaging window of
% time_avg_3d_sweep_<run>.nc. H0 = shelf depth (25 m for these runs).
%
% The model uses WENO horizontally and CATKE vertically, so nu is the CATKE
% vertical eddy viscosity. Three estimates are computed:
%
%   1. Ek_CATKE = <nu_z> / (f H0^2)
%        <nu_z> = mean CATKE viscosity over the shelf water column, if a
%        viscosity field was saved in the time-averaged output (searched for
%        by name: nu_names below). Skipped (with a message) otherwise.
%
%   2. Ek_surf = 1/2 (delta / H0)^2          (wind runs; from delta = sqrt(2 nu / f))
%        delta = surface Ekman-layer depth diagnosed from the model: depth of
%        the first sign change of the wind-induced cross-shelf flow
%        dU(z) = int_y (u_tau - u_0) dy at x_ek (as in plot_wind_ekman_sweep.m).
%        Also shown: the stratified theory  H_Ek = sqrt(|tau| / (rho0 N f))
%        (Zhao et al.; McWilliams et al. 2009), as Ek_th = 1/2 (H_Ek / H0)^2.
%
%   3. Ek_bot = 2 (r / (f H0))^2 = 2 R^2    (all runs; from r = sqrt(nu f / 2))
%        r = C_d U_b, U_b = rms near-bottom speed sqrt(<u^2 + v^2>) at the
%        lowest wet cell, averaged over the shelf (x_shelf, y >= 20 km).
%
% Panels: (a) Ek vs tau (log axis), (b) boundary-layer depths vs tau with H0.
% A table of all quantities is printed to the log.
%
% Example:
%   plot_ekman_number('cluster_Fnnie_1_19')

if nargin < 1 || isempty(cluster_name)
    cluster_name = 'cluster_Fnnie_1_19';
end

sweep_output_root = '/glade/work/nguyen/unc-shoals/sweep_output';
cluster_dir       = fullfile(sweep_output_root, cluster_name);

runs = { 'wind_m0p15', 'wind_m0p10', 'wind_m0p08', 'wind_m0p05', 'wind_m0p03', ...
         'Zs5_Ls40k', ...
         'wind_0p03', 'wind_0p05', 'wind_0p08', 'wind_0p10', 'wind_0p15' };

% Order here is MEANINGFUL (wind-stress axis / panel order), so this is NOT
% auto-discovered -- discover_sweeps returns directory order and would
% scramble the axis. Runs absent from this cluster are dropped and reported.
runs = keep_existing(cluster_dir, runs);

tau  = [-0.15, -0.10, -0.08, -0.05, -0.03, 0, 0.03, 0.05, 0.08, 0.10, 0.15];   % N/m^2
ref_run = 'Zs5_Ls40k';

% =========================================================================
% 🟢 SETTINGS
% =========================================================================
H0        = 25;                    % shelf depth (m) for these runs
x_ek      = 40;                    % km: cross-shelf position for the surface Ekman layer
y_ek      = [80, 200];             % km: alongshore range for the surface Ekman layer
x_shelf   = [0, 50];               % km: shelf region for nu_z and U_b
y_shelf   = [20, Inf];             % km: outside the 20 km southern nudging zone
Cd        = 2.5e-3;                % quadratic bottom drag coefficient
rho0      = 1024.0;
g         = 9.81;
eos_alpha = 1.67e-4;               % linear equation of state (Oceananigans defaults)
eos_beta  = 7.80e-4;
x_ambient = 10;                    % km: outer strip used for the background N
nu_names  = {'νz', 'nu_z', 'νu', 'nu_u', 'κu', 'kappa_u', 'Ku', 'viscosity', 'nu'};  % CATKE viscosity names to try
f         = 2 * (2*pi/86400) * sin(deg2rad(35.2480));   % ~8.44e-5 s^-1
fs        = 11;

% ---------- data ----------
nr = numel(runs);
prof = cell(1, nr);  Ub = nan(1, nr);  nuz = nan(1, nr);  N = nan(1, nr);  nu_var = '';
for k = 1:nr
    fname = find_time_avg(cluster_dir, runs{k});
    if isempty(fname), continue; end
    fprintf('  %-11s -> %s\n', runs{k}, fname);
    try
        [prof{k}, Ub(k), nuz(k), N(k), nv] = read_run(fname, x_ek, y_ek, x_shelf, y_shelf, ...
            nu_names, g, eos_alpha, eos_beta, x_ambient);
        if ~isempty(nv), nu_var = nv; end
    catch ME
        warning('Error processing %s: %s', runs{k}, ME.message);
    end
end
kref = find(strcmp(runs, ref_run), 1);
if isempty(prof{kref})
    error('Reference run %s could not be read; it is needed for the surface Ekman layer.', ref_run);
end

% 1. CATKE viscosity
Ek_catke = nuz / (f * H0^2);
if isempty(nu_var)
    fprintf('\nNo CATKE viscosity field found (tried: %s). Ek_CATKE skipped.\n', strjoin(nu_names, ', '));
else
    fprintf('\nUsing CATKE viscosity field ''%s''.\n', nu_var);
end

% 2. Surface Ekman layer (wind runs)
delta = nan(1, nr);
for k = 1:nr
    if tau(k) == 0 || isempty(prof{k}), continue; end
    delta(k) = ekman_depth(prof{k}, prof{kref}, sign(tau(k)));
end
Ek_surf = 0.5 * (delta / H0).^2;
H_Ek    = sqrt(abs(tau) ./ (rho0 * N * f));
H_Ek(tau == 0) = NaN;
Ek_th   = 0.5 * (H_Ek / H0).^2;

% 3. Bottom Ekman layer (all runs)
r_lin   = Cd * Ub;
Ek_bot  = 2 * (r_lin ./ (f * H0)).^2;
delta_b = 2 * r_lin / f;                       % bottom Ekman depth sqrt(2 nu / f) with nu = 2 r^2 / f

fprintf('\n%7s %10s %9s %9s %9s %9s %9s %10s %10s %10s\n', 'tau', 'U_b(m/s)', 'delta_s', 'H_Ek', 'delta_b', ...
    'Ek_surf', 'Ek_th', 'Ek_bot', 'nu_z', 'Ek_CATKE');
for k = 1:nr
    fprintf('%7.2f %10.3f %8.1fm %8.1fm %8.1fm %9.3g %9.3g %10.3g %10.3g %10.3g\n', tau(k), Ub(k), ...
        delta(k), H_Ek(k), delta_b(k), Ek_surf(k), Ek_th(k), Ek_bot(k), nuz(k), Ek_catke(k));
end

% ---------- plotting ----------
fig = figure('Color', 'w', 'Position', [100 100 760 360]);
tl  = tiledlayout(fig, 1, 2, 'TileSpacing', 'normal', 'Padding', 'compact');
col_s  = [0.0, 0.0, 0.7];    % surface (blue)
col_b  = [0.7, 0.0, 0.0];    % bottom (red)
col_c  = [0.0, 0.5, 0.2];    % CATKE (green)
style_axis = @(ax) set(ax, 'FontSize', fs, 'Box', 'on', 'XGrid', 'on', 'YGrid', 'on', ...
    'LineWidth', 1, 'TickDir', 'out', 'XColor', 'k', 'YColor', 'k');

% (a) Ekman numbers
ax1 = nexttile(tl); hold(ax1, 'on');
h = gobjects(0);  lab = {};
h(end+1) = plot(ax1, tau, Ek_surf, '-o', 'LineWidth', 1.5, 'Color', col_s, 'MarkerFaceColor', col_s, 'MarkerEdgeColor', 'none', 'MarkerSize', 5);
lab{end+1} = 'Surface (model \delta)';
h(end+1) = plot(ax1, tau, Ek_th, '--', 'LineWidth', 1.2, 'Color', col_s);
lab{end+1} = 'Surface (theory H_{Ek})';
h(end+1) = plot(ax1, tau, Ek_bot, '-s', 'LineWidth', 1.5, 'Color', col_b, 'MarkerFaceColor', col_b, 'MarkerEdgeColor', 'none', 'MarkerSize', 5);
lab{end+1} = 'Bottom (2R^2)';
if any(isfinite(Ek_catke))
    h(end+1) = plot(ax1, tau, Ek_catke, '-^', 'LineWidth', 1.5, 'Color', col_c, 'MarkerFaceColor', col_c, 'MarkerEdgeColor', 'none', 'MarkerSize', 5);
    lab{end+1} = 'CATKE \nu_z/(fH_0^2)';
end
set(ax1, 'YScale', 'log');
xlabel(ax1, '\tau (N m^{-2})', 'FontSize', fs);
ylabel(ax1, 'Ek = \nu / (f H_0^2)', 'FontSize', fs);
title(ax1, '(a) Ekman Number', 'FontSize', fs+2, 'FontWeight', 'bold');
xlim(ax1, [min(tau), max(tau)] + [-0.02, 0.02]);
style_axis(ax1); pbaspect(ax1, [1 1 1]);
legend(ax1, h, lab, 'Location', 'best', 'FontSize', fs - 2);

% (b) boundary-layer depths
ax2 = nexttile(tl); hold(ax2, 'on');
yline(ax2, H0, '-', 'Color', [0.5 0.5 0.5], 'LineWidth', 1, 'HandleVisibility', 'off');
text(ax2, max(tau), H0, 'H_0', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', 'FontSize', fs - 1, 'Color', [0.4 0.4 0.4]);
h2 = gobjects(0);  lab2 = {};
h2(end+1) = plot(ax2, tau, delta, '-o', 'LineWidth', 1.5, 'Color', col_s, 'MarkerFaceColor', col_s, 'MarkerEdgeColor', 'none', 'MarkerSize', 5);
lab2{end+1} = 'Surface \delta (model)';
h2(end+1) = plot(ax2, tau, H_Ek, '--', 'LineWidth', 1.2, 'Color', col_s);
lab2{end+1} = 'H_{Ek} (theory)';
h2(end+1) = plot(ax2, tau, delta_b, '-s', 'LineWidth', 1.5, 'Color', col_b, 'MarkerFaceColor', col_b, 'MarkerEdgeColor', 'none', 'MarkerSize', 5);
lab2{end+1} = 'Bottom 2r/f';
xlabel(ax2, '\tau (N m^{-2})', 'FontSize', fs);
ylabel(ax2, 'Boundary-layer depth (m)', 'FontSize', fs);
title(ax2, '(b) Ekman Layer Depth', 'FontSize', fs+2, 'FontWeight', 'bold');
xlim(ax2, [min(tau), max(tau)] + [-0.02, 0.02]);
ymax = max([delta, H_Ek, delta_b, H0], [], 'omitnan');
ylim(ax2, [0, 1.15 * ymax]);
style_axis(ax2); pbaspect(ax2, [1 1 1]);
legend(ax2, h2, lab2, 'Location', 'best', 'FontSize', fs - 2);

drawnow;
outdir = '/glade/work/nguyen/unc-shoals/animations';
if ~exist(outdir, 'dir'), outdir = pwd; end
outfile = fullfile(outdir, sprintf('%s_ekman_number.png', cluster_name));
exportgraphics(fig, outfile, 'Resolution', 300);
fprintf('Saved: %s\n', outfile);
end

% ================== helpers ==================

function delta = ekman_depth(P, P0, s)
% Depth of the first sign change of the wind-induced cross-shelf flow below the surface
dU = P.U - P0.U;
[~, kz] = sort(P.zc, 'descend');
dUs = dU(kz);  dzs = P.dz(kz);
k_end = find(s * dUs <= 0, 1, 'first');
if isempty(k_end), k_end = numel(dUs) + 1; end
delta = sum(dzs(1:k_end-1));
end

function [P, Ub, nuz, N, nu_var] = read_run(fname, x_ek, y_ek, x_shelf, y_shelf, nu_names, g, ea, eb, x_amb)
xc = try_ncread_(fname, {'xC','xc','x','x_caa'});  xc = xc(:);  xk = xc / 1e3;
yc = try_ncread_(fname, {'yC','yc','y','y_aca'});  yc = yc(:);  yk = yc / 1e3;
zc = try_ncread_(fname, {'zC','zc','z','z_aac'});  zc = zc(:);
dz = cell_width(fname, zc, {'zF','z_aaf','zf'});
dy = cell_width(fname, yc, {'yF','y_afa','yf'});
vinfo = ncinfo(fname, 'u_c');  t = vinfo.Size(end);

% Alongshore-integrated cross-shelf velocity profile at x_ek (surface Ekman layer)
[~, ix] = min(abs(xk - x_ek));
iy = find(yk >= y_ek(1) & yk <= y_ek(2));
u = squeeze(ncread(fname, 'u_c', [ix iy(1) 1 t], [1 numel(iy) Inf 1]));   % (Ny, Nz)
u(~isfinite(u)) = 0;
P.U = (dy(iy)' * u)';  P.zc = zc;  P.dz = dz;

% Shelf block: near-bottom rms speed and CATKE viscosity
ixs = find(xk >= x_shelf(1) & xk <= x_shelf(2));
iys = find(yk >= y_shelf(1) & yk <= y_shelf(2));
rd  = @(v) ncread(fname, v, [ixs(1) iys(1) 1 t], [numel(ixs) numel(iys) Inf 1]);
[~, kup] = sort(zc, 'ascend');                         % bottom -> top along dim 3
T   = rd('T');   T = T(:, :, kup);
ke2 = rd('uu') + rd('vv');  ke2 = ke2(:, :, kup);      % <u^2 + v^2>
wet = (T ~= 0) & isfinite(T);
Nzz = size(T, 3);  kk = reshape(1:Nzz, 1, 1, []);
k_bot = min(wet .* kk + ~wet * (Nzz + 1), [], 3);      % lowest wet level
okc = k_bot <= Nzz;
[ii, jj] = ndgrid(1:size(T, 1), 1:size(T, 2));
Ub = sqrt(mean(ke2(sub2ind(size(ke2), ii(okc), jj(okc), k_bot(okc))), 'omitnan'));

nuz = NaN;  nu_var = '';
info = ncinfo(fname);  vnames = {info.Variables.Name};
for c = 1:numel(nu_names)
    if any(strcmp(vnames, nu_names{c}))
        nu = rd(nu_names{c});  nu = nu(:, :, kup);
        DZ = reshape(dz(kup), 1, 1, []) .* wet;
        nuz = sum(nu .* DZ, 'all', 'omitnan') / sum(DZ, 'all');   % thickness-weighted shelf mean
        nu_var = nu_names{c};
        break;
    end
end

% Background N from the ambient offshore column (linear EOS, bulk)
ixa = find(xk >= max(xk) - x_amb);
Tc = ncread(fname, 'T', [ixa(1) iys(1) 1 t], [numel(ixa) numel(iys) Inf 1]);
Sc = ncread(fname, 'S', [ixa(1) iys(1) 1 t], [numel(ixa) numel(iys) Inf 1]);
b  = g * (ea * Tc - eb * Sc);  b(Tc == 0 | ~isfinite(Tc)) = NaN;
bp = squeeze(median(median(b, 1, 'omitnan'), 2, 'omitnan'));
ok = isfinite(bp);  N = NaN;
if sum(ok) >= 2
    [z_hi, i_hi] = max(zc(ok));  [z_lo, i_lo] = min(zc(ok));  bz = bp(ok);
    N2 = (bz(i_hi) - bz(i_lo)) / (z_hi - z_lo);
    if N2 > 0, N = sqrt(N2); end
end
end

function fname = find_time_avg(cluster_dir, sw_core)
fname = '';
d = dir(fullfile(cluster_dir, ['*' sw_core '*']));
d = d([d.isdir]);
d = d(~ismember({d.name}, {'.', '..'}));
if isempty(d)
    warning('Folder not found for %s', sw_core);
    return;
end
datadir = fullfile(cluster_dir, d(1).name);
for c = {['time_avg_3d_sweep_' sw_core '.nc'], ['time_avg_3d_' sw_core '.nc']}
    if isfile(fullfile(datadir, c{1})), fname = fullfile(datadir, c{1}); return; end
end
warning('No time_avg_3d file for %s in %s', sw_core, datadir);
end

function d = cell_width(fname, c, face_names)
for nm = face_names
    try
        fc = ncread(fname, nm{1});
        if numel(fc) == numel(c) + 1
            d = abs(diff(fc(:)));
            return;
        end
    catch
    end
end
if numel(c) == 1, d = 1; return; end
mid   = (c(1:end-1) + c(2:end)) / 2;
edges = [c(1) - (mid(1) - c(1)); mid; c(end) + (c(end) - mid(end))];
d     = abs(diff(edges));
end

function data = try_ncread_(fname, names)
for i = 1:numel(names)
    try
        data = ncread(fname, names{i});
        return;
    catch
    end
end
error('Could not find variable in %s', fname);
end
