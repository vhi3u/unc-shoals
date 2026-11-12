function plot_wind_ekman_sweep(cluster_name)
% plot_wind_ekman_sweep(cluster_name)
%
% 2-panel figure for the wind-stress sweep:
%   (a) Offshore volume transport Q (same definition as panel (e) of
%       plot_full_scatter_sweeps.m: u > 0 only, top 25 m, y = 80-200 km,
%       at x = 40 km)
%   (b) Ekman volume transport diagnosed FROM THE MODEL (left axis, solid),
%       the classical theory tau L / (rho0 f) for comparison (left axis,
%       dashed), and the Ekman share of the offshore transport,
%       100 * Q_Ek / Q (right axis; upwelling-favourable winds only)
%
% Model Ekman layer and transport (same x and y-range as Q):
%   dU(z)  = int_y [ u_tau(z) - u_0(z) ] dy      wind-induced cross-shelf flow
%            (u_0 = no-wind reference run, removes the circulation present
%            without wind, including its geostrophic part)
%   delta  = depth of the first sign change of dU below the surface
%            (base of the surface Ekman layer; the return flow lies below)
%   Q_Ek   = int_{-delta}^{0} dU dz               [m^3/s], signed
%            (> 0 offshore for upwelling winds, < 0 onshore for downwelling)
% The diagnosed delta is compared with the stratified Ekman depth
%   H_Ek = sqrt(|tau| / (rho0 N f))   (McWilliams et al. 2009; Zhao et al.)
% with N the background stratification of the ambient offshore column from
% the model's linear equation of state.
%
% Sign convention: tau > 0 is upwelling-favourable.
%
% Example:
%   plot_wind_ekman_sweep('cluster_Fnnie_1_19')

if nargin < 1 || isempty(cluster_name)
    cluster_name = 'cluster_Fnnie_1_19';
end

sweep_output_root = '/glade/work/nguyen/unc-shoals/sweep_output';
cluster_dir       = fullfile(sweep_output_root, cluster_name);

% Wind sweep (tau = 0 is the reference run)
group_wind = { 'wind_m0p15', 'wind_m0p10', 'wind_m0p08', 'wind_m0p05', 'wind_m0p03', ...
               'Zs5_Ls40k', ...
               'wind_0p03', 'wind_0p05', 'wind_0p08', 'wind_0p10', 'wind_0p15' };
wind_vals = [-0.15, -0.10, -0.08, -0.05, -0.03, 0, 0.03, 0.05, 0.08, 0.10, 0.15]; % N/m^2
ref_run   = 'Zs5_Ls40k';

% =========================================================================
% 🟢 SETTINGS
% =========================================================================
x_target_km = 40;                 % cross-shelf position (as in panel (e))
y_range_Q   = [80, 200];          % km, alongshore range (as in panel (e))
z_top_m     = 25;                 % panel (a): integrate the top 25 m
rho0        = 1024.0;
g           = 9.81;
eos_alpha   = 1.67e-4;            % linear equation of state (Oceananigans defaults)
eos_beta    = 7.80e-4;
x_ambient   = 10;                 % km: outer strip used for the background N
f           = 2 * (2*pi/86400) * sin(deg2rad(35.2480));   % ~8.44e-5 s^-1
L           = (y_range_Q(2) - y_range_Q(1)) * 1e3;        % alongshore length (m)
fs = 11;

% ---------- data ----------
fprintf('Processing wind sweep...\n');
nr = numel(group_wind);
Q  = nan(1, nr);  prof = cell(1, nr);  Nbg = nan(1, nr);
for k = 1:nr
    [Q(k), prof{k}, Nbg(k)] = read_run(cluster_dir, group_wind{k}, x_target_km, y_range_Q, ...
                                       z_top_m, g, eos_alpha, eos_beta, x_ambient);
end
kref = find(strcmp(group_wind, ref_run), 1);
if isempty(prof{kref})
    error('Reference run %s could not be read; it is needed for the Ekman diagnostic.', ref_run);
end

% Model Ekman layer depth and transport
QEk_mod = nan(1, nr);  delta = nan(1, nr);
for k = 1:nr
    if wind_vals(k) == 0
        QEk_mod(k) = 0;  delta(k) = 0;  continue;
    end
    if isempty(prof{k}), continue; end
    [QEk_mod(k), delta(k)] = ekman_layer(prof{k}, prof{kref}, sign(wind_vals(k)));
end

QEk_th = wind_vals * L / (rho0 * f);                                    % classical theory
H_Ek   = sqrt(abs(wind_vals) ./ (rho0 * Nbg * f));                       % stratified Ekman depth
share  = 100 * QEk_mod ./ Q;
share(wind_vals <= 0 | ~(Q > 0) | ~(QEk_mod > 0)) = NaN;                 % offshore Ekman only

% mSv
Q_mSv      = Q / 1e3;
QEkmod_mSv = QEk_mod / 1e3;
QEkth_mSv  = QEk_th / 1e3;

fprintf('\n%7s %9s %13s %14s %9s %9s %10s\n', 'tau', 'Q (mSv)', 'Q_Ek model', 'Q_Ek theory', 'delta', 'H_Ek', 'Ek share');
for k = 1:nr
    if isnan(share(k)), sh = '-'; else, sh = sprintf('%.0f %%', share(k)); end
    fprintf('%7.2f %9.2f %13.2f %14.2f %8.1fm %8.1fm %10s\n', wind_vals(k), Q_mSv(k), ...
        QEkmod_mSv(k), QEkth_mSv(k), delta(k), H_Ek(k), sh);
end
fprintf(['\nQ_Ek model = wind-induced cross-shelf transport above the first sign change of\n' ...
         'dU(z) = int_y (u_tau - u_0) dy at x = %g km; theory = tau L / (rho0 f).\n'], x_target_km);

% ---------- plotting ----------
fig = figure('Color', 'w', 'Position', [100 100 760 360]);
tl  = tiledlayout(fig, 1, 2, 'TileSpacing', 'normal', 'Padding', 'compact');

col_q   = [0.0, 0.0, 0.7];     % blue: offshore transport (as in plot_full_scatter_sweeps)
col_ek  = [0.0, 0.5, 0.2];     % green: Ekman transport
col_pct = [0.7, 0.0, 0.0];     % red: Ekman share

style_axis = @(ax) set(ax, 'FontSize', fs, 'Box', 'on', 'XGrid', 'on', 'YGrid', 'on', ...
    'LineWidth', 1, 'TickDir', 'out');

% (a) offshore volume transport
ax1 = nexttile(tl); hold(ax1, 'on');
plot(ax1, wind_vals, Q_mSv, '-o', 'LineWidth', 1.5, 'Color', col_q, ...
    'MarkerFaceColor', col_q, 'MarkerEdgeColor', 'none', 'MarkerSize', 5);
xlabel(ax1, '\tau (N m^{-2})', 'FontSize', fs);
ylabel(ax1, 'Offshore Volume Transport (mSv)', 'FontSize', fs);
title(ax1, '(a) Offshore Transport', 'FontSize', fs+2, 'FontWeight', 'bold');
xlim(ax1, get_padded_lims(wind_vals));
ylim(ax1, get_padded_lims(Q_mSv));
ax1.YAxis.Color = col_q;
style_axis(ax1);
pbaspect(ax1, [1 1 1]);

% (b) Ekman transport (left) and Ekman share of offshore transport (right)
ax2 = nexttile(tl);
yyaxis(ax2, 'left'); hold(ax2, 'on');
yline(ax2, 0, '-', 'Color', [0.6 0.6 0.6], 'LineWidth', 0.8, 'HandleVisibility', 'off');
h_th  = plot(ax2, wind_vals, QEkth_mSv, '--', 'LineWidth', 1.2, 'Color', col_ek);
h_mod = plot(ax2, wind_vals, QEkmod_mSv, '-^', 'LineWidth', 1.5, 'Color', col_ek, ...
    'MarkerFaceColor', col_ek, 'MarkerEdgeColor', 'none', 'MarkerSize', 5);
ylabel(ax2, 'Ekman Volume Transport (mSv)', 'FontSize', fs);
ylim(ax2, get_padded_lims([QEkmod_mSv, QEkth_mSv]));
ax2.YAxis(1).Color = col_ek;

yyaxis(ax2, 'right'); hold(ax2, 'on');
h_pct = plot(ax2, wind_vals, share, '-s', 'LineWidth', 1.5, 'Color', col_pct, ...
    'MarkerFaceColor', col_pct, 'MarkerEdgeColor', 'none', 'MarkerSize', 5);
ylabel(ax2, 'Ekman Share of Offshore Q (%)', 'FontSize', fs);
if any(isfinite(share))
    ylim(ax2, [0, max(100, max(share, [], 'omitnan') * 1.15)]);
end
ax2.YAxis(2).Color = col_pct;

xlabel(ax2, '\tau (N m^{-2})', 'FontSize', fs);
title(ax2, '(b) Ekman Transport', 'FontSize', fs+2, 'FontWeight', 'bold');
xlim(ax2, get_padded_lims(wind_vals));
style_axis(ax2);
pbaspect(ax2, [1 1 1]);
legend(ax2, [h_mod, h_th, h_pct], {'Q_{Ek}, model', '\tau L / \rho_0 f', 'Q_{Ek} / Q (%)'}, ...
    'Location', 'northwest', 'FontSize', fs - 1);

drawnow;

% ---------- save ----------
outdir = '/glade/work/nguyen/unc-shoals/animations';
if ~exist(outdir, 'dir'), outdir = pwd; end
outfile = fullfile(outdir, sprintf('%s_wind_ekman_sweep.png', cluster_name));
exportgraphics(fig, outfile, 'Resolution', 300);
fprintf('Saved: %s\n', outfile);
end

% ================== helpers ==================

function [QEk, delta] = ekman_layer(P, P0, s)
% Wind-induced cross-shelf transport above the first sign change of
% dU = U_tau - U_0 (alongshore-integrated), integrating from the surface down.
%   s = sign(tau): +1 expects offshore surface flow, -1 onshore
dU = P.U - P0.U;                               % (Nz x 1) [m^2/s]
[~, kz] = sort(P.zc, 'descend');               % surface first
dUs = dU(kz);  dzs = P.dz(kz);
k_end = find(s * dUs <= 0, 1, 'first');        % first level where the wind-induced flow reverses
if isempty(k_end), k_end = numel(dUs) + 1; end
if k_end == 1                                  % surface flow already opposite to Ekman
    QEk = 0;  delta = 0;
    return;
end
QEk   = sum(dUs(1:k_end-1) .* dzs(1:k_end-1));
delta = sum(dzs(1:k_end-1));
end

function [Q, P, N] = read_run(cluster_dir, sw, x_target_km, y_lim, z_top_m, g, ea, eb, x_amb)
% Q: offshore transport as in plot_full_scatter_sweeps.m panel (e)
% P: alongshore-integrated cross-shelf velocity profile U(z) = int_y u dy at x
% N: background stratification of the ambient offshore column (linear EOS)
Q = NaN;  P = [];  N = NaN;
sw_core = strrep(sw, 'sweep_', '');
d = dir(fullfile(cluster_dir, ['*' sw_core '*']));
d = d([d.isdir]);
d = d(~ismember({d.name}, {'.', '..'}));
if isempty(d)
    warning('Folder not found for %s', sw);
    return;
end
datadir = fullfile(cluster_dir, d(1).name);
fname = '';
for c = {['time_avg_3d_sweep_' sw_core '.nc'], ['time_avg_3d_' sw_core '.nc']}
    if isfile(fullfile(datadir, c{1})), fname = fullfile(datadir, c{1}); break; end
end
if isempty(fname)
    warning('No time_avg_3d file for %s in %s', sw, datadir);
    return;
end
fprintf('  %s -> %s\n', sw, fname);

try
    xc = try_ncread_(fname, {'xC','xc','x','x_caa'});  xc = xc(:);
    yc = try_ncread_(fname, {'yC','yc','y','y_aca'});  yc = yc(:);
    zc = try_ncread_(fname, {'zC','zc','z','z_aac'});  zc = zc(:);
    dz = cell_width(fname, zc, {'zF','z_aaf','zf'});
    dy = cell_width(fname, yc, {'yF','y_afa','yf'});
    [~, ix] = min(abs(xc / 1e3 - x_target_km));
    iy = find(yc / 1e3 >= y_lim(1) & yc / 1e3 <= y_lim(2));

    vinfo = ncinfo(fname, 'u_c');  vsize = vinfo.Size;  t = vsize(end);
    u = squeeze(ncread(fname, 'u_c', [ix 1 1 t], [1 vsize(2) vsize(3) 1]));    % (Ny, Nz)

    % (a) offshore Q exactly as in panel (e): u > 0, top z_top_m, uniform dz, trapz in y
    up = u;  up(up < 0) = 0;
    iz = find(zc >= -z_top_m);
    u_int = sum(up(:, iz), 2, 'omitnan') * abs(zc(1) - zc(2));
    Q = trapz(yc(iy), u_int(iy));

    % Alongshore-integrated profile for the Ekman diagnostic
    uu = u(iy, :);  uu(~isfinite(uu)) = 0;
    P.U  = (dy(iy)' * uu)';                    % (Nz x 1) [m^2/s]
    P.zc = zc;  P.dz = dz;

    % Background N from the ambient offshore column (linear EOS, bulk)
    ixa = find(xc / 1e3 >= max(xc / 1e3) - x_amb);
    Tc = ncread(fname, 'T', [ixa(1) iy(1) 1 t], [numel(ixa) numel(iy) Inf 1]);
    Sc = ncread(fname, 'S', [ixa(1) iy(1) 1 t], [numel(ixa) numel(iy) Inf 1]);
    b  = g * (ea * Tc - eb * Sc);  b(Tc == 0 | ~isfinite(Tc)) = NaN;
    bp = squeeze(median(median(b, 1, 'omitnan'), 2, 'omitnan'));
    ok = isfinite(bp);
    if sum(ok) >= 2
        [z_hi, i_hi] = max(zc(ok));  [z_lo, i_lo] = min(zc(ok));  bz = bp(ok);
        N2 = (bz(i_hi) - bz(i_lo)) / (z_hi - z_lo);
        if N2 > 0, N = sqrt(N2); end
    end
catch ME
    warning('Error processing %s: %s', sw, ME.message);
end
end

function d = cell_width(fname, c, face_names)
% Cell widths from face coordinates if present, else from center midpoints
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

function bnds = get_padded_lims(vals)
v_min = min(vals(:), [], 'omitnan');
v_max = max(vals(:), [], 'omitnan');
if isempty(v_min) || isnan(v_min)
    v_min = 0; v_max = 1;
end
dv = (v_max - v_min) * 0.1;
if dv == 0
    dv = max(1e-10, abs(v_max) * 0.05);
end
bnds = [v_min - dv, v_max + dv];
end
