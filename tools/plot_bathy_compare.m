function plot_bathy_compare(varargin)
% PLOT_BATHY_COMPARE  Fig. 1 bathymetry panel: current vs coast-tracking.
%
%   plot_bathy_compare()
%   plot_bathy_compare('Zs_3d', -15, 'outdir', '~/Desktop')
%   plot_bathy_compare('selftest_only', true)
%
% Recreates the layout of fig3 in MATLAB/shoal_stuff/plot_shoals.m -- 3D
% surface left, X-Z transect right, blue shoal / red background -- and stacks
% the two geometries so they read off against each other:
%
%   row 1  CURRENT        Zc fixed at -5 m
%   row 2  COAST TRACKS   Zc = Zs
%
% Options (name/value):
%   'Zs_list'       crest depths for the X-Z panels   default [-5 -10 -15]
%   'Zs_3d'         crest depth for the 3D panels     default -15
%   'shoal_length'  L_S [m]                           default 40e3
%   'sigma'         shoal half-width in y [m]         default 8e3
%   'Zsh'           shelf depth [m]                   default -25
%   'outdir'        where the PNG goes            default <repo>/figures
%   'selftest_only' run the check and return          default false
%
% ── WHY THIS FILE IS DANGEROUS, AND WHAT GUARDS IT ──────────────────────
% The bathymetry below is a hand translation of src/dshoal_vn_param_coastal.jl.
% A second copy of a function the model actually runs is exactly how
% MATLAB/shoal_stuff/plot_shoals.m ended up drawing a 100x300 km domain with
% the retired Hs/shelf_length parameterization long after the model had moved
% to Zs/Zc on 150x200 km -- the figure and the simulation diverged silently,
% and nothing complained.
%
% So this file ships with 180 reference values generated from the Julia
% function itself, and checks against them on every run. If someone edits the
% Julia and not this, you get an error naming the worst offender instead of a
% plausible-looking wrong figure. Regenerate the table after an INTENDED
% change to the Julia (the command is in the REF block below).

p = inputParser;
addParameter(p, 'Zs_list', [-5 -10 -15], @isnumeric);
addParameter(p, 'Zs_3d', -15, @isscalar);
addParameter(p, 'shoal_length', 40e3, @isscalar);
addParameter(p, 'sigma', 8e3, @isscalar);
addParameter(p, 'Zsh', -25.0, @isscalar);
addParameter(p, 'outdir', '', @ischar);
addParameter(p, 'selftest_only', false, @islogical);
parse(p, varargin{:});
o = p.Results;

% ── Domain, matching flow_over_shoals_sweep.jl ──────────────────────────
Lx = 150e3; Ly = 200e3; Lz = 50.0;
SBE = 12.0e3; ZC_FIXED = -5.0;

selftest(Ly);
if o.selftest_only, return; end

nx = 301; ny = 401;
x_vec = linspace(0, Lx, nx);
y_vec = linspace(0, Ly, ny);
y_shoal = Ly / 2;            % shoal centreline
y_bg    = Ly / 2 - 60e3;     % 7.5 sigma away: background only

cols = [0.12 0.47 0.87; 0.18 0.55 0.34; 0.52 0.40 0.75];
rows = {'CURRENT  —  Z_c fixed at -5 m', 'COAST TRACKS  —  Z_c = Z_S'};

fig = figure('Color', 'w', 'Position', [60 60 1700 1000]);
tl = tiledlayout(fig, 2, 9, 'TileSpacing', 'compact', 'Padding', 'compact');

for r = 1:2
    % Zc rule for this row
    if r == 1, zc = @(Zs) ZC_FIXED; else, zc = @(Zs) Zs; end

    % ── Left tile: 3D surface (5 of 9) ──────────────────────────────────
    ax = nexttile(tl, [1 5]);
    Z = shoal_bottom_grid(x_vec, y_vec, o.Zs_3d, zc(o.Zs_3d), Ly, ...
                          o.shoal_length, o.sigma, o.Zsh, SBE);
    surf(ax, x_vec/1e3, y_vec/1e3, Z, 'EdgeColor', 'none');
    try
        colormap(ax, cmocean('deep'));
    catch
        colormap(ax, sky(256));
    end
    clim(ax, [-Lz 0]);
    xlabel(ax, 'x (km)'); ylabel(ax, 'y (km)'); zlabel(ax, 'z (m)');
    title(ax, sprintf('%s\n3D  (Z_S = %g m)', rows{r}, -o.Zs_3d), ...
          'FontWeight', 'bold', 'FontSize', 11);
    view(ax, 45, 30); zlim(ax, [-Lz 0]); pbaspect(ax, [1.5 2 1]);
    set(ax, 'FontSize', 10); hold(ax, 'on');
    zs_line = shoal_bottom(x_vec, y_shoal, o.Zs_3d, zc(o.Zs_3d), Ly, ...
                           o.shoal_length, o.sigma, o.Zsh, SBE);
    zb_line = shoal_bottom(x_vec, y_bg, o.Zs_3d, zc(o.Zs_3d), Ly, ...
                           o.shoal_length, o.sigma, o.Zsh, SBE);
    plot3(ax, x_vec/1e3, repmat(y_shoal/1e3, 1, nx), zs_line, '-', ...
          'LineWidth', 2.5, 'Color', [0.15 0.45 0.85]);
    plot3(ax, x_vec/1e3, repmat(y_bg/1e3, 1, nx), zb_line, '-', ...
          'LineWidth', 2.5, 'Color', [0.85 0.15 0.15]);

    % ── Right tile: X-Z transects (4 of 9) ──────────────────────────────
    ax2 = nexttile(tl, [1 4]);
    hold(ax2, 'on');
    zbg = shoal_bottom(x_vec, y_bg, o.Zs_list(1), ZC_FIXED, Ly, ...
                       o.shoal_length, o.sigma, o.Zsh, SBE);
    plot(ax2, x_vec/1e3, zbg, 'LineWidth', 2.5, 'Color', [0.85 0.15 0.15], ...
         'DisplayName', sprintf('Background (y = %.0f km)', y_bg/1e3));
    for k = 1:numel(o.Zs_list)
        Zs = o.Zs_list(k);
        z = shoal_bottom(x_vec, y_shoal, Zs, zc(Zs), Ly, ...
                         o.shoal_length, o.sigma, o.Zsh, SBE);
        plot(ax2, x_vec/1e3, z, 'LineWidth', 2.5, 'Color', cols(k,:), ...
             'DisplayName', sprintf('Z_S = %g m', -Zs));
        % flat-crest extent: the L_S the model actually gets, not the input
        on = abs(z - Zs) < 0.05;
        if any(on)
            xi = x_vec(find(on, 1, 'first'))/1e3;
            xo = x_vec(find(on, 1, 'last'))/1e3;
            plot(ax2, [xi xo], [Zs+1.3 Zs+1.3], '-', 'LineWidth', 1.3, ...
                 'Color', cols(k,:), 'HandleVisibility', 'off');
            text(ax2, (xi+xo)/2, Zs+3.0, sprintf('L_S = %.1f km', xo-xi), ...
                 'Color', cols(k,:), 'FontSize', 9, 'HorizontalAlignment', 'center');
        end
    end
    xline(ax2, 63.5, ':', 'Color', [0.6 0.6 0.6], 'HandleVisibility', 'off');
    yline(ax2, 0, '--', 'Color', [0.4 0.4 0.4], 'HandleVisibility', 'off');
    xlabel(ax2, 'x (km)'); ylabel(ax2, 'z (m)');
    title(ax2, sprintf('%s\nX-Z transects', rows{r}), ...
          'FontWeight', 'bold', 'FontSize', 11);
    xlim(ax2, [0 100]); ylim(ax2, [-Lz 1]); grid(ax2, 'on');
    legend(ax2, 'Location', 'southwest', 'FontSize', 8);
    set(ax2, 'FontSize', 10);
end

title(tl, sprintf('Shoal bathymetry — L_S = %g km, \\sigma_S = %g km', ...
      o.shoal_length/1e3, o.sigma/1e3), 'FontSize', 12, 'FontWeight', 'bold');

% ── Numbers behind the labels ───────────────────────────────────────────
fprintf('\n  %5s %-9s %9s %10s %11s %12s\n', ...
        'Zs', 'variant', 'crest_in', 'crest_out', 'L_S actual', 'max|dh/dx|');
for k = 1:numel(o.Zs_list)
    Zs = o.Zs_list(k);
    for r = 1:2
        if r == 1, Zc = ZC_FIXED; nm = 'current'; else, Zc = Zs; nm = 'tracks'; end
        z = shoal_bottom(x_vec, y_shoal, Zs, Zc, Ly, o.shoal_length, ...
                         o.sigma, o.Zsh, SBE);
        on = abs(z - Zs) < 0.05;
        xi = x_vec(find(on, 1, 'first'))/1e3;
        xo = x_vec(find(on, 1, 'last'))/1e3;
        xin = 0:125:20e3;
        zin = shoal_bottom(xin, y_shoal, Zs, Zc, Ly, o.shoal_length, ...
                           o.sigma, o.Zsh, SBE);
        fprintf('  %5g %-9s %9.1f %10.1f %11.1f %12.5f\n', ...
                Zs, nm, xi, xo, xo-xi, max(abs(diff(zin))/125));
    end
end

% ── Save ────────────────────────────────────────────────────────────────
outdir = o.outdir;
if isempty(outdir)
    outdir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'figures');
end
if ~exist(outdir, 'dir'), mkdir(outdir); end
out = fullfile(outdir, 'bathy_current_vs_coast_tracking.png');
exportgraphics(fig, out, 'Resolution', 160);
fprintf('\nSaved: %s\n', out);
end

% ═══════════════════════════════════════════════════════════════════════
% Bathymetry — hand translation of src/dshoal_vn_param_coastal.jl.
% Keep these in step. The self-test below is what tells you if they are not.
% ═══════════════════════════════════════════════════════════════════════

function s = smooth_step_c(x, x0, w)
s = 0.5 * (1.0 + tanh((x - x0) / w));
end

function h = param_background_depth_c(x, Zsh, shelf_break_end, Zc, ...
                                      coastal_drop, coastal_mid, coastal_width)
h_deep = -50.0;
h0 = Zc;
h1 = Zc - coastal_drop;
h = h0 + (h1 - h0) * smooth_step_c(x, coastal_mid, coastal_width);
sb_mid   = 5.0e3 + (shelf_break_end - 5.0e3) / 2.0;
sb_width = (shelf_break_end - 5.0e3) / 2.8;
h = h + (Zsh - h1)     * smooth_step_c(x, sb_mid, sb_width);
h = h + (h_deep - Zsh) * smooth_step_c(x, 63.5e3, 5.0e3);
end

function w = param_shoal_window_c(y, y0, half_extent)
w = ones(size(y));
if half_extent >= y0, return; end
dy = abs(y - y0);
taper_start = 0.5 * half_extent;
w(dy >= half_extent) = 0.0;
m = (dy > taper_start) & (dy < half_extent);
w(m) = 0.5 * (1.0 + cos(pi * (dy(m) - taper_start) / (half_extent - taper_start)));
end

function z = shoal_bottom(x, y, Zs, Zc, Ly, shoal_length, sigma, Zsh, shelf_break_end)
% Vectorised over x; y may be scalar or the same size as x.
coastal_drop = 2.0; coastal_mid = 2.5e3; coastal_width = 1.5e3;
y0 = Ly / 2.0;
half_extent = Ly / 2.0;                 % Ly_shoal defaults to Ly
min_depth = max(Zc, Zs);                % shallowest feature requested

if isscalar(y), y = repmat(y, size(x)); end

hw = param_background_depth_c(x, Zsh, shelf_break_end, Zc, ...
                              coastal_drop, coastal_mid, coastal_width);
window = param_shoal_window_c(y, y0, half_extent);

x_taper_start = shoal_length;
x_taper_end   = shoal_length + 15.0e3;
taper = ones(size(x));
taper(x >= x_taper_end) = 0.0;
m = (x > x_taper_start) & (x < x_taper_end);
taper(m) = 0.5 * (1.0 + cos(pi * (x(m) - x_taper_start) / (x_taper_end - x_taper_start)));

gauss_y = exp(-((y - y0).^2) ./ (2 * sigma^2));
factor  = taper .* gauss_y .* window;
potential_height = max(0.0, Zs - hw);
z = min(min_depth, hw + potential_height .* factor);
end

function Z = shoal_bottom_grid(x_vec, y_vec, Zs, Zc, Ly, shoal_length, sigma, Zsh, sbe)
Z = zeros(numel(y_vec), numel(x_vec));
for j = 1:numel(y_vec)
    Z(j,:) = shoal_bottom(x_vec, y_vec(j), Zs, Zc, Ly, shoal_length, sigma, Zsh, sbe);
end
end

% ═══════════════════════════════════════════════════════════════════════
% Self-test against the Julia source.
%
% Regenerate REF after an INTENDED change to dshoal_vn_param_coastal.jl:
%
%   julia --project=. tools/plot_bathy_compare.jl --emit-ref
%
% and paste the output over reference_table() below. That script calls the
% Julia function directly, so it is the authority if the two ever disagree.
% ═══════════════════════════════════════════════════════════════════════

function selftest(Ly)
REF = reference_table();
worst = 0; wrow = [];
for i = 1:size(REF, 1)
    x = REF(i,1)*1e3; y = REF(i,2)*1e3; Zs = REF(i,3); Zc = REF(i,4);
    got = shoal_bottom(x, y, Zs, Zc, Ly, 40e3, 8e3, -25.0, 12.0e3);
    d = abs(got - REF(i,5));
    if d > worst, worst = d; wrow = [REF(i,:) got]; end
end
if worst > 1e-9
    error('plot_bathy_compare:drift', ...
        ['Bathymetry does not match the Julia source (max error %.3g m).\n' ...
         '  worst: x = %g km, y = %g km, Zs = %g, Zc = %g\n' ...
         '         julia = %.10f    matlab = %.10f\n' ...
         'This file and src/dshoal_vn_param_coastal.jl have diverged. Fix the\n' ...
         'translation, or regenerate REF if the Julia change was intended.'], ...
        worst, wrow(1), wrow(2), wrow(3), wrow(4), wrow(5), wrow(6));
end
fprintf('selftest: %d reference points match the Julia source (max err %.2g m)\n', ...
        size(REF,1), worst);
end

function REF = reference_table()
% Columns: x [km], y [km], Zs [m], Zc [m], bottom depth [m].
% Generated by: julia --project=. tools/plot_bathy_compare.jl --emit-ref
% Do not hand-edit.
REF = [
         0.0    100.0    -5.0    -5.0               -5.0;
         2.0    100.0    -5.0    -5.0               -5.0;
         5.0    100.0    -5.0    -5.0               -5.0;
         7.8    100.0    -5.0    -5.0               -5.0;
        10.0    100.0    -5.0    -5.0               -5.0;
        20.0    100.0    -5.0    -5.0               -5.0;
        40.0    100.0    -5.0    -5.0               -5.0;
        50.0    100.0    -5.0    -5.0     -20.0843051218;
        63.5    100.0    -5.0    -5.0              -37.5;
       100.0    100.0    -5.0    -5.0     -49.9999885912;
         0.0     92.0    -5.0    -5.0      -5.0349857357;
         2.0     92.0    -5.0    -5.0      -5.3058203631;
         5.0     92.0    -5.0    -5.0      -6.1658279265;
         7.8     92.0    -5.0    -5.0       -8.361073964;
        10.0     92.0    -5.0    -5.0     -11.2299398723;
        20.0     92.0    -5.0    -5.0     -12.8686715446;
        40.0     92.0    -5.0    -5.0     -12.8702004729;
        50.0     92.0    -5.0    -5.0     -22.0627089834;
        63.5     92.0    -5.0    -5.0              -37.5;
       100.0     92.0    -5.0    -5.0     -49.9999885912;
         0.0     40.0    -5.0    -5.0      -5.0889160402;
         2.0     40.0    -5.0    -5.0      -5.7772406432;
         5.0     40.0    -5.0    -5.0      -7.9629447766;
         7.8     40.0    -5.0    -5.0     -13.5421495905;
        10.0     40.0    -5.0    -5.0     -20.8333553199;
        20.0     40.0    -5.0    -5.0     -24.9981821681;
        40.0     40.0    -5.0    -5.0     -25.0020679304;
        50.0     40.0    -5.0    -5.0      -25.112406829;
        63.5     40.0    -5.0    -5.0              -37.5;
       100.0     40.0    -5.0    -5.0     -49.9999885912;
         0.0    100.0    -5.0    -5.0               -5.0;
         2.0    100.0    -5.0    -5.0               -5.0;
         5.0    100.0    -5.0    -5.0               -5.0;
         7.8    100.0    -5.0    -5.0               -5.0;
        10.0    100.0    -5.0    -5.0               -5.0;
        20.0    100.0    -5.0    -5.0               -5.0;
        40.0    100.0    -5.0    -5.0               -5.0;
        50.0    100.0    -5.0    -5.0     -20.0843051218;
        63.5    100.0    -5.0    -5.0              -37.5;
       100.0    100.0    -5.0    -5.0     -49.9999885912;
         0.0     92.0    -5.0    -5.0      -5.0349857357;
         2.0     92.0    -5.0    -5.0      -5.3058203631;
         5.0     92.0    -5.0    -5.0      -6.1658279265;
         7.8     92.0    -5.0    -5.0       -8.361073964;
        10.0     92.0    -5.0    -5.0     -11.2299398723;
        20.0     92.0    -5.0    -5.0     -12.8686715446;
        40.0     92.0    -5.0    -5.0     -12.8702004729;
        50.0     92.0    -5.0    -5.0     -22.0627089834;
        63.5     92.0    -5.0    -5.0              -37.5;
       100.0     92.0    -5.0    -5.0     -49.9999885912;
         0.0     40.0    -5.0    -5.0      -5.0889160402;
         2.0     40.0    -5.0    -5.0      -5.7772406432;
         5.0     40.0    -5.0    -5.0      -7.9629447766;
         7.8     40.0    -5.0    -5.0     -13.5421495905;
        10.0     40.0    -5.0    -5.0     -20.8333553199;
        20.0     40.0    -5.0    -5.0     -24.9981821681;
        40.0     40.0    -5.0    -5.0     -25.0020679304;
        50.0     40.0    -5.0    -5.0      -25.112406829;
        63.5     40.0    -5.0    -5.0              -37.5;
       100.0     40.0    -5.0    -5.0     -49.9999885912;
         0.0    100.0   -10.0    -5.0      -5.0889160402;
         2.0    100.0   -10.0    -5.0      -5.7772406432;
         5.0    100.0   -10.0    -5.0      -7.9629447766;
         7.8    100.0   -10.0    -5.0              -10.0;
        10.0    100.0   -10.0    -5.0              -10.0;
        20.0    100.0   -10.0    -5.0              -10.0;
        40.0    100.0   -10.0    -5.0              -10.0;
        50.0    100.0   -10.0    -5.0     -21.3343051218;
        63.5    100.0   -10.0    -5.0              -37.5;
       100.0    100.0   -10.0    -5.0     -49.9999885912;
         0.0     92.0   -10.0    -5.0      -5.0889160402;
         2.0     92.0   -10.0    -5.0      -5.7772406432;
         5.0     92.0   -10.0    -5.0      -7.9629447766;
         7.8     92.0   -10.0    -5.0     -11.3937272626;
        10.0     92.0   -10.0    -5.0     -14.2625931708;
        20.0     92.0   -10.0    -5.0     -15.9013248432;
        40.0     92.0   -10.0    -5.0     -15.9028537715;
        50.0     92.0   -10.0    -5.0     -22.8208723081;
        63.5     92.0   -10.0    -5.0              -37.5;
       100.0     92.0   -10.0    -5.0     -49.9999885912;
         0.0     40.0   -10.0    -5.0      -5.0889160402;
         2.0     40.0   -10.0    -5.0      -5.7772406432;
         5.0     40.0   -10.0    -5.0      -7.9629447766;
         7.8     40.0   -10.0    -5.0     -13.5421495905;
        10.0     40.0   -10.0    -5.0     -20.8333553199;
        20.0     40.0   -10.0    -5.0     -24.9981821681;
        40.0     40.0   -10.0    -5.0     -25.0020679304;
        50.0     40.0   -10.0    -5.0      -25.112406829;
        63.5     40.0   -10.0    -5.0              -37.5;
       100.0     40.0   -10.0    -5.0     -49.9999885912;
         0.0    100.0   -10.0   -10.0              -10.0;
         2.0    100.0   -10.0   -10.0              -10.0;
         5.0    100.0   -10.0   -10.0              -10.0;
         7.8    100.0   -10.0   -10.0              -10.0;
        10.0    100.0   -10.0   -10.0              -10.0;
        20.0    100.0   -10.0   -10.0              -10.0;
        40.0    100.0   -10.0   -10.0              -10.0;
        50.0    100.0   -10.0   -10.0     -21.3343051218;
        63.5    100.0   -10.0   -10.0              -37.5;
       100.0    100.0   -10.0   -10.0     -49.9999885912;
         0.0     92.0   -10.0   -10.0     -10.0327969916;
         2.0     92.0   -10.0   -10.0     -10.2950269111;
         5.0     92.0   -10.0   -10.0     -11.0530513982;
         7.8     92.0   -10.0   -10.0     -12.6458500683;
        10.0     92.0   -10.0   -10.0     -14.7179851745;
        20.0     92.0   -10.0   -10.0     -15.9015236027;
        40.0     92.0   -10.0   -10.0     -15.9028537715;
        50.0     92.0   -10.0   -10.0     -22.8208723081;
        63.5     92.0   -10.0   -10.0              -37.5;
       100.0     92.0   -10.0   -10.0     -49.9999885912;
         0.0     40.0   -10.0   -10.0       -10.08335336;
         2.0     40.0   -10.0   -10.0     -10.7498091487;
         5.0     40.0   -10.0   -10.0     -12.6763238971;
         7.8     40.0   -10.0   -10.0     -16.7244122919;
        10.0     40.0   -10.0   -10.0     -21.9907314024;
        20.0     40.0   -10.0   -10.0     -24.9986873141;
        40.0     40.0   -10.0   -10.0     -25.0020679304;
        50.0     40.0   -10.0   -10.0      -25.112406829;
        63.5     40.0   -10.0   -10.0              -37.5;
       100.0     40.0   -10.0   -10.0     -49.9999885912;
         0.0    100.0   -15.0    -5.0      -5.0889160402;
         2.0    100.0   -15.0    -5.0      -5.7772406432;
         5.0    100.0   -15.0    -5.0      -7.9629447766;
         7.8    100.0   -15.0    -5.0     -13.5421495905;
        10.0    100.0   -15.0    -5.0              -15.0;
        20.0    100.0   -15.0    -5.0              -15.0;
        40.0    100.0   -15.0    -5.0              -15.0;
        50.0    100.0   -15.0    -5.0     -22.5843051218;
        63.5    100.0   -15.0    -5.0              -37.5;
       100.0    100.0   -15.0    -5.0     -49.9999885912;
         0.0     92.0   -15.0    -5.0      -5.0889160402;
         2.0     92.0   -15.0    -5.0      -5.7772406432;
         5.0     92.0   -15.0    -5.0      -7.9629447766;
         7.8     92.0   -15.0    -5.0     -13.5421495905;
        10.0     92.0   -15.0    -5.0     -17.2952464694;
        20.0     92.0   -15.0    -5.0     -18.9339781418;
        40.0     92.0   -15.0    -5.0     -18.9355070701;
        50.0     92.0   -15.0    -5.0     -23.5790356327;
        63.5     92.0   -15.0    -5.0              -37.5;
       100.0     92.0   -15.0    -5.0     -49.9999885912;
         0.0     40.0   -15.0    -5.0      -5.0889160402;
         2.0     40.0   -15.0    -5.0      -5.7772406432;
         5.0     40.0   -15.0    -5.0      -7.9629447766;
         7.8     40.0   -15.0    -5.0     -13.5421495905;
        10.0     40.0   -15.0    -5.0     -20.8333553199;
        20.0     40.0   -15.0    -5.0     -24.9981821681;
        40.0     40.0   -15.0    -5.0     -25.0020679304;
        50.0     40.0   -15.0    -5.0      -25.112406829;
        63.5     40.0   -15.0    -5.0              -37.5;
       100.0     40.0   -15.0    -5.0     -49.9999885912;
         0.0    100.0   -15.0   -15.0              -15.0;
         2.0    100.0   -15.0   -15.0              -15.0;
         5.0    100.0   -15.0   -15.0              -15.0;
         7.8    100.0   -15.0   -15.0              -15.0;
        10.0    100.0   -15.0   -15.0              -15.0;
        20.0    100.0   -15.0   -15.0              -15.0;
        40.0    100.0   -15.0   -15.0              -15.0;
        50.0    100.0   -15.0   -15.0     -22.5843051218;
        63.5    100.0   -15.0   -15.0              -37.5;
       100.0    100.0   -15.0   -15.0     -49.9999885912;
         0.0     92.0   -15.0   -15.0     -15.0306082475;
         2.0     92.0   -15.0   -15.0      -15.284233459;
         5.0     92.0   -15.0   -15.0     -15.9402748698;
         7.8     92.0   -15.0   -15.0     -16.9306261726;
        10.0     92.0   -15.0   -15.0     -18.2060304767;
        20.0     92.0   -15.0   -15.0     -18.9343756607;
        40.0     92.0   -15.0   -15.0     -18.9355070701;
        50.0     92.0   -15.0   -15.0     -23.5790356327;
        63.5     92.0   -15.0   -15.0              -37.5;
       100.0     92.0   -15.0   -15.0     -49.9999885912;
         0.0     40.0   -15.0   -15.0     -15.0777906798;
         2.0     40.0   -15.0   -15.0     -15.7223776542;
         5.0     40.0   -15.0   -15.0     -17.3897030176;
         7.8     40.0   -15.0   -15.0     -19.9066749933;
        10.0     40.0   -15.0   -15.0      -23.148107485;
        20.0     40.0   -15.0   -15.0     -24.9991924601;
        40.0     40.0   -15.0   -15.0     -25.0020679305;
        50.0     40.0   -15.0   -15.0      -25.112406829;
        63.5     40.0   -15.0   -15.0              -37.5;
       100.0     40.0   -15.0   -15.0     -49.9999885912;
];
end
