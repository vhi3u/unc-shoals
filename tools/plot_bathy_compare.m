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
% Geometry comes from tools/dshoal_bottom.m and is checked against the
% Julia by tools/dshoal_selftest.m on every run. See BATHY_EXPLORER for
% the edit-and-rerun version of this figure.

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

dshoal_selftest();
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
    % One background per Zs, NOT one for the row. Zc is a single global
    % number: if Zc = Zs then the coast moves everywhere, including far from
    % the shoal, so the reference transect moves with it. Drawing a fixed
    % Zc = -5 background beside coast-tracking shoals implies a domain that
    % cannot exist, and hides that this change alters the control state.
    for k = 1:numel(o.Zs_list)
        Zs = o.Zs_list(k);
        zbg = shoal_bottom(x_vec, y_bg, Zs, zc(Zs), Ly, ...
                           o.shoal_length, o.sigma, o.Zsh, SBE);
        if r == 1 && k > 1, continue; end   % all identical when Zc is fixed
        if r == 1, nm = sprintf('Background (y = %.0f km)', y_bg/1e3);
        else,      nm = sprintf('Background, Z_c = %g m', Zs); end
        plot(ax2, x_vec/1e3, zbg, '--', 'LineWidth', 2.0, ...
             'Color', min(1, [0.85 0.15 0.15] + 0.10*(k-1)), 'DisplayName', nm);
    end
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
% Thin wrappers onto tools/dshoal_bottom.m, so the bathymetry exists in
% exactly one MATLAB file and dshoal_selftest guards exactly one copy.
% ═══════════════════════════════════════════════════════════════════════

function z = shoal_bottom(x, y, Zs, Zc, Ly, shoal_length, sigma, Zsh, sbe)
z = dshoal_bottom(x, y, 'Ly', Ly, 'Zs', Zs, 'Zc', Zc, 'Ls', shoal_length, ...
                  'sigma', sigma, 'Zsh', Zsh, 'shelf_break_end', sbe);
end

function Z = shoal_bottom_grid(x_vec, y_vec, Zs, Zc, Ly, shoal_length, sigma, Zsh, sbe)
[XX, YY] = meshgrid(x_vec, y_vec);
Z = dshoal_bottom(XX, YY, 'Ly', Ly, 'Zs', Zs, 'Zc', Zc, 'Ls', shoal_length, ...
                  'sigma', sigma, 'Zsh', Zsh, 'shelf_break_end', sbe);
end
