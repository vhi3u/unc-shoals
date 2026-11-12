%% ═══ BATHYMETRY EXPLORER ═══  edit the numbers, then Cmd+Enter this section
% Draws the shoal-shelf bathymetry for one parameter set. Everything this
% section needs is in this section, so Run Section (Cmd+Enter) redraws it.
%
% Geometry comes from tools/dshoal_bottom.m, a MATLAB twin of the Julia the
% model runs (src/dshoal_vn_param.jl). dshoal_selftest checks it against 180
% values emitted from that Julia, so a figure here cannot quietly disagree
% with a simulation -- see tools/dshoal_selftest.m for how to regenerate.
%
% Reference case: Zs = -5, Ls = 40 km, sigma = 8 km, Zsh = -25.

Zs    = -5.0;      % shoal crest depth [m]              (deeper = lower shoal)
Ls    = 40e3;      % shoal length before the taper [m]
sigma = 8e3;       % shoal half-width in y, Gaussian [m]
Zsh   = -25.0;     % shelf depth [m]
sbe   = 12e3;      % offshore end of the shelf break [m]  (inshore end of shelf)
rise  = 63.5e3;    % centre of the offshore ramp [m]      (offshore end of shelf)
riseW = 5.0e3;     % width of the offshore ramp [m]
                   %   SHELF LENGTH is rise - sbe. Vary `rise` to change it
                   %   while leaving Zsh alone; vary Zsh to change the depth
                   %   while leaving the length alone. Both Julia files
                   %   hardcode rise/riseW, so `rise` is exploratory until
                   %   it is exposed there -- same caveat as Zc below.

Zc    = -5.0;      % coast depth [m] -- set to Zs for "coast tracks the shoal"
                   %   NOTE the model's Julia hardcodes this to -5 and takes no
                   %   Zc argument, so anything else is exploratory only.


% ── Holding the shelf break fixed while varying Zsh (optional) ──────────
% Zsh alone preserves shelf length but NOT the break slope: the drop from
% -7 m to Zsh happens over the same 5->sbe km, so the break steepens in
% proportion. The tanh's peak slope is exactly 1.4*drop/(sbe - 5 km), so
% uncommenting these two lines pins it and keeps the shelf roughly put:
%
%   sbe  = 5e3 + 1.4*abs(Zsh - (Zc - 2))/0.0036;   % 0.0036 = the Zsh=-25 value
%   rise = 63.5e3 + (sbe - 12e3);
%
% Verified: break |dh/dx| stays at 0.00360 for Zsh = -20..-35, and the flat
% shelf holds to 33.0/32.1/31.2/30.4 km (8%) instead of drifting 32%. The
% residual is the offshore ramp, whose own drop |h_deep - Zsh| changes too;
% scale rise_width by that as well if you need it exact.

Lx = 150e3; Ly = 200e3; Lz = 50.0;     % domain, matching the sweep
y_shoal = Ly/2;                        % shoal centreline
y_bg    = Ly/2 - 60e3;                 % 7.5 sigma away: background only

% ── path (works whether the current folder is the repo root or tools/) ──
if exist('dshoal_bottom', 'file') ~= 2
    here = fileparts(mfilename('fullpath'));          % empty when Run Section
    if isempty(here), here = pwd; end
    addpath(here); addpath(fullfile(here, 'tools'));
    if exist('dshoal_bottom', 'file') ~= 2
        error(['dshoal_bottom.m not found. cd to the repo root (or tools/) ' ...
               'and run: addpath(''tools'')']);
    end
end
dshoal_selftest();

opts = {'Ly', Ly, 'Zs', Zs, 'Ls', Ls, 'sigma', sigma, 'Zsh', Zsh, ...
        'shelf_break_end', sbe, 'Zc', Zc, 'rise_mid', rise, 'rise_width', riseW};

nx = 401; ny = 501;
xv = linspace(0, Lx, nx);
yv = linspace(0, Ly, ny);
[XX, YY] = meshgrid(xv, yv);
Z  = dshoal_bottom(XX, YY, opts{:});
z_shoal = dshoal_bottom(xv, y_shoal, opts{:});
z_bg    = dshoal_bottom(xv, y_bg,    opts{:});

fig = figure('Color', 'w', 'Position', [60 60 1500 950]);
tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% ── (a) XY map ──────────────────────────────────────────────────────────
ax1 = nexttile(tl);
imagesc(ax1, xv/1e3, yv/1e3, Z); set(ax1, 'YDir', 'normal');
axis(ax1, 'equal', 'tight');
try, colormap(ax1, cmocean('deep')); catch, colormap(ax1, sky(256)); end
clim(ax1, [-Lz 0]); cb = colorbar(ax1); cb.Label.String = 'z (m)';
hold(ax1, 'on');
contour(ax1, xv/1e3, yv/1e3, Z, -45:5:-5, 'Color', [0.25 0.25 0.25], 'LineWidth', 0.6);
plot(ax1, [0 Lx/1e3], [y_shoal y_shoal]/1e3, '-', 'LineWidth', 2, 'Color', [0.15 0.45 0.85]);
plot(ax1, [0 Lx/1e3], [y_bg y_bg]/1e3,       '-', 'LineWidth', 2, 'Color', [0.85 0.15 0.15]);
xlabel(ax1, 'x (km)'); ylabel(ax1, 'y (km)');
title(ax1, 'XY map', 'FontWeight', 'bold'); set(ax1, 'FontSize', 11);

% ── (b) 3D surface ──────────────────────────────────────────────────────
ax2 = nexttile(tl);
surf(ax2, xv/1e3, yv/1e3, Z, 'EdgeColor', 'none');
try, colormap(ax2, cmocean('deep')); catch, colormap(ax2, sky(256)); end
clim(ax2, [-Lz 0]); view(ax2, 45, 30); zlim(ax2, [-Lz 0]); pbaspect(ax2, [1.5 2 1]);
hold(ax2, 'on');
plot3(ax2, xv/1e3, repmat(y_shoal/1e3,1,nx), z_shoal, '-', 'LineWidth', 2.5, 'Color', [0.15 0.45 0.85]);
plot3(ax2, xv/1e3, repmat(y_bg/1e3,1,nx),    z_bg,    '-', 'LineWidth', 2.5, 'Color', [0.85 0.15 0.15]);
xlabel(ax2, 'x (km)'); ylabel(ax2, 'y (km)'); zlabel(ax2, 'z (m)');
title(ax2, '3D bathymetry', 'FontWeight', 'bold'); set(ax2, 'FontSize', 11);

% ── (c) XZ transect, annotated like Fig. 1 ──────────────────────────────
ax3 = nexttile(tl); hold(ax3, 'on');
fill(ax3, [xv fliplr(xv)]/1e3, [z_shoal repmat(-Lz,1,nx)], [0.15 0.45 0.85], ...
     'FaceAlpha', 0.12, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(ax3, xv/1e3, z_shoal, 'LineWidth', 2.5, 'Color', [0.15 0.45 0.85], ...
     'DisplayName', sprintf('Shoal (y = %g km)', y_shoal/1e3));
plot(ax3, xv/1e3, z_bg, 'LineWidth', 2.5, 'Color', [0.85 0.15 0.15], ...
     'DisplayName', sprintf('Background (y = %g km)', y_bg/1e3));
yline(ax3, 0, '--', 'Color', [0.4 0.4 0.4], 'HandleVisibility', 'off');

% Z_S, L_S and Z_Sh, measured off the curve rather than echoed from the input
on = abs(z_shoal - Zs) < 0.05;
if any(on)
    xi = xv(find(on,1,'first'))/1e3;  xo = xv(find(on,1,'last'))/1e3;
    plot(ax3, [xi xo], [Zs Zs]+1.0, 'k-', 'LineWidth', 1.2, 'HandleVisibility','off');
    text(ax3, (xi+xo)/2, Zs+2.6, sprintf('L_S = %.1f km', xo-xi), ...
         'HorizontalAlignment','center', 'FontSize', 10);
    plot(ax3, [xi xi], [0 Zs], 'k-', 'LineWidth', 1.2, 'HandleVisibility','off');
    text(ax3, xi+1.5, Zs/2, sprintf('Z_S = %g m', -Zs), 'FontSize', 10);
end
x_sh = 58;
plot(ax3, [x_sh x_sh], [0 Zsh], 'k-', 'LineWidth', 1.2, 'HandleVisibility','off');
text(ax3, x_sh+1.5, Zsh/2, sprintf('Z_{Sh} = %g m', -Zsh), 'FontSize', 10);
xlabel(ax3, 'x (km)'); ylabel(ax3, 'z (m)');
title(ax3, 'X-Z transect', 'FontWeight', 'bold');
xlim(ax3, [0 100]); ylim(ax3, [-Lz 1]); grid(ax3, 'on');
legend(ax3, 'Location', 'southwest', 'FontSize', 9); set(ax3, 'FontSize', 11);

% ── (d) YZ transects ────────────────────────────────────────────────────
ax4 = nexttile(tl); hold(ax4, 'on');
x_cuts = [2e3 10e3 20e3 30e3 40e3 50e3];
cmap = parula(numel(x_cuts));
for k = 1:numel(x_cuts)
    plot(ax4, yv/1e3, dshoal_bottom(repmat(x_cuts(k),1,ny), yv, opts{:}), ...
         'LineWidth', 2, 'Color', cmap(k,:), ...
         'DisplayName', sprintf('x = %g km', x_cuts(k)/1e3));
end
yline(ax4, 0, '--', 'Color', [0.4 0.4 0.4], 'HandleVisibility', 'off');
xlabel(ax4, 'y (km)'); ylabel(ax4, 'z (m)');
title(ax4, 'Y-Z transects', 'FontWeight', 'bold');
ylim(ax4, [-Lz 1]); grid(ax4, 'on');
legend(ax4, 'Location', 'southeast', 'FontSize', 8); set(ax4, 'FontSize', 11);

title(tl, sprintf(['Z_S = %g m,  L_S = %g km,  \\sigma_S = %g km,  ' ...
      'Z_{Sh} = %g m,  Z_c = %g m'], Zs, Ls/1e3, sigma/1e3, Zsh, Zc), ...
      'FontSize', 13, 'FontWeight', 'bold');

% ── What the geometry actually came out as ──────────────────────────────
dxp = xv(2) - xv(1);
s_in  = max(abs(diff(z_shoal(xv <= 20e3))))  / dxp;
s_out = max(abs(diff(z_shoal(xv > 40e3 & xv <= 58e3)))) / dxp;
fprintf('\n  requested : Zs = %g m, Ls = %g km, Zsh = %g m, Zc = %g m\n', ...
        Zs, Ls/1e3, Zsh, Zc);
if any(on)
    fprintf('  actual    : crest %.1f to %.1f km  ->  L_S = %.1f km', xi, xo, xo-xi);
    if abs((xo-xi)*1e3 - Ls) > 1e3
        fprintf('   <-- NOT the %g km requested', Ls/1e3);
    end
    fprintf('\n');
else
    fprintf('  actual    : no flat crest at Zs (shoal never reaches it)\n');
end
fprintf('  max|dh/dx|: %.5f inshore (0-20 km),  %.5f on the shoal''s offshore flank\n', ...
        s_in, s_out);
fprintf('  water over the crest: %g m\n', -Zs);

% The shelf, measured off the background transect rather than assumed
flat = abs(z_bg - Zsh) < 0.05;
if any(flat)
    si = xv(find(flat,1,'first'))/1e3;  so = xv(find(flat,1,'last'))/1e3;
    fprintf('  shelf     : flat at %g m from %.1f to %.1f km  ->  length %.1f km\n', ...
            Zsh, si, so, so-si);
else
    fprintf('  shelf     : never flat at Zsh (break and ramp overlap)\n');
end
s_brk = max(abs(diff(z_bg(xv <= sbe + 5e3)))) / dxp;
fprintf('  shelf break: max|dh/dx| = %.5f   (drop %g m over %g-%g km)\n\n', ...
        s_brk, abs(Zsh - (Zc - 2)), 5, sbe/1e3);


%% ── Overlay a sweep of one parameter ──  edit, then Cmd+Enter this section
% Reuses the Zs/Ls/... set above for everything not being swept.
SWEEP_NAME = 'Zs';                  % 'Zs' | 'Ls' | 'sigma' | 'Zsh' | 'Zc'
SWEEP_VALS = [-5 -10 -15 -20];

figure('Color','w','Position',[80 80 950 560]); ax = axes; hold(ax,'on');
plot(ax, xv/1e3, dshoal_bottom(xv, y_bg, opts{:}), 'LineWidth', 2.5, ...
     'Color', [0.85 0.15 0.15], 'DisplayName', 'Background');
cmap = lines(numel(SWEEP_VALS));
fprintf('\n  %-8s %10s %10s %12s %12s\n', SWEEP_NAME, 'crest_in', 'crest_out', 'L_S actual', 'max|dh/dx|');
for k = 1:numel(SWEEP_VALS)
    o2 = opts;
    o2{find(strcmp(o2, SWEEP_NAME)) + 1} = SWEEP_VALS(k);
    crest = o2{find(strcmp(o2,'Zs')) + 1};
    zk = dshoal_bottom(xv, y_shoal, o2{:});
    plot(ax, xv/1e3, zk, 'LineWidth', 2.5, 'Color', cmap(k,:), ...
         'DisplayName', sprintf('%s = %g', SWEEP_NAME, SWEEP_VALS(k)));
    on = abs(zk - crest) < 0.05;
    if any(on)
        a = xv(find(on,1,'first'))/1e3; b = xv(find(on,1,'last'))/1e3;
        fprintf('  %-8g %10.1f %10.1f %12.1f %12.5f\n', SWEEP_VALS(k), a, b, b-a, ...
                max(abs(diff(zk(xv <= 20e3))))/(xv(2)-xv(1)));
    else
        fprintf('  %-8g %10s %10s %12s %12.5f\n', SWEEP_VALS(k), '-', '-', '-', ...
                max(abs(diff(zk(xv <= 20e3))))/(xv(2)-xv(1)));
    end
end
yline(ax, 0, '--', 'Color', [0.4 0.4 0.4], 'HandleVisibility', 'off');
xlabel(ax,'x (km)'); ylabel(ax,'z (m)'); xlim(ax,[0 100]); ylim(ax,[-Lz 1]);
grid(ax,'on'); legend(ax,'Location','southwest'); set(ax,'FontSize',12);
title(ax, sprintf('Sweeping %s', SWEEP_NAME), 'FontWeight','bold');
