%% show_bathymetry.m — the fixed bathymetry used in the model re-runs
% Open this file in the MATLAB editor and press Cmd+Enter (run section).
% Shows the two re-run tables, both trench-free with the shoal attached to
% the coast:
%   params_depthaxis    the geometry sweep: crest depth -5 / -10 / -15 m,
%                       crest length held at 37 km
%   params_pulse_front  the pulsed-wind / front sweep
% Two figures each: a 3-D view of every run (rotate it) and the cross-shore
% profile through the shoal centre.

% Find tools/ from this file's location, whether run as a section or a script.
try
    here = fileparts(matlab.desktop.editor.getActiveFilename);
catch
    here = fileparts(mfilename('fullpath'));
end
addpath(here);

close all
plot_bathymetry_sweeps('sweep_dir', fullfile(here, '..', 'src', 'sweep'), ...
                       'tables', {'params_depthaxis', 'params_pulse_front'});

%% The geometry-sweep fix: old (trench, what cluster_depthaxis_120 ran) vs new
close all
plot_bathymetry_sweeps('sweep_dir', fullfile(here, '..', 'src', 'sweep'), ...
                       'tables', {'params_depthaxis_asrun', 'params_depthaxis'});

%% Are the shoals what the tables say? Declared vs MEASURED, every run
% Measured from the bottom itself (shoal_bathymetry's second output). "Ls" in
% the old tables is shoal_length, which is where the crest ENDS, not how long
% it is; runs that set crest_length declare the length directly.
tables = dir(fullfile(here, '..', 'src', 'sweep', 'params*.toml'));
fprintf('\n%-26s %-20s %8s %8s | %10s %10s %11s %8s  %s\n', 'table', 'run', 'Zs', 'Ls(decl)', ...
        'crest (m)', 'crest (km)', 'starts (km)', 'width_y', 'problems');
for t = tables'
    runs = read_sweep_table(fullfile(t.folder, t.name));
    for k = 1:numel(runs)
        p = struct();
        for f = fieldnames(runs(k))', v = runs(k).(f{1}); if ~isempty(v) && isnumeric(v), p.(f{1}) = double(v); end, end
        [~, g] = shoal_bathymetry(0, 0, p);
        note = g.note;
        fprintf('%-26s %-20s %8.1f %8.1f | %10.2f %10.1f %11.1f %8.1f  %s\n', strrep(t.name, '.toml', ''), ...
                runs(k).run_label, g.declared_depth, g.declared_length / 1e3, g.crest_depth, ...
                g.crest_length / 1e3, g.crest_start / 1e3, g.width_y / 1e3, note);
    end
end

%% PROPOSAL for approval: crest_length (what Ls should mean)
% Same three crest depths, two ways of setting the length, profiles through
% the shoal centre (y = 100 km). Each crest is marked with its MEASURED start,
% end and length (shoal_bathymetry's second output).
%   left:  today, shoal_length = 40 km  (the crest ENDS at 40 km)
%   right: proposed, crest_length = 40 km (the crest IS 40 km long)
close all
zs_list = [-5 -10 -15];  cc = lines(3);  xs = 0:50:90e3;
fig = figure('Color', 'w', 'Name', 'Proposal: crest_length', 'NumberTitle', 'off', 'Position', [60 60 1400 560]);
tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
cases = {struct('shoal_length', 40e3), struct('crest_length', 40e3)};
names = {'TODAY: shoal\_length = 40 km (where the crest ends)', ...
         'PROPOSED: crest\_length = 40 km (how long the crest is)'};
for c = 1:2
    ax = nexttile(tl);  hold(ax, 'on');  grid(ax, 'on');
    for k = 1:3
        p = cases{c};  p.Zs = zs_list(k);
        [h, g] = shoal_bathymetry(xs, 100e3, p);
        plot(ax, xs/1e3, h, 'Color', cc(k,:), 'LineWidth', 2, ...
             'DisplayName', sprintf('Zs = %d m: crest %.1f km long', zs_list(k), g.crest_length/1e3));
        plot(ax, [g.crest_start g.crest_end]/1e3, g.crest_depth*[1 1] + 1.2, '|-', 'Color', cc(k,:), ...
             'LineWidth', 1.5, 'MarkerSize', 10, 'HandleVisibility', 'off');
        text(ax, (g.crest_start + g.crest_end)/2e3, g.crest_depth + 2.6, ...
             sprintf('%.1f km', g.crest_length/1e3), 'Color', cc(k,:), 'HorizontalAlignment', 'center', ...
             'FontWeight', 'bold');
    end
    xlim(ax, [0 90]);  ylim(ax, [-52 2]);
    xlabel(ax, 'x (km from coast)');  ylabel(ax, 'depth (m)');
    title(ax, names{c});  legend(ax, 'Location', 'southwest');
end
title(tl, 'Crest depth sweep: does "Ls = 40 km" give a 40 km crest?  (bars = measured crest)', 'FontWeight', 'bold');

fprintf('\n%-10s %-28s %10s %10s %10s %10s %8s\n', 'Zs (m)', 'length set by', 'crest (m)', 'start km', 'end km', 'length km', 'trench');
for c = 1:2
    for k = 1:3
        p = cases{c};  p.Zs = zs_list(k);
        [~, g] = shoal_bathymetry(0, 0, p);
        if c == 1, how = 'shoal_length = 40 km'; else, how = 'crest_length = 40 km'; end
        fprintf('%-10d %-28s %10.2f %10.2f %10.2f %10.2f %8.2f\n', zs_list(k), how, ...
                g.crest_depth, g.crest_start/1e3, g.crest_end/1e3, g.crest_length/1e3, g.trench);
    end
end

%% PROPOSAL, part 2: changing shelf depth, crest_length = 40 km, ONE fixed break
% The deep_geom setup (100 m basin, break slopes held fixed per shelf depth),
% crest_length = 40 km. The shelf break is FIXED for the whole sweep, at the
% smallest rise_mid that leaves every run at least 10 km of flat shelf between
% the shoal's taper and the break (fixed_break_rise_mid). The measured gap is
% then a CHECK. Two ways to hold "the same shoal" while the shelf deepens:
%   left:  same crest DEPTH              Zs = -5 m on every shelf
%   right: same HEIGHT above the shelf   Hs = Zsh - Zs = 10 m
% Bars = measured crest; open circles = end of the offshore taper; triangles
% = start of the shelf break.
close all
shelf = struct('Zsh', {-15, -25, -40, -50}, 'shelf_break_end', {8100, 12000, 17800, 21700}, ...
               'rise_width', {17000, 15000, 12000, 10000});
cc = lines(numel(shelf));  xs = 0:50:150e3;
fig = figure('Color', 'w', 'Name', 'Proposal: shelf depth', 'NumberTitle', 'off', 'Position', [60 60 1400 600]);
tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
titles = {'same crest DEPTH: Zs = -5 m', 'same HEIGHT above the shelf: Hs = 10 m'};
fprintf('\n%-8s %-18s %8s %9s %9s %9s %9s %8s  %s\n', 'Zsh', 'held fixed', 'Zs', 'start km', 'length km', ...
        'taper km', 'break km', 'gap km', 'problems');
for c = 1:2
    runs = shelf;
    for k = 1:numel(runs)
        runs(k).h_deep = -100;  runs(k).crest_length = 40e3;
        if c == 1, runs(k).Zs = -5; else, runs(k).Zs = runs(k).Zsh + 10; end
    end
    rm = fixed_break_rise_mid(runs, 10e3);            % one break for this sweep
    [runs.rise_mid] = deal(rm);
    ax = nexttile(tl);  hold(ax, 'on');  grid(ax, 'on');
    for k = 1:numel(runs)
        p = runs(k);
        [h, g] = shoal_bathymetry(xs, 100e3, p);
        plot(ax, xs/1e3, h, 'Color', cc(k,:), 'LineWidth', 2, ...
             'DisplayName', sprintf('shelf %d m: crest %.1f km at %g m, gap %.1f km', p.Zsh, g.crest_length/1e3, p.Zs, g.shelf_gap/1e3));
        plot(ax, [g.crest_start g.crest_end]/1e3, g.crest_depth*[1 1] + 1.5, '|-', 'Color', cc(k,:), ...
             'LineWidth', 1.5, 'MarkerSize', 8, 'HandleVisibility', 'off');
        plot(ax, g.taper_end/1e3, p.Zsh, 'o', 'Color', cc(k,:), 'MarkerSize', 8, 'LineWidth', 1.5, 'HandleVisibility', 'off');
        plot(ax, g.break_start/1e3, p.Zsh - 1, 'v', 'Color', cc(k,:), 'MarkerFaceColor', cc(k,:), 'HandleVisibility', 'off');
        if c == 1, how = 'crest depth -5 m'; else, how = 'height 10 m'; end
        fprintf('%-8d %-18s %8.1f %9.2f %9.2f %9.1f %9.1f %8.1f  %s\n', p.Zsh, how, p.Zs, g.crest_start/1e3, ...
                g.crest_length/1e3, g.taper_end/1e3, g.break_start/1e3, g.shelf_gap/1e3, g.note);
    end
    xline(ax, rm/1e3, 'k:', sprintf('rise\\_mid = %.1f km (all runs)', rm/1e3), 'LabelVerticalAlignment', 'bottom', 'HandleVisibility', 'off');
    xline(ax, 65, 'k--', 'jet offshore edge (65 km)', 'LabelVerticalAlignment', 'middle', 'HandleVisibility', 'off');
    fprintf('   -> one break for this sweep: rise_mid = %.1f km\n', rm/1e3);
    xlim(ax, [0 150]);  ylim(ax, [-102 2]);
    xlabel(ax, 'x (km from coast)');  ylabel(ax, 'depth (m)');  title(ax, titles{c});
    legend(ax, 'Location', 'southwest');
end
title(tl, 'Shelf-depth sweep: crest\_length = 40 km, one fixed break per sweep (>= 10 km flat shelf)  (o = taper end, \nabla = break start)', ...
      'FontWeight', 'bold');

%% PROPOSAL, part 2 in 3-D: the shelf-depth sweep
% Same runs as part 2 (crest_length = 40 km, one fixed break per sweep, 100 m
% basin): top row same crest DEPTH (Zs = -5 m), bottom row same HEIGHT above
% the shelf (Hs = 10 m). Rotate any panel.
close all
shelf = struct('Zsh', {-15, -25, -40, -50}, 'shelf_break_end', {8100, 12000, 17800, 21700}, ...
               'rise_width', {17000, 15000, 12000, 10000});
[X, Y] = ndgrid(0:1000:150e3, 0:1000:200e3);
fig = figure('Color', 'w', 'Name', 'Proposal: shelf depth (3-D)', 'NumberTitle', 'off', 'Position', [40 40 1700 820]);
tl = tiledlayout(fig, 2, numel(shelf), 'TileSpacing', 'compact', 'Padding', 'compact');
for c = 1:2
    runs = shelf;
    for k = 1:numel(runs)
        runs(k).h_deep = -100;  runs(k).crest_length = 40e3;
        if c == 1, runs(k).Zs = -5; else, runs(k).Zs = runs(k).Zsh + 10; end
    end
    rm = fixed_break_rise_mid(runs, 10e3);
    [runs.rise_mid] = deal(rm);
    for k = 1:numel(runs)
        p = runs(k);
        H = shoal_bathymetry(X, Y, p);
        [~, g] = shoal_bathymetry(0, 0, p);
        ax = nexttile(tl);
        surf(ax, X/1e3, Y/1e3, H, 'EdgeColor', 'none');
        colormap(ax, flipud(parula));  clim(ax, [-100 0]);  zlim(ax, [-100 0]);
        view(ax, -35, 35);  axis(ax, 'tight');  daspect(ax, [1 1 2.5]);
        lighting(ax, 'gouraud');  camlight(ax, 'headlight');  material(ax, 'dull');
        xlabel(ax, 'x (km)');  ylabel(ax, 'y (km)');  zlabel(ax, 'z (m)');
        title(ax, sprintf('shelf %d m, crest %g m: %.1f km long, gap %.1f km', p.Zsh, p.Zs, ...
              g.crest_length/1e3, g.shelf_gap/1e3), 'FontSize', 9);
    end
end
title(tl, 'Shelf-depth sweep, proposed geometry (crest\_length = 40 km, one fixed break per row). Top: same crest depth (-5 m). Bottom: same height above the shelf (10 m).', ...
      'FontWeight', 'bold');

%% The reference run: Zs = -5 m, Ls = 40 km (sweep defaults, 25 m shelf)
% Left / top: the reference run as it has always been built.
% Right / bottom: the same shoal with the shelf break moved just far enough
% offshore to leave the minimum 10 km of flat shelf (fixed_break_rise_mid).
close all
ref = struct('Zs', -5, 'crest_length', 40e3);          % everything else: sweep defaults
fixed = ref;  fixed.rise_mid = fixed_break_rise_mid(ref, 10e3);
cases = {ref, fixed};
names = {'reference run as built (rise\_mid = 63.5 km)', ...
         sprintf('with >= 10 km flat shelf (rise\\_mid = %.1f km)', fixed.rise_mid/1e3)};
xs = 0:50:150e3;  [X, Y] = ndgrid(0:1000:150e3, 0:1000:200e3);
fig = figure('Color', 'w', 'Name', 'Reference run', 'NumberTitle', 'off', 'Position', [40 40 1500 850]);
tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
fprintf('\n%-50s %9s %9s %9s %9s %9s %9s %9s  %s\n', 'case', 'crest m', 'start km', 'length km', ...
        'width km', 'taper km', 'break km', 'gap km', 'problems');
for c = 1:2
    p = cases{c};
    [h, g] = shoal_bathymetry(xs, 100e3, p);
    ax = nexttile(tl, c);  hold(ax, 'on');  grid(ax, 'on');
    plot(ax, xs/1e3, h, 'k', 'LineWidth', 2);
    plot(ax, [g.crest_start g.crest_end]/1e3, g.crest_depth*[1 1] + 1.5, '|-', 'Color', [0 0.45 0.75], 'LineWidth', 1.5, 'MarkerSize', 10);
    text(ax, (g.crest_start + g.crest_end)/2e3, g.crest_depth + 3.5, sprintf('crest %.1f km at %.1f m', g.crest_length/1e3, g.crest_depth), ...
         'HorizontalAlignment', 'center', 'Color', [0 0.45 0.75], 'FontWeight', 'bold');
    plot(ax, g.taper_end/1e3, -25, 'o', 'Color', [0.85 0.33 0.1], 'MarkerSize', 9, 'LineWidth', 1.5);
    plot(ax, g.break_start/1e3, -26, 'v', 'Color', [0.47 0.67 0.19], 'MarkerFaceColor', [0.47 0.67 0.19], 'MarkerSize', 8);
    plot(ax, [g.taper_end g.break_start]/1e3, [-29 -29], '-', 'Color', [0.47 0.67 0.19], 'LineWidth', 3);
    text(ax, (g.taper_end + g.break_start)/2e3, -32, sprintf('flat shelf %.1f km', g.shelf_gap/1e3), ...
         'HorizontalAlignment', 'center', 'Color', [0.47 0.67 0.19], 'FontWeight', 'bold');
    xline(ax, 65, 'k--', 'jet offshore edge', 'LabelVerticalAlignment', 'bottom', 'HandleVisibility', 'off');
    xlim(ax, [0 150]);  ylim(ax, [-52 2]);  xlabel(ax, 'x (km from coast)');  ylabel(ax, 'depth (m)');
    title(ax, [names{c} '  --  o taper end, \nabla break start']);
    if isempty(g.note), status = 'ok'; else, status = g.note; end
    fprintf('%-50s %9.2f %9.2f %9.2f %9.1f %9.1f %9.1f %9.1f  %s\n', strrep(strrep(names{c}, '\_', '_'), '\\', ''), g.crest_depth, ...
            g.crest_start/1e3, g.crest_length/1e3, g.width_y/1e3, g.taper_end/1e3, g.break_start/1e3, g.shelf_gap/1e3, status);

    ax = nexttile(tl, c + 2);
    surf(ax, X/1e3, Y/1e3, shoal_bathymetry(X, Y, p), 'EdgeColor', 'none');
    colormap(ax, flipud(parula));  clim(ax, [-50 0]);  zlim(ax, [-50 0]);
    view(ax, -35, 35);  axis(ax, 'tight');  daspect(ax, [1 1 1.25]);
    lighting(ax, 'gouraud');  camlight(ax, 'headlight');  material(ax, 'dull');
    xlabel(ax, 'x (km)');  ylabel(ax, 'y (km)');  zlabel(ax, 'z (m)');  title(ax, names{c});
end
title(tl, 'Reference run, Zs = -5 m, Ls = 40 km: profile through the shoal centre (top) and 3-D (bottom)', 'FontWeight', 'bold');

%% CANDIDATES for a new reference shoal: width, length and crest shape
% Four shoals on the reference shelf (25 m, break at 63.5 km), crest at -5 m:
%   A  current reference      sigma 8 km,   40 km flat crest
%   B  narrow                 sigma 2.5 km, 40 km flat crest
%   C  narrow + short         sigma 2.5 km, 20 km flat crest
%   D  narrow + short + slope sigma 2.5 km, 20 km crest deepening -5 -> -15 m
% Rows: 3-D view, plan view (depth contours), profile through the shoal
% centre. Each title gives the regime numbers for the inflow jet
% (v0 = 0.2 m/s, f = 8.42e-5 1/s, Ld = 4.2 km), W = alongshore width at half
% height:
%   Ro    = v0/(f W)          shoal Rossby number: steered (small) vs separating
%   d/Ro  = blocking / Ro     Taylor-column parameter: > ~1 flow goes around
%   Ld/W                      how much stratification matters around it
close all
cand = { struct('Zs', -5, 'crest_length', 40e3, 'sigma', 8e3), ...
         struct('Zs', -5, 'crest_length', 40e3, 'sigma', 2.5e3), ...
         struct('Zs', -5, 'crest_length', 20e3, 'sigma', 2.5e3), ...
         struct('Zs', -5, 'crest_length', 20e3, 'sigma', 2.5e3, 'tip_depth', -15) };
names = {'A: current reference', 'B: narrow', 'C: narrow + short', 'D: narrow + short + sloping crest'};
v0 = 0.2;  f0 = 8.42e-5;  Ld = 4.2e3;  Zsh = -25;
nc = numel(cand);
[X, Y] = ndgrid(0:500:100e3, 50e3:500:150e3);
xs = 0:50:100e3;
fig = figure('Color', 'w', 'Name', 'Candidate reference shoals', 'NumberTitle', 'off', 'Position', [30 30 1750 1000]);
tl = tiledlayout(fig, 3, nc, 'TileSpacing', 'compact', 'Padding', 'compact');
fprintf('\n%-36s %7s %7s %7s %8s %8s %7s %7s %7s %7s  %s\n', 'candidate', 'crest m', 'tip m', 'len km', ...
        'W km', 'gap km', 'delta', 'Ro', 'd/Ro', 'Ld/W', 'problems');
for c = 1:nc
    p = cand{c};
    H = shoal_bathymetry(X, Y, p);
    [h, g] = shoal_bathymetry(xs, 100e3, p);
    delta = (g.crest_depth - Zsh) / abs(Zsh);
    Ro = v0 / (f0 * g.width_y);
    lbl = sprintf('%s\nRo = %.2f, \\delta/Ro = %.1f, L_d/W = %.2f', names{c}, Ro, delta / Ro, Ld / g.width_y);

    ax = nexttile(tl, c);                                  % 3-D
    surf(ax, X/1e3, Y/1e3, H, 'EdgeColor', 'none');
    colormap(ax, flipud(parula));  clim(ax, [-50 0]);  zlim(ax, [-50 0]);
    view(ax, -35, 35);  axis(ax, 'tight');  daspect(ax, [1 1 1.5]);
    lighting(ax, 'gouraud');  camlight(ax, 'headlight');  material(ax, 'dull');
    xlabel(ax, 'x (km)');  ylabel(ax, 'y (km)');  zlabel(ax, 'z (m)');  title(ax, lbl, 'FontSize', 9);

    ax = nexttile(tl, nc + c);                             % plan view
    contourf(ax, X/1e3, Y/1e3, H, -50:2.5:-5, 'LineColor', 'none');
    hold(ax, 'on');  contour(ax, X/1e3, Y/1e3, H, [-24 -20 -15 -10 -6], 'k', 'LineWidth', 0.5);
    colormap(ax, flipud(parula));  clim(ax, [-50 0]);  axis(ax, 'equal', 'tight');
    xlabel(ax, 'x (km)');  ylabel(ax, 'y (km)');
    title(ax, sprintf('plan view: width %.1f km at half height', g.width_y/1e3), 'FontSize', 9);

    ax = nexttile(tl, 2*nc + c);                           % profile
    hold(ax, 'on');  grid(ax, 'on');
    plot(ax, xs/1e3, h, 'k', 'LineWidth', 2);
    plot(ax, [g.crest_start g.crest_end]/1e3, [g.crest_depth g.tip_depth] + 1.5, '|-', 'Color', [0 0.45 0.75], ...
         'LineWidth', 1.5, 'MarkerSize', 8);
    plot(ax, g.taper_end/1e3, Zsh, 'o', 'Color', [0.85 0.33 0.1], 'LineWidth', 1.5);
    plot(ax, g.break_start/1e3, Zsh - 1, 'v', 'Color', [0.47 0.67 0.19], 'MarkerFaceColor', [0.47 0.67 0.19]);
    xlim(ax, [0 100]);  ylim(ax, [-52 2]);  xlabel(ax, 'x (km from coast)');  ylabel(ax, 'depth (m)');
    title(ax, sprintf('crest %.1f km, %.0f -> %.0f m; flat shelf %.1f km', g.crest_length/1e3, ...
          g.crest_depth, g.tip_depth, g.shelf_gap/1e3), 'FontSize', 9);
    fprintf('%-36s %7.2f %7.2f %7.1f %8.1f %8.1f %7.2f %7.2f %7.1f %7.2f  %s\n', names{c}, g.crest_depth, g.tip_depth, ...
            g.crest_length/1e3, g.width_y/1e3, g.shelf_gap/1e3, delta, Ro, delta/Ro, Ld/g.width_y, g.note);
end
title(tl, 'Candidate reference shoals on the reference shelf (25 m, break at 63.5 km, crest -5 m)', 'FontWeight', 'bold');
