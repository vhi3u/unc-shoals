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
