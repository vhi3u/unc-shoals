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
