%% show_bathymetry.m — the fixed bathymetry used in the model re-runs
% Open this file in the MATLAB editor and press Cmd+Enter (run section).
% Shows params_pulse_front.toml, the re-run table: the reference shoal
% attached to the coast, no trench. Two figures: a 3-D view of every run
% (rotate it) and the cross-shore profile through the shoal centre.

% Find tools/ from this file's location, whether run as a section or a script.
try
    here = fileparts(matlab.desktop.editor.getActiveFilename);
catch
    here = fileparts(mfilename('fullpath'));
end
addpath(here);

close all
plot_bathymetry_sweeps('sweep_dir', fullfile(here, '..', 'src', 'sweep'), ...
                       'tables', {'params_pulse_front'});
