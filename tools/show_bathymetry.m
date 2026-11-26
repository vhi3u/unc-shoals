%% show_bathymetry.m — every sweep's bathymetry, as interactive figures
% Open this file in the MATLAB editor and press Cmd+Enter (run section).
% Two figures per sweep table: a 3-D view of every run (rotate it) and the
% cross-shore profile through the shoal centre. Runs with a trench between
% the coast and the shoal are titled in red.

% Find tools/ from this file's location, whether run as a section or a script.
try
    here = fileparts(matlab.desktop.editor.getActiveFilename);
catch
    here = fileparts(mfilename('fullpath'));
end
addpath(here);

close all
plot_bathymetry_sweeps('sweep_dir', fullfile(here, '..', 'src', 'sweep'));

%% The trench fix: the live pulse/front table (attached shoal, no trench)
% next to the table cluster_pulsefront_120 was actually run from (trench in red)
close all
plot_bathymetry_sweeps('sweep_dir', fullfile(here, '..', 'src', 'sweep'), ...
                       'tables', {'params_pulse_front', 'params_pulse_front_asrun'});

%% Just some tables (edit the list), e.g. the deep-domain sweeps
close all
plot_bathymetry_sweeps('sweep_dir', fullfile(here, '..', 'src', 'sweep'), ...
                       'tables', {'params_deep_geom', 'params_deep_wind'});
