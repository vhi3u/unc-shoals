function offshore_flux_bulk_full(cluster_name)
% offshore_flux_bulk_full(cluster_name)
%
% Runs offshore_flux (Full Model Extent) for every sweep.
%
% PNGs are written to: /glade/work/nguyen/unc-shoals/animations/

if nargin < 1 || isempty(cluster_name)
    cluster_name = 'cluster_Cnnie_1_15';
end

sweep_output_root = '/glade/work/nguyen/unc-shoals/sweep_output';
cluster_dir       = fullfile(sweep_output_root, cluster_name);

% =====================================================================
% OPTIONS
% =====================================================================
% Sweeps are discovered from the cluster directory, not hardcoded. The
% old list held whichever labels were current when this was written, so
% any later cluster matched nothing and every run was silently skipped.
[sweep_labels, sweep_tags] = discover_sweeps(cluster_dir);
sweeps = sweep_labels;

outdir = '/glade/work/nguyen/unc-shoals/animations';

% Fixed clims for presentation consistency
climuT = [-6.5e-3, 6.5e-3]; 
climuS = [-3.5e-3, 3.5e-3]; 
% =====================================================================

addpath(fileparts(mfilename('fullpath')));

fprintf('\n=== offshore_flux_bulk_full: %d sweeps ===\n\n', numel(sweeps));

failed = {};

for k = 1:numel(sweeps)
    sw      = sweeps{k};
    % Find actual folder with wildcard to handle sweep_001_... numbering
    sw_core = strrep(sw, 'sweep_', '');
    d = dir(fullfile(cluster_dir, ['*' sw_core '*']));
    d = d([d.isdir]);
    d = d(~ismember({d.name}, {'.', '..'}));
    if isempty(d)
        fprintf('  SKIP: No folder found matching *%s*\n', sw);
        failed{end+1} = sw; continue;
    end
    datadir = fullfile(cluster_dir, d(1).name);

    fprintf('[%d/%d] %s\n', k, numel(sweeps), sw);

    if ~isfile(fullfile(datadir, ['time_avg_3d_' sw '.nc'])) && ...
       ~isfile(fullfile(datadir, ['zavg_xy_' sw '.nc']))
        fprintf('  SKIP: NC file not found\n');
        failed{end+1} = sw; continue;
    end

    try
        offshore_flux(sw, ...
            'datadir', datadir, ...
            'outdir',  outdir,  ...
            'climuT',  climuT,  ...
            'climuS',  climuS);
    catch ME
        fprintf('  ERROR: %s\n', ME.message);
        failed{end+1} = sw;
    end
    fprintf('\n');
end

fprintf('=== Done: %d/%d succeeded ===\n', numel(sweeps)-numel(failed), numel(sweeps));
end
