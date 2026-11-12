function vis_froude_bulk(cluster_name)
% vis_froude_bulk(cluster_name)
%
% Runs vis_froude for every sweep under:
%   /glade/work/nguyen/unc-shoals/sweep_output/<cluster_name>/
%
% PNGs are written to: /glade/work/nguyen/unc-shoals/animations/
%
% Example:
%   vis_froude_bulk('cluster13_111_117')

if nargin < 1 || isempty(cluster_name)
    cluster_name = 'cluster10_81_89';
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
climFr = [0, 1.5]; % Focus on the transition around Fr=1
% =====================================================================

addpath(fileparts(mfilename('fullpath')));

fprintf('\n=== vis_froude_bulk: %d sweeps in %s ===\n\n', numel(sweeps), cluster_name);

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

    % vis_froude looks for run_tag.nc or vnshoals_run_tag.nc or *run_tag*.nc
    % We just need to check if the directory exists and has some .nc files
    if ~exist(datadir, 'dir')
        fprintf('  SKIP: Directory not found\n');
        failed{end+1} = sw; continue; %#ok<SAGROW>
    end
    
    nc_files = dir(fullfile(datadir, '*.nc'));
    if isempty(nc_files)
        fprintf('  SKIP: No NC files found in %s\n', datadir);
        failed{end+1} = sw; continue; %#ok<SAGROW>
    end

    try
        % Call the visualization function
        vis_froude(sw, ...
            'datadir', datadir, ...
            'outdir',  outdir,  ...
            'clim',    climFr);
    catch ME
        fprintf('  ERROR: %s\n', ME.message);
        failed{end+1} = sw; %#ok<SAGROW>
    end
    fprintf('\n');
end

fprintf('=== Done: %d/%d succeeded ===\n', numel(sweeps)-numel(failed), numel(sweeps));

if ~isempty(failed)
    fprintf('Failed:\n');
    for j = 1:numel(failed)
        fprintf('  - %s\n', failed{j});
    end
end

end
