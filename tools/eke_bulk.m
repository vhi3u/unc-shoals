function eke_bulk(cluster_name)
% eke_bulk(cluster_name)
%
% Runs eke_clipped for every sweep under:
%   /glade/work/nguyen/unc-shoals/sweep_output/<cluster_name>/
%
% PNGs are written to: /glade/work/nguyen/unc-shoals/animations/
%
% Example:
%   eke_bulk('cluster10_81_89')

if nargin < 1 || isempty(cluster_name)
    cluster_name = 'cluster10_81_89';
end

sweep_output_root = '/glade/work/nguyen/unc-shoals/sweep_output';
cluster_dir       = fullfile(sweep_output_root, cluster_name);

% =====================================================================
% OPTIONS
% =====================================================================

% Sweeps are discovered from the cluster directory, not hardcoded. The old
% list held whichever labels were current when this was written, so any
% later cluster matched nothing and every run was silently skipped.
[sweep_labels, sweep_tags] = discover_sweeps(cluster_dir);
sweeps = sweep_labels;

outdir = '/glade/work/nguyen/unc-shoals/animations';

% Fixed clims for presentation consistency (Presentation quality)
climKE  = [0, 3e-3]; 
climEKE = [0, 3e-5]; % Dynamic (free) limits% =====================================================================

addpath(fileparts(mfilename('fullpath')));

fprintf('\n=== eke_bulk: %d sweeps ===\n\n', numel(sweeps));

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
        failed{end+1} = sw; continue; %#ok<SAGROW>
    end

    try
        eke_clipped(sw, ...
            'datadir', datadir, ...
            'outdir',  outdir, ...
            'climKE',  climKE, ...
            'climEKE', climEKE);
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
