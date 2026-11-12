function vishoals_8panel_bulk(cluster_name)
% vishoals_8panel_bulk(cluster_name)
%
% Runs vishoals for every sweep with vel='u' and vort='Ro':
%   /glade/work/nguyen/unc-shoals/sweep_output/<cluster_name>/
%
% MP4s (or GIFs if want_mp4 = false) are written to:
%   /glade/work/nguyen/unc-shoals/animations/
%
% Example:
%   vishoals_8panel_bulk('cluster10_81_89')

if nargin < 1 || isempty(cluster_name)
    cluster_name = 'cluster_Cnnie_1_15';
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
sweeps = sweep_tags;

last_days  = 50;
fps        = 15;
frame_skip = 1;
want_mp4   = true;    % true -> MP4 only, false -> GIF only
clipped    = true;    % clip XY to x=0-100 km, y=20-200 km (XZ clipped to same x)

% Fixed clims for cross-sweep comparison (presentation)
% climv  = [-0.06, 0.06];   % u velocity (m/s)
climv = []; % Empty to allow auto-scanning
climu = [-0.1, 0.1];
climRo = [-0.5, 0.5];
climKE = [];

outdir = '/glade/work/nguyen/unc-shoals/animations';
% =====================================================================

addpath(fileparts(mfilename('fullpath')));

fprintf('\n=== vishoals_8panel_bulk: %d sweeps ===\n\n', numel(sweeps));

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

    if ~isfile(fullfile(datadir, ['top_' sw '.nc']))
        fprintf('  SKIP: top_%s.nc not found\n', sw);
        failed{end+1} = sw; continue; %#ok<SAGROW>
    end
    if ~isfile(fullfile(datadir, ['midy_' sw '.nc']))
        fprintf('  SKIP: midy_%s.nc not found\n', sw);
        failed{end+1} = sw; continue; %#ok<SAGROW>
    end

    try
        extra = {};
        if ~isempty(climu),  extra = [extra, {'climu',  climu}];  end
        if ~isempty(climv),  extra = [extra, {'climv',  climv}];  end
        if ~isempty(climRo), extra = [extra, {'climRo', climRo}]; end
        if ~isempty(climKE), extra = [extra, {'climKE', climKE}]; end

        vishoals(sw, ...
            'datadir',    datadir,    ...
            'outdir',     outdir,     ...
            'vel',        'u',        ...
            'vort',       'Ro',       ...
            'last_days',  last_days,  ...
            'fps',        fps,        ...
            'frame_skip', frame_skip, ...
            'mp4',        want_mp4,   ...
            'isopycnals', false,      ...
            'full_panel', true,       ...
            'clipped',    clipped,    ...
            extra{:});
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
