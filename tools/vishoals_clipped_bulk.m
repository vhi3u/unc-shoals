function vishoals_clipped_bulk(cluster_name)
% vishoals_clipped_bulk(cluster_name)
%
% Runs vishoals_clipped for every sweep under:
%   /glade/work/nguyen/unc-shoals/sweep_output/<cluster_name>/
%
% Each sweep lives in its own subdirectory:
%   <cluster_dir>/<sweep_name>/top_<sweep_name>.nc
%   <cluster_dir>/<sweep_name>/midy_<sweep_name>.nc
%
% GIFs are written to: <cluster_dir>/animations/<sweep_name>/
%
% Example:
%   vishoals_clipped_bulk('cluster10_81_89')

if nargin < 1 || isempty(cluster_name)
    cluster_name = 'cluster10_81_89';
end

sweep_output_root = '/glade/work/nguyen/unc-shoals/sweep_output';
cluster_dir       = fullfile(sweep_output_root, cluster_name);

% =====================================================================
% OPTIONS
% =====================================================================

sweeps = { ...
    'sweep_Hs00_0k', ...
    'Zs15_Ls40k', ...
    'Zs10_Ls40k', ...
    'Zs5_Ls40k', ...
    'Zs10_Ls20k', ...
    'Zs10_Ls30k', ...
    'Zsh_m10', ...
    'Zsh_m50', ...
    'wind_m0p05', ...
    'wind_m0p03', ...
    'wind_m0p01', ...
    'wind_0p01', ...
    'wind_0p03', ...
    'wind_0p05', ...
    'v0_0p15', ...
    'v0_0p20', ...
};

last_days  = 50;
fps        = 5;
frame_skip = 1;
want_mp4   = false;

% Fixed clims for cross-sweep comparison (presentation)
% climu  = [-0.06, 0.06];   % (m/s)
% climKE = [0,     6e-3];   % (m^2/s^2)
climu = [];
climKE = [];

outdir_root = '/glade/work/nguyen/unc-shoals/animations';
% =====================================================================

addpath(fileparts(mfilename('fullpath')));

fprintf('\n=== vishoals_clipped_bulk: %d sweeps ===\n\n', numel(sweeps));

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
    outdir  = outdir_root;

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
        % Build optional clim args — only pass when non-empty to avoid
        % failing the numel==2 validator in vishoals_clipped
        extra = {};
        if ~isempty(climu),  extra = [extra, {'climu',  climu}];  end
        if ~isempty(climKE), extra = [extra, {'climKE', climKE}]; end

        vishoals_clipped(sw, ...
            'datadir',    datadir,    ...
            'outdir',     outdir,     ...
            'last_days',  last_days,  ...
            'fps',        fps,        ...
            'frame_skip', frame_skip, ...
            'mp4',        want_mp4,   ...
            extra{:});

        % Generate bathymetry XZ PNG
        plot_topo_xz(sw, 'datadir', datadir);

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