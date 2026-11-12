function vishoals_still_bulk(cluster_name)
% vishoals_still_bulk(cluster_name)
%
% Runs vishoals_still (8-panel still image, LAST snapshot) for every sweep
% found in the cluster directory. Run lengths may differ between sweeps.
% This includes bathymetry contours and nudging regions.

if nargin < 1 || isempty(cluster_name)
    cluster_name = 'cluster_Cnnie_1_15';
end

sweep_output_root = '/glade/work/nguyen/unc-shoals/sweep_output';
cluster_dir       = fullfile(sweep_output_root, cluster_name);

% Sweeps are DISCOVERED from the cluster directory rather than hardcoded.
% The previous hardcoded list held the 19 params_HsLs.toml labels, so any
% cluster using different run_labels -- e.g. the res250 resolution check,
% whose runs are R500_Zs05, R250_Zs05, ... -- matched nothing and every
% entry was reported as "SKIP: No folder found".
d_all = dir(fullfile(cluster_dir, 'sweep_*'));
d_all = d_all([d_all.isdir]);
d_all = d_all(~ismember({d_all.name}, {'.', '..'}));
if isempty(d_all)
    error('vishoals_still_bulk:noSweeps', ...
          'No sweep_* directories found in %s', cluster_dir);
end
sweeps = cell(1, numel(d_all));
for k = 1:numel(d_all)
    % sweep_001_R500_Zs05 -> R500_Zs05  (the label the .nc files are named by)
    nm = d_all(k).name;
    tok = regexp(nm, '^sweep_\d+_(.+)$', 'tokens', 'once');
    if isempty(tok)
        sweeps{k} = regexprep(nm, '^sweep_', '');
    else
        sweeps{k} = tok{1};
    end
end
sweeps = sweeps(~cellfun(@isempty, sweeps));

% Fixed clims for cross-sweep comparison
climv  = [];
% climu = [-0.1, 0.1];
climRo = [-1, 1];
climu = [];
% climRo = [];

clipped = true;   % clip XY to x=0-100 km, y=20-200 km (XZ clipped to same x)

outdir = '/glade/work/nguyen/unc-shoals/animations';

fprintf('\n=== vishoals_still_bulk: %d sweeps ===\n\n', numel(sweeps));

failed = {};

for k = 1:numel(sweeps)
    sw      = sweeps{k};
    % Find actual folder with wildcard to handle sweep_001_... numbering
    sw_core = sw;
    d = dir(fullfile(cluster_dir, ['*' sw_core '*']));
    d = d([d.isdir]);
    d = d(~ismember({d.name}, {'.', '..'}));
    if isempty(d)
        fprintf('  SKIP: No folder found matching *%s*\n', sw);
        failed{end+1} = sw; continue;
    end
    datadir = fullfile(cluster_dir, d(1).name);

    fprintf('[%d/%d] %s\n', k, numel(sweeps), sw);

    if ~isfile(fullfile(datadir, ['top_' sw '.nc'])) || ~isfile(fullfile(datadir, ['midy_' sw '.nc']))
        fprintf('  SKIP: NC files not found\n');
        failed{end+1} = sw; continue;
    end

    try
        vishoals_still(sw, ...
            'datadir',    datadir,    ...
            'outdir',     outdir,     ...
            'vel',        'u',        ...
            'vort',       'Ro',       ...
            'day',        Inf,        ...  % last snapshot in the file
            'Ls',         20,         ...
            'full_panel', true,       ...
            'clipped',    clipped,    ...
            'climu',      climu,      ...
            'climRo',     climRo,     ...
            'isopycnals', true);
            % 'climv',      climv);
    catch ME
        fprintf('  ERROR: %s\n', ME.message);
        failed{end+1} = sw;
    end
    fprintf('\n');
end

fprintf('=== Done: %d/%d succeeded ===\n', numel(sweeps)-numel(failed), numel(sweeps));
end
