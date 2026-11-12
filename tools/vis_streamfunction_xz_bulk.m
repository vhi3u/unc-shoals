function vis_streamfunction_xz_bulk(cluster_name)
% vis_streamfunction_xz_bulk(cluster_name)
%
% Runs vis_streamfunction_xz for every sweep in the cluster.

if nargin < 1 || isempty(cluster_name)
    cluster_name = 'cluster_Cnnie_1_15';
end

sweep_output_root = '/glade/work/nguyen/unc-shoals/sweep_output';
cluster_dir       = fullfile(sweep_output_root, cluster_name);

% Sweeps are discovered from the cluster directory, not hardcoded. The old
% list held whichever labels were current when this was written, so any
% later cluster matched nothing and every run was silently skipped.
[sweep_labels, sweep_tags] = discover_sweeps(cluster_dir);
sweeps = sweep_labels;

outdir = '/glade/work/nguyen/unc-shoals/animations';

fprintf('\n=== vis_streamfunction_xz_bulk: %d sweeps ===\n\n', numel(sweeps));

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
    try
        % Run the streamfunction plot
        % You can optionally pass 'y_range' or 'background' here if you want to override defaults
        vis_streamfunction_xz(sw, ...
            'datadir', datadir, ...
            'outdir',  outdir);
    catch ME
        fprintf('  ERROR: %s\n', ME.message);
        failed{end+1} = sw;
    end
end

if ~isempty(failed)
    fprintf('\nFailed sweeps:\n');
    for i = 1:numel(failed)
        fprintf('  %s\n', failed{i});
    end
else
    fprintf('\nAll sweeps completed successfully!\n');
end

end
