function vis_density_gradients_bulk(cluster_name, outdir)
% vis_density_gradients_bulk(cluster_name, outdir)
%
% Runs vis_density_gradients for all sweeps in a cluster.

SWEEP_OUTPUT = '/glade/work/nguyen/unc-shoals/sweep_output';
datadir_base = fullfile(SWEEP_OUTPUT, cluster_name);

% Sweeps are discovered from the cluster directory, not hardcoded. The old
% list held whichever labels were current when this was written, so any
% later cluster matched nothing and every run was silently skipped.
[sweep_labels, sweep_tags] = discover_sweeps(datadir_base);
sweeps = sweep_labels;

if nargin < 2, outdir = ''; end

fprintf('=== Running Density Gradient Bulk: %s ===\n', cluster_name);

for i = 1:numel(sweeps)
    run_tag = sweeps{i};
    fprintf('[%d/%d] %s\n', i, numel(sweeps), run_tag);
    
    % Find actual folder with wildcard to handle sweep_001_... numbering
    sw_core = strrep(run_tag, 'sweep_', '');
    d = dir(fullfile(datadir_base, ['*' sw_core '*']));
    d = d([d.isdir]);
    d = d(~ismember({d.name}, {'.', '..'}));
    if isempty(d)
        fprintf('  SKIP: No folder found matching *%s*\n', run_tag);
        continue;
    end
    datadir = fullfile(datadir_base, d(1).name);
    
    try
        vis_density_gradients(run_tag, 'datadir', datadir, 'outdir', outdir);
    catch ME
        fprintf('  ERROR: %s\n', ME.message);
    end
end

fprintf('=== Done ===\n');

end
