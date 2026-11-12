function vis_density_gradients_bulk(cluster_name, outdir)
% vis_density_gradients_bulk(cluster_name, outdir)
%
% Runs vis_density_gradients for all sweeps in a cluster.

SWEEP_OUTPUT = '/glade/work/nguyen/unc-shoals/sweep_output';
datadir_base = fullfile(SWEEP_OUTPUT, cluster_name);

sweeps = { ...
    'Zs20_Ls40k', ...
    'Zs15_Ls40k', ...
    'Zs10_Ls40k', ...
    'Zs5_Ls40k', ...
    'Zs10_Ls20k', ...
    'Zs10_Ls30k', ...
    'Zsh_m10', ...
    'Zsh_m50', ...
    'sigma_5k', ...
    'sigma_20k', ...
    'wind_0p01', ...
    'wind_0p05', ...
};

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
