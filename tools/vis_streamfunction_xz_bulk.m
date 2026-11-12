function vis_streamfunction_xz_bulk(cluster_name)
% vis_streamfunction_xz_bulk(cluster_name)
%
% Runs vis_streamfunction_xz for every sweep in the cluster.

if nargin < 1 || isempty(cluster_name)
    cluster_name = 'cluster_Cnnie_1_15';
end

sweep_output_root = '/glade/work/nguyen/unc-shoals/sweep_output';
cluster_dir       = fullfile(sweep_output_root, cluster_name);

sweeps = { ...
    'Zs15_Ls40k', ...
    'Zs10_Ls40k', ...
    'Zs5_Ls40k', ...
    'Zs5_Ls20k', ...
    'Zs5_Ls30k', ...
    'Zsh_m10', ...
    'Zsh_m50', ...
    'wind_m0p05', ...
    'wind_m0p03', ...
    'wind_m0p01', ...
    'wind_0p01', ...
    'wind_0p03', ...
    'wind_0p05', ...
    'v0_0p10', ...
    'v0_0p15', ...
};

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
