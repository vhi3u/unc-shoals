function vis_u_mean_bulk(cluster_name, z_target)
% vis_u_mean_bulk(cluster_name, z_target)
%
% Bulk script to plot time-averaged U-velocity for a cluster of simulations.
% Specifically targets the last time window in time_avg_3d_sweep_N.nc.
%
% z_target: optional depth (m) for the XY slice. Default = 0 (surface).
%
% Example:
%   vis_u_mean_bulk('cluster10_81_89', -20)

if nargin < 1 || isempty(cluster_name)
    cluster_name = 'cluster10_81_89';
end
if nargin < 2 || isempty(z_target)
    z_target = 0;
end

% Set up paths
sweep_output_root = '/glade/work/nguyen/unc-shoals/sweep_output';
cluster_dir       = fullfile(sweep_output_root, cluster_name);

% Use a subfolder for the specific depth to keep things organized
if z_target == 0
    subfolder = 'u_mean_surface';
else
    subfolder = sprintf('u_mean_z%d', round(abs(z_target)));
end
outdir = fullfile('/glade/work/nguyen/unc-shoals/animations', subfolder);

if ~exist(outdir, 'dir'), mkdir(outdir); end

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

% Fixed color limits for comparison (adjust if needed)
clim_u = [-0.15, 0.15]; 

fprintf('\n=== Starting vis_u_mean_bulk for %s (z = %.1f m) ===\n', cluster_name, z_target);
fprintf('Found %d sweeps.\n\n', numel(sweeps));

failed = {};

for k = 1:numel(sweeps)
    sw = sweeps{k};
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
    
    fprintf('[%d/%d] Processing %s...\n', k, numel(sweeps), sw);
    
    nc_file = fullfile(datadir, ['time_avg_3d_' sw '.nc']);
    
    if ~exist(nc_file, 'file')
        fprintf('  SKIP: File not found: %s\n', nc_file);
        failed{end+1} = sw;
        continue;
    end
    
    try
        % Call the plotting function
        vis_u_mean(sw, ...
            'datadir',  datadir,  ...
            'outdir',   outdir,   ...
            'clim',     clim_u,   ...
            'z_target', z_target);
        fprintf('  Done.\n');
    catch ME
        fprintf('  ERROR: %s\n', ME.message);
        failed{end+1} = sw;
    end
end

fprintf('\n=== Bulk Processing Complete ===\n');
fprintf('Succeeded: %d/%d\n', numel(sweeps) - numel(failed), numel(sweeps));
if ~isempty(failed)
    fprintf('Failed sweeps:\n');
    for j = 1:numel(failed)
        fprintf('  - %s\n', failed{j});
    end
end

end
