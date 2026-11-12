function ke_vs_mke_bulk(cluster_name, outdir_user)
% ke_vs_mean_bulk(cluster_name, [outdir])
%
% Bulk script to plot Total Kinetic Energy vs Mean Kinetic Energy (MKE)
% for a cluster of simulations.
%
% Specifically targets the last time window in time_avg_3d_sweep_N.nc.
%
% Panels:
% 1. Total KE: 0.5 * (<uu> + <vv> + <ww>)
% 2. Mean KE:  0.5 * (<u>^2 + <v>^2 + <w>^2)
%
% Both panels use the same color scale (clims) to allow direct comparison.
%
% Example:
%   ke_vs_mke_bulk('cluster13_111_117', '/glade/work/nguyen/unc-shoals/animations/')

if nargin < 1 || isempty(cluster_name)
    cluster_name = 'cluster13_111_117';
end

if nargin < 2 || isempty(outdir_user)
    outdir = '/glade/work/nguyen/unc-shoals/animations/';
else
    outdir = outdir_user;
end

dataroot = '/glade/work/nguyen/unc-shoals/sweep_output';
cluster_dir = fullfile(dataroot, cluster_name);

if ~exist(outdir, 'dir'), mkdir(outdir); end

% Define sweeps for cluster13_111_117
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

% Optional: search directory for any other sweeps if they exist
% (Uncomment if you want to catch all folders starting with 'sweep_')
% d = dir(fullfile(cluster_dir, 'sweep_*'));
% sweeps = {d.name};

% Fixed color limits for comparison
% Using same limit for both to see relative magnitude
clim_shared = [0, 5e-3]; 

fprintf('\n=== Starting ke_vs_mke_bulk for %s ===\n', cluster_name);
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
    
    try
        plot_ke_mke_pair(sw, datadir, outdir, clim_shared);
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

function plot_ke_mke_pair(run_tag, datadir, outdir, clim)
    % Internal plotting function for one sweep
    
    fname = fullfile(datadir, ['time_avg_3d_' run_tag '.nc']);
    if ~isfile(fname)
        % Try alternative naming if sweep folder ≠ run_tag in file
        possible = dir(fullfile(datadir, 'time_avg_3d_*.nc'));
        if isempty(possible)
            error('No time_avg_3d_*.nc file found in %s', datadir);
        end
        fname = fullfile(datadir, possible(1).name);
    end

    % Load coordinates (using robust try logic)
    xc = try_ncread_(fname, {'xC','x_caa','xc'});
    yc = try_ncread_(fname, {'yC','y_aca','yc'});
    xc_km = xc / 1000;
    yc_km = yc / 1000;
    
    % Dynamic clipping (matching eke_clipped.m)
    Ly_km = yc_km(end);
    x_clip = [0, 60];
    y_clip = [Ly_km/2 - 50, Ly_km/2 + 50];

    % Load fields (last time step)
    u  = load_last_3d(fname, 'u_c');  if isempty(u),  u  = load_last_3d(fname, 'u'); end
    v  = load_last_3d(fname, 'v_c');  if isempty(v),  v  = load_last_3d(fname, 'v'); end
    w  = load_last_3d(fname, 'w_c');  if isempty(w),  w  = load_last_3d(fname, 'w'); end
    uu = load_last_3d(fname, 'uu');
    vv = load_last_3d(fname, 'vv');
    ww = load_last_3d(fname, 'ww');

    if isempty(u) || isempty(uu)
        error('Required variables (u, uu) not found in %s', fname);
    end

    % Mask 0 as NaN (land/immersed)
    u(u==0)=NaN; v(v==0)=NaN; w(w==0)=NaN;
    uu(uu==0)=NaN; vv(vv==0)=NaN; ww(ww==0)=NaN;

    % Calculations
    ke_total_3d = 0.5 * (uu + vv + ww);
    ke_mean_3d  = 0.5 * (u.^2 + v.^2 + w.^2);
    
    % Depth average (Z is dimension 3)
    ke_total_zavg = squeeze(mean(ke_total_3d, 3, 'omitnan'));
    ke_mean_zavg  = squeeze(mean(ke_mean_3d, 3, 'omitnan'));

    % Get indices for color limit (clipped region only)
    ix = find(xc_km >= x_clip(1) & xc_km <= x_clip(2));
    iy = find(yc_km >= y_clip(1) & yc_km <= y_clip(2));
    ke_total_sub = ke_total_zavg(ix, iy);
    
    % Use dynamic clim based on Max Total KE to ensure fair comparison
    cmax = max(ke_total_sub(:), [], 'omitnan');
    if isempty(cmax) || isnan(cmax) || cmax == 0, cmax = 1e-3; end
    clim_dyn = [0, cmax];

    % Prep plotting
    [Xc, Yc] = meshgrid(xc_km, yc_km);
    fig = figure('Color','w','Position',[100 100 1100 500], 'Visible', 'off');
    tl = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'tight');

    % Panel 1: Total KE
    ax1 = nexttile(tl);
    pcolor(ax1, Xc, Yc, ke_total_zavg');
    shading(ax1, 'flat'); axis(ax1, 'equal', 'tight');
    colormap(ax1, turbo); caxis(ax1, clim_dyn);
    cb1 = colorbar(ax1); cb1.Label.String = 'm^2/s^2';
    title(ax1, 'Total Kinetic Energy', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel(ax1, 'x (km)'); ylabel(ax1, 'y (km)');
    xlim(ax1, x_clip); ylim(ax1, y_clip);
    grid(ax1, 'on'); ax1.Layer = 'top';

    % Panel 2: Mean KE
    ax2 = nexttile(tl);
    pcolor(ax2, Xc, Yc, ke_mean_zavg');
    shading(ax2, 'flat'); axis(ax2, 'equal', 'tight');
    colormap(ax2, turbo); caxis(ax2, clim_dyn);
    cb2 = colorbar(ax2); cb2.Label.String = 'm^2/s^2';
    title(ax2, 'Mean Kinetic Energy', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel(ax2, 'x (km)'); ylabel(ax2, 'y (km)');
    xlim(ax2, x_clip); ylim(ax2, y_clip);
    grid(ax2, 'on'); ax2.Layer = 'top';

    sgtitle(tl, sprintf('%s: Total vs Mean KE (Depth-Averaged)', run_tag), ...
        'FontSize', 16, 'FontWeight', 'bold', 'Interpreter', 'none');

    % Save
    outfile = fullfile(outdir, [run_tag '_ke_comparison.png']);
    exportgraphics(fig, outfile, 'Resolution', 200);
    close(fig);
end

function data = load_last_3d(fn, varname)
    try
        vinfo = ncinfo(fn, varname);
        vsize = vinfo.Size;
        nd = numel(vsize);
        if nd == 4
            data = ncread(fn, varname, [1 1 1 vsize(4)], [vsize(1) vsize(2) vsize(3) 1]);
        elseif nd == 3
            if vsize(3) > 10 % Likely Nz
                data = ncread(fn, varname);
            else % Likely Nt
                data = ncread(fn, varname, [1 1 vsize(3)], [vsize(1) vsize(2) 1]);
            end
        else
            data = ncread(fn, varname);
        end
        data = squeeze(data);
    catch
        data = [];
    end
end

function data = try_ncread_(fname, names)
for i = 1:numel(names)
    try
        data = ncread(fname, names{i});
        return;
    catch
    end
end
info = ncinfo(fname);
avail = strjoin({info.Variables.Name}, ', ');
error('Could not find variable. Tried: {%s}. Available: %s', ...
    strjoin(names, ', '), avail);
end
