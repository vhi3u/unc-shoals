function plot_offshore_velocity_sweeps(cluster_name)
% plot_offshore_velocity_sweeps(cluster_name)
%
% Creates two plots stacking line plots of offshore velocity (u) at x = 50 km.
% 1. Effect of Hs: sweep_Hs05_Ls20k, sweep_Hs10_Ls20k, sweep_Hs15_Ls20k
% 2. Effect of Ls: sweep_Hs15_Ls20k, sweep_Hs15_Ls30k, sweep_Hs15_Ls40k
%
% The script loads time-averaged 3D data, depth-integrates the u-velocity,
% and extracts the profile at x = 50 km (m^2/s).

if nargin < 1 || isempty(cluster_name)
    cluster_name = 'cluster13_111_117';
end

sweep_output_root = '/glade/work/nguyen/unc-shoals/sweep_output';
cluster_dir       = fullfile(sweep_output_root, cluster_name);

% Define Sweep Groups
group1 = { ...
    'sweep_Hs05_Ls20k', ...
    'sweep_Hs10_Ls20k', ...
    'sweep_Hs15_Ls20k', ...
};

group2 = { ...
    'sweep_Hs15_Ls20k', ...
    'sweep_Hs15_Ls30k', ...
    'sweep_Hs15_Ls40k', ...
};

% Custom Display Names for Legends (LaTeX format)
group1_names = {'$H_s = 5$ m', '$H_s = 10$ m', '$H_s = 15$ m'};
group2_names = {'$L_s = 20$ km', '$L_s = 30$ km', '$L_s = 40$ km'};

% Parameters
x_target_km = 50;
y_clip_km   = [50, 150];
fs = 18; % Global font size

% Colors for plotting
colors = [
    0.0, 0.45, 0.74; % Blue
    0.85, 0.33, 0.10; % Orange
    0.93, 0.69, 0.13; % Yellow
];

% Initialize Figures
fig = figure('Color','w', 'Position', [100 100 1000 900]);
tl = tiledlayout(2, 1, 'TileSpacing', 'Compact', 'Padding', 'Compact');

%% Plot Group 1: Effect of Hs
ax1 = nexttile(tl);
hold(ax1, 'on');
for i = 1:numel(group1)
    sw = group1{i};
    [yc_km, u_profile] = get_u_profile(cluster_dir, sw, x_target_km);
    
    if ~isempty(u_profile)
        plot(ax1, yc_km, u_profile, 'LineWidth', 3, 'Color', colors(i,:), ...
            'DisplayName', group1_names{i});
    end
end
grid(ax1, 'on'); box(ax1, 'on');
ylabel(ax1, '$\int \bar{u} \, dz$ (m$^2$/s)', 'FontSize', fs, 'Interpreter', 'latex');
legend(ax1, 'Location', 'best', 'Interpreter', 'latex', 'FontSize', fs);
set(ax1, 'FontSize', fs);
xlim(ax1, y_clip_km); % Clip spatial y-coordinate as requested
ylim(ax1, 'auto');     % Auto-scale velocity axis

%% Plot Group 2: Effect of Ls
ax2 = nexttile(tl);
hold(ax2, 'on');
for i = 1:numel(group2)
    sw = group2{i};
    [yc_km, u_profile] = get_u_profile(cluster_dir, sw, x_target_km);
    
    if ~isempty(u_profile)
        plot(ax2, yc_km, u_profile, 'LineWidth', 3, 'Color', colors(i,:), ...
            'DisplayName', group2_names{i});
    end
end
grid(ax2, 'on'); box(ax2, 'on');
xlabel(ax2, 'y (km)', 'FontSize', fs);
ylabel(ax2, '$\int \bar{u} \, dz$ (m$^2$/s)', 'FontSize', fs, 'Interpreter', 'latex');
legend(ax2, 'Location', 'best', 'Interpreter', 'latex', 'FontSize', fs);
set(ax2, 'FontSize', fs);
xlim(ax2, y_clip_km); % Clip spatial y-coordinate as requested
ylim(ax2, 'auto');     % Auto-scale velocity axis

% Removed titles for a cleaner look
% sgtitle(tl, ...)

drawnow;

% Save the figure
outdir = '/glade/work/nguyen/unc-shoals/animations';
if ~exist(outdir, 'dir'), outdir = pwd; end
outfile = fullfile(outdir, 'offshore_velocity_comparison.png');
exportgraphics(fig, outfile, 'Resolution', 300);
fprintf('Saved comparison plot: %s\n', outfile);

end

%% Helper: Extract U Profile
function [yc_km, u_profile] = get_u_profile(cluster_dir, sw, x_target_km)
    yc_km = [];
    u_profile = [];
    
    datadir = fullfile(cluster_dir, sw);
    fname = fullfile(datadir, ['time_avg_3d_' sw '.nc']);
    
    if ~isfile(fname)
        warning('File not found: %s', fname);
        return;
    end
    
    try
        % Load coordinates (expanded names for cluster 13 compatibility)
        xc = try_ncread_(fname, {'xC','xc','x','x_caa'});
        yc = try_ncread_(fname, {'yC','yc','y','y_aca'});
        zc = try_ncread_(fname, {'zC','zc','z','z_aac'});
        
        xc_km = xc / 1e3;
        yc_km = yc / 1e3;
        
        % Calculate dz
        if numel(zc) > 1
            dz = abs(zc(2) - zc(1));
        else
            dz = 1;
        end
        
        % Find nearest x index
        [~, ix] = min(abs(xc_km - x_target_km));
        
        % Load U velocity (time-averaged)
        % Dimensions order in MITgcm NC: (x, y, z, t)
        [u_raw, varname] = load_var_try_names(fname, {'u_c','u'});
        
        vinfo = ncinfo(fname, varname);
        vsize = vinfo.Size;
        nd = numel(vsize);
        
        % Extraction logic: 
        % 1. Time-averaged: We take the last available time step from the 
        %    time_avg_3d file, which represents the final averaging window.
        
        if nd == 4
            % (Nx, Ny, Nz, Nt) -> [1, Ny, Nz, 1]
            u_raw_slice = ncread(fname, varname, [ix 1 1 vsize(4)], [1 vsize(2) vsize(3) 1]);
            u_3d  = squeeze(u_raw_slice); % Result: [Ny, Nz]
            
            % Depth-integration: Sum over the Nz dimension * dz
            u_profile = sum(u_3d, 2, 'omitnan') * dz;
            
        elseif nd == 3
             % Check if 3rd dim is Nz or Nt
             if vsize(3) > 10 % Likely (Nx, Ny, Nz) - already time averaged
                 u_raw_slice = ncread(fname, varname, [ix 1 1], [1 vsize(2) vsize(3)]);
                 u_3d  = squeeze(u_raw_slice); % Result: [Ny, Nz]
                 u_profile = sum(u_3d, 2, 'omitnan') * dz;
             else % Likely (Nx, Ny, Nt) - 2D field, take last t
                 u_raw_slice = ncread(fname, varname, [ix 1 vsize(3)], [1 vsize(2) 1]);
                 u_profile = squeeze(u_raw_slice); % Result: [Ny, 1]
             end
        else
            error('Unexpected variable dimensions for %s', varname);
        end
        
        % Mask land (0) to NaN for plotting
        u_profile(u_profile == 0) = NaN;
        
    catch ME
        warning('Error reading %s: %s', sw, ME.message);
    end
end

%% Helper: load_var_try_names
function [data, matched_name] = load_var_try_names(fname, names)
    for i = 1:numel(names)
        try
            ncinfo(fname, names{i}); % Check existence
            data = ncread(fname, names{i});
            matched_name = names{i};
            return;
        catch
        end
    end
    error('Could not find variables (tried: %s) in %s', strjoin(names, ', '), fname);
end

%% Helper: try_ncread_
function data = try_ncread_(fname, names)
    for i = 1:numel(names)
        try
            data = ncread(fname, names{i});
            return;
        catch
        end
    end
    error('Could not find coordinates (tried: %s) in %s', strjoin(names, ', '), fname);
end
