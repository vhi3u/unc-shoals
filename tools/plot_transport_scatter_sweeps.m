function plot_transport_scatter_sweeps(cluster_name)
% plot_transport_scatter_sweeps(cluster_name)
%
% Creates a 4-panel figure showing offshore volume transport (m^3/s) 
% across parameter sweeps:
%
% Panels:
% (a) Shoal Length (Ls): 20, 30, 40 km
% (b) Shoal Depth (Zs): -15, -10, -5 m
% (c) Inflow Velocity (v0): 0.10, 0.15, 0.20 m/s
% (d) Alongshore Wind Stress (tau): -0.15 to +0.15 N/m^2
%
% Both Upstream (y=50-100 km) and Downstream (y=100-150 km) transports
% are plotted on each panel for direct comparison.
%
% Data is extracted at x = 60 km.
% Transport Q = integrate( u * H ) dy, where H is local depth.

if nargin < 1 || isempty(cluster_name)
    cluster_name = 'cluster13_111_117';
end

sweep_output_root = '/glade/work/nguyen/unc-shoals/sweep_output';
cluster_dir       = fullfile(sweep_output_root, cluster_name);

% Define Sweep Groups and numerical values
group_Ls = { ...
    'sweep_Zs5_Ls20k', ...
    'sweep_Zs5_Ls30k', ...
    'sweep_Zs5_Ls40k', ...
};
Ls_vals = [20, 30, 40]; % in km

group_Zs = { ...
    'sweep_Zs15_Ls40k', ...
    'sweep_Zs10_Ls40k', ...
    'sweep_Zs5_Ls40k', ...
};
Zs_vals = [-15, -10, -5]; % in meters

group_wind = { ...
    'sweep_wind_m0p15', ...
    'sweep_wind_m0p10', ...
    'sweep_wind_m0p05', ...
    'sweep_Zs5_Ls40k', ...
    'sweep_wind_0p05', ...
    'sweep_wind_0p10', ...
    'sweep_wind_0p15', ...
};
wind_vals = [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]; % in N/m^2

group_v0 = { ...
    'sweep_v0_0p10', ...
    'sweep_v0_0p15', ...
    'sweep_Zs5_Ls40k', ...
};
v0_vals = [0.10, 0.15, 0.20]; % in m/s (sweep_Zs5_Ls40k has baseline v0 = 0.20)

% Parameters
x_target_km = 60;
y_upstream   = [50, 100];
y_downstream = [100, 150];
fs = 13; % Global font size

% Pre-allocate storage
Q_up_Ls   = zeros(size(Ls_vals));
Q_down_Ls = zeros(size(Ls_vals));
Q_up_Zs   = zeros(size(Zs_vals));
Q_down_Zs = zeros(size(Zs_vals));
Q_up_wind   = zeros(size(wind_vals));
Q_down_wind = zeros(size(wind_vals));
Q_up_v0   = zeros(size(v0_vals));
Q_down_v0 = zeros(size(v0_vals));

%% Helper function to process group
function [Q_up_out, Q_down_out] = process_group(group_names)
    Q_up_out = zeros(size(group_names));
    Q_down_out = zeros(size(group_names));
    for idx = 1:numel(group_names)
        [Q_u, Q_d] = calculate_transport(cluster_dir, group_names{idx}, x_target_km, y_upstream, y_downstream);
        Q_up_out(idx)   = Q_u;
        Q_down_out(idx) = Q_d;
    end
end

fprintf('Processing Ls sweep...\n');
[Q_up_Ls, Q_down_Ls] = process_group(group_Ls);

fprintf('Processing Zs sweep...\n');
[Q_up_Zs, Q_down_Zs] = process_group(group_Zs);

fprintf('Processing Wind sweep...\n');
[Q_up_wind, Q_down_wind] = process_group(group_wind);

fprintf('Processing v0 sweep...\n');
[Q_up_v0, Q_down_v0] = process_group(group_v0);

%% Plotting (2x4 layout or 4 panels with upstream & downstream)
fig = figure('Color','w', 'Position', [100 100 1000 850]);
tl = tiledlayout(2, 2, 'TileSpacing', 'loose', 'Padding', 'compact');

% Colors
col_up   = [0.85, 0.325, 0.098]; % Reddish for Upstream
col_down = [0, 0.447, 0.741];    % Blue for Downstream

function plot_param_panel(ax, x_vals, Q_u, Q_d, x_lbl, title_str)
    hold(ax, 'on');
    h_u = plot(ax, x_vals, Q_u, '-o', 'LineWidth', 2, 'Color', col_up, ...
        'MarkerSize', 8, 'MarkerFaceColor', col_up, 'MarkerEdgeColor', 'k', ...
        'DisplayName', 'Upstream ($y=50-100$ km)');
    h_d = plot(ax, x_vals, Q_d, '-s', 'LineWidth', 2, 'Color', col_down, ...
        'MarkerSize', 8, 'MarkerFaceColor', col_down, 'MarkerEdgeColor', 'k', ...
        'DisplayName', 'Downstream ($y=100-150$ km)');
    t = title(ax, title_str, 'FontSize', fs, 'Interpreter', 'latex');
    t.Units = 'normalized'; t.Position(2) = 1.05;
    ylabel(ax, 'Transport (m$^3$/s)', 'FontSize', fs, 'Interpreter', 'latex');
    xlabel(ax, x_lbl, 'FontSize', fs, 'Interpreter', 'latex');
    grid(ax, 'on'); box(ax, 'on'); set(ax, 'FontSize', fs);
    ax.YAxis.Exponent = 4;
    pbaspect(ax, [1 1 1]);
end

% Panel 1: Ls
ax1 = nexttile(tl);
plot_param_panel(ax1, Ls_vals, Q_up_Ls, Q_down_Ls, 'Shoal Length $L_s$ (km)', '(a) Shoal Length');

% Panel 2: Zs
ax2 = nexttile(tl);
plot_param_panel(ax2, Zs_vals, Q_up_Zs, Q_down_Zs, 'Shoal Peak Depth $Z_s$ (m)', '(b) Shoal Height / Depth');

% Panel 3: Inflow velocity v0
ax3 = nexttile(tl);
plot_param_panel(ax3, v0_vals, Q_up_v0, Q_down_v0, 'Inflow Velocity $v_0$ (m/s)', '(c) Inflow Velocity');

% Panel 4: Wind Stress
ax4 = nexttile(tl);
plot_param_panel(ax4, wind_vals, Q_up_wind, Q_down_wind, 'Alongshore Wind Stress $\tau_y$ (N/m$^2$)', '(d) Wind Stress');

% Unified Legend
lgd = legend(ax1, 'Orientation', 'horizontal', 'Interpreter', 'latex', 'FontSize', fs);
lgd.Layout.Tile = 'north';

drawnow;

% Save
outdir = '/glade/work/nguyen/unc-shoals/animations';
if ~exist(outdir, 'dir'), outdir = pwd; end
outfile = fullfile(outdir, 'transport_scatter_comparison.png');
exportgraphics(fig, outfile, 'Resolution', 300);
fprintf('Saved scatter plot: %s\n', outfile);

end

%% Helper: Calculate Transport
function [Q_up, Q_down] = calculate_transport(cluster_dir, sw, x_target_km, y_up_lim, y_down_lim)
    Q_up = 0; Q_down = 0;
    
    % Find actual folder with wildcard to handle sweep_001_... numbering
    sw_core = strrep(sw, 'sweep_', '');
    d = dir(fullfile(cluster_dir, ['*' sw_core '*']));
    d = d([d.isdir]);
    d = d(~ismember({d.name}, {'.', '..'}));
    if isempty(d)
        warning('Folder not found matching *%s*', sw);
        return;
    end
    datadir = fullfile(cluster_dir, d(1).name);
    fname = fullfile(datadir, ['time_avg_3d_' sw '.nc']);
    
    if ~isfile(fname)
        warning('File not found: %s', fname);
        return;
    end
    
    try
        % Load Coordinates
        xc = try_ncread_(fname, {'xC','xc','x','x_caa'});
        yc = try_ncread_(fname, {'yC','yc','y','y_aca'});
        zc = try_ncread_(fname, {'zC','zc','z','z_aac'});
        
        xc_km = xc / 1e3;
        yc_km = yc / 1e3;
        
        % Calculate dz (assume uniform spacing)
        if numel(zc) > 1
            dz = abs(zc(1) - zc(2));
        else
            dz = 1; % Fallback for single layer
        end
        
        % Find nearest x index
        [~, ix] = min(abs(xc_km - x_target_km));
        
        % Check if variable names are u or u_c
        try
            vinfo = ncinfo(fname, 'u_c');
            varname = 'u_c';
        catch
            varname = 'u';
        end
        
        vinfo = ncinfo(fname, varname);
        vsize = vinfo.Size;
        
        % Load U velocity (time-averaged, last timestep)
        % Dimensions: (Nx, Ny, Nz, Nt)
        u_slice = ncread(fname, varname, [ix 1 1 vsize(4)], [1 vsize(2) vsize(3) 1]);
        u_2d = squeeze(u_slice); % (Ny, Nz)
        
        % Load bathymetry or estimate depth
        try
            bot_z = ncread(fname, 'bottom_height', [ix 1], [1 vsize(2)]);
            H = abs(squeeze(bot_z));
        catch
            % Use mask-derived depth if bottom_height not available
            mask = ~isnan(u_2d) & (u_2d ~= 0);
            H = sum(mask, 2) * dz;
        end
        
        % Transport Calculation: Q = integral(integral(u dz) dy)
        % First, depth-integrated transport (m^2/s)
        u_depth_int = sum(u_2d, 2, 'omitnan') * dz;
        
        % Now integrate over y
        % Upstream y mask
        iy_up = find(yc_km >= y_up_lim(1) & yc_km <= y_up_lim(2));
        if ~isempty(iy_up)
            Q_up = trapz(yc(iy_up), u_depth_int(iy_up));
        end
        
        % Downstream y mask
        iy_down = find(yc_km >= y_down_lim(1) & yc_km <= y_down_lim(2));
        if ~isempty(iy_down)
            Q_down = trapz(yc(iy_down), u_depth_int(iy_down));
        end
        
    catch ME
        warning('Error processing %s: %s', sw, ME.message);
    end
end

%% Helper: Robust ncread
function data = try_ncread_(fname, names)
    for i = 1:numel(names)
        try
            data = ncread(fname, names{i});
            return;
        catch
        end
    end
    error('Could not find variable in %s', fname);
end
