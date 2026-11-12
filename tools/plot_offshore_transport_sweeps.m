function plot_offshore_transport_sweeps(cluster_name)
% plot_offshore_transport_sweeps(cluster_name)
%
% Generates two separate figures analyzing offshore transport at x = 50 km:
% 1. OFFSHORE TRANSPORT PROFILES: Line plots of integrated u-velocity (m^2/s) vs y.
% 2. TOTAL VOLUME TRANSPORT SUMMARY: Total flux (m^3/s) vs sweep parameters (Hs, Ls).
%
% Groups:
% 1. Effect of Hs: sweep_Hs05_Ls20k, sweep_Hs10_Ls20k, sweep_Hs15_Ls20k
% 2. Effect of Ls: sweep_Hs15_Ls20k, sweep_Hs15_Ls30k, sweep_Hs15_Ls40k

if nargin < 1 || isempty(cluster_name)
    cluster_name = 'cluster13_111_117';
end

sweep_output_root = '/glade/work/nguyen/unc-shoals/sweep_output';
cluster_dir       = fullfile(sweep_output_root, cluster_name);

% Define Sweep Groups
group1 = { ...
    'sweep_Hs05_Ls40k', ...
    'sweep_Hs10_Ls40k', ...
    'sweep_Hs15_Ls40k', ...
};

group2 = { ...
    'sweep_Hs15_Ls20k', ...
    'sweep_Hs15_Ls30k', ...
    'sweep_Hs15_Ls40k', ...
};

% Parameter values for summary plotting
Hs_vals = [5, 10, 15];
Ls_vals = [20, 30, 40];

% Custom Display Names for Legends (LaTeX format)
group1_names = {'$H_s = 5$ m', '$H_s = 10$ m', '$H_s = 15$ m'};
group2_names = {'$L_s = 20$ km', '$L_s = 30$ km', '$L_s = 40$ km'};

% Parameters
x_target_km = 50;
y_clip_km   = [50, 150];
fs = 18; 

% Colors (Consistent across plots)
colors = [
    0.0, 0.45, 0.74; % Blue
    0.85, 0.33, 0.10; % Orange
    0.93, 0.69, 0.13; % Yellow
];

%% Data Collection
fprintf('Collecting data for transport analysis...\n');

trans1_profiles = cell(size(group1));
trans1_totals   = zeros(size(group1));
yc1_km          = [];

for i = 1:numel(group1)
    [yc, p, t] = get_u_transport(cluster_dir, group1{i}, x_target_km, y_clip_km);
    trans1_profiles{i} = p;
    trans1_totals(i)   = t;
    if isempty(yc1_km), yc1_km = yc; end
end

trans2_profiles = cell(size(group2));
trans2_totals   = zeros(size(group2));
yc2_km          = [];

for i = 1:numel(group2)
    [yc, p, t] = get_u_transport(cluster_dir, group2{i}, x_target_km, y_clip_km);
    trans2_profiles{i} = p;
    trans2_totals(i)   = t;
    if isempty(yc2_km), yc2_km = yc; end
end

%% ========================================================================
%% FIGURE 1: Transport Profiles (m^2/s)
%% ========================================================================
fig1 = figure('Color','w', 'Position', [100 100 900 800]);
tl1 = tiledlayout(fig1, 2, 1, 'TileSpacing', 'Compact', 'Padding', 'Compact');

% Panel 1: Hs variations
ax1a = nexttile(tl1); hold(ax1a, 'on');
for i = 1:numel(group1)
    if ~isempty(trans1_profiles{i})
        plot(ax1a, yc1_km, trans1_profiles{i}, 'LineWidth', 3, 'Color', colors(i,:), ...
            'DisplayName', group1_names{i});
    end
end
grid(ax1a,'on'); box(ax1a,'on');
ylabel(ax1a, '$\int \bar{u} \, dz$ (m$^2$/s)', 'FontSize', fs, 'Interpreter', 'latex');
legend(ax1a, 'Location', 'best', 'Interpreter', 'latex', 'FontSize', fs-2);
set(ax1a, 'FontSize', fs); xlim(ax1a, y_clip_km);
title(ax1a, 'Offshore Transport Profiles vs $H_s$', 'FontSize', fs, 'Interpreter', 'latex');

% Panel 2: Ls variations
ax1b = nexttile(tl1); hold(ax1b, 'on');
for i = 1:numel(group2)
    if ~isempty(trans2_profiles{i})
        plot(ax1b, yc2_km, trans2_profiles{i}, 'LineWidth', 3, 'Color', colors(i,:), ...
            'DisplayName', group2_names{i});
    end
end
grid(ax1b,'on'); box(ax1b,'on');
xlabel(ax1b, 'y (km)', 'FontSize', fs);
ylabel(ax1b, '$\int \bar{u} \, dz$ (m$^2$/s)', 'FontSize', fs, 'Interpreter', 'latex');
legend(ax1b, 'Location', 'best', 'Interpreter', 'latex', 'FontSize', fs-2);
set(ax1b, 'FontSize', fs); xlim(ax1b, y_clip_km);
title(ax1b, 'Offshore Transport Profiles vs $L_s$', 'FontSize', fs, 'Interpreter', 'latex');

%% ========================================================================
%% FIGURE 2: Total Volume Transport Summary (m^3/s)
%% ========================================================================
fig2 = figure('Color','w', 'Position', [100 100 900 450]);
tl2 = tiledlayout(fig2, 1, 2, 'TileSpacing', 'Compact', 'Padding', 'Compact');

% Panel 1: Total vs Hs
ax2a = nexttile(tl2);
plot(ax2a, Hs_vals, trans1_totals, '-o', 'LineWidth', 3, 'MarkerSize', 10, ...
    'Color', colors(1,:), 'MarkerFaceColor', colors(1,:));
grid(ax2a, 'on'); box(ax2a, 'on');
xlabel(ax2a, 'Shoal Height, $H_s$ (m)', 'FontSize', fs, 'Interpreter', 'latex');
ylabel(ax2a, 'Total Transport (m$^3$/s)', 'FontSize', fs, 'Interpreter', 'latex');
set(ax2a, 'FontSize', fs, 'XTick', Hs_vals);
title(ax2a, 'Transport vs $H_s$', 'FontSize', fs, 'Interpreter', 'latex');
ymin1 = min(0, min(trans1_totals)*1.2);
ymax1 = max(0, max(trans1_totals)*1.2);
if ymax1 <= ymin1, ymax1 = ymin1 + 1; end
ylim(ax2a, [ymin1, ymax1]);

% Panel 2: Total vs Ls
ax2b = nexttile(tl2);
plot(ax2b, Ls_vals, trans2_totals, '-s', 'LineWidth', 3, 'MarkerSize', 10, ...
    'Color', colors(2,:), 'MarkerFaceColor', colors(2,:));
grid(ax2b, 'on'); box(ax2b, 'on');
xlabel(ax2b, 'Shelf Length, $L_s$ (km)', 'FontSize', fs, 'Interpreter', 'latex');
ylabel(ax2b, 'Total Transport (m$^3$/s)', 'FontSize', fs, 'Interpreter', 'latex');
set(ax2b, 'FontSize', fs, 'XTick', Ls_vals);
title(ax2b, 'Transport vs $L_s$', 'FontSize', fs, 'Interpreter', 'latex');
ymin2 = min(0, min(trans2_totals)*1.2);
ymax2 = max(0, max(trans2_totals)*1.2);
if ymax2 <= ymin2, ymax2 = ymin2 + 1; end
ylim(ax2b, [ymin2, ymax2]);

drawnow;

%% Saving
outdir = '/glade/work/nguyen/unc-shoals/animations';
if ~exist(outdir, 'dir'), outdir = pwd; end

outfile1 = fullfile(outdir, 'offshore_transport_profiles.png');
exportgraphics(fig1, outfile1, 'Resolution', 300);
fprintf('Saved Figure 1: %s\n', outfile1);

outfile2 = fullfile(outdir, 'offshore_transport_summary.png');
exportgraphics(fig2, outfile2, 'Resolution', 300);
fprintf('Saved Figure 2: %s\n', outfile2);

end

%% Helper: Extract U Transport (Integrated U and Total Volume Transport)
function [yc_km, u_transport, total_T] = get_u_transport(cluster_dir, sw, x_target_km, y_clip_km)
    yc_km = []; u_transport = []; total_T = 0;
    
    datadir = fullfile(cluster_dir, sw);
    fname = fullfile(datadir, ['time_avg_3d_' sw '.nc']);
    
    if ~isfile(fname), warning('File not found: %s', fname); return; end
    
    try
        % Load coordinates
        xc = try_ncread_(fname, {'xC','xc','x','x_caa'});
        yc = try_ncread_(fname, {'yC','yc','y','y_aca'});
        zc = try_ncread_(fname, {'zC','zc','z','z_aac'});
        
        xc_km = xc / 1e3; yc_km = yc / 1e3;
        
        % Spacings
        if numel(zc) > 1, dz = abs(zc(2) - zc(1)); else, dz = 1; end
        if numel(yc) > 1, dy = abs(yc(2) - yc(1)); else, dy = 1; end
        
        % Nearest x index
        [~, ix] = min(abs(xc_km - x_target_km));
        
        % Load U velocity
        [u_raw, varname] = load_var_try_names(fname, {'u_c','u'});
        vsize = ncinfo(fname, varname).Size;
        
        if numel(vsize) == 4
            u_raw_slice = ncread(fname, varname, [ix 1 1 vsize(4)], [1 vsize(2) vsize(3) 1]);
            u_3d  = squeeze(u_raw_slice);
            u_transport = sum(u_3d, 2, 'omitnan') * dz;
        elseif numel(vsize) == 3
             if vsize(3) > 10 % Likely (Nx, Ny, Nz)
                 u_raw_slice = ncread(fname, varname, [ix 1 1], [1 vsize(2) vsize(3)]);
                 u_3d  = squeeze(u_raw_slice); 
                 u_transport = sum(u_3d, 2, 'omitnan') * dz;
             else % Likely (Nx, Ny, Nt)
                 u_raw_slice = ncread(fname, varname, [ix 1 vsize(3)], [1 vsize(2) 1]);
                 u_transport = squeeze(u_raw_slice); 
             end
        end
        
        u_transport(u_transport == 0) = NaN;
        
        % Total Volume Transport (m^3/s) in clipped range
        iy = find(yc_km >= y_clip_km(1) & yc_km <= y_clip_km(2));
        total_T = sum(u_transport(iy), 'omitnan') * dy;
        
    catch ME
        warning('Error reading %s: %s', sw, ME.message);
    end
end

%% Helper: load_var_try_names
function [data, matched_name] = load_var_try_names(fname, names)
    for i = 1:numel(names)
        try
            ncinfo(fname, names{i}); 
            data = ncread(fname, names{i});
            matched_name = names{i};
            return;
        catch
        end
    end
    error('Could not find variables in %s', fname);
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
    error('Coordinate variable not found in %s', fname);
end
