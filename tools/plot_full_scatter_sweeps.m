function plot_full_scatter_sweeps(cluster_name, eke_mode)
% plot_full_scatter_sweeps(cluster_name, eke_mode)
%
% Creates a 5-panel figure showing volume transport (mSv) and
% mean Eddy Kinetic Energy (EKE) at x = x_target_km across parameter
% sweeps of shoal length (Ls), shoal height (Zs), inflow velocity (v0),
% shelf depth (Zsh), and wind stress (tau).
%
% Transport integration range: y = y_range_Q (top 25 m, u > 0 only).
% Runs that are missing or fail to load are plotted as gaps (NaN).

if nargin < 1 || isempty(cluster_name)
    cluster_name = 'cluster_Cnnie_1_15';
end
if nargin < 2 || isempty(eke_mode)
    eke_mode = 'domain'; % 'domain' or 'slice'
end

if strcmpi(eke_mode, 'slice')
    eke_label = 'Mean EKE (m^2 s^{-2})';
else
    eke_label = 'Mean EKE (m^2 s^{-2})';
end

sweep_output_root = '/glade/work/nguyen/unc-shoals/sweep_output';
cluster_dir       = fullfile(sweep_output_root, cluster_name);

% Define Sweep Groups and numerical values
group_Zs = { 'Zs15_Ls40k', 'Zs10_Ls40k', 'Zs5_Ls40k' };
Zs_vals = [-15, -10, -5]; % in meters

group_Ls = { 'Zs5_Ls20k', 'Zs5_Ls30k', 'Zs5_Ls40k' };
Ls_vals = [20, 30, 40]; % in km

group_v0 = { 'v0_0p10', 'v0_0p15', 'Zs5_Ls40k' };
v0_vals = [0.10, 0.15, 0.20]; % in m/s

group_Zsh = { 'Zsh_m10', 'Zs5_Ls40k', 'Zsh_m50' };
Zsh_vals = [-10, -25, -50]; % in meters (baseline Zsh = -25 m)

group_wind = { 'wind_m0p15', 'wind_m0p10', 'wind_m0p08', 'wind_m0p05', 'wind_m0p03', ...
               'Zs5_Ls40k', ...
               'wind_0p03', 'wind_0p05', 'wind_0p08', 'wind_0p10', 'wind_0p15' };
wind_vals = [-0.15, -0.10, -0.08, -0.05, -0.03, 0, 0.03, 0.05, 0.08, 0.10, 0.15]; % in N/m^2

% Parameters
x_target_km = 40; % where shelf and shoal connect. 
y_range_Q   = [80, 200];  % Upstream domain for volume transport
y_range_EKE = [80, 200]; % Downstream wake region for EKE
fs = 11; % Global font size

% Helper function to process a group
function [Q_out, EKE_out] = process_group(group_names)
    Q_out = zeros(size(group_names));
    EKE_out = zeros(size(group_names));
    for idx = 1:numel(group_names)
        [Q, EKE] = calculate_metrics(cluster_dir, group_names{idx}, x_target_km, y_range_Q, y_range_EKE, eke_mode);
        Q_out(idx) = Q;
        EKE_out(idx) = EKE;
    end
end

fprintf('Processing Ls sweep...\n');
[Q_Ls, EKE_Ls] = process_group(group_Ls);
fprintf('Processing Zs sweep...\n');
[Q_Zs, EKE_Zs] = process_group(group_Zs);
fprintf('Processing V0 sweep...\n');
[Q_v0, EKE_v0] = process_group(group_v0);
fprintf('Processing Zsh sweep...\n');
[Q_Zsh, EKE_Zsh] = process_group(group_Zsh);
fprintf('Processing Wind sweep...\n');
[Q_wind, EKE_wind] = process_group(group_wind);

% Convert Volume Transport to mSv (1 mSv = 10^3 m^3/s)
Q_Ls = Q_Ls / 1e3;
Q_Zs = Q_Zs / 1e3;
Q_v0 = Q_v0 / 1e3;
Q_Zsh = Q_Zsh / 1e3;
Q_wind = Q_wind / 1e3;

%% Plotting
fig = figure('Color','w', 'Position', [100 100 975 580]);
tl = tiledlayout(2, 3, 'TileSpacing', 'normal', 'Padding', 'compact');

col_q   = [0, 0.0, 0.7, 0.7];      % Blue for Total Transport Q
col_eke = [0.7, 0.0, 0.0, 0.7];    % Red for EKE

style_axis = @(ax) (set(ax, 'FontSize', fs, 'Box', 'on', 'XGrid', 'on', 'YGrid', 'on', ...
    'LineWidth', 1, 'TickDir', 'out'));

function plot_panel(ax, x_vals, Q_vals, EKE_vals, x_label)
    yyaxis(ax, 'left');
    hold(ax, 'on');
    plot(ax, x_vals, Q_vals, '-o', 'LineWidth', 1.5, 'Color', col_q, 'MarkerFaceColor', col_q(1:3), 'MarkerEdgeColor', 'none', 'MarkerSize', 5);
    ylabel(ax, 'Volume Transport (mSv)', 'FontSize', fs);
    ax.YAxis(1).Color = col_q(1:3);
    ax.YAxis(1).Exponent = 0;
    ylim(ax, get_padded_lims(Q_vals));
    
    yyaxis(ax, 'right');
    hold(ax, 'on');
    plot(ax, x_vals, EKE_vals, '-s', 'LineWidth', 1.5, 'Color', col_eke, 'MarkerFaceColor', col_eke(1:3), 'MarkerEdgeColor', 'none', 'MarkerSize', 5);
    ylabel(ax, eke_label, 'FontSize', fs);
    ax.YAxis(2).Color = col_eke(1:3);
    ylim(ax, get_padded_lims(EKE_vals));
    
    xlabel(ax, x_label, 'FontSize', fs);
    xlim(ax, get_padded_lims(x_vals));
    style_axis(ax);
    pbaspect(ax, [1 1 1]);
end

ax1 = nexttile(tl);
plot_panel(ax1, Ls_vals, Q_Ls, EKE_Ls, 'L_s (km)');
title(ax1, '(a) Shoal Length', 'FontSize', fs+2, 'FontWeight', 'bold');

ax2 = nexttile(tl);
plot_panel(ax2, Zs_vals, Q_Zs, EKE_Zs, 'Z_s (m)');
title(ax2, '(b) Shoal Height', 'FontSize', fs+2, 'FontWeight', 'bold');

ax3 = nexttile(tl);
plot_panel(ax3, v0_vals, Q_v0, EKE_v0, 'v_0 (m s^{-1})');
title(ax3, '(c) Inflow Velocity', 'FontSize', fs+2, 'FontWeight', 'bold');

ax4 = nexttile(tl);
plot_panel(ax4, Zsh_vals, Q_Zsh, EKE_Zsh, 'Z_{sh} (m)');
title(ax4, '(d) Shelf Depth', 'FontSize', fs+2, 'FontWeight', 'bold');

ax5 = nexttile(tl);
plot_panel(ax5, wind_vals, Q_wind, EKE_wind, '\tau (N m^{-2})');
title(ax5, '(e) Wind Stress', 'FontSize', fs+2, 'FontWeight', 'bold');

% Unified horizontal legend at the top
h_leg_q   = plot(ax1, NaN, NaN, '-o', 'LineWidth', 1.5, 'Color', col_q, 'MarkerSize', 6, 'MarkerFaceColor', col_q(1:3));
h_leg_eke = plot(ax1, NaN, NaN, '-s', 'LineWidth', 1.5, 'Color', col_eke, 'MarkerSize', 6, 'MarkerFaceColor', col_eke(1:3));

lgd = legend(ax1, [h_leg_q, h_leg_eke], {'Q (mSv)', eke_label}, ...
    'Orientation', 'horizontal');
lgd.Layout.Tile = 'north';

drawnow;

% Save
outdir = '/glade/work/nguyen/unc-shoals/animations';
if ~exist(outdir, 'dir'), outdir = pwd; end
outfile = fullfile(outdir, 'full_scatter_sweeps.png');
exportgraphics(fig, outfile, 'Resolution', 300);
fprintf('Saved scatter plot: %s\n', outfile);

end

%% Helper: Calculate Metrics
function [Q, EKE] = calculate_metrics(cluster_dir, sw, x_target_km, y_lim_Q, y_lim_EKE, eke_mode)
    Q = NaN; EKE = NaN;
    
    % Find actual folder with wildcard to handle sweep_001_... numbering
    sw_core = strrep(sw, 'sweep_', '');
    d = dir(fullfile(cluster_dir, ['*' sw_core '*']));
    d = d([d.isdir]);
    d = d(~ismember({d.name}, {'.', '..'}));
    if isempty(d)
        warning('Folder not found for %s', sw);
        return;
    end
    datadir = fullfile(cluster_dir, d(1).name);
    % Output files may or may not carry the 'sweep_' prefix
    % (e.g. time_avg_3d_sweep_Zs5_Ls40k.nc vs time_avg_3d_Zs5_Ls40k.nc)
    candidates = { fullfile(datadir, ['time_avg_3d_sweep_' sw_core '.nc']), ...
                   fullfile(datadir, ['time_avg_3d_' sw_core '.nc']) };
    fname = '';
    for c = 1:numel(candidates)
        if isfile(candidates{c}), fname = candidates{c}; break; end
    end
    if isempty(fname)
        found = dir(fullfile(datadir, 'time_avg_3d_*.nc'));
        if ~isempty(found)
            fname = fullfile(datadir, found(1).name);
        else
            nc_here = dir(fullfile(datadir, '*.nc'));
            warning('No time_avg_3d_*.nc in %s. NetCDF files present: %s', ...
                datadir, strjoin({nc_here.name}, ', '));
            return;
        end
    end
    fprintf('  %s -> %s\n', sw, fname);
    
    try
        % Load Coordinates
        xc = try_ncread_(fname, {'xC','xc','x','x_caa'});
        yc = try_ncread_(fname, {'yC','yc','y','y_aca'});
        zc = try_ncread_(fname, {'zC','zc','z','z_aac'}); zc = zc(:);
        
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
        
        % Load velocities
        u_2d = try_load_slice(fname, {'u_c', 'u'}, ix);
        v_2d = try_load_slice(fname, {'v_c', 'v'}, ix);
        w_2d = try_load_slice(fname, {'w_c', 'w'}, ix);
        
        % Load uu, vv, ww
        uu_2d = try_load_slice(fname, {'uu'}, ix);
        vv_2d = try_load_slice(fname, {'vv'}, ix);
        ww_2d = try_load_slice(fname, {'ww'}, ix);
        
        % Depth-integrated quantities (for transport Q)
        % Integrate over the upper 25 m to avoid cancellation from the deep return flow
        % Only calculate offshore flux using the values upstream when u > 0
        u_2d_pos = u_2d;
        u_2d_pos(u_2d_pos < 0) = 0;
        
        iz_upper = find(zc >= -25);
        if ~isempty(iz_upper)
            u_depth_int = sum(u_2d_pos(:, iz_upper), 2, 'omitnan') * dz;
        else
            u_depth_int = zeros(size(u_2d, 1), 1);
        end
        
        % Now integrate over y for Q
        Q = 0;
        iy_Q = find(yc_km >= y_lim_Q(1) & yc_km <= y_lim_Q(2));
        if ~isempty(iy_Q)
            Q = trapz(yc(iy_Q), u_depth_int(iy_Q));
        end
        
        % Calculate EKE at x = x_target_km
        v_2d = try_load_slice(fname, {'v_c', 'v'}, ix);
        w_2d = try_load_slice(fname, {'w_c', 'w'}, ix);
        uu_2d = try_load_slice(fname, {'uu'}, ix);
        vv_2d = try_load_slice(fname, {'vv'}, ix);
        ww_2d = try_load_slice(fname, {'ww'}, ix);
        
        if ~isempty(v_2d) && ~isempty(w_2d) && ~isempty(uu_2d) && ~isempty(vv_2d) && ~isempty(ww_2d)
            u_2d_masked = u_2d; u_2d_masked(u_2d_masked == 0) = NaN;
            v_2d_masked = v_2d; v_2d_masked(v_2d_masked == 0) = NaN;
            w_2d_masked = w_2d; w_2d_masked(w_2d_masked == 0) = NaN;
            
            uu_2d_masked = uu_2d; uu_2d_masked(uu_2d_masked == 0) = NaN;
            vv_2d_masked = vv_2d; vv_2d_masked(vv_2d_masked == 0) = NaN;
            ww_2d_masked = ww_2d; ww_2d_masked(ww_2d_masked == 0) = NaN;
            
            EKE_2d = 0.5 * (uu_2d_masked - u_2d_masked.^2 + ...
                            vv_2d_masked - v_2d_masked.^2 + ...
                            ww_2d_masked - w_2d_masked.^2);
                            
            if strcmpi(eke_mode, 'slice')
                iy_EKE = find(yc_km >= y_lim_EKE(1) & yc_km <= y_lim_EKE(2));
                if ~isempty(iy_EKE)
                    EKE = mean(EKE_2d(iy_EKE, :), 'all', 'omitnan');
                else
                    EKE = NaN;
                end
            else
                % Domain-wide: Mean EKE across full y extent (y >= 20) at x = x_target_km
                iy_EKE = find(yc_km >= 20);
                if ~isempty(iy_EKE)
                    EKE = mean(EKE_2d(iy_EKE, :), 'all', 'omitnan');
                else
                    EKE = NaN;
                end
            end
        else
            EKE = NaN;
            warning('Could not compute EKE for %s, returning NaN', sw);
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

%% Helper: Load slice at specific x index
function data = try_load_slice(fname, names, ix)
    for i = 1:numel(names)
        data = load_slice(fname, names{i}, ix);
        if ~isempty(data)
            return;
        end
    end
    data = [];
end

function data = load_slice(fname, varname, ix)
    try
        vinfo = ncinfo(fname, varname);
        vsize = vinfo.Size;
        nd = numel(vsize);
        if nd == 4
            data = ncread(fname, varname, [ix 1 1 vsize(4)], [1 vsize(2) vsize(3) 1]);
        elseif nd == 3
            if vsize(3) > 10
                data = ncread(fname, varname, [ix 1 1], [1 vsize(2) vsize(3)]);
            else
                data = ncread(fname, varname, [ix 1 vsize(3)], [1 vsize(2) 1]);
            end
        else
            data = ncread(fname, varname);
        end
        data = squeeze(data);
    catch
        data = [];
    end
end

%% Helper: Load 3D chunk
function data = try_load_3d(fname, names, start_x, start_y, count_x, count_y)
    for i = 1:numel(names)
        try
            vinfo = ncinfo(fname, names{i});
            vsize = vinfo.Size;
            nd = numel(vsize);
            if nd == 4
                data = ncread(fname, names{i}, [start_x start_y 1 1], [count_x count_y Inf 1]);
            elseif nd == 3
                if vsize(3) > 10
                    data = ncread(fname, names{i}, [start_x start_y 1], [count_x count_y Inf]);
                else
                    data = ncread(fname, names{i}, [start_x start_y vsize(3)], [count_x count_y 1]);
                end
            else
                data = ncread(fname, names{i});
            end
            data = squeeze(data);
            if ~isempty(data), return; end
        catch
        end
    end
    data = [];
end


%% Helper: get padded limits
function bnds = get_padded_lims(vals)
    v_min = min(vals(:), [], 'omitnan');
    v_max = max(vals(:), [], 'omitnan');
    if isempty(v_min) || isnan(v_min)
        v_min = 0;
        v_max = 1;
    end
    dv = (v_max - v_min) * 0.1;
    if dv == 0
        dv = max(1e-10, abs(v_max) * 0.05);
    end
    bnds = [v_min - dv, v_max + dv];
end
