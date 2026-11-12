function vis_offshore_3panel(cluster_name, varargin)
% vis_offshore_3panel(cluster_name, ...)
%
% Generates 5 figures (each 3 panels) for depth-averaged 
% offshore flux (uT or uS) across the 5 sweep comparisons.
%
% Spatial clip: x in [0, 60] km, y in [50, 150] km.

if nargin < 1 || isempty(cluster_name)
    cluster_name = 'cluster_Cnnie_1_15';
end

%% ---------- Options ----------
p = inputParser;
addParameter(p, 'dataroot', '/glade/work/nguyen/unc-shoals/sweep_output', @ischar);
addParameter(p, 'outdir',   '/glade/work/nguyen/unc-shoals/animations/', @ischar);
addParameter(p, 'var_type', 'uT', @ischar); % 'uT' or 'uS'
addParameter(p, 'clim',     [], @(v)isempty(v) || (isnumeric(v) && numel(v)==2));
addParameter(p, 'full_domain', false, @islogical);
parse(p, varargin{:});

dataroot = p.Results.dataroot;
outdir   = p.Results.outdir;
var_type = p.Results.var_type;
clim_in  = p.Results.clim;
full_domain = p.Results.full_domain;

if ~exist(outdir, 'dir'), mkdir(outdir); end

%% ---------- Define Sweep Groups ----------
group_Zs   = { 'Zs15_Ls40k', 'Zs10_Ls40k', 'Zs5_Ls40k' };
titles_Zs  = { 'Z_s = -15 m', 'Z_s = -10 m', 'Z_s = -5 m' };

group_Ls   = { 'Zs5_Ls20k', 'Zs5_Ls30k', 'Zs5_Ls40k' };
titles_Ls  = { 'L_s = 20 km', 'L_s = 30 km', 'L_s = 40 km' };

group_v0   = { 'v0_0p10', 'v0_0p15', 'Zs5_Ls40k' };
titles_v0  = { 'v_0 = 0.10 m/s', 'v_0 = 0.15 m/s', 'v_0 = 0.20 m/s' };

group_wind = { 'wind_m0p05', 'Zs5_Ls40k', 'wind_0p05' };
titles_wind= { '\tau = -0.05 N/m^2', '\tau = 0 N/m^2', '\tau = 0.05 N/m^2' };

group_shelf = { 'Zsh_m10', 'Zs5_Ls40k', 'Zsh_m50' };
titles_shelf = { 'H_{shelf} = -10 m', 'H_{shelf} = -25 m', 'H_{shelf} = -50 m' };

groups       = {group_Zs, group_Ls, group_v0, group_wind, group_shelf};
group_titles = {titles_Zs, titles_Ls, titles_v0, titles_wind, titles_shelf};
group_names  = {'Zs', 'Ls', 'v0', 'wind', 'Shelf'};

%% ---------- Generate Plots ----------
for g = 1:numel(groups)
    plot_3panel_group(groups{g}, group_names{g}, group_titles{g}, cluster_name, dataroot, outdir, var_type, clim_in, full_domain);
end

end

function plot_3panel_group(run_tags, group_name, panel_titles, cluster, dataroot, outdir, var_type, clim_in, full_domain)
    N = numel(run_tags);
    data_cells = cell(1, N);
    Xc_cells = cell(1, N);
    Yc_cells = cell(1, N);
    h_bottom_cells = cell(1, N);
    
    for i = 1:N
        run_tag = run_tags{i};
        if ~startsWith(run_tag, 'sweep_')
            run_tag = ['sweep_' run_tag];
        end
        datadir = fullfile(dataroot, cluster, run_tag);
        fname = fullfile(datadir, ['time_avg_3d_' run_tag '.nc']);
        
        if ~isfile(fname)
            % try to find actual folder with wildcard to handle numbering (e.g. sweep_001_Hs...)
            sw_core = strrep(run_tag, 'sweep_', '');
            d = dir(fullfile(dataroot, cluster, ['*' sw_core '*']));
            d = d([d.isdir] & ~ismember({d.name}, {'.', '..'}));
            if ~isempty(d)
                datadir = fullfile(dataroot, cluster, d(1).name);
                fname = fullfile(datadir, ['time_avg_3d_' run_tag '.nc']);
            end
        end

        if ~isfile(fname)
            warning('File not found: %s. Skipping.', fname);
            continue;
        end
        
        fprintf('Processing %s for %s (%s sweep)...\n', run_tag, upper(var_type), group_name);
        
        % Load Coordinates
        xc = try_ncread_(fname, {'xC','xc','x','x_caa'});
        yc = try_ncread_(fname, {'yC','yc','y','y_aca'});
        xc_km = xc / 1e3;
        yc_km = yc / 1e3;
        
        if full_domain
            x_bnds = [min(xc_km), max(xc_km)];
            y_bnds = [min(yc_km), max(yc_km)];
        else
            x_bnds = [0, 60];
            y_bnds = [50, 150];
        end
        
        % Mappings for clipping
        ix = find(xc_km >= x_bnds(1) & xc_km <= x_bnds(2));
        iy = find(yc_km >= y_bnds(1) & yc_km <= y_bnds(2));
        
        [Xc, Yc] = meshgrid(xc_km(ix), yc_km(iy));
        Xc_cells{i} = Xc;
        Yc_cells{i} = Yc;
        
        % Load Bathymetry
        fname_top = fullfile(datadir, ['top_' run_tag '.nc']);
        if isfile(fname_top)
            try
                h_bot = ncread(fname_top, 'bottom_height');
                if ndims(h_bot) > 2, h_bot = squeeze(h_bot(:,:,1,1)); end
                h_bottom_cells{i} = h_bot(ix, iy)';
            catch
                h_bottom_cells{i} = [];
            end
        else
            h_bottom_cells{i} = [];
        end
        
        % Load required variables
        if strcmpi(var_type, 'uT')
            uT_raw = load_last_3d_clip(fname, 'uT', ix, iy);
            if ~isempty(uT_raw)
                data_cells{i} = squeeze(mean(uT_raw, 3, 'omitnan'))';
            else
                % load fallback
                uT_zavg = load_last_processed_clip(fname, 'uT', ix, iy);
                if isempty(uT_zavg), uT_zavg = load_last_processed_clip(fname, 'uT_zavg', ix, iy); end
                data_cells{i} = uT_zavg';
            end
            var_label = '\langle uT \rangle_z  [°C·m/s]';
        elseif strcmpi(var_type, 'uS')
            uS_raw = load_last_3d_clip(fname, 'uS', ix, iy);
            if ~isempty(uS_raw)
                data_cells{i} = squeeze(mean(uS_raw, 3, 'omitnan'))';
            else
                uS_zavg = load_last_processed_clip(fname, 'uS', ix, iy);
                if isempty(uS_zavg), uS_zavg = load_last_processed_clip(fname, 'uS_zavg', ix, iy); end
                data_cells{i} = uS_zavg';
            end
            var_label = '\langle uS \rangle_z  [psu·m/s]';
        else
            error('Unknown var_type: %s. Use ''uT'' or ''uS''.', var_type);
        end
        
        % Mask Land
        if ~isempty(data_cells{i})
            data_cells{i}(data_cells{i} == 0) = NaN;
        end
    end
    
    %% ---------- Figure Setup ----------
    if full_domain
        fig = figure('Color','w','Position', [80 400 675 300]);
    else
        fig = figure('Color','w','Position', [80 400 675 300]);
    end
    tl = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    % Color limits: Centered symmetric
    if ~isempty(clim_in)
        clim_eff = clim_in;
    else
        all_max = 0;
        for i = 1:N
            if ~isempty(data_cells{i})
                all_max = max(all_max, max(abs(data_cells{i}(:)), [], 'omitnan'));
            end
        end
        if all_max == 0, all_max = 1e-4; end
        clim_eff = [-all_max, all_max];
    end
    
    % Plot Panels
    for i = 1:N
        ax = nexttile(tl);
        if isempty(data_cells{i}), continue; end
        
        pcolor(ax, Xc_cells{i}, Yc_cells{i}, data_cells{i});
        shading flat; axis tight equal; hold on;
        set(ax, 'YDir', 'normal');
        colormap(ax, cmocean('balance')); 
        caxis(ax, clim_eff);
        
        % Add bathymetry contours
        if ~isempty(h_bottom_cells{i})
            contour(ax, Xc_cells{i}, Yc_cells{i}, h_bottom_cells{i}, -60:5:-10, 'LineColor', 'k', 'LineWidth', 1.0, 'ShowText', 'off');
        end
        
        title(ax, panel_titles{i}, 'FontSize', 16, 'Interpreter', 'tex', 'FontWeight', 'bold');
        if i == 1
            ylabel(ax, 'y (km)', 'FontSize', 14);
        end
        xlabel(ax, 'x (km)', 'FontSize', 12);
        set(ax, 'FontSize', 12);
        xlim(ax, x_bnds);
        ylim(ax, y_bnds);
        grid on;
    end
    
    cb = colorbar;
    cb.Layout.Tile = 'east';
    ylabel(cb, var_label, 'FontSize', 16);
    
    drawnow;
    
    %% ---------- Save ----------
    if full_domain
        suffix = '_full';
    else
        suffix = '';
    end
    outfile = fullfile(outdir, [lower(var_type) '_3panel_sweep_' group_name suffix '.png']);
    print(fig, outfile, '-dpng', '-r200');
    fprintf('Saved 3-panel %s plot for %s: %s\n', upper(var_type), group_name, outfile);
end

%% ================== Helpers ==================

function data = load_last_3d_clip(fn, varname, ix, iy)
    try
        vinfo = ncinfo(fn, varname);
        vsize = vinfo.Size;
        data = ncread(fn, varname, [ix(1) iy(1) 1 vsize(4)], [numel(ix) numel(iy) vsize(3) 1]);
        data = squeeze(data);
    catch
        data = [];
    end
end

function data = load_last_processed_clip(fn, varname, ix, iy)
    try
        vinfo = ncinfo(fn, varname);
        vsize = vinfo.Size;
        if numel(vsize) == 3 % (Nx, Ny, Nt)
            data = ncread(fn, varname, [ix(1) iy(1) vsize(3)], [numel(ix) numel(iy) 1]);
        else
            data = ncread(fn, varname, [ix(1) iy(1)], [numel(ix) numel(iy)]);
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
    error('Coordinate variable not found.');
end
