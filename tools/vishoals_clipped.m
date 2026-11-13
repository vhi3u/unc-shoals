function vishoals_clipped(run_tag, varargin)
% vishoals(run_tag, 'outdir', OUTDIR, 'fps', FPS, 'frame_skip', N, ...
%   'mp4', true/false, ...
%   'isopycnals', true/false, 'rho_levels', 24:0.2:26, 'iso_color','k', ...
%   'iso_linewidth', 0.8, 'clabel_every', 10, ...
%   'climT',[Tmin Tmax],'climS',[Smin Smax],'climv',[vmin vmax],'climRo',[lo hi], ...
%   'datadir', '/path/to/nc/files')
%
% Reads from two pre-sliced NetCDF files produced by flow_over_shoals_hpc.jl:
%   top_<run_tag>.nc   — surface XY slice (u_c, v_c, w_c, T, S, Ro)
%   midy_<run_tag>.nc  — mid-y  XZ slice (u_c, v_c, w_c, T, S, Ro)
%
% Top row (XY @ surface):   T, S, v, Ro
% Bottom row (XZ @ y~mid):  T, S, v, Ro  + optional σ0 isopycnals
%
% Requires: cmocean, GSW Toolbox (for isopycnals)

% ---------- options ----------
p = inputParser;
addParameter(p, 'datadir', '', @ischar);
addParameter(p, 'outdir', '', @ischar);
addParameter(p, 'fps', 5, @(x)isnumeric(x)&&isscalar(x)&&x>0);
addParameter(p, 'frame_skip', 1, @(x)isnumeric(x)&&isscalar(x)&&x>=1);
addParameter(p, 'mp4', false, @(x)islogical(x)&&isscalar(x));
% Isopycnal options
addParameter(p, 'isopycnals', true, @(x)islogical(x)&&isscalar(x));
addParameter(p, 'rho_levels', [23:0.4:24.0, 24.1:0.4:26.4, 26.6:0.8:28], @(v)isnumeric(v)&&isvector(v));
addParameter(p, 'iso_color', 'k', @ischar);
addParameter(p, 'iso_linewidth', 0.8, @(x)isnumeric(x)&&isscalar(x)&&x>0);
addParameter(p, 'clabel_every', 10, @(x)isnumeric(x)&&isscalar(x)&&x>=1);
% Velocity component
addParameter(p, 'vel', 'v', @(x)ischar(x) && ismember(x,{'u','v','Ro'}));
% Kinetic Energy/Vorticity component
addParameter(p, 'vort', 'KE', @(x)ischar(x) && ismember(x,{'Ro','PV','KE'}));
% Manual color limits (optional)
addParameter(p, 'climT', [], @(v)isnumeric(v) && numel(v)==2);
addParameter(p, 'climS', [], @(v)isnumeric(v) && numel(v)==2);
addParameter(p, 'climv', [], @(v)isnumeric(v) && numel(v)==2);
addParameter(p, 'climRo', [], @(v)isnumeric(v) && numel(v)==2);
addParameter(p, 'scan_skip_days', 10, @(x)isnumeric(x) && isscalar(x) && x>=0);
addParameter(p, 'last_days', Inf, @(x)isnumeric(x)&&isscalar(x)&&x>0);

parse(p, varargin{:});
fps          = p.Results.fps;
frame_skip   = p.Results.frame_skip;
want_mp4     = p.Results.mp4;
want_iso     = p.Results.isopycnals;
rho_levels   = p.Results.rho_levels;
iso_color    = p.Results.iso_color;
iso_lw       = p.Results.iso_linewidth;
clabel_every = p.Results.clabel_every;
climT_user   = p.Results.climT;
climS_user   = p.Results.climS;
climv_user   = p.Results.climv;
climRo_user  = p.Results.climRo;
vel_comp     = p.Results.vel;
vort_comp    = p.Results.vort;
scan_skip    = p.Results.scan_skip_days;
last_days    = p.Results.last_days;

% Velocity field name and labels
if strcmp(vel_comp, 'u')
    vel_ncvar = 'u_c';
    vel_label_xy = 'U Velocity';
    vel_label_xz = 'U Velocity';
    vel_cb_label = '(m s^{-1})';
elseif strcmp(vel_comp, 'Ro')
    vel_ncvar = 'Ro';
    vel_label_xy = 'Vorticity/f';
    vel_label_xz = 'Vorticity/f';
    vel_cb_label = '(Ro)';
else
    vel_ncvar = 'v_c';
    vel_label_xy = 'V Velocity';
    vel_label_xz = 'V Velocity';
    vel_cb_label = '(m s^{-1})';
end

% Vorticity/KE field name and labels
if strcmp(vort_comp, 'PV')
    vort_ncvar = 'PV';
    vort_label_xy = 'Ertel PV';
    vort_label_xz = 'Ertel PV';
    vort_cb_label = '(s^{-3})';
elseif strcmp(vort_comp, 'KE')
    vort_ncvar = 'KE';
    vort_label_xy = 'Kinetic Energy';
    vort_label_xz = 'Kinetic Energy';
    vort_cb_label = '(m^2 s^{-2})';
else
    vort_ncvar = 'Ro';
    vort_label_xy = 'Vorticity/f';
    vort_label_xz = 'Vorticity/f';
    vort_cb_label = '(Ro)';
end

% ---------- build filenames ----------
datadir = p.Results.datadir;
if isempty(datadir), datadir = pwd; end

fname_top  = fullfile(datadir, ['top_' run_tag '.nc']);
fname_midy = fullfile(datadir, ['midy_' run_tag '.nc']);

if ~isfile(fname_top),  error('Surface file not found: %s', fname_top); end
if ~isfile(fname_midy), error('Mid-y file not found: %s', fname_midy); end

fprintf('Reading surface XY from: %s\n', fname_top);
fprintf('Reading XZ transect from: %s\n', fname_midy);

% ---------- detect coordinate names ----------
% Oceananigans NetCDFWriter uses xC/yC/zC for Center-located fields.
% Fall back to other names if needed.
xc_top  = try_ncread_(fname_top, {'xC','x_caa','xc'});
yc_top  = try_ncread_(fname_top, {'yC','y_aca','yc'});
xc_midy = try_ncread_(fname_midy, {'xC','x_caa','xc'});
zc_midy = try_ncread_(fname_midy, {'zC','z_aac','zc'});
time    = ncread(fname_top, 'time');
Nt      = numel(time);

% Calculate start index for scanning scan_skip_days
t_days = (time(1:Nt) - time(1)) / 86400;
total_days = t_days(end);
days_seconds = scan_skip * 86400;
t_rel = time - time(1);
scan_start_idx = find(t_rel >= days_seconds, 1, 'first');
if isempty(scan_start_idx), scan_start_idx = 1; end

% Determine animation start from last_days
if isfinite(last_days)
    start_day  = max(0, total_days - last_days);
    anim_start = find(t_days >= start_day, 1, 'first');
    if isempty(anim_start), anim_start = 1; end
else
    anim_start = 1;
end

% Also check midy has same time
time_midy = ncread(fname_midy, 'time');
Nt_midy   = numel(time_midy);
if Nt ~= Nt_midy
    warning('top and midy files have different number of timesteps (%d vs %d). Using min.', Nt, Nt_midy);
    Nt = min(Nt, Nt_midy);
end

% Dynamic clipping bounds
Ly_km     = yc_top(end) / 1000;
y_clip_km = [Ly_km/2 - 50, Ly_km/2 + 50];
x_clip_km = [0, 50]; % x is [0, 50] km

ix_top = find((xc_top/1000) >= x_clip_km(1) & (xc_top/1000) <= x_clip_km(2));
iy_top = find((yc_top/1000) >= y_clip_km(1) & (yc_top/1000) <= y_clip_km(2));
ix_midy = find((xc_midy/1000) >= x_clip_km(1) & (xc_midy/1000) <= x_clip_km(2));

nx_top_off = ix_top(1);  nx_top_cnt = numel(ix_top);
ny_top_off = iy_top(1);  ny_top_cnt = numel(iy_top);
nx_midy_off = ix_midy(1); nx_midy_cnt = numel(ix_midy);
nz_xz = numel(zc_midy);

Nx_top = nx_top_cnt; Ny_top = ny_top_cnt;
Nx_xz  = nx_midy_cnt; Nz_xz = nz_xz;

xc_km_top  = xc_top(ix_top) / 1000;
yc_km_top  = yc_top(iy_top) / 1000;
xc_km_midy = xc_midy(ix_midy) / 1000;

% XY grids in km
[Xc, Yc] = meshgrid(xc_km_top, yc_km_top);
% XZ grids: x in km, z in meters
[Xz, Zz] = meshgrid(xc_km_midy, zc_midy);

% ---------- TEOS-10 prep for isopycnals ----------
lat = 35.2480;
lon = 0;
p_db = gsw_p_from_z(zc_midy(:), lat);  % [Nz×1] dbar
P    = repmat(p_db, 1, Nx_xz);         % [Nz×Nx] for XZ
% For surface: use z=0 (surface)
p_surf = gsw_p_from_z(0, lat);
P_xy   = p_surf * ones(Ny_top, Nx_top);

% ---------- readers ----------
% Surface file: data is (Nx, Ny, 1, time)
read_surf = @(varname, t) squeeze(ncread(fname_top, varname, ...
    [nx_top_off ny_top_off 1 t], [nx_top_cnt ny_top_cnt 1 1]));

% XZ file: data is (Nx, 1, Nz, time)
read_xz = @(varname, t) squeeze(ncread(fname_midy, varname, ...
    [nx_midy_off 1 1 t], [nx_midy_cnt 1 Inf 1]));

    function A = surf_tracer(varname, t)
        A = read_surf(varname, t);
        A(A==0) = NaN;  % mask land/immersed cells
    end

    function A = xz_tracer(varname, t)
        A = read_xz(varname, t);
        A(A==0) = NaN;
    end

% ---------- scan color limits ----------
getmin = @(x) min(x(:), [], 'omitnan');
getmax = @(x) max(x(:), [], 'omitnan');

mins = struct('Txy', inf, 'Sxy', inf, 'vxy', inf, 'Roxy', inf, ...
              'Txz', inf, 'Sxz', inf, 'vxz', inf, 'Roxz', inf);
maxs = struct('Txy',-inf, 'Sxy',-inf, 'vxy',-inf, 'Roxy',-inf, ...
              'Txz',-inf, 'Sxz',-inf, 'vxz',-inf, 'Roxz',-inf);

need_auto_T  = isempty(climT_user);
need_auto_S  = isempty(climS_user);
need_auto_v  = isempty(climv_user);
need_auto_Ro = isempty(climRo_user);

if need_auto_T || need_auto_S || need_auto_v || need_auto_Ro
    % When last_days is active, scan only the animation window
    if isfinite(last_days)
        scan_from = anim_start;
    else
        scan_from = scan_start_idx;
    end
    fprintf('Scanning frames %d to %d for color limits...\n', scan_from, Nt);
    for t = scan_from:frame_skip:Nt
        if need_auto_T
            Txy = surf_tracer('T', t); Txz = xz_tracer('T', t);
            mins.Txy = min(mins.Txy, getmin(Txy)); maxs.Txy = max(maxs.Txy, getmax(Txy));
            mins.Txz = min(mins.Txz, getmin(Txz)); maxs.Txz = max(maxs.Txz, getmax(Txz));
        end
        if need_auto_S
            Sxy = surf_tracer('S', t); Sxz = xz_tracer('S', t);
            mins.Sxy = min(mins.Sxy, getmin(Sxy)); maxs.Sxy = max(maxs.Sxy, getmax(Sxy));
            mins.Sxz = min(mins.Sxz, getmin(Sxz)); maxs.Sxz = max(maxs.Sxz, getmax(Sxz));
        end
        if need_auto_v
            vxy = read_surf(vel_ncvar, t); vxz = read_xz(vel_ncvar, t);
            mins.vxy = min(mins.vxy, getmin(vxy)); maxs.vxy = max(maxs.vxy, getmax(vxy));
            mins.vxz = min(mins.vxz, getmin(vxz)); maxs.vxz = max(maxs.vxz, getmax(vxz));
        end
        if need_auto_Ro
            rxy = read_surf(vort_ncvar, t); rxz = read_xz(vort_ncvar, t);
            mins.Roxy = min(mins.Roxy, getmin(rxy)); maxs.Roxy = max(maxs.Roxy, getmax(rxy));
            mins.Roxz = min(mins.Roxz, getmin(rxz)); maxs.Roxz = max(maxs.Roxz, getmax(rxz));
        end
    end
end

pad = @(lo,hi) [lo - 0.02*(hi-lo+eps), hi + 0.02*(hi-lo+eps)];
if isempty(climT_user)
    climT_xy = pad(mins.Txy, maxs.Txy);
    climT_xz = pad(mins.Txz, maxs.Txz);
else
    climT_xy = climT_user; climT_xz = climT_user;
end
if isempty(climS_user)
    climS_xy = pad(mins.Sxy, maxs.Sxy);
    climS_xz = pad(mins.Sxz, maxs.Sxz);
else
    climS_xy = climS_user; climS_xz = climS_user;
end
if isempty(climv_user)
    vm_xy = max(abs([mins.vxy, maxs.vxy])); climv_xy = [-vm_xy, vm_xy];
    vm_xz = max(abs([mins.vxz, maxs.vxz])); climv_xz = [-vm_xz, vm_xz];
else
    climv_xy = climv_user; climv_xz = climv_user;
end
if isempty(climRo_user)
    if strcmp(vort_comp, 'KE')
        climRo_xy = [0, maxs.Roxy];
        climRo_xz = [0, maxs.Roxz];
    else
        rm_xy = max(abs([mins.Roxy, maxs.Roxy])); climRo_xy = [-rm_xy, rm_xy];
        rm_xz = max(abs([mins.Roxz, maxs.Roxz])); climRo_xz = [-rm_xz, rm_xz];
    end
else
    climRo_xy = climRo_user; climRo_xz = climRo_user;
end

% ---------- figure / panels ----------
fig = figure('Color','w','Position',[80 50 900 700]);
tl  = tiledlayout(fig, 3, 4, 'TileSpacing','compact','Padding','compact');

    function [ax,h,cb] = panel_xy(tileIdx, titleStr, clim, cmap, cbLabel, isFirstCol)
        ax = nexttile(tl, tileIdx, [2 1]);
        h  = pcolor(ax, Xc, Yc, nan(size(Xc)));
        shading(ax,'flat'); axis(ax,'tight'); set(ax,'YDir','normal');
        set(ax, 'Layer', 'top'); box(ax, 'on'); set(ax, 'LineWidth', 0.8, 'XColor', 'k', 'YColor', 'k');
        shading interp
        colormap(ax, cmap); caxis(ax, clim);
        cb = colorbar(ax);
        cb.FontSize = 14; cb.Title.String = cbLabel; cb.Title.FontSize = 16; ;
        title(ax, titleStr, 'FontSize', 18, 'FontWeight', 'bold');
        if isFirstCol
            ylabel(ax, 'y (km)', 'FontSize', 16, 'FontWeight', 'bold');
        else
            yticklabels(ax, []);
        end
        xticklabels(ax, []);
        set(ax, 'FontSize', 14);
    end

    function [ax,h,cb] = panel_xz(tileIdx, clim, cmap, cbLabel, isFirstCol)
        ax = nexttile(tl, tileIdx, [1 1]);
        h  = pcolor(ax, Xz, Zz, nan(size(Xz)));
        shading(ax,'flat'); set(ax,'YDir','normal'); axis(ax,'tight');
        set(ax, 'Layer','top'); set(ax, 'NextPlot','add');
        box(ax, 'on'); set(ax, 'LineWidth', 0.8, 'XColor', 'k', 'YColor', 'k');
        set(ax, 'Color', [0.9 0.9 0.9]); % Make NaN immersed cells gray
        colormap(ax, cmap); caxis(ax, clim);
        cb = colorbar(ax);
        cb.FontSize = 14; cb.Title.String = cbLabel; cb.Title.FontSize = 16;
        xlabel(ax, 'x (km)', 'FontSize', 16, 'FontWeight', 'bold');
        if isFirstCol
            ylabel(ax, 'z (m)', 'FontSize', 16, 'FontWeight', 'bold');
        else
            yticklabels(ax, []);
        end
        set(ax, 'FontSize', 14);
    end

% Top rows: surface XY (each spans 2 rows)
[ax1,h1,cb1] = panel_xy(1, 'Temperature',  climT_xy,  cmocean('thermal'), '(°C)', true);
[ax2,h2,cb2] = panel_xy(2, 'Salinity',     climS_xy,  cmocean('haline'),  '(psu)', false);
[ax3,h3,cb3] = panel_xy(3, vel_label_xy,   climv_xy,  cmocean('balance'), vel_cb_label, false);
if strcmp(vort_comp, 'KE')
    vort_cmap = pmkmp(100,'swtth');
    % vort_cmap = jet;
else
    vort_cmap = cmocean('balance');
end
[ax4,h4,cb4] = panel_xy(4, vort_label_xy,  climRo_xy, vort_cmap, vort_cb_label, false);

% Bottom row: XZ transect
[ax5,h5,cb5] = panel_xz(9,  climT_xz,  cmocean('thermal'), '(°C)', true);
[ax6,h6,cb6] = panel_xz(10, climS_xz,  cmocean('haline'), '(psu)', false);
[ax7,h7,cb7] = panel_xz(11, climv_xz,  cmocean('balance'), vel_cb_label, false);
[ax8,h8,cb8] = panel_xz(12, climRo_xz, vort_cmap, vort_cb_label, false);

drawnow;
axs = [ax1, ax2, ax3, ax4];
cbs = [cb1, cb2, cb3, cb4];
for i = 1:4
    axP = axs(i).Position;
    % =========================================================================
    % 🟢 EDIT COLORBAR POSITIONS FOR TOP PANELS (XY) HERE
    %
    % axP(1) = left edge of panel     axP(3) = width of panel
    % axP(2) = bottom edge of panel   axP(4) = height of panel
    % =========================================================================
    cw = 0.02; % Colorbar width
    ch = axP(4) * 0.18; % Colorbar height
    
    if i == 1 || i == 2
        % Adjust T and S (Panels 1 & 2). 
        % Decrease the '+ 0.005' to move LEFT. Increase to move RIGHT.
        xpos = axP(1) + axP(3) - cw + 0.01; 
    else
        % Adjust U Vel and KE (Panels 3 & 4).
        % Decrease the '- 0.01' to move further LEFT.
        xpos = axP(1) + axP(3) - cw + 0.0; 
    end
    
    % Adjust vertical position (all 4 panels). 
    % Decrease '- 0.08' to move UP. Increase to move DOWN.
    ypos = axP(2) + axP(4) - ch - 0.04; 
    cbs(i).Position = [xpos, ypos, cw, ch];
    cbs(i).YAxisLocation = 'left'; % tick labels inside to the left
    cbs(i).FontSize = 11;
    cbs(i).Title.FontSize = 11;
    
    if i == 1 || i == 2
        % Calculate exactly 3 ticks rounded to nearest 0.5
        cmin = cbs(i).Limits(1); cmax = cbs(i).Limits(2);
        step = round(((cmax - cmin) / 2) * 2) / 2; 
        if step < 0.5, step = 0.5; end
        mid = round((cmin + cmax)/2 * 2) / 2;
        ticks = [mid - step, mid, mid + step];
        
        % Expand limits if necessary so all 3 ticks are visible
        caxis(axs(i), [min(cmin, ticks(1)), max(cmax, ticks(3))]);
        cbs(i).Ticks = ticks;
    end
end

axs2 = [ax5, ax6, ax7, ax8];
cbs2 = [cb5, cb6, cb7, cb8];
for i = 1:4
    axP = axs2(i).Position;
    % =========================================================================
    % 🟢 EDIT COLORBAR POSITIONS FOR BOTTOM PANELS (XZ) HERE
    % =========================================================================
    cw = 0.015; % Colorbar width
    ch = axP(4) * 0.35; % Colorbar height
    
    if i == 1 || i == 2
        % Adjust T and S (Panels 1 & 2).
        xpos = axP(1) + 0.01; 
    elseif i == 3
        % Adjust U Vel (Panel 3).
        xpos = axP(1) + 0.01; 
    else
        % Adjust KE (Panel 4) - shifted to the right
        xpos = axP(1) + 0.03; 
    end
    
    % Adjust vertical position.
    % Increase '+ 0.04' to move UP. Decrease to move DOWN.
    ypos = axP(2) + 0.02;  
    cbs2(i).Position = [xpos, ypos, cw, ch];
    cbs2(i).YAxisLocation = 'right'; % tick labels inside to the right
    cbs2(i).FontSize = 11;
    cbs2(i).Title.FontSize = 11;
    
    if i == 1 || i == 2
        % Calculate exactly 3 ticks rounded to nearest 0.5
        cmin = cbs2(i).Limits(1); cmax = cbs2(i).Limits(2);
        step = round(((cmax - cmin) / 2) * 2) / 2; 
        if step < 0.5, step = 0.5; end
        mid = round((cmin + cmax)/2 * 2) / 2;
        ticks = [mid - step, mid, mid + step];
        
        % Expand limits if necessary so all 3 ticks are visible
        caxis(axs2(i), [min(cmin, ticks(1)), max(cmax, ticks(3))]);
        cbs2(i).Ticks = ticks;
    end
    % Text is left as default black so it's visible against the white background
end

run_label = run_tag;
if isfinite(last_days)
    supt_str = sprintf('%s  —  day %.1f / %.1f  [last %g days]', ...
        run_label, t_days(anim_start), total_days, last_days);
else
    supt_str = sprintf('%s — surface & XZ transect (day 0 of %.1f)', ...
        run_label, total_days);
end
% supt = sgtitle(tl, supt_str, ...
%     'FontSize', 12, 'FontWeight', 'bold', 'Interpreter', 'none');
drawnow;

% Isopycnal contour handles
hIso1 = []; hIso2 = []; hIso3 = []; hIso4 = [];
hIso5 = []; hIso6 = []; hIso7 = []; hIso8 = [];
set(ax1, 'NextPlot','add');
set(ax2, 'NextPlot','add');
set(ax3, 'NextPlot','add');
set(ax4, 'NextPlot','add');

% ---------- output directory ----------
outdir = p.Results.outdir;
if isempty(outdir)
    outdir = fullfile(fileparts(mfilename('fullpath')), 'animations');
end
if ~exist(outdir,'dir'), mkdir(outdir); end

giffile = fullfile(outdir, [run_tag '_clipped.gif']);
% Delete stale GIF to avoid GIF89a append errors
if exist(giffile, 'file')
    delete(giffile);
    fprintf('Deleted stale GIF: %s\n', giffile);
end
if want_mp4
    % Was a bare VideoWriter(..., 'MPEG-4'), which THROWS on Linux rather
    % than falling back, so this function could not run on Casper at all.
    mp4_path = fullfile(outdir, [run_tag '_clipped.mp4']);
    [vwriter, avi_path] = video_writer_open(mp4_path, fps);
end

% ---------- animate ----------
delay    = 1/fps;
is_first = true;

for t = anim_start:frame_skip:Nt
    % Surface XY — transpose for pcolor (Ny×Nx)
    Txy  = surf_tracer('T', t)';
    Sxy  = surf_tracer('S', t)';
    vxy  = read_surf(vel_ncvar, t)';
    Roxy = read_surf(vort_ncvar, t)';
    vxy(isnan(Txy)) = NaN; Roxy(isnan(Txy)) = NaN; % Apply land mask

    set(h1,'CData', Txy);
    set(h2,'CData', Sxy);
    set(h3,'CData', vxy);
    set(h4,'CData', Roxy);

    % XZ transect — transpose for pcolor (Nz×Nx)
    Txz  = xz_tracer('T', t)';
    Sxz  = xz_tracer('S', t)';
    vxz  = read_xz(vel_ncvar, t)';
    Roxz = read_xz(vort_ncvar, t)';
    vxz(isnan(Txz)) = NaN; Roxz(isnan(Txz)) = NaN; % Apply land mask

    set(h5,'CData', Txz);
    set(h6,'CData', Sxz);
    set(h7,'CData', vxz);
    set(h8,'CData', Roxz);

    % --- TEOS-10 σ0 isopycnals ---
    if want_iso
        % XY surface isopycnals
        bad_xy = ~isfinite(Txy) | ~isfinite(Sxy);
        Tsurf = Txy; Ssurf = Sxy;
        Tsurf(bad_xy) = NaN; Ssurf(bad_xy) = NaN;
        SA_xy = gsw_SA_from_SP(Ssurf, P_xy, lon, lat);
        CT_xy = gsw_CT_from_t(SA_xy, Tsurf, P_xy);
        sigma0_xy = gsw_sigma0(SA_xy, CT_xy);

        if ~isempty(hIso1) && isgraphics(hIso1), delete(hIso1); end
        hIso1 = [];
        fin_xy = sigma0_xy(isfinite(sigma0_xy));
        if ~isempty(fin_xy) && range(fin_xy) > 0
            [C1, hIso1] = contour(ax1, Xc, Yc, sigma0_xy, rho_levels, ...
                'Color', iso_color, 'LineWidth', iso_lw);
            if mod(t, clabel_every) == 1 && ~isempty(C1)
                clabel(C1, hIso1, 'FontSize', 7, 'Color', iso_color, 'LabelSpacing', 400);
            end
        end

        if ~isempty(hIso2) && isgraphics(hIso2), delete(hIso2); end
        hIso2 = [];
        if ~isempty(fin_xy) && range(fin_xy) > 0
            [C2, hIso2] = contour(ax2, Xc, Yc, sigma0_xy, rho_levels, ...
                'Color', iso_color, 'LineWidth', iso_lw);
            if mod(t, clabel_every) == 1 && ~isempty(C2)
                clabel(C2, hIso2, 'FontSize', 7, 'Color', iso_color, 'LabelSpacing', 400);
            end
        end

        if ~isempty(hIso3) && isgraphics(hIso3), delete(hIso3); end
        hIso3 = [];
        if ~isempty(fin_xy) && range(fin_xy) > 0
            [C3, hIso3] = contour(ax3, Xc, Yc, sigma0_xy, rho_levels, ...
                'Color', iso_color, 'LineWidth', iso_lw);
            if mod(t, clabel_every) == 1 && ~isempty(C3)
                clabel(C3, hIso3, 'FontSize', 7, 'Color', iso_color, 'LabelSpacing', 400);
            end
        end

        if ~isempty(hIso4) && isgraphics(hIso4), delete(hIso4); end
        hIso4 = [];
        if ~isempty(fin_xy) && range(fin_xy) > 0
            [C4, hIso4] = contour(ax4, Xc, Yc, sigma0_xy, rho_levels, ...
                'Color', iso_color, 'LineWidth', iso_lw);
            if mod(t, clabel_every) == 1 && ~isempty(C4)
                clabel(C4, hIso4, 'FontSize', 7, 'Color', iso_color, 'LabelSpacing', 400);
            end
        end

        % XZ cross-section isopycnals
        bad = ~isfinite(Txz) | ~isfinite(Sxz);
        Ttmp = Txz; Stmp = Sxz;
        Ttmp(bad) = NaN; Stmp(bad) = NaN;
        SA = gsw_SA_from_SP(Stmp, P, lon, lat);
        CT = gsw_CT_from_t(SA, Ttmp, P);
        sigma0_xz = gsw_sigma0(SA, CT);

        if ~isempty(hIso5) && isgraphics(hIso5), delete(hIso5); end
        [C5, hIso5] = contour(ax5, Xz, Zz, sigma0_xz, rho_levels, ...
            'Color', iso_color, 'LineWidth', iso_lw);
        if mod(t, clabel_every) == 1 && ~isempty(C5)
            clabel(C5, hIso5, 'FontSize', 8, 'Color', iso_color, 'LabelSpacing', 400);
        end

        if ~isempty(hIso6) && isgraphics(hIso6), delete(hIso6); end
        [C6, hIso6] = contour(ax6, Xz, Zz, sigma0_xz, rho_levels, ...
            'Color', iso_color, 'LineWidth', iso_lw);
        if mod(t, clabel_every) == 1 && ~isempty(C6)
            clabel(C6, hIso6, 'FontSize', 8, 'Color', iso_color, 'LabelSpacing', 400);
        end

        if ~isempty(hIso7) && isgraphics(hIso7), delete(hIso7); end
        [C7, hIso7] = contour(ax7, Xz, Zz, sigma0_xz, rho_levels, ...
            'Color', iso_color, 'LineWidth', iso_lw);
        if mod(t, clabel_every) == 1 && ~isempty(C7)
            clabel(C7, hIso7, 'FontSize', 8, 'Color', iso_color, 'LabelSpacing', 400);
        end

        if ~isempty(hIso8) && isgraphics(hIso8), delete(hIso8); end
        [C8, hIso8] = contour(ax8, Xz, Zz, sigma0_xz, rho_levels, ...
            'Color', iso_color, 'LineWidth', iso_lw);
        if mod(t, clabel_every) == 1 && ~isempty(C8)
            clabel(C8, hIso8, 'FontSize', 8, 'Color', iso_color, 'LabelSpacing', 400);
        end
    end

    % Update title
%     if isfinite(last_days)
%         set(supt,'String', sprintf('%s  —  day %.1f / %.1f  [last %g days]', ...
%             run_label, t_days(t), total_days, last_days));
%     else
%         set(supt,'String', sprintf('%s  —  day %.1f / %.1f', ...
%             run_label, t_days(t), total_days));
%     end
    drawnow;

    % Write frame
    fr   = getframe(fig);
    [A, map] = rgb2ind(fr.cdata, 256);
    if is_first
        imwrite(A, map, giffile, 'gif', 'LoopCount', Inf, 'DelayTime', delay);
        is_first = false;
    else
        imwrite(A, map, giffile, 'gif', 'WriteMode', 'append', 'DelayTime', delay);
    end
    if want_mp4
        writeVideo(vwriter, fr);
    end
end

fprintf('Wrote GIF: %s\n', giffile);
if want_mp4
    % Reports the file it actually wrote. The old line announced an MP4
    % unconditionally, so a fallback would have been reported as an MP4.
    video_writer_finish(vwriter, mp4_path, avi_path, fps);
end
close all;
end

% ================== helpers ==================

function data = try_ncread_(fname, names)
% Try reading coordinate from file using a list of possible variable names
for i = 1:numel(names)
    try
        data = ncread(fname, names{i});
        return;
    catch
        % try next name
    end
end
% If nothing worked, list what's in the file
info = ncinfo(fname);
avail = strjoin({info.Variables.Name}, ', ');
error('Could not find coordinate variable. Tried: {%s}. Available: %s', ...
    strjoin(names, ', '), avail);
end
