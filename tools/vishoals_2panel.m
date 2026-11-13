function vishoals_2panel(run_tag, varargin)
% vishoals_2panel(run_tag, 'outdir', OUTDIR, 'fps', FPS, 'frame_skip', N, ...
%   'mp4', true/false, ...
%   'last_days', 50, ...
%   'isopycnals', true/false, 'rho_levels', 24:0.2:26, 'iso_color','k', ...
%   'iso_linewidth', 0.8, 'clabel_every', 10, ...
%   'climu',[umin umax], 'climKE',[KEmin KEmax], ...
%   'datadir', '/path/to/nc/files')
%
% Like vishoals.m but:
%   - Animates only the LAST <last_days> days (default 50).
%   - Shows the full domain (no clipping).
%   - Only two tracers: Velocity/Vorticity and Kinetic Energy.

% ---------- options ----------
p = inputParser;
addParameter(p, 'datadir',     '',    @ischar);
addParameter(p, 'outdir',      '',    @ischar);
addParameter(p, 'fps',         5,     @(x)isnumeric(x)&&isscalar(x)&&x>0);
addParameter(p, 'frame_skip',  1,     @(x)isnumeric(x)&&isscalar(x)&&x>=1);
addParameter(p, 'mp4',         false, @(x)islogical(x)&&isscalar(x));
addParameter(p, 'last_days',   50,    @(x)isnumeric(x)&&isscalar(x)&&x>0);
% Isopycnal options (kept for optional overlay)
addParameter(p, 'isopycnals',   false, @(x)islogical(x)&&isscalar(x));
addParameter(p, 'rho_levels',   24:0.2:26, @(v)isnumeric(v)&&isvector(v));
addParameter(p, 'iso_color',    'k',  @ischar);
addParameter(p, 'iso_linewidth',0.8,  @(x)isnumeric(x)&&isscalar(x)&&x>0);
addParameter(p, 'clabel_every', 10,   @(x)isnumeric(x)&&isscalar(x)&&x>=1);
addParameter(p, 'climu',  [], @(v)isnumeric(v)&&numel(v)==2);
addParameter(p, 'climv',  [], @(v)isnumeric(v)&&numel(v)==2);
% climRo default [-1 1], not auto. An auto scale is per-run, so a weak-wind
% case rescales to fill the same colormap as a strong one and reads as
% equally energetic -- Ro panels were not comparable between runs. Pass
% 'climRo', [] to restore auto-scaling (the validator accepts empty).
addParameter(p, 'climRo', [-1, 1], @(v)isempty(v) || (isnumeric(v)&&numel(v)==2));
addParameter(p, 'climKE', [], @(v)isnumeric(v)&&numel(v)==2);
addParameter(p, 'velocity_field', 'u', @(s) ismember(s, {'u', 'v', 'Ro'}));

parse(p, varargin{:});
fps          = p.Results.fps;
frame_skip   = p.Results.frame_skip;
want_mp4     = p.Results.mp4;
want_iso     = p.Results.isopycnals;
rho_levels   = p.Results.rho_levels;
iso_color    = p.Results.iso_color;
iso_lw       = p.Results.iso_linewidth;
clabel_every = p.Results.clabel_every;
climu_user   = p.Results.climu;
climv_user   = p.Results.climv;
climRo_user  = p.Results.climRo;
climKE_user  = p.Results.climKE;
last_days    = p.Results.last_days;
vel_field    = p.Results.velocity_field;

% ---------- build filenames ----------
datadir = p.Results.datadir;
if isempty(datadir), datadir = pwd; end

fname_top  = fullfile(datadir, ['top_'  run_tag '.nc']);
fname_midy = fullfile(datadir, ['midy_' run_tag '.nc']);

if ~isfile(fname_top),  error('Surface file not found: %s', fname_top);  end
if ~isfile(fname_midy), error('Midy file not found: %s', fname_midy);  end

fprintf('Reading surface XY from: %s\n', fname_top);
fprintf('Reading midy XZ from: %s\n', fname_midy);

% ---------- detect coordinate names ----------
xc_top = try_ncread_(fname_top, {'xC','x_caa','xc'});
yc_top = try_ncread_(fname_top, {'yC','y_aca','yc'});
xc_mid = try_ncread_(fname_midy, {'xC','x_caa','xc'});
zc_mid = try_ncread_(fname_midy, {'zC','z_aac','zc'});
time   = ncread(fname_top, 'time');
Nt     = numel(time);

% ---------- determine temporal window: last <last_days> days ----------
t_days     = (time(1:Nt) - time(1)) / 86400;   % elapsed days
total_days = t_days(end);
start_day  = max(0, total_days - last_days);     % start of window
anim_start = find(t_days >= start_day, 1, 'first');
if isempty(anim_start), anim_start = 1; end
anim_idx   = anim_start : frame_skip : Nt;      % frames to animate
fprintf('Animating frames %d to %d (days %.1f – %.1f of %.1f total).\n', ...
    anim_idx(1), anim_idx(end), t_days(anim_idx(1)), t_days(anim_idx(end)), total_days);

% ---------- coordinate arrays ----------
xc_km_top = xc_top / 1000;
yc_km_top = yc_top / 1000;

% Meshgrid for plotting
[Xc, Yc] = meshgrid(xc_km_top, yc_km_top);   % [Ny x Nx]

x_lims = [xc_km_top(1), xc_km_top(end)];
y_lims = [yc_km_top(1), yc_km_top(end)];

xc_km_mid = xc_mid / 1000;
zc_m_mid  = zc_mid;
[Xc_xz, Zc_xz] = meshgrid(xc_km_mid, zc_m_mid);
x_lims_xz = [xc_km_mid(1), xc_km_mid(end)];
z_lims_xz = [zc_m_mid(1), zc_m_mid(end)];

% ---------- full domain readers ----------
%  surface (Nx, Ny, 1, time) → (Ny × Nx) after transpose
nx_top_cnt = numel(xc_km_top);
ny_top_cnt = numel(yc_km_top);

read_surf = @(varname, t) squeeze(ncread(fname_top, varname, ...
    [1, 1, 1, t], [nx_top_cnt, ny_top_cnt, 1, 1]))';

    function A = surf_tracer(varname, t)
        A = read_surf(varname, t);
        A(A==0) = NaN;
    end

    function A = midy_tracer(varname, t)
        info = ncinfo(fname_midy, varname);
        st = ones(1, length(info.Size));
        ct = info.Size;
        st(end) = t;
        ct(end) = 1;
        A = squeeze(ncread(fname_midy, varname, st, ct))';
        A(A==0) = NaN;
    end

% ---------- scan color limits (full window) ----------
getmin = @(x) min(x(:), [], 'omitnan');
getmax = @(x) max(x(:), [], 'omitnan');

mins = struct('vel', inf, 'KE', inf);
maxs = struct('vel',-inf, 'KE',-inf);

if strcmpi(vel_field, 'u')
    clim_vel_user = climu_user;
elseif strcmpi(vel_field, 'v')
    clim_vel_user = climv_user;
else
    clim_vel_user = climRo_user;
end

need_auto_vel = isempty(clim_vel_user);
need_auto_KE  = isempty(climKE_user);

if strcmpi(vel_field, 'Ro')
    tracer_name = 'Ro';
else
    tracer_name = [vel_field '_c'];
end

if need_auto_vel || need_auto_KE
    fprintf('Scanning %d frames for color limits (full window)...\n', numel(anim_idx));
    for ti = anim_idx
        if need_auto_vel
            uxy = surf_tracer(tracer_name, ti);
            mins.vel = min(mins.vel, getmin(uxy));
            maxs.vel = max(maxs.vel, getmax(uxy));
        end
        if need_auto_KE
            KExy = surf_tracer('KE', ti);
            mins.KE = min(mins.KE, getmin(KExy));
            maxs.KE = max(maxs.KE, getmax(KExy));
        end
    end
end

if isempty(clim_vel_user)
    vm = max(abs([mins.vel, maxs.vel]));  clim_vel = [-vm, vm];
else
    clim_vel = clim_vel_user;
end
if isempty(climKE_user)
    climKE = [0, maxs.KE * 1.02];
else
    climKE = climKE_user;
end

fprintf('clim_%s  = [%.4f, %.4f]\n', vel_field, clim_vel(1), clim_vel(2));
fprintf('climKE = [%.4f, %.4f]\n', climKE(1), climKE(2));

% ---------- figure / panels ----------
%  1 row x 2 cols: U velocity | Kinetic Energy  (surface XY only)
fig = figure('Color','w','Position',[80 400 450 466]);
tl  = tiledlayout(fig, 1, 2, 'TileSpacing','compact','Padding','compact');

    function [ax,h,cb] = panel_xy(col, titleStr, clim, cmap, cbLabel)
        ax = nexttile(tl, col);
        h  = pcolor(ax, Xc, Yc, nan(size(Xc)));
        shading(ax,'flat'); axis(ax,'tight'); set(ax,'YDir','normal');
        set(ax, 'Layer', 'top'); box(ax, 'on'); set(ax, 'LineWidth', 0.8, 'XColor', 'k', 'YColor', 'k');
        colormap(ax, cmap); caxis(ax, clim);
        cb = colorbar(ax);
        cb.FontSize = 14; cb.Title.String = cbLabel; cb.Title.FontSize = 16; cb.Title.FontWeight = 'bold';
        title(ax, titleStr, 'FontSize', 18, 'FontWeight', 'bold');
        if col == 1
            ylabel(ax, 'y (km)', 'FontSize', 16, 'FontWeight', 'bold');
        else
            yticklabels(ax, []);
        end
        xlabel(ax, 'x (km)', 'FontSize', 16, 'FontWeight', 'bold');
        set(ax, 'FontSize', 14);
        xlim(ax, x_lims); ylim(ax, y_lims);
    end

if strcmpi(vel_field, 'v')
    vel_title = 'V Velocity';
    cb_label = '(m s^{-1})';
elseif strcmpi(vel_field, 'u')
    vel_title = 'U Velocity';
    cb_label = '(m s^{-1})';
else
    vel_title = 'Rossby Number (Ro)';
    cb_label = '';
end
vel_cmap  = cmocean('balance');

[ax1, h1, cb1] = panel_xy(1, vel_title,     clim_vel, vel_cmap, cb_label);
[ax2, h2, cb2] = panel_xy(2, 'Kinetic Energy', climKE, turbo,              '(m^2 s^{-2})');

drawnow;
axs = [ax1, ax2];
cbs = [cb1, cb2];
for i = 1:2
    axP = axs(i).Position;
    cw = 0.03; 
    ch = axP(4) * 0.18; % slightly larger to accommodate big text
    xpos = axP(1) + axP(3) - cw - 0.01; % flushed to the right with padding
    ypos = axP(2) + axP(4) - ch - 0.12; % tucked into the top-right, pulled down to avoid title clip
    cbs(i).Position = [xpos, ypos, cw, ch];
    cbs(i).YAxisLocation = 'left'; % tick labels inside to the left
    cbs(i).FontSize = 11;
    cbs(i).Title.FontSize = 11;
end

run_label = run_tag;
supt = sgtitle(tl, sprintf('%s (XY) —  day %.1f / %.1f  [last %g days]', ...
    run_label, t_days(anim_idx(1)), total_days, last_days), ...
    'FontSize', 16, 'FontWeight', 'bold', 'Interpreter', 'none');

%  1 row x 2 cols: U velocity | Kinetic Energy  (midy XZ only)
fig2 = figure('Color','w','Position',[80 100 450 233]);
tl2  = tiledlayout(fig2, 1, 2, 'TileSpacing','compact','Padding','compact');

    function [ax,h,cb] = panel_xz(col, titleStr, clim, cmap, cbLabel)
        ax = nexttile(tl2, col);
        h  = pcolor(ax, Xc_xz, Zc_xz, nan(size(Xc_xz)));
        shading(ax,'flat'); axis(ax,'tight'); set(ax,'YDir','normal');
        set(ax, 'Layer','top'); set(ax, 'NextPlot','add');
        box(ax, 'on'); set(ax, 'LineWidth', 0.8, 'XColor', 'k', 'YColor', 'k');
        colormap(ax, cmap); caxis(ax, clim);
        cb = colorbar(ax);
        cb.FontSize = 14; cb.Title.String = cbLabel; cb.Title.FontSize = 16; cb.Title.FontWeight = 'bold';
        title(ax, titleStr, 'FontSize', 18, 'FontWeight', 'bold');
        if col == 1
            ylabel(ax, 'z (m)', 'FontSize', 16, 'FontWeight', 'bold');
        else
            yticklabels(ax, []);
        end
        xlabel(ax, 'x (km)', 'FontSize', 16, 'FontWeight', 'bold');
        set(ax, 'FontSize', 14);
        xlim(ax, x_lims_xz); ylim(ax, z_lims_xz);
    end

[ax1_xz, h1_xz, cb1_xz] = panel_xz(1, vel_title, clim_vel, vel_cmap, cb_label);
[ax2_xz, h2_xz, cb2_xz] = panel_xz(2, 'Kinetic Energy', climKE, turbo, '(m^2 s^{-2})');

drawnow;
axs2 = [ax1_xz, ax2_xz];
cbs2 = [cb1_xz, cb2_xz];
for i = 1:2
    axP = axs2(i).Position;
    cw = 0.018; 
    ch = axP(4) * 0.35; % smaller relative height for short XZ plots
    xpos = axP(1) + 0.015; % tucked into the bottom left corner
    ypos = axP(2) + 0.15; % pull up enough to avoid overlapping x-axis labels
    cbs2(i).Position = [xpos, ypos, cw, ch];
    cbs2(i).YAxisLocation = 'right'; % tick labels inside to the right
    cbs2(i).FontSize = 11;
    cbs2(i).Title.FontSize = 11;
    cbs2(i).Color = 'w'; % usually dark in bottom left
end

supt2 = sgtitle(tl2, sprintf('%s (XZ) —  day %.1f / %.1f  [last %g days]', ...
    run_label, t_days(anim_idx(1)), total_days, last_days), ...
    'FontSize', 16, 'FontWeight', 'bold', 'Interpreter', 'none');
drawnow;

% ---------- output ----------
outdir = p.Results.outdir;
if isempty(outdir)
    outdir = fullfile(fileparts(mfilename('fullpath')), 'animations');
end
if ~exist(outdir,'dir'), mkdir(outdir); end

giffile = fullfile(outdir, [run_tag '_2panel_xy.gif']);
giffile2 = fullfile(outdir, [run_tag '_2panel_xz.gif']);

if want_mp4
    % Both were bare VideoWriter(..., 'MPEG-4') calls, which THROW on Linux
    % instead of falling back, so this function could not run on Casper.
    mp4_path_xy = fullfile(outdir, [run_tag '_2panel_xy.mp4']);
    mp4_path_xz = fullfile(outdir, [run_tag '_2panel_xz.mp4']);
    [vwriter,  avi_path_xy] = video_writer_open(mp4_path_xy, fps);
    [vwriter2, avi_path_xz] = video_writer_open(mp4_path_xz, fps);
end

% ---------- animate ----------
delay    = 1/fps;
is_first = true;

for ti = anim_idx
    % Surface XY — (Ny × Nx) after transpose in reader
    uxy  = surf_tracer(tracer_name, ti);
    KExy = surf_tracer('KE',  ti);
    set(h1, 'CData', uxy);
    set(h2, 'CData', KExy);

    % Midy XZ
    uxz  = midy_tracer(tracer_name, ti);
    KExz = midy_tracer('KE',  ti);
    set(h1_xz, 'CData', uxz);
    set(h2_xz, 'CData', KExz);

    % Update titles
    set(supt, 'String', sprintf('%s (XY) —  day %.1f / %.1f  [last %g days]', ...
        run_label, t_days(ti), total_days, last_days));
    set(supt2, 'String', sprintf('%s (XZ) —  day %.1f / %.1f  [last %g days]', ...
        run_label, t_days(ti), total_days, last_days));
    drawnow;

    % Write frame XY
    fr   = getframe(fig);
    [A, map] = rgb2ind(fr.cdata, 256);
    
    % Write frame XZ
    fr2  = getframe(fig2);
    [A2, map2] = rgb2ind(fr2.cdata, 256);

    if is_first
        imwrite(A, map, giffile, 'gif', 'LoopCount', Inf, 'DelayTime', delay);
        imwrite(A2, map2, giffile2, 'gif', 'LoopCount', Inf, 'DelayTime', delay);
        is_first = false;
    else
        imwrite(A, map, giffile, 'gif', 'WriteMode', 'append', 'DelayTime', delay);
        imwrite(A2, map2, giffile2, 'gif', 'WriteMode', 'append', 'DelayTime', delay);
    end
    if want_mp4
        writeVideo(vwriter, fr);
        writeVideo(vwriter2, fr2);
    end
end


fprintf('Wrote XY GIF: %s\n', giffile);
fprintf('Wrote XZ GIF: %s\n', giffile2);
if want_mp4
    video_writer_finish(vwriter,  mp4_path_xy, avi_path_xy, fps);
    video_writer_finish(vwriter2, mp4_path_xz, avi_path_xz, fps);
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
info  = ncinfo(fname);
avail = strjoin({info.Variables.Name}, ', ');
error('Could not find coordinate variable. Tried: {%s}. Available: %s', ...
    strjoin(names, ', '), avail);
end
