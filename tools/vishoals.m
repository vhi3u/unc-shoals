function vishoals(run_tag, varargin)
% vishoals(run_tag, 'outdir', OUTDIR, 'fps', FPS, 'frame_skip', N, 'sample_hours', H, ...
%   'mp4', true/false, ...
%   'isopycnals', true/false, 'rho_levels', 24:0.2:26, 'iso_color','k', ...
%   'iso_linewidth', 0.8, 'clabel_every', 10, ...
%   'climT',[Tmin Tmax],'climS',[Smin Smax],'climv',[vmin vmax],'climRo',[lo hi], ...
%   'datadir', '/path/to/nc/files')
%
% 'sample_hours': animation sampling interval in hours (e.g. 24 = one frame per
%   day). The output interval of the files is detected from their time axis and
%   the stride set accordingly; overrides 'frame_skip'. Default [] = every
%   saved output.
%
% Reads from two pre-sliced NetCDF files produced by flow_over_shoals_hpc.jl:
%   top_<run_tag>.nc   — surface XY slice (u_c, v_c, w_c, T, S, Ro)
%   midy_<run_tag>.nc  — mid-y  XZ slice (u_c, v_c, w_c, T, S, Ro)
%
% Top row (XY @ surface):   T, S, v, Ro
% Bottom row (XZ @ y~mid):  T, S, v, Ro  + optional σ0 isopycnals
%
% Runs without temperature and salinity (e.g. buoyancy-only runs such as
% periodic_buoy70) are detected automatically: if top_<run_tag>.nc has no T/S,
% the figure shows only u, v, vorticity/f and kinetic energy (XY + XZ,
% 4 columns, same panel size as the full_panel layout), with no isopycnals.
%
% Requires: cmocean, GSW Toolbox (for isopycnals)

% ---------- options ----------
p = inputParser;
addParameter(p, 'datadir', '', @ischar);
addParameter(p, 'outdir', '', @ischar);
addParameter(p, 'fps', 5, @(x)isnumeric(x)&&isscalar(x)&&x>0);
addParameter(p, 'frame_skip', 1, @(x)isnumeric(x)&&isscalar(x)&&x>=1);
addParameter(p, 'sample_hours', [], @(x)isempty(x) || (isnumeric(x)&&isscalar(x)&&x>0));
addParameter(p, 'mp4', false, @(x)islogical(x)&&isscalar(x));
addParameter(p, 'full_panel', false, @(x)islogical(x)&&isscalar(x));
addParameter(p, 'clipped', false, @(x)islogical(x)&&isscalar(x));
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
addParameter(p, 'climu', [], @(v)isnumeric(v) && numel(v)==2);
addParameter(p, 'climv', [], @(v)isnumeric(v) && numel(v)==2);
% climRo default [-1 1], not auto. An auto scale is per-run, so a weak-wind
% case rescales to fill the same colormap as a strong one and reads as
% equally energetic -- Ro panels were not comparable between runs. Pass
% 'climRo', [] to restore auto-scaling (the validator accepts empty).
addParameter(p, 'climRo', [-1, 1], @(v)isempty(v) || (isnumeric(v) && numel(v)==2));
addParameter(p, 'climKE', [], @(v)isempty(v) || (isnumeric(v) && numel(v)==2));
addParameter(p, 'no_outlier', false, @(x)islogical(x)&&isscalar(x));
% 'raster': how the field panels are drawn. imagesc is one image object per
% panel; pcolor is one vector quad per DATA CELL, which under the painters
% renderer is 487,152 quads a frame on a 500 m grid -- and -nodisplay forces
% painters, so no renderer flag avoids it. Measured on cluster_weakwinds:
% print() per frame 3.612 s with pcolor, 0.456 s with imagesc, a 7.9x
% difference, and the cost is nearly independent of dpi because it is
% per-primitive rather than per-pixel. 'pcolor' is kept as an escape hatch.
addParameter(p, 'raster', 'imagesc', @(x)ischar(x) && ismember(x, {'imagesc','pcolor'}));
% Frame resolution. Was a hardcoded 100, which is soft for a 12-panel figure:
% axis labels and colorbar ticks are small, go mushy in the render, and the
% H.264 encode blurs them again. Under pcolor raising it was nearly free
% (print cost was per-primitive -- -r50 only 14% cheaper than -r100), and
% imagesc leaves headroom at 0.456 s/frame.
addParameter(p, 'frame_dpi', 150, @(x)isnumeric(x)&&isscalar(x)&&x>=50);
addParameter(p, 'scan_skip_days', 10, @(x)isnumeric(x) && isscalar(x) && x>=0);
addParameter(p, 'last_days', Inf, @(x)isnumeric(x)&&isscalar(x)&&x>0);

parse(p, varargin{:});
fps          = p.Results.fps;
frame_skip   = p.Results.frame_skip;
want_mp4     = p.Results.mp4;
full_panel   = p.Results.full_panel;
clipped      = p.Results.clipped;
want_iso     = p.Results.isopycnals;
rho_levels   = p.Results.rho_levels;
iso_color    = p.Results.iso_color;
iso_lw       = p.Results.iso_linewidth;
clabel_every = p.Results.clabel_every;
climT_user   = p.Results.climT;
climS_user   = p.Results.climS;
climu_user   = p.Results.climu;
climv_user   = p.Results.climv;
climRo_user  = p.Results.climRo;
climKE_user  = p.Results.climKE;
no_outlier   = p.Results.no_outlier;
use_imagesc  = strcmp(p.Results.raster, 'imagesc');
frame_dpi    = p.Results.frame_dpi;
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

% ---------- runs without temperature / salinity (e.g. periodic_buoy70) ----------
info_top = ncinfo(fname_top);
has_TS   = all(ismember({'T', 'S'}, {info_top.Variables.Name}));
no_TS    = ~has_TS;
if no_TS
    fprintf('No T/S in %s: plotting u, v, vorticity and kinetic energy only (no isopycnals).\n', fname_top);
    full_panel = true;     % use the u / v / Ro / KE colour scaling of the full-panel layout
    want_iso   = false;
end

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

% Sampling interval: 'sample_hours' -> stride over the saved outputs
dt_out_h = median(diff(time)) / 3600;
if ~isempty(p.Results.sample_hours)
    frame_skip = max(1, round(p.Results.sample_hours / dt_out_h));
end
fprintf('Saved output every %.3g h; animating every %d output(s) = %.3g h per frame\n', ...
    dt_out_h, frame_skip, frame_skip * dt_out_h);

% Also check midy has same time
time_midy = ncread(fname_midy, 'time');
Nt_midy   = numel(time_midy);
if Nt ~= Nt_midy
    warning('top and midy files have different number of timesteps (%d vs %d). Using min.', Nt, Nt_midy);
    Nt = min(Nt, Nt_midy);
end

Nx_top = numel(xc_top); Ny_top = numel(yc_top);
Nx_xz  = numel(xc_midy); Nz_xz = numel(zc_midy);

xc_km_top  = xc_top / 1000;
yc_km_top  = yc_top / 1000;
xc_km_midy = xc_midy / 1000;

ix_top = 1:Nx_top;
iy_top = 1:Ny_top;
ix_xz  = 1:Nx_xz;
clip_x = [0 100];   % km, applied to both XY and XZ so columns line up
clip_y = [20 200];  % km, XY only
if clipped
    ix_top = find(xc_km_top >= clip_x(1) & xc_km_top <= clip_x(2));
    iy_top = find(yc_km_top >= clip_y(1) & yc_km_top <= clip_y(2));
    xc_km_top = xc_km_top(ix_top);
    yc_km_top = yc_km_top(iy_top);
    Nx_top = numel(ix_top);
    Ny_top = numel(iy_top);

    ix_xz = find(xc_km_midy >= clip_x(1) & xc_km_midy <= clip_x(2));
    xc_km_midy = xc_km_midy(ix_xz);
    Nx_xz = numel(ix_xz);
end

% XY grids in km
[Xc, Yc] = meshgrid(xc_km_top, yc_km_top);
% XZ grids: x in km, z in meters
[Xz, Zz] = meshgrid(xc_km_midy, zc_midy);

% ---------- TEOS-10 prep for isopycnals ----------
lat = 35.2480;
lon = 0;
if want_iso                            % only needed for isopycnals (needs GSW)
    p_db = gsw_p_from_z(zc_midy(:), lat);  % [Nz×1] dbar
    P    = repmat(p_db, 1, Nx_xz);         % [Nz×Nx] for XZ
    % For surface: use z=0 (surface)
    p_surf = gsw_p_from_z(0, lat);
    P_xy   = p_surf * ones(Ny_top, Nx_top);
end

% ---------- readers ----------
% Surface file: data is (Nx, Ny, 1, time) — z is singleton
read_surf = @(varname, t) squeeze(ncread(fname_top, varname, ...
    [ix_top(1) iy_top(1) 1 t], [numel(ix_top) numel(iy_top) 1 1]));

% XZ file: data is (Nx, 1, Nz, time) — y is singleton
read_xz = @(varname, t) squeeze(ncread(fname_midy, varname, ...
    [ix_xz(1) 1 1 t], [numel(ix_xz) 1 Inf 1]));

    function A = surf_tracer(varname, t)
        try
            A = read_surf(varname, t);
            A(A==0) = NaN;  % mask land/immersed cells
        catch
            A = nan(Ny_top, Nx_top);
        end
    end

    function A = xz_tracer(varname, t)
        try
            A = read_xz(varname, t);
            A(A==0) = NaN;
        catch
            A = nan(Nz_xz, Nx_xz);
        end
    end

    function A = read_surf_safe(varname, t)
        try, A = read_surf(varname, t); catch, A = nan(Ny_top, Nx_top); end
    end
    
    function A = read_xz_safe(varname, t)
        try, A = read_xz(varname, t); catch, A = nan(Nz_xz, Nx_xz); end
    end

% Trim boundary cells (i=1, i=end, j=1, j=end) to avoid numerical artifacts in color limits
trim = @(x) x(2:end-1, 2:end-1);

if no_outlier
    getmin = @(x) prctile(reshape(trim(x), [], 1), 1);
    getmax = @(x) prctile(reshape(trim(x), [], 1), 99);
else
    getmin = @(x) min(reshape(trim(x), [], 1), [], 'omitnan');
    getmax = @(x) max(reshape(trim(x), [], 1), [], 'omitnan');
end

mins = struct('Txy', inf, 'Sxy', inf, 'vxy', inf, 'Roxy', inf, ...
              'Txz', inf, 'Sxz', inf, 'vxz', inf, 'Roxz', inf, ...
              'uxy', inf, 'uxz', inf, 'v_cxy', inf, 'v_cxz', inf, ...
              'KExy', inf, 'KExz', inf);
maxs = struct('Txy',-inf, 'Sxy',-inf, 'vxy',-inf, 'Roxy',-inf, ...
              'Txz',-inf, 'Sxz',-inf, 'vxz',-inf, 'Roxz',-inf, ...
              'uxy',-inf, 'uxz',-inf, 'v_cxy',-inf, 'v_cxz',-inf, ...
              'KExy',-inf, 'KExz',-inf);

need_auto_T  = isempty(climT_user);
need_auto_S  = isempty(climS_user);
if strcmp(vel_comp, 'u')
    need_auto_v = isempty(climu_user);
else
    need_auto_v = isempty(climv_user);
end
need_auto_Ro = isempty(climRo_user);

if full_panel || need_auto_T || need_auto_S || need_auto_v || need_auto_Ro
    % When last_days is active, scan only the animation window
    % However, we still respect scan_skip_days (scan_start_idx) so we don't scan spin-up artifacts
    if isfinite(last_days)
        scan_from = max(anim_start, scan_start_idx);
    else
        scan_from = scan_start_idx;
    end
    fprintf('Scanning frames %d to %d for color limits...\n', scan_from, Nt);
    for t = scan_from:frame_skip:Nt
        if has_TS && (full_panel || need_auto_T)
            Txy = surf_tracer('T', t); Txz = xz_tracer('T', t);
            mins.Txy = min(mins.Txy, getmin(Txy)); maxs.Txy = max(maxs.Txy, getmax(Txy));
            mins.Txz = min(mins.Txz, getmin(Txz)); maxs.Txz = max(maxs.Txz, getmax(Txz));
        end
        if has_TS && (full_panel || need_auto_S)
            Sxy = surf_tracer('S', t); Sxz = xz_tracer('S', t);
            mins.Sxy = min(mins.Sxy, getmin(Sxy)); maxs.Sxy = max(maxs.Sxy, getmax(Sxy));
            mins.Sxz = min(mins.Sxz, getmin(Sxz)); maxs.Sxz = max(maxs.Sxz, getmax(Sxz));
        end
        if need_auto_v && ~full_panel
            vxy = read_surf_safe(vel_ncvar, t); vxz = read_xz_safe(vel_ncvar, t);
            mins.vxy = min(mins.vxy, getmin(vxy)); maxs.vxy = max(maxs.vxy, getmax(vxy));
            mins.vxz = min(mins.vxz, getmin(vxz)); maxs.vxz = max(maxs.vxz, getmax(vxz));
        end
        if need_auto_Ro && ~full_panel
            rxy = read_surf_safe(vort_ncvar, t); rxz = read_xz_safe(vort_ncvar, t);
            mins.Roxy = min(mins.Roxy, getmin(rxy)); maxs.Roxy = max(maxs.Roxy, getmax(rxy));
            mins.Roxz = min(mins.Roxz, getmin(rxz)); maxs.Roxz = max(maxs.Roxz, getmax(rxz));
        end
        if full_panel
            uxy = read_surf_safe('u_c', t); uxz = read_xz_safe('u_c', t);
            mins.uxy = min(mins.uxy, getmin(uxy)); maxs.uxy = max(maxs.uxy, getmax(uxy));
            mins.uxz = min(mins.uxz, getmin(uxz)); maxs.uxz = max(maxs.uxz, getmax(uxz));
            
            vcxy = read_surf_safe('v_c', t); vcxz = read_xz_safe('v_c', t);
            mins.v_cxy = min(mins.v_cxy, getmin(vcxy)); maxs.v_cxy = max(maxs.v_cxy, getmax(vcxy));
            mins.v_cxz = min(mins.v_cxz, getmin(vcxz)); maxs.v_cxz = max(maxs.v_cxz, getmax(vcxz));
            
            roxy = read_surf_safe('Ro', t); roxz = read_xz_safe('Ro', t);
            mins.Roxy = min(mins.Roxy, getmin(roxy)); maxs.Roxy = max(maxs.Roxy, getmax(roxy));
            mins.Roxz = min(mins.Roxz, getmin(roxz)); maxs.Roxz = max(maxs.Roxz, getmax(roxz));
            
            kexy = read_surf_safe('KE', t); kexz = read_xz_safe('KE', t);
            mins.KExy = min(mins.KExy, getmin(kexy)); maxs.KExy = max(maxs.KExy, getmax(kexy));
            mins.KExz = min(mins.KExz, getmin(kexz)); maxs.KExz = max(maxs.KExz, getmax(kexz));
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

if full_panel
    if isempty(climu_user)
        um_xy = max(abs([mins.uxy, maxs.uxy])); clim_uxy = [-um_xy, um_xy];
        um_xz = max(abs([mins.uxz, maxs.uxz])); clim_uxz = [-um_xz, um_xz];
    else
        clim_uxy = climu_user; clim_uxz = climu_user;
    end
    
    if isempty(climv_user)
        vm_xy = max(abs([mins.v_cxy, maxs.v_cxy])); clim_vcxy = [-vm_xy, vm_xy];
        vm_xz = max(abs([mins.v_cxz, maxs.v_cxz])); clim_vcxz = [-vm_xz, vm_xz];
    else
        clim_vcxy = climv_user; clim_vcxz = climv_user;
    end
    
    if isempty(climRo_user)
        rm_xy = max(abs([mins.Roxy, maxs.Roxy])); clim_roxy = [-rm_xy, rm_xy];
        rm_xz = max(abs([mins.Roxz, maxs.Roxz])); clim_roxz = [-rm_xz, rm_xz];
    else
        clim_roxy = climRo_user; clim_roxz = climRo_user;
    end
    
    if isempty(climKE_user)
        clim_kexy = [0, maxs.KExy]; clim_kexz = [0, maxs.KExz];
    else
        clim_kexy = climKE_user; clim_kexz = climKE_user;
    end
else
    if strcmp(vel_comp, 'u')
        clim_vel_user = climu_user;
    else
        clim_vel_user = climv_user;
    end

    if isempty(clim_vel_user)
        vm_xy = max(abs([mins.vxy, maxs.vxy])); climv_xy = [-vm_xy, vm_xy];
        vm_xz = max(abs([mins.vxz, maxs.vxz])); climv_xz = [-vm_xz, vm_xz];
    else
        climv_xy = clim_vel_user; climv_xz = clim_vel_user;
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
end

if full_panel
    fprintf('\nColor limits (XY):\n');
    if has_TS
        fprintf('  T:  [%.4g, %.4g]\n', climT_xy(1), climT_xy(2));
        fprintf('  S:  [%.4g, %.4g]\n', climS_xy(1), climS_xy(2));
    end
    fprintf('  u:  [%.4g, %.4g]\n', clim_uxy(1), clim_uxy(2));
    fprintf('  v:  [%.4g, %.4g]\n', clim_vcxy(1), clim_vcxy(2));
    fprintf('  Ro: [%.4g, %.4g]\n', clim_roxy(1), clim_roxy(2));
    fprintf('  KE: [%.4g, %.4g]\n', clim_kexy(1), clim_kexy(2));

    fprintf('Color limits (XZ):\n');
    if has_TS
        fprintf('  T:  [%.4g, %.4g]\n', climT_xz(1), climT_xz(2));
        fprintf('  S:  [%.4g, %.4g]\n', climS_xz(1), climS_xz(2));
    end
    fprintf('  u:  [%.4g, %.4g]\n', clim_uxz(1), clim_uxz(2));
    fprintf('  v:  [%.4g, %.4g]\n', clim_vcxz(1), clim_vcxz(2));
    fprintf('  Ro: [%.4g, %.4g]\n', clim_roxz(1), clim_roxz(2));
    fprintf('  KE: [%.4g, %.4g]\n\n', clim_kexz(1), clim_kexz(2));
else
    fprintf('\nColor limits (XY):\n');
    fprintf('  T:  [%.4g, %.4g]\n', climT_xy(1), climT_xy(2));
    fprintf('  S:  [%.4g, %.4g]\n', climS_xy(1), climS_xy(2));
    fprintf('  v:  [%.4g, %.4g]\n', climv_xy(1), climv_xy(2));
    fprintf('  Ro: [%.4g, %.4g]\n', climRo_xy(1), climRo_xy(2));

    fprintf('Color limits (XZ):\n');
    fprintf('  T:  [%.4g, %.4g]\n', climT_xz(1), climT_xz(2));
    fprintf('  S:  [%.4g, %.4g]\n', climS_xz(1), climS_xz(2));
    fprintf('  v:  [%.4g, %.4g]\n', climv_xz(1), climv_xz(2));
    fprintf('  Ro: [%.4g, %.4g]\n\n', climRo_xz(1), climRo_xz(2));
end

% ---------- figure / panels ----------
if no_TS
    % 4 columns (u, v, Ro, KE) at the per-panel width of the 6-column layout
    fig = figure('Color','w','Position',[80 50 1350 * 4/6 700]);
    tl  = tiledlayout(fig, 3, 4, 'TileSpacing','compact','Padding','compact');
elseif full_panel
    fig = figure('Color','w','Position',[80 50 1350 700]);
    tl  = tiledlayout(fig, 3, 6, 'TileSpacing','compact','Padding','compact');
else
    fig = figure('Color','w','Position',[80 50 900 700]);
    tl  = tiledlayout(fig, 3, 4, 'TileSpacing','compact','Padding','compact');
end

    function h = raster_(ax, XX, YY, C, smooth_mode)
        % One image object, or one quad per cell. Limits are set the same way
        % either way: imagesc + axis tight extends half a cell past the data,
        % which shifts every tick (the XZ axis went 0..-50 to -5..-45 in the
        % render comparison). Pinning them to the data range keeps the axes
        % identical to the pcolor figure.
        xv = XX(1,:); yv = YY(:,1);
        if use_imagesc
            h = imagesc(ax, xv, yv, C);
            % A surface simply omits a NaN face, so the axes colour shows
            % through. An IMAGE instead maps NaN to the first colormap entry,
            % which rendered the immersed bathymetry as the darkest colour of
            % each colormap rather than the grey set below. AlphaData makes
            % those cells transparent, restoring the grey.
            set(h, 'AlphaData', ~isnan(C));
        else
            h = pcolor(ax, XX, YY, C);
            shading(ax, smooth_mode);
        end
        xlim(ax, [min(xv) max(xv)]);
        ylim(ax, [min(yv) max(yv)]);
    end

    function setC_(h, C)
        % CData and the NaN mask have to move together. AlphaData is static
        % once set, so updating CData alone would stamp the first frame's
        % land mask onto every later frame.
        set(h, 'CData', C);
        if strcmp(get(h, 'Type'), 'image')
            set(h, 'AlphaData', ~isnan(C));
        end
    end

    function [ax,h,cb] = panel_xy(tileIdx, titleStr, clim, cmap, cbLabel, isFirstCol)
        ax = nexttile(tl, tileIdx, [2 1]);
        % Was pcolor + 'shading interp'. At 0.70 x 1.16 px/cell on this grid a
        % cell renders at about one pixel, so the interpolation was being
        % thrown away by the rasterizer while costing 26% more than flat.
        h  = raster_(ax, Xc, Yc, nan(size(Xc)), 'interp');
        set(ax,'YDir','normal');
        set(ax, 'Color', [0.9 0.9 0.9]);   % masked cells grey, as in the XZ row
        set(ax, 'Layer', 'top'); box(ax, 'on'); set(ax, 'LineWidth', 0.8, 'XColor', 'k', 'YColor', 'k');
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
        % Stays unsmoothed on purpose. At 3.94 px/cell the stepping IS visible
        % here, but it is the real GridFittedBottom staircase at dz = 1 m --
        % 'shading interp' would render the bathymetry as a smooth ramp the
        % model never integrates. imagesc also draws the surface row that
        % pcolor+flat discards, since pcolor colours each face from its
        % lower-left corner and so never shows the last row or column.
        h  = raster_(ax, Xz, Zz, nan(size(Xz)), 'flat');
        set(ax,'YDir','normal');
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

if no_TS
    kmap = pmkmp(100,'swtth');
    [ax3,h3,cb3]    = panel_xy(1, 'Cross-shore Vel.', clim_uxy,  cmocean('balance'), '(m s^{-1})', true);
    [ax4,h4,cb4]    = panel_xy(2, 'Along-shore Vel.', clim_vcxy, cmocean('balance'), '(m s^{-1})', false);
    [ax9,h9,cb9]    = panel_xy(3, 'Vorticity/f',      clim_roxy, cmocean('balance'), '(Ro)', false);
    [ax10,h10,cb10] = panel_xy(4, 'Kinetic Energy',   clim_kexy, kmap, '(m^2 s^{-2})', false);

    [ax7,h7,cb7]    = panel_xz(9,  clim_uxz,  cmocean('balance'), '(m s^{-1})', true);
    [ax8,h8,cb8]    = panel_xz(10, clim_vcxz, cmocean('balance'), '(m s^{-1})', false);
    [ax12,h12,cb12] = panel_xz(11, clim_roxz, cmocean('balance'), '(Ro)', false);
    [ax13,h13,cb13] = panel_xz(12, clim_kexz, kmap, '(m^2 s^{-2})', false);
elseif full_panel
    [ax1,h1,cb1] = panel_xy(1, 'Temperature',  climT_xy,  cmocean('thermal'), '(°C)', true);
    [ax2,h2,cb2] = panel_xy(2, 'Salinity',     climS_xy,  cmocean('haline'),  '(psu)', false);
    [ax3,h3,cb3] = panel_xy(3, 'Cross-shore Vel.', clim_uxy, cmocean('balance'), '(m s^{-1})', false);
    [ax4,h4,cb4] = panel_xy(4, 'Along-shore Vel.', clim_vcxy, cmocean('balance'), '(m s^{-1})', false);
    [ax9,h9,cb9] = panel_xy(5, 'Vorticity/f', clim_roxy, cmocean('balance'), '(Ro)', false);
    kmap = pmkmp(100,'swtth');
    [ax10,h10,cb10] = panel_xy(6, 'Kinetic Energy', clim_kexy, kmap, '(m^2 s^{-2})', false);

    [ax5,h5,cb5] = panel_xz(13, climT_xz,  cmocean('thermal'), '(°C)', true);
    [ax6,h6,cb6] = panel_xz(14, climS_xz,  cmocean('haline'), '(psu)', false);
    [ax7,h7,cb7] = panel_xz(15, clim_uxz, cmocean('balance'), '(m s^{-1})', false);
    [ax8,h8,cb8] = panel_xz(16, clim_vcxz, cmocean('balance'), '(m s^{-1})', false);
    [ax12,h12,cb12] = panel_xz(17, clim_roxz, cmocean('balance'), '(Ro)', false);
    [ax13,h13,cb13] = panel_xz(18, clim_kexz, kmap, '(m^2 s^{-2})', false);
else
    % Top rows: surface XY (each spans 2 rows)
    [ax1,h1,cb1] = panel_xy(1, 'Temperature',  climT_xy,  cmocean('thermal'), '(°C)', true);
    [ax2,h2,cb2] = panel_xy(2, 'Salinity',     climS_xy,  cmocean('haline'),  '(psu)', false);
    [ax3,h3,cb3] = panel_xy(3, vel_label_xy,   climv_xy,  cmocean('balance'), vel_cb_label, false);
    if strcmp(vort_comp, 'KE')
        vort_cmap = pmkmp(100,'swtth');
    else
        vort_cmap = cmocean('balance');
    end
    [ax4,h4,cb4] = panel_xy(4, vort_label_xy,  climRo_xy, vort_cmap, vort_cb_label, false);

    % Bottom row: XZ transect
    [ax5,h5,cb5] = panel_xz(9,  climT_xz,  cmocean('thermal'), '(°C)', true);
    [ax6,h6,cb6] = panel_xz(10, climS_xz,  cmocean('haline'), '(psu)', false);
    [ax7,h7,cb7] = panel_xz(11, climv_xz,  cmocean('balance'), vel_cb_label, false);
    [ax8,h8,cb8] = panel_xz(12, climRo_xz, vort_cmap, vort_cb_label, false);
end

drawnow;
% ctype(i) = which quantity column i shows (1 T, 2 S, 3 u, 4 v, 5 Ro, 6 KE);
% drives the colorbar position / tick rules below
if no_TS
    axs = [ax3, ax4, ax9, ax10];
    cbs = [cb3, cb4, cb9, cb10];
    n_cols = 4;
    ctype  = [3 4 5 6];
elseif full_panel
    axs = [ax1, ax2, ax3, ax4, ax9, ax10];
    cbs = [cb1, cb2, cb3, cb4, cb9, cb10];
    n_cols = 6;
    ctype  = 1:6;
else
    axs = [ax1, ax2, ax3, ax4];
    cbs = [cb1, cb2, cb3, cb4];
    n_cols = 4;
    ctype  = 1:4;
end

for i = 1:n_cols
    ci = ctype(i);   % quantity shown in this column
    axP = axs(i).Position;
    % =========================================================================
    % 🟢 EDIT COLORBAR POSITIONS FOR TOP PANELS (XY) HERE
    %
    % axP(1) = left edge of panel     axP(3) = width of panel
    % axP(2) = bottom edge of panel   axP(4) = height of panel
    % =========================================================================
    cw = 0.02; % Colorbar width
    ch = axP(4) * 0.18; % Colorbar height
    
    if ci == 1     % Temperature
        xpos = axP(1) + axP(3) - cw + 0.02;
        ypos = axP(2) + axP(4) - ch - 0.04;
    elseif ci == 2 % Salinity
        xpos = axP(1) + axP(3) - cw + 0.01;
        ypos = axP(2) + axP(4) - ch - 0.04;
    elseif ci == 3 % Cross-shore Vel (u)
        xpos = axP(1) + axP(3) - cw + 0.0;
        ypos = axP(2) + axP(4) - ch - 0.04;
    elseif ci == 4 % Along-shore Vel (v)
        xpos = axP(1) + axP(3) - cw + 0.0;
        ypos = axP(2) + axP(4) - ch - 0.04;
    elseif ci == 5 % Vorticity
        xpos = axP(1) + axP(3) - cw + 0.0;
        ypos = axP(2) + axP(4) - ch - 0.04;
    elseif ci == 6 % Kinetic Energy
        xpos = axP(1) + axP(3) - cw - 0.01;
        ypos = axP(2) + axP(4) - ch - 0.04;
    else
        xpos = axP(1) + axP(3) - cw;
        ypos = axP(2) + axP(4) - ch - 0.04;
    end 
    cbs(i).Position = [xpos, ypos, cw, ch];
    cbs(i).YAxisLocation = 'left'; % tick labels inside to the left
    cbs(i).FontSize = 11;
    cbs(i).Title.FontSize = 11;
    
    if ci == 1 || ci == 2
        % Calculate exactly 3 ticks, ensuring they fall inside the limits
        cmin = ceil(cbs(i).Limits(1) * 100) / 100; 
        cmax = floor(cbs(i).Limits(2) * 100) / 100;
        mid = round((cmin + cmax) / 2, 2);
        
        ticks = unique([cmin, mid, cmax]);
        cbs(i).Ticks = ticks;
        cbs(i).TickLabels = sprintfc('%.2f', ticks);
    elseif ci == 3 || ci == 4 || ci == 5
        % U, V, Ro: exactly 3 ticks; 2 decimals for narrow ranges (as in vishoals_still.m)
        L1 = cbs(i).Limits(1); L2 = cbs(i).Limits(2);
        if (L2 - L1) < 0.21
            cmin = ceil(L1 * 100) / 100;  cmax = floor(L2 * 100) / 100;
            mid = round((cmin + cmax) / 2, 2);  fmt = '%.2f';
        else
            cmin = ceil(L1 * 10) / 10;  cmax = floor(L2 * 10) / 10;
            mid = round((cmin + cmax) / 2, 1);  fmt = '%.1f';
        end
        ticks = unique([cmin, mid, cmax]);
        cbs(i).Ticks = ticks;
        cbs(i).TickLabels = sprintfc(fmt, ticks);
    end
end

if no_TS
    axs2 = [ax7, ax8, ax12, ax13];
    cbs2 = [cb7, cb8, cb12, cb13];
elseif full_panel
    axs2 = [ax5, ax6, ax7, ax8, ax12, ax13];
    cbs2 = [cb5, cb6, cb7, cb8, cb12, cb13];
else
    axs2 = [ax5, ax6, ax7, ax8];
    cbs2 = [cb5, cb6, cb7, cb8];
end

% Force identical x-limits between each XY panel and the XZ panel below it
xlim_common = [max(min(xc_km_top), min(xc_km_midy)), min(max(xc_km_top), max(xc_km_midy))];
for i = 1:n_cols
    linkaxes([axs(i), axs2(i)], 'x');
    xlim(axs(i), xlim_common);
end
for i = 1:n_cols
    ci = ctype(i);   % quantity shown in this column
    axP = axs2(i).Position;
    % =========================================================================
    % 🟢 EDIT COLORBAR POSITIONS FOR BOTTOM PANELS (XZ) HERE
    % =========================================================================
    cw = 0.015; % Colorbar width
    ch = axP(4) * 0.35; % Colorbar height
    
    if ci == 1     % Temperature
        xpos = axP(1) + 0.03;
        ypos = axP(2) + 0.02;
    elseif ci == 2 % Salinity
        xpos = axP(1) + 0.015;
        ypos = axP(2) + 0.02;
    elseif ci == 3 % Cross-shore Vel (u)
        xpos = axP(1) + 0.005;
        ypos = axP(2) + 0.02;
    elseif ci == 4 % Along-shore Vel (v)
        xpos = axP(1) + 0.00;
        ypos = axP(2) + 0.02;
    elseif ci == 5 % Vorticity
        xpos = axP(1) + 0.0;
        ypos = axP(2) + 0.02;
    elseif ci == 6 % Kinetic Energy
        xpos = axP(1) + 0.02;
        ypos = axP(2) + 0.02;
    else
        xpos = axP(1) + 0.01;
        ypos = axP(2) + 0.02;
    end  
    cbs2(i).Position = [xpos, ypos, cw, ch];
    cbs2(i).YAxisLocation = 'right'; % tick labels inside to the right
    cbs2(i).FontSize = 11;
    cbs2(i).Title.FontSize = 11;
    
    if ci == 1 || ci == 2
        % Calculate exactly 3 ticks, ensuring they fall inside the limits
        cmin = ceil(cbs2(i).Limits(1) * 100) / 100; 
        cmax = floor(cbs2(i).Limits(2) * 100) / 100;
        mid = round((cmin + cmax) / 2, 2);
        
        ticks = unique([cmin, mid, cmax]);
        cbs2(i).Ticks = ticks;
        cbs2(i).TickLabels = sprintfc('%.2f', ticks);
    elseif ci == 3 || ci == 4 || ci == 5
        % U, V, Ro: exactly 3 ticks; 2 decimals for narrow ranges (as in vishoals_still.m)
        L1 = cbs2(i).Limits(1); L2 = cbs2(i).Limits(2);
        if (L2 - L1) < 0.21
            cmin = ceil(L1 * 100) / 100;  cmax = floor(L2 * 100) / 100;
            mid = round((cmin + cmax) / 2, 2);  fmt = '%.2f';
        else
            cmin = ceil(L1 * 10) / 10;  cmax = floor(L2 * 10) / 10;
            mid = round((cmin + cmax) / 2, 1);  fmt = '%.1f';
        end
        ticks = unique([cmin, mid, cmax]);
        cbs2(i).Ticks = ticks;
        cbs2(i).TickLabels = sprintfc(fmt, ticks);
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
hIso9 = []; hIso10= []; hIso11= [];
hIso12= []; hIso13= [];
for k = 1:numel(axs)
    set(axs(k), 'NextPlot','add');
end

% ---------- output directory ----------
outdir = p.Results.outdir;
if isempty(outdir)
    outdir = fullfile(fileparts(mfilename('fullpath')), 'animations');
end
if ~exist(outdir,'dir'), mkdir(outdir); end

clip_str = '';
if clipped
    clip_str = '_clipped';
end

if no_TS
    giffile = fullfile(outdir, [run_tag '_8panel_uvRoKE' clip_str '.gif']);
    mp4_filename = [run_tag '_8panel_uvRoKE' clip_str '.mp4'];
elseif full_panel
    giffile = fullfile(outdir, [run_tag '_14panel' clip_str '.gif']);
    mp4_filename = [run_tag '_14panel' clip_str '.mp4'];
else
    giffile = fullfile(outdir, [run_tag '_8panel' clip_str '.gif']);
    mp4_filename = [run_tag '_8panel' clip_str '.mp4'];
end

% Delete stale GIF to avoid GIF89a append errors (GIF only written when mp4 is off)
if ~want_mp4 && exist(giffile, 'file')
    delete(giffile);
    fprintf('Deleted stale GIF: %s\n', giffile);
end
if want_mp4
    % MATLAB's MPEG-4 profile does not exist on Linux; video_writer_open
    % falls back to Motion-JPEG AVI and video_writer_finish transcodes it.
    mp4_path = fullfile(outdir, mp4_filename);
    [vwriter, avi_path] = video_writer_open(mp4_path, fps);
end

% ---------- animate ----------
delay    = 1/fps;
is_first = true;
frame_size = [];
crop_box   = [];
set(fig, 'InvertHardcopy', 'off', 'PaperPositionMode', 'auto');   % keep white bg, on-screen layout

for t = anim_start:frame_skip:Nt
    if no_TS
        % No T/S: mask land / immersed cells where both velocity components are zero
        uxy  = read_surf_safe('u_c', t)';  vcxy = read_surf_safe('v_c', t)';
        roxy = read_surf_safe('Ro', t)';   kexy = read_surf_safe('KE', t)';
        land_xy = (uxy == 0 & vcxy == 0);
        uxy(land_xy) = NaN; vcxy(land_xy) = NaN; roxy(land_xy) = NaN; kexy(land_xy) = NaN;
        setC_(h3, uxy);  setC_(h4, vcxy);
        setC_(h9, roxy); setC_(h10, kexy);

        uxz  = read_xz_safe('u_c', t)';  vcxz = read_xz_safe('v_c', t)';
        roxz = read_xz_safe('Ro', t)';   kexz = read_xz_safe('KE', t)';
        land_xz = (uxz == 0 & vcxz == 0);
        uxz(land_xz) = NaN; vcxz(land_xz) = NaN; roxz(land_xz) = NaN; kexz(land_xz) = NaN;
        setC_(h7, uxz);   setC_(h8, vcxz);
        setC_(h12, roxz); setC_(h13, kexz);
    else
    % Surface XY — transpose for pcolor (Ny×Nx)
    Txy  = surf_tracer('T', t)';
    Sxy  = surf_tracer('S', t)';
    
    if full_panel
        uxy = read_surf_safe('u_c', t)';
        vcxy = read_surf_safe('v_c', t)';
        roxy = read_surf_safe('Ro', t)';
        kexy = read_surf_safe('KE', t)';
        
        uxy(isnan(Txy)) = NaN; vcxy(isnan(Txy)) = NaN;
        roxy(isnan(Txy)) = NaN; kexy(isnan(Txy)) = NaN;
        
        setC_(h1, Txy); setC_(h2, Sxy);
        setC_(h3, uxy); setC_(h4, vcxy);
        setC_(h9, roxy); setC_(h10, kexy);
    else
        vxy  = read_surf_safe(vel_ncvar, t)';
        Roxy = read_surf_safe(vort_ncvar, t)';
        vxy(isnan(Txy)) = NaN; Roxy(isnan(Txy)) = NaN;
        
        setC_(h1, Txy);
        setC_(h2, Sxy);
        setC_(h3, vxy);
        setC_(h4, Roxy);
    end

    % XZ transect — transpose for pcolor (Nz×Nx)
    Txz  = xz_tracer('T', t)';
    Sxz  = xz_tracer('S', t)';
    
    if full_panel
        uxz = read_xz_safe('u_c', t)';
        vcxz = read_xz_safe('v_c', t)';
        roxz = read_xz_safe('Ro', t)';
        kexz = read_xz_safe('KE', t)';
        
        uxz(isnan(Txz)) = NaN; vcxz(isnan(Txz)) = NaN;
        roxz(isnan(Txz)) = NaN; kexz(isnan(Txz)) = NaN;
        
        setC_(h5, Txz); setC_(h6, Sxz);
        setC_(h7, uxz); setC_(h8, vcxz);
        setC_(h12, roxz); setC_(h13, kexz);
    else
        vxz  = read_xz_safe(vel_ncvar, t)';
        Roxz = read_xz_safe(vort_ncvar, t)';
        vxz(isnan(Txz)) = NaN; Roxz(isnan(Txz)) = NaN;
        
        setC_(h5, Txz);
        setC_(h6, Sxz);
        setC_(h7, vxz);
        setC_(h8, Roxz);
    end
    end   % if no_TS

    % --- TEOS-10 σ0 isopycnals ---
    if want_iso
        % XY surface isopycnals
        bad_xy = ~isfinite(Txy) | ~isfinite(Sxy);
        Tsurf = Txy; Ssurf = Sxy;
        Tsurf(bad_xy) = NaN; Ssurf(bad_xy) = NaN;
        SA_xy = gsw_SA_from_SP(Ssurf, P_xy, lon, lat);
        CT_xy = gsw_CT_from_t(SA_xy, Tsurf, P_xy);
        sigma0_xy = gsw_sigma0(SA_xy, CT_xy);

        fin_xy = sigma0_xy(isfinite(sigma0_xy));
        if ~isempty(fin_xy) && range(fin_xy) > 0
            for k = 1:numel(axs)
                delete(findobj(axs(k), 'Type', 'contour'));
                [C_xy, h_xy] = contour(axs(k), Xc, Yc, sigma0_xy, rho_levels, ...
                    'Color', iso_color, 'LineWidth', iso_lw);
                if mod(t, clabel_every) == 1 && ~isempty(C_xy)
                    clabel(C_xy, h_xy, 'FontSize', 7, 'Color', iso_color, 'LabelSpacing', 400);
                end
            end
        end

        % XZ cross-section isopycnals
        bad = ~isfinite(Txz) | ~isfinite(Sxz);
        Ttmp = Txz; Stmp = Sxz;
        Ttmp(bad) = NaN; Stmp(bad) = NaN;
        SA = gsw_SA_from_SP(Stmp, P, lon, lat);
        CT = gsw_CT_from_t(SA, Ttmp, P);
        sigma0_xz = gsw_sigma0(SA, CT);

        for k = 1:numel(axs2)
            delete(findobj(axs2(k), 'Type', 'contour'));
            [C_xz, h_xz] = contour(axs2(k), Xz, Zz, sigma0_xz, rho_levels, ...
                'Color', iso_color, 'LineWidth', iso_lw);
            if mod(t, clabel_every) == 1 && ~isempty(C_xz)
                clabel(C_xz, h_xz, 'FontSize', 8, 'Color', iso_color, 'LabelSpacing', 400);
            end
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
    % No drawnow here: print renders the figure itself, so the flush was a
    % second render per frame. Measured with identical work either side, it
    % cost -2.6 ms, i.e. nothing, while print dominated. It matters now that
    % print has dropped from 3.6 s to 0.46 s -- at that point the stray render
    % was about half the remaining frame time.
    %
    % Frame capture stays on the print engine (as exportgraphics does for the
    % stills) rather than getframe: on headless MATLAB (-nodisplay, e.g. Casper)
    % getframe captures a screen-shrunk figure, which shifts colorbars/labels.
    img = print(fig, '-RGBImage', sprintf('-r%d', frame_dpi));
    % Trim the white margins as exportgraphics does for the stills; the crop box
    % is fixed from the first frame so every frame has the same size
    if isempty(crop_box)
        nonwhite = any(img < 250, 3);
        rr = find(any(nonwhite, 2));  cc = find(any(nonwhite, 1));
        pad = 4;
        crop_box = [max(1, rr(1)-pad), min(size(img,1), rr(end)+pad), ...
                    max(1, cc(1)-pad), min(size(img,2), cc(end)+pad)];
        crop_box(2) = crop_box(1) + 2*floor((crop_box(2)-crop_box(1)+1)/2) - 1;   % even height
        crop_box(4) = crop_box(3) + 2*floor((crop_box(4)-crop_box(3)+1)/2) - 1;   % even width
        frame_size  = size(img, [1 2]);
    elseif ~isequal(size(img, [1 2]), frame_size)
        img = imresize(img, frame_size);
    end
    img = img(crop_box(1):crop_box(2), crop_box(3):crop_box(4), :);
    if want_mp4
        writeVideo(vwriter, img);
    else
        [A, map] = rgb2ind(img, 256);
        if is_first
            imwrite(A, map, giffile, 'gif', 'LoopCount', Inf, 'DelayTime', delay);
            is_first = false;
        else
            imwrite(A, map, giffile, 'gif', 'WriteMode', 'append', 'DelayTime', delay);
        end
    end
end

if want_mp4
    video_writer_finish(vwriter, mp4_path, avi_path, fps);
else
    fprintf('Wrote GIF: %s\n', giffile);
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
