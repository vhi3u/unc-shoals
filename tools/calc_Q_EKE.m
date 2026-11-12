function [Q_diag_mSv, Q_Ek_mSv, EKE] = calc_Q_EKE(run_name, varargin)
% [Q_diag_mSv, Q_Ek_mSv, EKE] = calc_Q_EKE(run_name, 'datadir', '/glade/work/nguyen/unc-shoals/nc')
%
% Calculates and prints:
%   1. Diagnosed Offshore Volume Transport (Q_diag in mSv) at x = 40 km, y = [80, 200] km (u > 0, z >= -25m)
%   2. Theoretical Offshore Ekman Volume Transport (Q_Ek in mSv) for alongshore wind stress tau_y
%   3. Mean Eddy Kinetic Energy (EKE in m^2/s^2)

% ---------- Options ----------
p = inputParser;
addParameter(p, 'datadir', '/glade/work/nguyen/unc-shoals/nc', @ischar);
addParameter(p, 'x_target', 40, @(x)isnumeric(x)&&isscalar(x));
addParameter(p, 'y_range', [80, 200], @(v)isnumeric(v)&&numel(v)==2);
addParameter(p, 'tau', [], @(x)isempty(x)||(isnumeric(x)&&isscalar(x)));
parse(p, varargin{:});

datadir     = p.Results.datadir;
x_target_km = p.Results.x_target;
y_range     = p.Results.y_range;
tau_user    = p.Results.tau;

if nargin < 1 || isempty(run_name)
    run_name = 'sweep_Hs20_Ls40k';
end

% ---------- Locate NetCDF File ----------
fname = find_run_ncfile(datadir, run_name);

if isempty(fname) || ~isfile(fname)
    error('Could not locate NetCDF data file for run "%s" in datadir: %s', run_name, datadir);
end

% Check for metadata file
meta_fname = fullfile(fileparts(fname), ['sweep_metadata_' run_name '.nc']);
if ~isfile(meta_fname)
    meta_fname = fullfile(datadir, ['sweep_metadata_' run_name '.nc']);
end

fprintf('=== Processing Metrics for Run: %s ===\n', run_name);
fprintf('  Data File : %s\n', fname);
fprintf('  Target X  : %.1f km | Y Range: %.1f - %.1f km\n', x_target_km, y_range(1), y_range(2));

% ---------- Load Coordinates ----------
xc = try_ncread_(fname, {'xC','xc','x','x_caa'}); xc_km = xc(:) / 1e3;
yc = try_ncread_(fname, {'yC','yc','y','y_aca'}); yc_km = yc(:) / 1e3;
zc = try_ncread_(fname, {'zC','zc','z','z_aac'}); zc = zc(:);

if numel(zc) > 1
    dz = abs(zc(1) - zc(2));
else
    dz = 1;
end

[~, ix] = min(abs(xc_km - x_target_km));

% ---------- Calculate Diagnosed Offshore Volume Transport (Q_diag) ----------
u_2d  = try_load_slice(fname, {'u_c', 'u'}, ix);
v_2d  = try_load_slice(fname, {'v_c', 'v'}, ix);
w_2d  = try_load_slice(fname, {'w_c', 'w'}, ix);

uu_2d = try_load_slice(fname, {'uu'}, ix);
vv_2d = try_load_slice(fname, {'vv'}, ix);
ww_2d = try_load_slice(fname, {'ww'}, ix);

% Positive offshore velocity (u > 0) in upper 25 m
u_pos = u_2d;
u_pos(u_pos < 0) = 0;

iz_upper = find(zc >= -25);
if isempty(iz_upper), iz_upper = 1:size(u_2d, 2); end

u_depth_int = sum(u_pos(:, iz_upper), 2, 'omitnan') * dz;

iy_Q = find(yc_km >= y_range(1) & yc_km <= y_range(2));
if isempty(iy_Q), iy_Q = find(yc_km >= 20); end

Q_diag_m3s = trapz(yc(iy_Q), u_depth_int(iy_Q));
Q_diag_mSv = Q_diag_m3s / 1e3; % Convert m^3/s to mSv

% ---------- Calculate Offshore Ekman Volume Transport (Q_Ek) ----------
if ~isempty(tau_user)
    tau_y = tau_user;
elseif isfile(meta_fname)
    try
        tau_y = ncreadatt(meta_fname, '/', 'wind_stress');
    catch
        tau_y = parse_tau_from_name(run_name);
    end
else
    tau_y = parse_tau_from_name(run_name);
end

f    = 2 * (2*pi/86400) * sin(deg2rad(35.2480)); % Coriolis parameter (~ 8.44e-5 s^-1)
rho0 = 1024.0;                                   % Reference density (kg/m^3)
delta_Y_m = (y_range(2) - y_range(1)) * 1e3;     % Alongshore distance in meters (120,000 m)

% Positive offshore Ekman transport (tau_y > 0 drives +x offshore Ekman transport)
tau_y_pos = max(0, tau_y);
Q_Ek_m3s = (tau_y_pos * delta_Y_m) / (rho0 * f);
Q_Ek_mSv = Q_Ek_m3s / 1e3; % Convert m^3/s to mSv

% Eddy component Q_eddy
Q_eddy_mSv = Q_diag_mSv - Q_Ek_mSv;

% ---------- Calculate Mean Eddy Kinetic Energy (EKE) ----------
if ~isempty(v_2d) && ~isempty(w_2d) && ~isempty(uu_2d) && ~isempty(vv_2d) && ~isempty(ww_2d)
    u_m = u_2d; u_m(u_m == 0) = NaN;
    v_m = v_2d; v_m(v_m == 0) = NaN;
    w_m = w_2d; w_m(w_m == 0) = NaN;
    
    uu_m = uu_2d; uu_m(uu_m == 0) = NaN;
    vv_m = vv_2d; vv_m(vv_m == 0) = NaN;
    ww_m = ww_2d; ww_m(ww_m == 0) = NaN;
    
    EKE_2d = 0.5 * ((uu_m - u_m.^2) + (vv_m - v_m.^2) + (ww_m - w_m.^2));
    
    iy_EKE = find(yc_km >= y_range(1) & yc_km <= y_range(2));
    if isempty(iy_EKE), iy_EKE = find(yc_km >= 20); end
    
    EKE = mean(EKE_2d(iy_EKE, :), 'all', 'omitnan');
else
    EKE = NaN;
    warning('Velocity variance fields (uu, vv, ww) missing or empty.');
end

% ---------- Print Formatted Summary ----------
fprintf('\n---------------- RESULTS ----------------\n');
fprintf('  Run Name                     : %s\n', run_name);
fprintf('  Alongshore Wind Stress (tau_y): %.3f N/m^2\n', tau_y);
fprintf('  Diagnosed Offshore Q (mSv)   : %.4f mSv (%.3e m^3/s)\n', Q_diag_mSv, Q_diag_m3s);
fprintf('  Theoretical Ekman Q_Ek (mSv) : %.4f mSv (%.3e m^3/s)\n', Q_Ek_mSv, Q_Ek_m3s);
fprintf('  Eddy Transport Q_eddy (mSv)  : %.4f mSv\n', Q_eddy_mSv);
fprintf('  Mean EKE (m^2/s^2)           : %.4e m^2/s^2\n', EKE);
fprintf('  sqrt(EKE) (m/s)              : %.4f m/s\n', sqrt(EKE));
fprintf('-----------------------------------------\n\n');

end

% ================== Helper Functions ==================

function tau = parse_tau_from_name(run_tag)
tau = 0;
if contains(run_tag, 'wind_m0p05'), tau = -0.05;
elseif contains(run_tag, 'wind_m0p03'), tau = -0.03;
elseif contains(run_tag, 'wind_m0p01'), tau = -0.01;
elseif contains(run_tag, 'wind_0p01'),  tau = 0.01;
elseif contains(run_tag, 'wind_0p03'),  tau = 0.03;
elseif contains(run_tag, 'wind_0p05'),  tau = 0.05;
end
end

function fname = find_run_ncfile(datadir, run_name)
fname = '';
candidates = {
    fullfile(datadir, ['time_avg_3d_' run_name '.nc']), ...
    fullfile(datadir, run_name, ['time_avg_3d_' run_name '.nc']), ...
    fullfile(datadir, [run_name '.nc']), ...
    fullfile(datadir, ['top_' run_name '.nc'])
};

for i = 1:numel(candidates)
    if isfile(candidates{i})
        fname = candidates{i};
        return;
    end
end

d = dir(fullfile(datadir, ['*' run_name '*']));
for i = 1:numel(d)
    if d(i).isdir && ~ismember(d(i).name, {'.', '..'})
        sub_fname = fullfile(datadir, d(i).name, ['time_avg_3d_' run_name '.nc']);
        if isfile(sub_fname)
            fname = sub_fname;
            return;
        end
    elseif contains(d(i).name, '.nc')
        fname = fullfile(datadir, d(i).name);
        return;
    end
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
error('Could not find coordinate variable in %s', fname);
end

function data = try_load_slice(fname, names, ix)
for i = 1:numel(names)
    try
        vinfo = ncinfo(fname, names{i});
        vsize = vinfo.Size;
        nd = numel(vsize);
        if nd == 4
            data = ncread(fname, names{i}, [ix 1 1 vsize(4)], [1 vsize(2) vsize(3) 1]);
        elseif nd == 3
            if vsize(3) > 10
                data = ncread(fname, names{i}, [ix 1 1], [1 vsize(2) vsize(3)]);
            else
                data = ncread(fname, names{i}, [ix 1 vsize(3)], [1 vsize(2) 1]);
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
