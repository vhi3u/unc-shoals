function eke_map(run_tag, varargin)
% eke_map(run_tag, 'datadir', '/path/to/nc', ...)
%
% Plots depth-averaged Eddy Kinetic Energy (EKE) as a 2D XY heatmap.
%
% EKE = 0.5 * ( <uu>_t - <u>_t^2  +  <vv>_t - <v>_t^2  +  <ww>_t - <w>_t^2 )
%
% where <·>_t is the time average over selected snapshots,
% and all fields are already depth-averaged (from zavg_xy_<run_tag>.nc).
%
% Reads:
%   zavg_xy_<run_tag>.nc — u_zavg, v_zavg, w_zavg, uu_zavg, vv_zavg, ww_zavg
%
% Layout: single pcolor plot of EKE(x,y)
%
% Requires: cmocean

% ---------- options ----------
p = inputParser;
addParameter(p, 'datadir',    '', @ischar);
addParameter(p, 'outdir',     '', @ischar);
addParameter(p, 'clim_eke',   [], @(v)isnumeric(v) && numel(v)==2);
addParameter(p, 'time_range', [], @(v)isnumeric(v) && numel(v)==2);  % [t1 t2] in days
addParameter(p, 'log_scale',  false, @islogical);  % use log10 color scale
parse(p, varargin{:});

datadir = p.Results.datadir;
if isempty(datadir), datadir = pwd; end
outdir = p.Results.outdir;
if isempty(outdir), outdir = fullfile(fileparts(mfilename('fullpath')), 'animations'); end
if ~exist(outdir,'dir'), mkdir(outdir); end

clim_eke_user = p.Results.clim_eke;
time_range    = p.Results.time_range;
use_log       = p.Results.log_scale;

% ---------- filename ----------
fname = fullfile(datadir, ['time_avg_3d_' run_tag '.nc']);
if ~isfile(fname)
    fname = fullfile(datadir, ['zavg_xy_' run_tag '.nc']);
end
if ~isfile(fname), error('Output file not found for tag: %s', run_tag); end
fprintf('Reading data from: %s\n', fname);

% ---------- coordinates ----------
xc = try_ncread_(fname, {'xC','x_caa','xc'});
yc = try_ncread_(fname, {'yC','y_aca','yc'});
time = ncread(fname, 'time');

xc_km = xc / 1e3;
yc_km = yc / 1e3;

% ---------- load and process ----------
% Helper to load, get last time step, and depth-average if needed
u_mean  = load_last_processed(fname, {'u_c', 'u', 'u_zavg'});
v_mean  = load_last_processed(fname, {'v_c', 'v', 'v_zavg'});
w_mean  = load_last_processed(fname, {'w_c', 'w', 'w_zavg'});
uu_mean = load_last_processed(fname, {'uu_c', 'uu', 'uu_zavg'});
vv_mean = load_last_processed(fname, {'vv_c', 'vv', 'vv_zavg'});
ww_mean = load_last_processed(fname, {'ww_c', 'ww', 'ww_zavg'});

% ---------- compute EKE ----------
% EKE = 0.5 * ( <uu> - <u>^2 + <vv> - <v>^2 + <ww> - <w>^2 )
EKE = 0.5 * ( (uu_mean - u_mean.^2) + ...
              (vv_mean - v_mean.^2) + ...
              (ww_mean - w_mean.^2) );

% Mask zeros / immersed boundary
EKE(EKE == 0) = NaN;

% Clamp any small negative values from numerical noise
EKE = max(EKE, 0);

avg_label = 'Last Time-Averaged Window (Depth-Averaged)';
fprintf('EKE size: %s, range: [%.2e, %.2e] m²/s²\n', ...
    mat2str(size(EKE)), min(EKE(:),[],'omitnan'), max(EKE(:),[],'omitnan'));

% ---------- color limits ----------
if isempty(clim_eke_user)
    if use_log
        valid = EKE(EKE > 0 & ~isnan(EKE));
        if isempty(valid)
            clim_eke = [-6 -2];
        else
            clim_eke = [floor(log10(min(valid))), ceil(log10(max(valid)))];
        end
    else
        clim_eke = [0, max(EKE(:), [], 'omitnan')];
        if clim_eke(2) == 0 || isnan(clim_eke(2))
            clim_eke = [0 1];
        end
    end
else
    clim_eke = clim_eke_user;
end

% ---------- figure ----------
fig = figure('Color','w', 'Position', [100 100 700 600]);

if use_log
    plot_data = log10(max(EKE, 1e-20));  % avoid log(0)
    pcolor(xc_km, yc_km, plot_data.');
    shading flat; axis tight equal;
    colormap(gca, cmocean('thermal'));
    caxis(clim_eke);
    cb = colorbar;
    cb.Label.String = 'log_{10} EKE  [m²/s²]';
else
    pcolor(xc_km, yc_km, EKE.');
    shading flat; axis tight equal;
    colormap(gca, cmocean('thermal'));
    caxis(clim_eke);
    cb = colorbar;
    cb.Label.String = 'EKE  [m²/s²]';
end

cb.Label.FontSize = 12;
xlabel('x [km]', 'FontSize', 12);
ylabel('y [km]', 'FontSize', 12);
title(sprintf('%s: Depth-Averaged EKE\n%s', run_tag, avg_label), ...
    'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'none');

drawnow;

% ---------- save ----------
outfile = fullfile(outdir, [run_tag '_eke_map.png']);
set(fig, 'PaperPositionMode', 'auto');
print(fig, outfile, '-dpng', '-r200');
fprintf('Saved figure: %s\n', outfile);
end

% ================== helpers ==================

function data = load_last_processed(fn, varname_or_list)
    % Accept either a single name (string) or list of names to try (cell array)
    if ischar(varname_or_list)
        varnames = {varname_or_list};
    else
        varnames = varname_or_list;
    end

    for i = 1:numel(varnames)
        try
            varname = varnames{i};
            vinfo = ncinfo(fn, varname);
            vsize = vinfo.Size;
            raw = ncread(fn, varname);

            nd = numel(vsize);
            if nd == 4
                % (Nx, Ny, Nz, Nt) -> Mean over z, last t
                data = mean(raw(:,:,:,end), 3, 'omitnan');
            elseif nd == 3
                % (Nx, Ny, Nt) -> last t
                data = raw(:,:,end);
            elseif nd == 2
                % Already 2D
                data = raw;
            else
                data = raw;
            end
            data = squeeze(data);
            return;  % Success, exit loop
        catch
        end
    end
    % All names failed
    data = [];
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
error('Could not find coordinate variable. Tried: {%s}. Available: %s', ...
    strjoin(names, ', '), avail);
end
