function offshore_flux(run_tag, varargin)
% offshore_flux(run_tag, 'datadir', '/path/to/nc', ...)
%
% Plots depth-averaged & time-averaged XY maps of offshore heat and salt flux.
%
% Reads:
%   time_avg_3d_<run_tag>.nc — 3D time-averaged uT, uS
%
% Layout (1×2 tiles):
%   XY  ⟨uT⟩_{z,t}     |   XY  ⟨uS⟩_{z,t}
%
% Requires: cmocean

% ---------- options ----------
p = inputParser;
addParameter(p, 'datadir',    '', @ischar);
addParameter(p, 'outdir',     '', @ischar);
addParameter(p, 'climuT',     [-6.5e-3, 6.5e-3], @(v)isnumeric(v) && numel(v)==2);
addParameter(p, 'climuS',     [-3.5e-3, 3.5e-3], @(v)isnumeric(v) && numel(v)==2);
addParameter(p, 'time_range', [], @(v)isnumeric(v) && numel(v)==2);  % [t1 t2] in days
parse(p, varargin{:});

datadir = p.Results.datadir;
if isempty(datadir), datadir = pwd; end
outdir = p.Results.outdir;
if isempty(outdir), outdir = fullfile(fileparts(mfilename('fullpath')), 'animations'); end
if ~exist(outdir,'dir'), mkdir(outdir); end

climuT_user = p.Results.climuT;
climuS_user = p.Results.climuS;
time_range  = p.Results.time_range;

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

Nt = numel(time);

% ---------- time selection ----------
time_days = (time - time(1)) / 86400;

if isempty(time_range)
    tidx = 1:Nt;
    avg_label = sprintf('%.1f-%.1f days', time_days(1), time_days(end));
else
    tidx = find(time_days >= time_range(1) & time_days <= time_range(2));
    if isempty(tidx), error('No snapshots in selected time_range'); end
    avg_label = sprintf('%.1f-%.1f days', time_range(1), time_range(2));
end

% ---------- load and process ----------
% We need u, T, S, uT, uS in 3D to do the subtraction correctly before depth-averaging.
u  = load_last_3d(fname, 'u_c'); if isempty(u), u = load_last_3d(fname, 'u'); end
T  = load_last_3d(fname, 'T');
S  = load_last_3d(fname, 'S');
uT_raw = load_last_3d(fname, 'uT');
uS_raw = load_last_3d(fname, 'uS');

if isempty(uT_raw) || isempty(u) || isempty(T)
    warning('Required variables (u, T, uT) not found for calculation. Falling back to existing depth-averaged variables.');
    uT_zavg = load_last_processed(fname, 'uT'); if isempty(uT_zavg), uT_zavg = load_last_processed(fname, 'uT_zavg'); end
    uS_zavg = load_last_processed(fname, 'uS'); if isempty(uS_zavg), uS_zavg = load_last_processed(fname, 'uS_zavg'); end
else
    % Offshore Flux Calculation
    uT_offshore = uT_raw;
    uS_offshore = uS_raw;
    
    % Depth Averaging
    uT_zavg = squeeze(mean(uT_offshore, 3, 'omitnan'));
    uS_zavg = squeeze(mean(uS_offshore, 3, 'omitnan'));
end

% Mask zeros (immersed boundary)
if ~isempty(uT_zavg), uT_zavg(uT_zavg == 0) = NaN; end
if ~isempty(uS_zavg), uS_zavg(uS_zavg == 0) = NaN; end

avg_label = 'Last Time-Averaged Window';
fprintf('Processed flux maps size: %s\n', mat2str(size(uT_zavg)));

% ---------- color limits ----------
y_idx_interior = yc_km >= 10 & yc_km <= 290;
sym_clim = @(A) safe_clim([-1 1] * max(abs(A(:)), [], 'omitnan'), [-1 1]);

% ---------- figure ----------
fig = figure('Color','w', 'Position', [80 400 450 466]);
tl  = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Panel 1: Heat flux
if ~isempty(uT_zavg)
    ax1 = nexttile(tl, 1);
    pcolor(ax1, xc_km, yc_km, uT_zavg.');
    shading flat; axis tight equal; set(gca, 'YDir', 'normal');
    colormap(ax1, cmocean('balance')); 
    if isempty(climuT_user)
        uT_subset = uT_zavg(:, y_idx_interior);
        climuT = sym_clim(uT_subset); 
    else
        climuT = climuT_user; 
    end
    caxis(climuT);
    cb1 = colorbar(ax1, 'Location', 'eastoutside');
    cb1.FontSize = 16; cb1.Label.String = '\langle uT \rangle_z  [°C·m/s]'; cb1.Label.FontSize = 18;
    title(ax1, 'Heat Flux', 'FontSize', 18, 'FontWeight', 'bold');
    xlabel('x (km)', 'FontSize', 18); ylabel('y (km)', 'FontSize', 18);
    set(ax1, 'FontSize', 16);
end

% Panel 2: Salt flux
if ~isempty(uS_zavg)
    ax2 = nexttile(tl, 2);
    pcolor(ax2, xc_km, yc_km, uS_zavg.');
    shading flat; axis tight equal; set(gca, 'YDir', 'normal');
    colormap(ax2, cmocean('balance'));
    if isempty(climuS_user)
        uS_subset = uS_zavg(:, y_idx_interior);
        climuS = sym_clim(uS_subset); 
    else
        climuS = climuS_user; 
    end
    caxis(climuS);
    cb2 = colorbar(ax2, 'Location', 'eastoutside');
    cb2.FontSize = 16; cb2.Label.String = '\langle uS \rangle_z  [psu·m/s]'; cb2.Label.FontSize = 18;
    title(ax2, 'Salt Flux', 'FontSize', 18, 'FontWeight', 'bold');
    xlabel('x (km)', 'FontSize', 18); ylabel('y (km)', 'FontSize', 18);
    set(ax2, 'FontSize', 16);
end

sgtitle(tl, sprintf('%s: Depth & Time Averaged Net Fluxes\n%s', ...
    run_tag, avg_label), 'FontSize', 16, 'FontWeight','bold', 'Interpreter','none');

drawnow;

% ---------- save ----------
outfile = fullfile(outdir, [run_tag '_flux_xy.png']);
print(fig, outfile, '-dpng', '-r200');
fprintf('Saved figure: %s\n', outfile);
end

% ================== helpers ==================

function data = load_last_3d(fn, varname)
    % Loads the last time step of a variable, keeping 3D [Nx, Ny, Nz]
    try
        vinfo = ncinfo(fn, varname);
        vsize = vinfo.Size;
        nd = numel(vsize);
        
        if nd == 4
            % (Nx, Ny, Nz, Nt) -> last t
            data = ncread(fn, varname, [1 1 1 vsize(4)], [vsize(1) vsize(2) vsize(3) 1]);
        elseif nd == 3
            % (Nx, Ny, Nz) or (Nx, Ny, Nt)
            % Guessing if it's 3D spatial or 2D+time
            if vsize(3) > 10 % Likely Nz
                data = ncread(fn, varname);
            else % Likely Nt for a surface field
                data = ncread(fn, varname, [1 1 vsize(3)], [vsize(1) vsize(2) 1]);
            end
        else
            data = ncread(fn, varname);
        end
        data = squeeze(data);
    catch
        data = [];
    end
end

function data = load_last_processed(fn, varname)
    try
        vinfo = ncinfo(fn, varname);
        vsize = vinfo.Size;
        nd = numel(vsize);
        
        if nd == 4
            % (Nx, Ny, Nz, Nt) -> Mean over z, last t
            raw = ncread(fn, varname, [1 1 1 vsize(4)], [vsize(1) vsize(2) vsize(3) 1]);
            data = mean(raw, 3, 'omitnan');
        elseif nd == 3
            % (Nx, Ny, Nt) or (Nx, Ny, Nz)
            if vsize(3) < 10 && vsize(3) > 1 % probably Nt
                data = ncread(fn, varname, [1 1 vsize(3)], [vsize(1) vsize(2) 1]);
            else % probably Nz
                raw = ncread(fn, varname);
                data = mean(raw, 3, 'omitnan');
            end
        else
            data = ncread(fn, varname);
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
info = ncinfo(fname);
avail = strjoin({info.Variables.Name}, ', ');
error('Could not find coordinate variable. Tried: {%s}. Available: %s', ...
    strjoin(names, ', '), avail);
end

function clim = safe_clim(clim_in, fallback)
if any(isnan(clim_in)) || any(isinf(clim_in)) || clim_in(1) >= clim_in(2)
    clim = fallback;
else
    clim = clim_in;
end
end
