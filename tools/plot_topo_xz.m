function plot_topo_xz(run_tag, varargin)
% plot_topo_xz(run_tag, 'datadir', DATADIR, 'y_slice', Y_KM)
%
% Generates a clean PNG of the bathymetry in the X-Z plane.
%
% Referencing vishoals_clipped_bulk.m for syntax and formatting.
%
% Usage:
%   plot_topo_xz('sweep_Hs15_Ls20k');

% ---------- options ----------
p = inputParser;
addParameter(p, 'datadir', '', @ischar);
addParameter(p, 'y_slice', [], @(x)isnumeric(x)&&isscalar(x)); % y-coordinate to slice at (km)
parse(p, varargin{:});

datadir = p.Results.datadir;
if isempty(datadir), datadir = pwd; end
y_slice_km = p.Results.y_slice;

% ---------- filename guessing ----------
% Try different possible files containing bathymetry
fnames = {
    fullfile(datadir, ['midy_' run_tag '.nc']), ...
    fullfile(datadir, ['top_'  run_tag '.nc']), ...
    fullfile(datadir, ['time_avg_3d_' run_tag '.nc']), ...
    'bottom_height_TEST_clean441.nc'
};

% ... (search logic remains)
fname = '';
for i = 1:numel(fnames)
    if isfile(fnames{i})
        fname = fnames{i};
        break;
    end
end

if isempty(fname)
    error('Could not find NetCDF file for run_tag: %s (tried midy, top, etc.)', run_tag);
end

fprintf('Reading bathymetry from: %s\n', fname);

% ---------- coordinates and data ----------
xc = try_ncread(fname, {'xC','x_caa','xc'});
zc = try_ncread(fname, {'zC','z_aac','zc'});
xc_km = xc / 1000;

% Load velocity field to find the topography mask
% Note: Oceananigans NC files are typically (Nx, Ny, Nz, Nt)
try
    % Read u_c at mid-y (index 1 for midy slices) and t=1
    u = squeeze(ncread(fname, 'u_c', [1 1 1 1], [Inf 1 Inf 1]));
catch
    % Try alternative variable if u_c is missing
    u = squeeze(ncread(fname, 'T', [1 1 1 1], [Inf 1 Inf 1]));
end

% Topography profile H(x): find the transition from water to ground (or v.v.)
% In Oceananigans, immersed points are often 0 or NaN. 
% We look for the first non-zero, non-NaN index from the TOP (surface) down.
H_xz = zeros(size(xc));
for i = 1:numel(xc)
    col = u(i, :);
    % We assume surface is at the end of the column (index Nz) or start?
    % In Oceananigans, z indices usually go from bottom to top (1 to Nz).
    % So we search from Nz down to 1 for the bottom.
    idx = find(col ~= 0 & ~isnan(col), 1, 'first'); 
    
    if isempty(idx)
        H_xz(i) = min(zc); % No water found
    else
        % idx is the deepest water point index? 
        % Wait, if we are looking for the BOTTOM, we want the first LAND index
        % searching from TOP down.
        % zc goes from e.g. -50 to 0. 
        % index 1 is -50, index Nz is 0.
        % We search from index 1 up for the first non-land point.
        % Actually, let's just find the first index that is NOT land.
        idx = find(col ~= 0 & ~isnan(col), 1, 'first');
        if isempty(idx)
            H_xz(i) = max(zc); % Should not happen if all is land
        else
            H_xz(i) = zc(idx);
        end
    end
end

% ---------- figure ----------
% Half the width of vishoals_clipped.m figure [80 80 1000 600]
% so this aligns with a single panel from that layout.
fig = figure('Color','w', 'Position', [80 80 500 300]);
ax  = axes(fig, 'Color', 'w');
hold(ax, 'on');

% Step-profile for seafloor (pixelated look)
% Shift xc (centers) to edges to define the cell boundaries
dx = diff(xc_km);
if isempty(dx), dx = 0.5; end % Fallback
x_edges = [xc_km(1)-dx(1)/2; xc_km(1:end-1)+dx/2; xc_km(end)+dx(end)/2];

% Generate stair-step coordinates
x_stairs = reshape([x_edges(1:end-1)'; x_edges(2:end)'], [], 1);
z_stairs = reshape([H_xz'; H_xz'], [], 1);

% Define seafloor patch area
fill_color = [0.4 0.4 0.4]; % Gray
z_floor    = min(zc) - 5;
x_patch = [x_stairs(1); x_stairs; x_stairs(end)];
z_patch = [z_floor;   z_stairs;  z_floor];

% Fill ground with stair-step boundary
patch(ax, x_patch, z_patch, fill_color, 'EdgeColor', [0.2 0.2 0.2], 'LineWidth', 1.0);

% Formatting
xlabel(ax, 'x (km)', 'FontSize', 18);
ylabel(ax, 'z (m)',  'FontSize', 18);

set(ax, 'FontSize', 16, 'LineWidth', 1.2, 'TickDir', 'out', 'Box', 'off', 'Layer', 'top');

grid(ax, 'off');

% Clip to x = [0, 60] km
xlim(ax, [0, 60]);
ylim(ax, [min(zc)-2,  2]);

% ---------- save ----------
outdir = '/glade/work/nguyen/unc-shoals/animations';
if ~exist(outdir, 'dir'), mkdir(outdir); end
outfile = fullfile(outdir, [run_tag '_topo_xz.fig']);

% Save as .fig for manual MATLAB interaction
savefig(fig, outfile);

fprintf('Saved FIG: %s\n', outfile);
close(fig); 
end

% ================== helper ==================
function data = try_ncread(fname, names)
for i = 1:numel(names)
    try
        data = ncread(fname, names{i});
        return;
    catch
    end
end
error('Could not find any of these variables in %s: %s', fname, strjoin(names, ', '));
end