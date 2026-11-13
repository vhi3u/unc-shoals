function vishoals_8panel_bulk(cluster_name)
% vishoals_8panel_bulk(cluster_name)
%
% Animates every sweep in a cluster with vishoals, in the full_panel layout:
%   /glade/work/nguyen/unc-shoals/sweep_output/<cluster_name>/
%
% MP4s (or GIFs if want_mp4 = false) are written to:
%   /glade/work/nguyen/unc-shoals/animations/
%
% This is the bulk counterpart of run_vishoals.sh -- same function, same
% layout, one animation per run instead of one run.
%
% HOW MANY PANELS: three different numbers are in circulation and only one of
% them is current. The figure draws TWELVE -- six columns (T, S, u cross-shore,
% v along-shore, Ro, KE), each as a surface XY panel and a mid-y XZ panel,
% from tiledlayout(3, 6) at vishoals.m:448. The '8panel' in this function's
% name predates that layout, and vishoals.m:737 tags the output file
% '_14panel'. Both are left alone deliberately: renaming the function would
% break run_vishoals_8panel_bulk.sh, and changing the filename tag would
% orphan every MP4 already sitting in animations/.
%
% Example:
%   vishoals_8panel_bulk('cluster_weakwinds')

if nargin < 1 || isempty(cluster_name)
    cluster_name = 'cluster_Cnnie_1_15';
end

sweep_output_root = '/glade/work/nguyen/unc-shoals/sweep_output';
cluster_dir       = fullfile(sweep_output_root, cluster_name);

% =====================================================================
% OPTIONS
% =====================================================================

% Sweeps are discovered from the cluster directory, not hardcoded. The old
% list held whichever labels were current when this was written, so any
% later cluster matched nothing and every run was silently skipped.
[sweep_labels, sweep_tags] = discover_sweeps(cluster_dir);
sweeps = sweep_tags;

last_days  = 50;
fps        = 15;
frame_skip = 1;
% Animate daily, from 6-hourly output. snapshot_interval stays at 6 h in
% flow_over_shoals_sweep.jl because the inertial period is 20.74 h and daily
% SAMPLING would alias it to a spurious 6.35-day signal in the mesoscale band
% -- but that constraint is about the diagnostics, not the movie. Striding
% here cuts 201 frames to 51 and leaves the NetCDF untouched.
% Set [] to animate every saved frame.
sample_hours = 24;
want_mp4   = true;    % true -> MP4 only, false -> GIF only
clipped    = true;    % clip XY to x=0-100 km, y=20-200 km (XZ clipped to same x)

% Fixed clims for cross-sweep comparison (presentation)
% climv  = [-0.06, 0.06];   % u velocity (m/s)
climv = []; % Empty to allow auto-scanning
climu = [-0.1, 0.1];
climRo = [-1, 1];    % same Ro scale as every other vorticity figure
climKE = [];

outdir = '/glade/work/nguyen/unc-shoals/animations';
% =====================================================================

addpath(fileparts(mfilename('fullpath')));

fprintf('\n=== vishoals_8panel_bulk: %d sweeps ===\n\n', numel(sweeps));

failed = {};

for k = 1:numel(sweeps)
    sw      = sweeps{k};
    % Find actual folder with wildcard to handle sweep_001_... numbering
    sw_core = strrep(sw, 'sweep_', '');
    d = dir(fullfile(cluster_dir, ['*' sw_core '*']));
    d = d([d.isdir]);
    d = d(~ismember({d.name}, {'.', '..'}));
    if isempty(d)
        fprintf('  SKIP: No folder found matching *%s*\n', sw);
        failed{end+1} = sw; continue;
    end
    datadir = fullfile(cluster_dir, d(1).name);

    fprintf('[%d/%d] %s\n', k, numel(sweeps), sw);

    if ~isfile(fullfile(datadir, ['top_' sw '.nc']))
        fprintf('  SKIP: top_%s.nc not found\n', sw);
        failed{end+1} = sw; continue; %#ok<SAGROW>
    end
    if ~isfile(fullfile(datadir, ['midy_' sw '.nc']))
        fprintf('  SKIP: midy_%s.nc not found\n', sw);
        failed{end+1} = sw; continue; %#ok<SAGROW>
    end

    try
        extra = {};
        if ~isempty(climu),  extra = [extra, {'climu',  climu}];  end
        if ~isempty(climv),  extra = [extra, {'climv',  climv}];  end
        if ~isempty(climRo), extra = [extra, {'climRo', climRo}]; end
        if ~isempty(climKE), extra = [extra, {'climKE', climKE}]; end

        % 'vel' and 'vort' are deliberately NOT passed. They choose the
        % single velocity and vorticity column of the FOUR-column layout and
        % are read only in the non-full_panel branch (vishoals.m:810 and :837;
        % the auto-climit scan at :309 and :314 is gated on ~full_panel).
        % full_panel pins columns 3-6 to u_c, v_c, Ro and KE, so the
        % 'vel','u' / 'vort','Ro' that used to sit here had no effect -- u and
        % Ro were drawn as columns 3 and 5 regardless. Removing them leaves
        % every frame identical and stops the call reading as if the layout
        % were selectable here.
        vishoals(sw, ...
            'datadir',    datadir,    ...
            'outdir',     outdir,     ...
            'last_days',  last_days,  ...
            'fps',        fps,        ...
            'frame_skip', frame_skip, ...
            'sample_hours', sample_hours, ...
            'mp4',        want_mp4,   ...
            'isopycnals', false,      ...
            'full_panel', true,       ...
            'clipped',    clipped,    ...
            extra{:});
    catch ME
        fprintf('  ERROR: %s\n', ME.message);
        failed{end+1} = sw; %#ok<SAGROW>
    end
    fprintf('\n');
end

fprintf('=== Done: %d/%d succeeded ===\n', numel(sweeps)-numel(failed), numel(sweeps));
if ~isempty(failed)
    fprintf('Failed:\n');
    for j = 1:numel(failed)
        fprintf('  - %s\n', failed{j});
    end
end
end
