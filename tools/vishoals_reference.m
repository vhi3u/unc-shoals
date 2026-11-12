function vishoals_reference(cluster_name, varargin)
% vishoals_reference(cluster_name, ...)
%
% Full-panel animation (vishoals.m, 'full_panel' = true) of the sweep
% REFERENCE case only: no wind, Zs = -5 m, Ls = 40 km (run Zs5_Ls40k).
%
% Finds the run folder sweep_NNN_Zs5_Ls40k in
%   /glade/work/nguyen/unc-shoals/sweep_output/<cluster_name>/
% and calls vishoals('sweep_Zs5_Ls40k', 'datadir', <that folder>, ...).
%
% Defaults (override any of them, or pass any other vishoals option):
%   'full_panel', true, 'clipped', true, 'mp4', true, 'fps', 15,
%   'outdir', '/glade/work/nguyen/unc-shoals/animations'
%
% Examples:
%   vishoals_reference('cluster_Fnnie_1_19')
%   vishoals_reference('cluster_Fnnie_1_19', 'mp4', false, 'last_days', 50)

if nargin < 1 || isempty(cluster_name)
    cluster_name = 'cluster_Fnnie_1_19';
end

sweep_root = '/glade/work/nguyen/unc-shoals/sweep_output';
ref_core   = 'Zs5_Ls40k';                 % no wind, Zs = -5 m, Ls = 40 km
run_tag    = ['sweep_' ref_core];         % files: top_sweep_Zs5_Ls40k.nc, midy_sweep_Zs5_Ls40k.nc

% ---------- locate the reference run folder ----------
cluster_dir = fullfile(sweep_root, cluster_name);
d = dir(fullfile(cluster_dir, ['*' ref_core '*']));
d = d([d.isdir]);
d = d(~ismember({d.name}, {'.', '..'}));
if isempty(d)
    error('No folder matching *%s* in %s', ref_core, cluster_dir);
end
datadir = fullfile(cluster_dir, d(1).name);
fprintf('Reference run: %s\n', datadir);

% ---------- defaults, overridden by anything passed in ----------
opts = struct('full_panel', true, 'clipped', true, 'mp4', true, 'fps', 15, ...
              'outdir', '/glade/work/nguyen/unc-shoals/animations');
if mod(numel(varargin), 2) ~= 0
    error('Options must be name-value pairs.');
end
for k = 1:2:numel(varargin)
    opts.(varargin{k}) = varargin{k+1};
end
names = fieldnames(opts);
args  = cell(1, 2 * numel(names));
for k = 1:numel(names)
    args{2*k-1} = names{k};
    args{2*k}   = opts.(names{k});
end

vishoals(run_tag, 'datadir', datadir, args{:});
end
