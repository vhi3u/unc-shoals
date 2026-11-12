function kept = keep_existing(cluster_dir, wanted)
% KEEP_EXISTING  Filter a hand-ordered run list down to what is on disk.
%
%   kept = keep_existing(cluster_dir, wanted)
%
% `wanted` is a cell array of run labels or run tags, in an order that
% MATTERS -- an x-axis sorted by wind stress, or the rows of a multi-panel
% figure. Entries with no matching directory under cluster_dir are dropped
% and reported; the rest keep their original order.
%
% This is the counterpart to DISCOVER_SWEEPS. Use discover_sweeps where the
% script means "every sweep in this cluster" and order is irrelevant. Use
% this where re-ordering or re-grouping would change what the figure means.
% Running discover_sweeps on, say, plot_ekman_number would hand back runs in
% directory order and scramble the wind-stress axis.
%
% Accepts either naming convention and returns entries exactly as supplied,
% so the caller's downstream code is unaffected.

    if nargin < 2
        error('keep_existing:args', 'Usage: keep_existing(cluster_dir, wanted)');
    end
    if ~isfolder(cluster_dir)
        error('keep_existing:missingDir', ...
              'Cluster directory does not exist:\n  %s', cluster_dir);
    end

    [labels, tags] = discover_sweeps(cluster_dir);

    kept    = {};
    missing = {};
    for k = 1:numel(wanted)
        w    = wanted{k};
        bare = regexprep(w, '^sweep_', '');
        if any(strcmp(bare, labels)) || any(strcmp(w, tags)) || any(strcmp(w, labels))
            kept{end+1} = w; %#ok<AGROW>
        else
            missing{end+1} = w; %#ok<AGROW>
        end
    end

    if ~isempty(missing)
        fprintf('keep_existing: %d of %d requested run(s) not present, skipped:\n', ...
                numel(missing), numel(wanted));
        fprintf('    %s\n', strjoin(missing, ', '));
    end
    if isempty(kept)
        error('keep_existing:nothingLeft', ...
              ['None of the %d requested runs exist in:\n  %s\n' ...
               'Available: %s'], numel(wanted), cluster_dir, strjoin(labels, ', '));
    end
end
