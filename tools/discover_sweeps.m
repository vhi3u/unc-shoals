function [labels, tags] = discover_sweeps(cluster_dir)
% DISCOVER_SWEEPS  List the sweeps actually present in a cluster directory.
%
%   [labels, tags] = discover_sweeps(cluster_dir)
%
%   labels  bare run labels, e.g. 'R500_Zs05', 'Zs5_Ls40k'
%   tags    run tags as the model names its files, e.g. 'sweep_R500_Zs05'
%
% Replaces the hardcoded sweep lists that used to sit at the top of every
% bulk script. Those lists held the labels of whichever sweep was current
% when the script was written, so any later cluster with different labels
% matched nothing and the script reported "SKIP: No folder found" for every
% entry -- silently producing no output rather than failing.
%
% Directory layout this expects, as written by sweep_driver.jl:
%
%   <cluster_dir>/sweep_001_R500_Zs05/top_sweep_R500_Zs05.nc
%                                     midy_sweep_R500_Zs05.nc
%
% so the directory carries a zero-padded index that the filenames do not.
% Older clusters predate the index (sweep_Zs5_Ls40k/), and those are handled
% too: there the directory name IS the run tag.
%
% Results are sorted by directory name, which for indexed clusters means
% they come back in the order the sweep table declared them.
%
% See also KEEP_EXISTING, which filters a hand-ordered list instead -- use
% that where the ORDER or GROUPING of runs carries meaning (an x-axis in
% wind-stress order, say, or rows of a multi-panel figure). Auto-discovery
% is only correct when the script genuinely means "every sweep in here".

    if nargin < 1 || isempty(cluster_dir)
        error('discover_sweeps:noDir', 'cluster_dir is required.');
    end
    if ~isfolder(cluster_dir)
        error('discover_sweeps:missingDir', ...
              'Cluster directory does not exist:\n  %s', cluster_dir);
    end

    d = dir(fullfile(cluster_dir, 'sweep_*'));
    d = d([d.isdir]);
    d = d(~ismember({d.name}, {'.', '..'}));

    if isempty(d)
        error('discover_sweeps:noSweeps', ...
              'No sweep_* directories found in:\n  %s', cluster_dir);
    end

    [~, order] = sort({d.name});
    d = d(order);

    labels = cell(1, numel(d));
    tags   = cell(1, numel(d));
    for k = 1:numel(d)
        nm  = d(k).name;
        tok = regexp(nm, '^sweep_\d+_(.+)$', 'tokens', 'once');
        if isempty(tok)
            % no index segment: the directory name is already the tag
            tags{k}   = nm;
            labels{k} = regexprep(nm, '^sweep_', '');
        else
            labels{k} = tok{1};
            tags{k}   = ['sweep_' tok{1}];
        end
    end

    keep   = ~cellfun(@isempty, labels);
    labels = labels(keep);
    tags   = tags(keep);

    fprintf('discover_sweeps: %d sweep(s) in %s\n', numel(labels), cluster_dir);
end
