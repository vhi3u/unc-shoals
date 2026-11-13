function R = eke_geometry_controls(cluster_name, varargin)
% EKE_GEOMETRY_CONTROLS  Is the Fig. 4b shoal-height trend actually Z_S?
%
%   eke_geometry_controls('cluster_geometry')
%   R = eke_geometry_controls('cluster_geometry', 'window', 10);
%
% Zs sets the crest depth, but the shoal is built by filling the background
% up to that level, so a deeper crest ALSO gets a shorter flat crest and a
% steeper inshore flank. Fig. 4b's two endpoints therefore differ in three
% ways at once:
%
%   label            Zs     Ls    crest    inshore |dh/dx|
%   Zs05_ref       -5.0   40.0    40.4 km  0
%   Zs15_ref      -15.0   40.0    32.4 km  0.00349
%
% params_geomcontrol.toml adds three runs that each change ONE of the three
% and hold the other two. This script reduces every run to a single EKE
% number and forms the contrasts that separate them -- a map sweep cannot
% answer this, because the question is about ratios between specific pairs.
%
% EKE matches vis_eke_sweeps.m, so the numbers are directly comparable with
% Fig. 4b rather than needing rescaling:
%
%   EKE = 1/2 [ (<uu> - <u>^2) + (<vv> - <v>^2) ]        horizontal only
%
% from the ONLINE 10-day averages in time_avg_3d_<tag>.nc, depth-averaged as
% (1/H) int EKE dz, then area-averaged over the Fig. 4b window
% x = 0-100 km, y = 20-200 km (the 20 km southern nudging zone is excluded).
%
% Options:
%   'sweep_root'  default /glade/work/nguyen/unc-shoals/sweep_output
%   'window'      which 10-day average to use; [] = last (default)
%   'x_box'       km, cross-shelf averaging range   default [0 100]
%   'y_box'       km, alongshore range              default [20 200]
%   'outdir'      where the PNG goes

p = inputParser;
addParameter(p, 'sweep_root', '/glade/work/nguyen/unc-shoals/sweep_output', @ischar);
addParameter(p, 'window', [], @(x)isempty(x) || (isnumeric(x)&&isscalar(x)));
addParameter(p, 'x_box', [0 100], @(v)isnumeric(v)&&numel(v)==2);
addParameter(p, 'y_box', [20 200], @(v)isnumeric(v)&&numel(v)==2);
addParameter(p, 'outdir', '/glade/work/nguyen/unc-shoals/animations', @ischar);
parse(p, varargin{:});
o = p.Results;

cluster_dir = fullfile(o.sweep_root, cluster_name);
[labels, tags] = discover_sweeps(cluster_dir);

fprintf('\n╔══════════════════════════════════════════════════════════════════╗\n');
fprintf('║ EKE geometry controls: %-41s ║\n', cluster_name);
fprintf('╚══════════════════════════════════════════════════════════════════╝\n');
fprintf('  EKE = 1/2[(<uu>-<u>^2) + (<vv>-<v>^2)], depth-averaged,\n');
fprintf('  area-averaged over x = %g-%g km, y = %g-%g km\n\n', ...
        o.x_box(1), o.x_box(2), o.y_box(1), o.y_box(2));

eke    = nan(1, numel(labels));
nwin   = nan(1, numel(labels));
spread = nan(1, numel(labels));
all_w  = cell(1, numel(labels));
Zs   = nan(1, numel(labels));   % crest depth        [m]
Lc   = nan(1, numel(labels));   % flat crest length  [km]
Sf   = nan(1, numel(labels));   % inshore flank |dh/dx|
for k = 1:numel(labels)
    f = find_tavg_(cluster_dir, labels{k}, tags{k});
    if isempty(f)
        fprintf('  %-16s no time_avg_3d file -- skipped\n', labels{k});
        continue;
    end
    [eke(k), nwin(k), ew] = eke_one_(f, o);
    [Zs(k), Lc(k), Sf(k)] = geom_one_(f, nwin(k));
    all_w{k} = ew; %#ok<AGROW>
    spread(k) = range_after_spinup_(ew);
    fprintf('  %-16s EKE = %.4e   sqrt(EKE) = %.4f m/s   window %d of %d\n', ...
            labels{k}, eke(k), sqrt(eke(k)), nwin(k), numel(ew));
end

% ── how big is a difference that means nothing? ──────────────────────────
% Each run is n = 1, so a contrast on its own cannot be told apart from
% internal variability. The 10-day averaging windows are not independent
% samples, but their spread after spin-up is the cheapest available floor on
% run-to-run noise -- a contrast inside that band is not evidence.
fprintf('\n── EKE per 10-day window (the run is still adjusting early on) ──\n');
for k = 1:numel(labels)
    ew = all_w{k};
    fprintf('  %-16s', labels{k});
    fprintf(' %9.3e', ew);
    fprintf('   last-3 spread %5.1f%%\n', 100*spread(k));
end
fprintf(['  ^ read left to right. EKE falls by more than a decade from the\n' ...
         '    early windows, so that decline is spin-up, not variability --\n' ...
         '    which is why the floor below uses only the last three.\n']);
typ = median(spread(~isnan(spread)));
fprintf(['\n  Typical within-run spread: %.1f%%. Treat a contrast below roughly\n' ...
         '  this as indistinguishable from noise, not as a null result.\n'], 100*typ);

% Geometry is MEASURED from the model's own wet mask, not taken from the
% design table in params_geomcontrol.toml. If a control did not land where it
% was meant to, that shows up here rather than silently skewing the scatter.
fprintf('\n── geometry as the model actually built it ──────────────────────\n');
fprintf('  %-16s %8s %12s %14s\n', 'run', 'Zs (m)', 'crest (km)', 'inshore |dh/dx|');
for k = 1:numel(labels)
    fprintf('  %-16s %8.1f %12.1f %14.5f\n', labels{k}, Zs(k), Lc(k), Sf(k));
end

% ── the contrasts the sweep exists to form ───────────────────────────────
% Which pairs those are depends on the cluster: params_geomcontrol.toml
% breaks the depth/crest/flank collinearity, params_res250.toml crosses the
% same depth axis with dx. Hardcoding the geomcontrol pairs printed five n/a
% rows under a "how to read it" block about controls that cluster does not
% contain.
[C, ticks, howto, head] = contrast_spec_(labels);

fprintf('\n── contrasts ────────────────────────────────────────────────────\n');
fprintf('  %-34s %10s %8s\n', 'what it isolates', 'ratio', 'change');
ratio = nan(1, size(C,1));
for i = 1:size(C,1)
    a = find(strcmp(labels, C{i,1}), 1);
    b = find(strcmp(labels, C{i,2}), 1);
    if isempty(a) || isempty(b) || isnan(eke(a)) || isnan(eke(b))
        fprintf('  %-34s %10s\n', C{i,3}, 'n/a');
        continue;
    end
    ratio(i) = eke(a) / eke(b);
    chg = 100*abs(ratio(i)-1);
    if isfinite(typ) && chg < 100*typ, verdict = '  <- within noise';
    else,                              verdict = ''; end
    fprintf('  %-34s %10.3f %+7.0f%%%s\n', C{i,3}, ratio(i), 100*(ratio(i)-1), verdict);
end

if isempty(C)
    fprintf(['  none -- no recognised contrast pair in this cluster.\n' ...
             '  Pairs are matched on run_label: the geomcontrol set (Zs05_ref,\n' ...
             '  Zs15_ref, Zs15_Lsfix, Zs05_Lsmatch, Zs15_slopefix) or the\n' ...
             '  resolution set (R<dx>_Zs<depth>). The per-run EKE and geometry\n' ...
             '  tables above are unaffected.\n']);
end

fprintf('\n── how to read it ───────────────────────────────────────────────\n');
fprintf('%s', howto);

% ── figure ───────────────────────────────────────────────────────────────
fig = figure('Color','w','Position',[80 80 max(1150, 420 + 80*size(C,1)) 450]);
tl = tiledlayout(fig, 1, 2, 'TileSpacing','compact','Padding','compact');

ax = nexttile(tl);
bar(ax, eke, 'FaceColor', [0.30 0.45 0.70], 'BaseValue', min(eke)/3);
set(ax, 'XTick', 1:numel(labels), 'XTickLabel', strrep(labels, '_', '\_'), ...
    'XTickLabelRotation', 30, 'FontSize', 11, 'YScale', 'log');
ylim(ax, [min(eke)/3, max(eke)*3]);
ylabel(ax, 'EKE (m^2 s^{-2})', 'FontWeight','bold');
title(ax, 'Depth-averaged EKE (log)', 'FontWeight','bold');
grid(ax,'on');

ax2 = nexttile(tl);
hold(ax2,'on');
if size(C,1) <= 5
    cols = [0.55 0.55 0.55; 0.85 0.35 0.15; 0.95 0.60 0.20; 0.20 0.55 0.35; 0.30 0.45 0.70];
else
    cols = lines(size(C,1));
end
for i = 1:numel(ratio)
    % BaseValue 1, so a bar grows UP for an increase and DOWN for a decrease
    % and its length is the size of the effect either way.
    bar(ax2, i, ratio(i), 'FaceColor', cols(min(i,size(cols,1)),:), 'BaseValue', 1);
end
yline(ax2, 1, 'k--', 'no effect', 'LineWidth', 1.2);
% Log axis, because these are RATIOS: x2 and /2 are the same size of effect
% in opposite directions, and only a log axis renders them symmetrically
% about 1. On a linear axis a halving looks half as important as a doubling.
rr = ratio(isfinite(ratio) & ratio > 0);
set(ax2, 'XTick', 1:size(C,1), 'XTickLabel', ticks, ...
    'XTickLabelRotation', 30, 'FontSize', 11, 'YScale', 'log');
if ~isempty(rr)
    m = max(max(rr), 1/min(rr)) * 1.6;
    ylim(ax2, [1/m, m]);
end
ylabel(ax2, 'EKE ratio (log)', 'FontWeight','bold');
title(ax2, 'Each contrast isolates one property', 'FontWeight','bold');
grid(ax2,'on');

title(tl, sprintf('%s  —  %s', strrep(cluster_name,'_','\_'), head), ...
      'FontWeight','bold', 'FontSize', 13);

if ~exist(o.outdir,'dir'), mkdir(o.outdir); end
out = fullfile(o.outdir, sprintf('%s_eke_geometry_controls.png', cluster_name));
exportgraphics(fig, out, 'Resolution', 200);
fprintf('\nSaved: %s\n', out);
close(fig);

% ── scatter: EKE against each covarying property ─────────────────────────
% The three properties move together across the ORIGINAL Fig. 4b pair, so a
% scatter against any one of them in isolation would look convincing. Showing
% all three side by side is the point: the driver is the one the five runs
% collapse onto, and the controls exist precisely to break the collinearity
% that makes a single-panel scatter meaningless here.
props = {Zs,  'Z_S (m)',               'crest depth'
         Lc,  'flat crest length (km)','crest length'
         Sf,  'inshore |dh/dx|',       'flank slope'};

fig2 = figure('Color','w','Position',[80 80 1350 460]);
tl2  = tiledlayout(fig2, 1, 3, 'TileSpacing','compact','Padding','compact');
mk   = {'o','s','^','d','v','p','h'};
cc   = lines(numel(labels));

for j = 1:size(props,1)
    ax = nexttile(tl2); hold(ax,'on'); grid(ax,'on');
    x = props{j,1};
    % Join each control pair, so which property a contrast moved is visible
    % rather than having to be matched up from the legend.
    for i = 1:size(C,1)
        a = find(strcmp(labels, C{i,1}), 1);
        b = find(strcmp(labels, C{i,2}), 1);
        if isempty(a) || isempty(b), continue; end
        plot(ax, [x(a) x(b)], [eke(a) eke(b)], '-', ...
             'Color', [0.6 0.6 0.6 0.5], 'LineWidth', 1.0, 'HandleVisibility','off');
    end
    for k = 1:numel(labels)
        if isfinite(spread(k)) && numel(all_w{k}) > 2
            % Last three windows, matching range_after_spinup_. These used to
            % span everything after window 1, which includes the spin-up
            % transient at ~2.7e-4 -- an order of magnitude above the
            % equilibrated values, so every bar ran off the top of the axes
            % and the markers were squashed onto the baseline.
            e2 = all_w{k}(max(1,end-2):end);
            plot(ax, [x(k) x(k)], [min(e2) max(e2)], '-', 'Color', [0.35 0.35 0.35], ...
                 'LineWidth', 1.2, 'HandleVisibility', 'off');
        end
        plot(ax, x(k), eke(k), mk{min(k,numel(mk))}, 'MarkerSize', 11, ...
             'MarkerFaceColor', cc(k,:), 'MarkerEdgeColor', 'k', ...
             'LineWidth', 0.8, 'DisplayName', strrep(labels{k},'_','\_'));
    end
    % EKE spans a decade across these runs, so a linear axis puts everything
    % on the baseline.
    set(ax, 'YScale', 'log');
    xlabel(ax, props{j,2}, 'FontWeight','bold');
    if j == 1
        ylabel(ax, 'EKE (m^2 s^{-2})', 'FontWeight','bold');
        legend(ax, 'Location','best', 'FontSize', 9);
    end
    title(ax, sprintf('EKE vs %s', props{j,3}), 'FontWeight','bold');
    set(ax, 'FontSize', 11);
    % Spearman rank correlation: monotonic association without assuming a
    % functional form, and robust at n = 5 where a fitted line would not be.
    ok = isfinite(x) & isfinite(eke);
    if sum(ok) >= 3
        rho = rank_corr_(x(ok), eke(ok));
        text(ax, 0.04, 0.96, sprintf('\\rho_s = %+.2f', rho), 'Units','normalized', ...
             'VerticalAlignment','top', 'FontSize', 11, 'FontWeight','bold');
    end
end
title(tl2, sprintf(['%s  —  EKE against each covarying property ' ...
      '(n = %d, so \\rho_s is indicative only)'], ...
      strrep(cluster_name,'_','\_'), numel(labels)), 'FontWeight','bold', 'FontSize', 13);

out2 = fullfile(o.outdir, sprintf('%s_eke_geometry_scatter.png', cluster_name));
exportgraphics(fig2, out2, 'Resolution', 200);
fprintf('Saved: %s\n', out2);
close(fig2);

R = struct('labels', {labels}, 'eke', eke, 'contrasts', {C}, 'ratio', ratio, ...
           'Zs', Zs, 'crest_km', Lc, 'flank', Sf);
end

function [C, ticks, howto, head] = contrast_spec_(labels)
% Which pairs to form, chosen from the run_labels that are present.
%
% Two clusters ask different questions with the same machinery:
%   params_geomcontrol.toml  separate depth from the crest length and flank
%                            slope that Zs varies along with it
%   params_res250.toml       cross that same depth axis with dx, so the depth
%                            contrast can be formed twice and compared
% The RATIO of those two depth contrasts is the answer to "does Fig. 4b
% survive dx/2": at 1.0 the trend is resolution-independent whatever its
% size; away from 1.0 the trend is partly numerical.

geom  = {'Zs05_ref','Zs15_ref','Zs15_Lsfix','Zs05_Lsmatch','Zs15_slopefix'};
tok   = regexp(labels, '^R(\d+)_Zs(\d+)$', 'tokens', 'once');
isres = ~cellfun(@isempty, tok);

if any(ismember(geom, labels)) || ~any(isres)
    C = {
     'Zs15_ref',      'Zs05_ref',     'Fig 4b pair (CONFOUNDED: depth+crest+flank)'
     'Zs15_Lsfix',    'Zs15_ref',     'CREST LENGTH, deep side  (32.4 -> 40.6 km)'
     'Zs05_Lsmatch',  'Zs05_ref',     'CREST LENGTH, shallow side (40.4 -> 32.9 km)'
     'Zs15_slopefix', 'Zs15_Lsfix',   'FLANK SLOPE (0.00349 -> 0.00165)'
     'Zs15_Lsfix',    'Zs05_ref',     'DEPTH with crest length controlled'
    };
    ticks = {'Fig 4b (conf.)','crest, deep','crest, shallow','flank slope','depth (ctrl)'};
    head  = 'geometry controls for the Fig. 4b shoal-height axis';
    howto = [ ...
     '  The confounded pair is row 1. If rows 2-4 are all near 1.0, crest\n' ...
     '  length and flank slope are not driving it and the Fig. 4b axis is\n' ...
     '  shoal height after all -- row 5 should then match row 1.\n' ...
     '  If row 2 or 4 is comparable in size to row 1, the axis is doing\n' ...
     '  more than one job and Fig. 4b needs re-cutting.\n' ...
     '  Rows 2 and 3 are the same test from opposite sides: they should\n' ...
     '  move in OPPOSITE directions by a similar factor. If they do not,\n' ...
     '  the effect depends on which run was altered, which would mean the\n' ...
     '  control itself is not clean.\n'];
    if ~any(ismember(geom, labels))
        % Neither naming scheme: report the runs, claim no contrasts.
        C = cell(0,3); ticks = {};
        head  = 'per-run EKE (no contrast pairs recognised)';
        howto = '  No contrasts were formed, so there is nothing here to read.\n';
    end
    return;
end

% ── resolution cluster ───────────────────────────────────────────────────
dxv = nan(1, numel(labels));
zsv = nan(1, numel(labels));
for k = find(isres(:)')
    dxv(k) = str2double(tok{k}{1});
    zsv(k) = -str2double(tok{k}{2});
end
dxs = sort(unique(dxv(isres)), 'descend');   % coarse first: 500, 250
zss = sort(unique(zsv(isres)), 'descend');   % shallow first: -5, -10, -15

has_ = @(d,z) any(isres & dxv == d & zsv == z);
lab_ = @(d,z) labels{find(isres & dxv == d & zsv == z, 1)};

C = cell(0,3); ticks = {};

% 1. dx at FIXED geometry -- pure numerics, no physics in it
for z = zss(:)'
    if has_(dxs(1), z) && has_(dxs(end), z)
        C(end+1,:) = {lab_(dxs(end), z), lab_(dxs(1), z), ...
            sprintf('RESOLUTION, Zs = %g m (%g->%g)', z, dxs(1), dxs(end))};
        ticks{end+1} = sprintf('dx, Zs%g', z);
    end
end

% 2. the Fig. 4b depth axis, formed once per resolution. Adjacent rows are
%    the same physical contrast at the two dx, so the comparison that matters
%    is between neighbours rather than against 1.0.
for i = 1:numel(zss)
    for j = i+1:numel(zss)
        for d = dxs(:)'
            if has_(d, zss(i)) && has_(d, zss(j))
                C(end+1,:) = {lab_(d, zss(j)), lab_(d, zss(i)), ...
                    sprintf('DEPTH %g/%g m at dx = %g m', zss(j), zss(i), d)};
                ticks{end+1} = sprintf('%g/%g @%g', zss(j), zss(i), d);
            end
        end
    end
end

head  = 'does the shoal-height trend survive \Deltax/2?';
howto = [ ...
 '  Rows 1-3 are dx at FIXED geometry, so they contain no physics: their\n' ...
 '  size is how far from converged each case is, and they should shrink as\n' ...
 '  Ld/dx grows (0.1, 0.8, 1.5 at Zs = -5, -10, -15 m).\n' ...
 '  The rest are the Fig. 4b depth axis, formed separately at each dx.\n' ...
 '  COMPARE EACH ADJACENT PAIR, not each row against 1.0: equal ratios mean\n' ...
 '  that depth contrast is resolution-independent and can be reported as\n' ...
 '  physics; a pair differing by more than the within-run spread means that\n' ...
 '  contrast is partly a resolution artifact and its 500 m value cannot be\n' ...
 '  quoted.\n' ...
 '  A non-zero row 1-3 BOUNDS the sensitivity; it does not remove it. dx is\n' ...
 '  not converged at 250 m either.\n'];
end

function r = rank_corr_(x, y)
% Spearman without the Statistics Toolbox: Pearson on the ranks.
[~, ix] = sort(x(:)); rx = zeros(size(x(:))); rx(ix) = 1:numel(x);
[~, iy] = sort(y(:)); ry = zeros(size(y(:))); ry(iy) = 1:numel(y);
rx = rx - mean(rx); ry = ry - mean(ry);
r = sum(rx.*ry) / sqrt(sum(rx.^2) * sum(ry.^2));
end

function [zs, crest_km, flank] = geom_one_(f, it)
% Shoal geometry on the centreline. <uu> is <u^2>: strictly positive in
% water, exactly 0 in immersed cells, so it gives the bottom depth h(x).
%
% Zs is READ from sweep_metadata, not inferred. Deriving it was wrong twice
% over: min(h) is the DEEPEST point, i.e. the 50 m offshore basin, which
% reported Zs = -50 m and a 79 km "crest" for every run; and max(h) would
% pick the -5 m coastal ramp, which is shallower than the crest whenever
% Zs < -5. The crest is neither extreme -- it is the plateau between them --
% so it needs the requested depth to be identified at all.
xc = try_nc_(f, {'xC','x_caa','xc'});
yc = try_nc_(f, {'yC','y_aca','yc'});
zc = try_nc_(f, {'zC','z_aac','zc'});
[~, jc] = min(abs(yc - mean(yc)));          % shoal centreline
uu = squeeze(ncread(f, 'uu', [1 jc 1 it], [numel(xc) 1 numel(zc) 1]));
dz = uniform_dz_(zc);
h  = -sum(uu > 0, 2) * dz;                  % bottom depth h(x) <= 0
x_km = xc(:) / 1e3;

zs = meta_att_(f, 'Zs');
if isnan(zs)
    warning('eke_geometry_controls:noMeta', ...
            'No sweep_metadata beside %s; crest and flank cannot be measured.', f);
    crest_km = NaN; flank = NaN; return;
end

% Longest CONTIGUOUS run at the crest depth. Contiguity matters: the coastal
% ramp also sits at -5 m, so at Zs = -5 the crest and the coast are the same
% depth and a non-contiguous test would merge them into one huge "crest".
on = abs(h - zs) < dz * 1.5;
[i1, i2] = longest_run_(on);
if isempty(i1)
    crest_km = NaN; flank = NaN; return;
end
crest_km = x_km(i2) - x_km(i1);

% Inshore flank: steepest gradient shoreward of the crest.
if i1 > 2
    flank = max(abs(diff(h(1:i1))) ./ diff(x_km(1:i1) * 1e3));
else
    flank = 0;   % the crest reaches the coast; there is no inshore flank
end
end

function [i1, i2] = longest_run_(mask)
i1 = []; i2 = [];
d = diff([false; mask(:); false]);
s = find(d == 1); e = find(d == -1) - 1;
if isempty(s), return; end
[~, b] = max(e - s);
i1 = s(b); i2 = e(b);
end

function v = meta_att_(f, name)
% Global attribute from sweep_metadata_<tag>.nc, which sweep_driver.jl writes
% beside the output with the REQUESTED parameters for the run.
v = NaN;
d = fileparts(f);
[~, base] = fileparts(f);
tag = regexprep(base, '^time_avg_3d_', '');
mf = fullfile(d, ['sweep_metadata_' tag '.nc']);
if ~isfile(mf), return; end
try, v = ncreadatt(mf, '/', name); catch, end
if ischar(v), v = str2double(v); end
end

% ================== helpers ==================

function f = find_tavg_(cluster_dir, label, tag)
d = dir(fullfile(cluster_dir, 'sweep_*'));
d = d([d.isdir]);
f = '';
for k = 1:numel(d)
    if ~isempty(regexp(d(k).name, [regexptranslate('escape', label) '$'], 'once'))
        cand = fullfile(cluster_dir, d(k).name, ['time_avg_3d_' tag '.nc']);
        if isfile(cand), f = cand; return; end
    end
end
end

function [e, it, ew] = eke_one_(f, o)
xc = try_nc_(f, {'xC','x_caa','xc'});
yc = try_nc_(f, {'yC','y_aca','yc'});
zc = try_nc_(f, {'zC','z_aac','zc'});
nt = numel(ncread(f, 'time'));
it = o.window; if isempty(it), it = nt; end
it = max(1, min(nt, it));

ix = find(xc/1e3 >= o.x_box(1) & xc/1e3 <= o.x_box(2));
iy = find(yc/1e3 >= o.y_box(1) & yc/1e3 <= o.y_box(2));
nx = numel(ix); ny = numel(iy); nz = numel(zc);
st = [ix(1) iy(1) 1 it]; ct = [nx ny nz 1];

u  = squeeze(ncread(f, 'u_c', st, ct));
v  = squeeze(ncread(f, 'v_c', st, ct));
uu = squeeze(ncread(f, 'uu',  st, ct));
vv = squeeze(ncread(f, 'vv',  st, ct));

% <uu> is <u^2> and is strictly positive in water, exactly 0 in the immersed
% cells, so it doubles as the wet mask -- no separate bathymetry read.
wet = uu > 0;

e3 = 0.5 * ((uu - u.^2) + (vv - v.^2));
% Tiny negatives are round-off in <u^2> - <u>^2, not physical; they would
% otherwise bias the depth integral low.
e3(~isfinite(e3) | e3 < 0) = 0;
e3(~wet) = 0;

% dz is uniform here (Lz/Nz = 50/50 = 1 m) but take it from the coordinate
% so a different Nz does not silently rescale the answer.
dz = uniform_dz_(zc);
H  = sum(wet, 3) * dz;                 % local water depth
col = sum(e3, 3) * dz;                 % int EKE dz   [m^3/s^2]
col(H <= 0) = NaN;
e = mean(col(H > 0) ./ H(H > 0), 'omitnan');   % (1/H) int EKE dz, area mean

if nargout > 2
    ew = nan(1, nt);
    for t = 1:nt
        ew(t) = eke_window_(f, o, ix, iy, nz, t);
    end
end
end

function e = eke_window_(f, o, ix, iy, nz, t) %#ok<INUSL>
st = [ix(1) iy(1) 1 t]; ct = [numel(ix) numel(iy) nz 1];
u  = squeeze(ncread(f, 'u_c', st, ct));
v  = squeeze(ncread(f, 'v_c', st, ct));
uu = squeeze(ncread(f, 'uu',  st, ct));
vv = squeeze(ncread(f, 'vv',  st, ct));
wet = uu > 0;
e3 = 0.5 * ((uu - u.^2) + (vv - v.^2));
e3(~isfinite(e3) | e3 < 0) = 0;
e3(~wet) = 0;
zc = try_nc_(f, {'zC','z_aac','zc'});
dz = uniform_dz_(zc);
H  = sum(wet, 3) * dz;
col = sum(e3, 3) * dz;
e = mean(col(H > 0) ./ H(H > 0), 'omitnan');
end

function s = range_after_spinup_(ew)
% Fractional peak-to-peak spread over the EQUILIBRATED windows only.
%
% Taking it over all windows after the first gave ~410% for every run and
% flagged every contrast as noise. That was measuring the initial-adjustment
% transient, not variability: EKE falls monotonically by more than a decade
% from the early windows to the last, so peak-to-peak across them is a trend.
% The last three windows are the closest thing here to repeated samples of
% the same state, and even they are consecutive rather than independent, so
% this stays a floor and not an interval.
if numel(ew) < 3, s = NaN; return; end
e2 = ew(max(1, end-2):end);
s = (max(e2) - min(e2)) / mean(e2, 'omitnan');
end

function dz = uniform_dz_(zc)
d = abs(diff(zc(:)));
dz = median(d);
if any(abs(d - dz) > 1e-6 * dz)
    warning('eke_geometry_controls:nonUniformZ', ...
            'z spacing is not uniform (%.4g to %.4g m); using the median %.4g m.', ...
            min(d), max(d), dz);
end
end

function data = try_nc_(fname, names)
for i = 1:numel(names)
    try, data = ncread(fname, names{i}); return; catch, end
end
error('No coordinate among {%s} in %s. Available: %s', strjoin(names, ', '), ...
      fname, strjoin({ncinfo(fname).Variables.Name}, ', '));
end
