function results = bench_animation(run_tag, varargin)
% BENCH_ANIMATION  Measure where vishoals spends its animation time.
%
%   bench_animation('sweep_wind_m0p020', 'cluster', 'cluster_weakwinds')
%   bench_animation('sweep_Zs5_Ls40k', 'cluster', 'cluster_Fnnie_1_19', 'n_frames', 20)
%   r = bench_animation(tag, 'cluster', c, 'dpi_list', [100 72 50], 'endtoend', true);
%
% Answers one question: is the animation slow because of the PRINT ENGINE, the
% NetCDF reads, or the video encoder? The three have completely different
% fixes, and guessing wrong wastes effort.
%
% It does NOT modify or call into the animation path by default, so it is safe
% to run while a vishoals_8panel_bulk job is in flight -- though the two will
% contend for CPU, so a quiet node gives cleaner numbers.
%
% WHAT IT MEASURES
%
% vishoals makes two passes over the data, not one:
%
%   1. a colour-limit scan (vishoals.m:289-298) that reads T, S, u, v, Ro and
%      KE for both slices on every frame and renders nothing. Under
%      'full_panel' this pass is unconditional, so it runs even when every
%      clim is supplied.
%   2. the animation pass, which reads the same fields again and then, per
%      frame, sets CData, calls drawnow, captures with
%      print(fig,'-RGBImage') and hands the array to writeVideo.
%
% Stage timings are reported per frame and as a share of the total, then
% projected to the real frame counts so the numbers are directly comparable
% with a production run.
%
% Options:
%   'cluster'    cluster name under sweep_root               default ''
%   'datadir'    explicit directory holding top_/midy_ .nc    default ''
%   'sweep_root' default /glade/work/nguyen/unc-shoals/sweep_output
%   'n_frames'   frames to time, taken from the END           default 12
%   'dpi_list'   print resolutions to compare                 default [100 72 50]
%   'clipped'    match the bulk script's clipping             default true
%   'endtoend'   also time real vishoals calls at 6 h vs 24 h default false
%   'outdir'     scratch dir for the throwaway video          default <root>/out/bench
%
% See also VISHOALS, VISHOALS_8PANEL_BULK.

p = inputParser;
addParameter(p, 'cluster',    '', @ischar);
addParameter(p, 'datadir',    '', @ischar);
addParameter(p, 'sweep_root', '/glade/work/nguyen/unc-shoals/sweep_output', @ischar);
addParameter(p, 'n_frames',   12, @(x)isnumeric(x)&&isscalar(x)&&x>=2);
addParameter(p, 'dpi_list',   [100 72 50], @(v)isnumeric(v)&&isvector(v));
addParameter(p, 'clipped',    true, @(x)islogical(x)&&isscalar(x));
addParameter(p, 'endtoend',   false, @(x)islogical(x)&&isscalar(x));
addParameter(p, 'outdir',     '', @ischar);
parse(p, varargin{:});
o = p.Results;

% ── locate the run ────────────────────────────────────────────────────────
datadir = o.datadir;
if isempty(datadir)
    if isempty(o.cluster)
        error('bench_animation:needLocation', ...
              'Pass ''cluster'' or ''datadir''.');
    end
    d = dir(fullfile(o.sweep_root, o.cluster, '*'));
    d = d([d.isdir] & ~ismember({d.name}, {'.','..'}));
    for k = 1:numel(d)
        cand = fullfile(o.sweep_root, o.cluster, d(k).name);
        if isfile(fullfile(cand, ['top_' run_tag '.nc']))
            datadir = cand; break;
        end
    end
    if isempty(datadir)
        error('bench_animation:notFound', 'No top_%s.nc under %s', ...
              run_tag, fullfile(o.sweep_root, o.cluster));
    end
end
f_top  = fullfile(datadir, ['top_'  run_tag '.nc']);
f_midy = fullfile(datadir, ['midy_' run_tag '.nc']);
assert(isfile(f_top),  'Missing %s', f_top);
assert(isfile(f_midy), 'Missing %s', f_midy);

outdir = o.outdir;
if isempty(outdir)
    outdir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'out', 'bench');
end
if ~exist(outdir, 'dir'), mkdir(outdir); end

% ── grid and frame selection, matching vishoals ───────────────────────────
xc = try_nc_(f_top,  {'xC','x_caa','xc'});
yc = try_nc_(f_top,  {'yC','y_aca','yc'});
zc = try_nc_(f_midy, {'zC','z_aac','zc'});
time = ncread(f_top, 'time');
Nt   = numel(time);
dt_out_h = median(diff(time)) / 3600;

ix = 1:numel(xc); iy = 1:numel(yc);
if o.clipped
    ix = find(xc/1e3 >=  0 & xc/1e3 <= 100);
    iy = find(yc/1e3 >= 20 & yc/1e3 <= 200);
end
Nx = numel(ix); Ny = numel(iy); Nz = numel(zc);

nf     = min(o.n_frames, Nt);
frames = (Nt - nf + 1):Nt;

fprintf('\n╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║ bench_animation: %-45s ║\n', run_tag);
fprintf('╚════════════════════════════════════════════════════════════════╝\n');
fprintf('  data        : %s\n', datadir);
fprintf('  output every: %.2f h   (%d saved frames, %.1f days)\n', ...
        dt_out_h, Nt, (time(end)-time(1))/86400);
fprintf('  XY panels   : %d x %d%s\n', Nx, Ny, ternary_(o.clipped, '  (clipped)', ''));
fprintf('  XZ panels   : %d x %d\n', Nx, Nz);
fprintf('  timing      : %d frames (the last %d saved)\n\n', nf, nf);

% ── readers, matching vishoals' slicing ───────────────────────────────────
rd_xy = @(v,t) safe_(@() squeeze(ncread(f_top,  v, [ix(1) iy(1) 1 t], [Nx Ny 1 1]))', Ny, Nx);
rd_xz = @(v,t) safe_(@() squeeze(ncread(f_midy, v, [ix(1) 1 1 t], [Nx 1 Nz 1]))', Nz, Nx);
FIELDS = {'T','S','u_c','v_c','Ro','KE'};

% safe_() NaN-fills a failed read, which would hide a wrong variable name --
% and because painters emits no quads for NaN cells, empty panels would make
% print() look cheap and invalidate the whole measurement. Check up front.
have_ = {ncinfo(f_top).Variables.Name};
miss_ = FIELDS(~ismember(FIELDS, have_));
if ~isempty(miss_)
    error('bench_animation:missingFields', ...
          'Not in %s: %s\nAvailable: %s', f_top, strjoin(miss_, ', '), ...
          strjoin(have_, ', '));
end

% ══ PHASE 1: the colour-limit scan (vishoals.m:289-298) — reads only ══════
t0 = tic;
for t = frames
    for k = 1:numel(FIELDS)
        rd_xy(FIELDS{k}, t);
        rd_xz(FIELDS{k}, t);
    end
end
t_scan = toc(t0) / nf;

% ══ build the figure exactly as vishoals' full_panel branch does ══════════
[Xc, Yc] = meshgrid(xc(ix)/1e3, yc(iy)/1e3);
[Xz, Zz] = meshgrid(xc(ix)/1e3, zc);

fig = figure('Color','w','Position',[80 50 1350 700]);
tl  = tiledlayout(fig, 3, 6, 'TileSpacing','compact','Padding','compact');
hxy = gobjects(1,6); hxz = gobjects(1,6);
cm  = {cmocean('thermal'), cmocean('haline'), cmocean('balance'), ...
       cmocean('balance'), cmocean('balance'), pmkmp(100,'swtth')};
for k = 1:6                                  % XY row, each spanning [2 1]
    ax = nexttile(tl, k, [2 1]);
    hxy(k) = pcolor(ax, Xc, Yc, nan(size(Xc)));
    shading(ax,'flat'); colormap(ax, cm{k}); colorbar(ax);
    set(ax,'NextPlot','add');
end
for k = 1:6                                  % XZ row
    ax = nexttile(tl, 12+k);
    hxz(k) = pcolor(ax, Xz, Zz, nan(size(Xz)));
    shading(ax,'flat'); colormap(ax, cm{k}); colorbar(ax);
end
drawnow;

% First print is always slower (engine init) -- discard it.
print(fig, '-RGBImage', '-r100');

% ══ PHASE 2: per-frame stages at the production dpi ═══════════════════════
dpi0 = o.dpi_list(1);
acc = struct('read',0,'cdata',0,'drawnow',0,'print',0,'crop',0,'write',0);
vid = fullfile(outdir, sprintf('bench_%s.avi', run_tag));
vw  = VideoWriter(vid, 'Motion JPEG AVI'); vw.Quality = 95; vw.FrameRate = 15;
open(vw);
crop = []; fsz = [];
for t = frames
    t1 = tic;
    xyd = cell(1,6); xzd = cell(1,6);
    for k = 1:numel(FIELDS)
        xyd{k} = rd_xy(FIELDS{k}, t);
        xzd{k} = rd_xz(FIELDS{k}, t);
    end
    acc.read = acc.read + toc(t1);

    t1 = tic;
    for k = 1:6
        set(hxy(k), 'CData', xyd{k});
        set(hxz(k), 'CData', xzd{k});
    end
    acc.cdata = acc.cdata + toc(t1);

    t1 = tic; drawnow;                             acc.drawnow = acc.drawnow + toc(t1);
    t1 = tic; img = print(fig,'-RGBImage',sprintf('-r%d',dpi0));
                                                   acc.print   = acc.print   + toc(t1);
    t1 = tic;
    if isempty(crop)
        nw = any(img < 250, 3);
        rr = find(any(nw,2)); cc = find(any(nw,1)); pad = 4;
        crop = [max(1,rr(1)-pad), min(size(img,1),rr(end)+pad), ...
                max(1,cc(1)-pad), min(size(img,2),cc(end)+pad)];
        crop(2) = crop(1) + 2*floor((crop(2)-crop(1)+1)/2) - 1;
        crop(4) = crop(3) + 2*floor((crop(4)-crop(3)+1)/2) - 1;
        fsz = size(img,[1 2]);
    elseif ~isequal(size(img,[1 2]), fsz)
        img = imresize(img, fsz);
    end
    img = img(crop(1):crop(2), crop(3):crop(4), :);
    acc.crop = acc.crop + toc(t1);

    t1 = tic; writeVideo(vw, img);                 acc.write = acc.write + toc(t1);
end
close(vw);
if isfile(vid), delete(vid); end

st = fieldnames(acc);
for k = 1:numel(st), acc.(st{k}) = acc.(st{k}) / nf; end
per_frame = sum(struct2array_(acc));

% ── drawnow: is it redundant before print? ────────────────────────────────
% Same work on both sides, differing only by the drawnow call. CData is
% perturbed each iteration so neither path can skip a re-render, and the
% data is already in memory so no read is counted.
base = xyd{1};
t0 = tic;
for i = 1:nf
    set(hxy(1), 'CData', base * (1 + 1e-9*i));
    drawnow;
    print(fig, '-RGBImage', sprintf('-r%d', dpi0));
end
t_drawnow_yes = toc(t0) / nf;
t0 = tic;
for i = 1:nf
    set(hxy(1), 'CData', base * (1 + 1e-9*i));
    print(fig, '-RGBImage', sprintf('-r%d', dpi0));
end
t_drawnow_no = toc(t0) / nf;

% ── print cost vs dpi ─────────────────────────────────────────────────────
dpi_t = zeros(size(o.dpi_list));
for j = 1:numel(o.dpi_list)
    t0 = tic;
    for t = frames, print(fig, '-RGBImage', sprintf('-r%d', o.dpi_list(j))); end
    dpi_t(j) = toc(t0) / nf;
end
close(fig);

% ══ report ════════════════════════════════════════════════════════════════
n_6h  = round(50*24/6) + 1;
n_24h = round(50*24/24) + 1;

fprintf('── per-frame stages, animation pass, -r%d ────────────────────────\n', dpi0);
fprintf('  %-10s %10s %8s\n', 'stage', 'ms/frame', 'share');
for k = 1:numel(st)
    fprintf('  %-10s %10.1f %7.1f%%\n', st{k}, 1e3*acc.(st{k}), ...
            100*acc.(st{k})/per_frame);
end
fprintf('  %-10s %10.1f\n\n', 'TOTAL', 1e3*per_frame);

fprintf('── the clim scan is a SECOND pass over the same data ─────────────\n');
fprintf('  read-only pass        %8.1f ms/frame\n', 1e3*t_scan);
fprintf('  share of a full run   %8.1f%%\n\n', 100*t_scan/(t_scan+per_frame));

fprintf('── print cost vs dpi ────────────────────────────────────────────\n');
for j = 1:numel(o.dpi_list)
    fprintf('  -r%-4d %8.1f ms/frame   %5.2fx vs -r%d\n', ...
            o.dpi_list(j), 1e3*dpi_t(j), dpi_t(j)/dpi_t(1), o.dpi_list(1));
end
fprintf('\n── is drawnow redundant before print? ───────────────────────────\n');
fprintf('  set + drawnow + print  %8.1f ms/frame\n', 1e3*t_drawnow_yes);
fprintf('  set +           print  %8.1f ms/frame\n', 1e3*t_drawnow_no);
fprintf('  drawnow costs          %8.1f ms/frame  (%.1f%% of a frame)\n\n', ...
        1e3*(t_drawnow_yes - t_drawnow_no), ...
        100*(t_drawnow_yes - t_drawnow_no)/per_frame);

fprintf('── projected wall clock for one 50-day run ──────────────────────\n');
full_6h  = n_6h  * (t_scan + per_frame);
full_24h = n_24h * (t_scan + per_frame);
fprintf('  6-hourly  (%3d frames)  %6.1f min\n', n_6h,  full_6h/60);
fprintf('  daily     (%3d frames)  %6.1f min   <- sample_hours, 24\n', n_24h, full_24h/60);
fprintf('  daily + -r72            %6.1f min   (approx)\n', ...
        n_24h*(t_scan + per_frame - acc.print + dpi_t(min(2,end)))/60);
fprintf('  ... x6 runs, serially   %6.1f min  vs  %.1f min as a PBS array\n\n', ...
        6*full_24h/60, full_24h/60);

results = struct('run_tag', run_tag, 'datadir', datadir, 'n_frames', nf, ...
                 'dt_out_h', dt_out_h, 'stages', acc, 'per_frame', per_frame, ...
                 'scan_per_frame', t_scan, 'dpi_list', o.dpi_list, ...
                 'dpi_time', dpi_t, ...
                 'drawnow_yes', t_drawnow_yes, 'drawnow_no', t_drawnow_no);

% ══ optional: time the real thing ═════════════════════════════════════════
if o.endtoend
    fprintf('── end-to-end vishoals, last 5 days ─────────────────────────────\n');
    for sh = [NaN 24]
        args = {'datadir', datadir, 'outdir', outdir, 'full_panel', true, ...
                'isopycnals', false, 'mp4', true, 'fps', 15, 'clipped', o.clipped, ...
                'last_days', 5};
        if ~isnan(sh), args = [args, {'sample_hours', sh}]; end %#ok<AGROW>
        t0 = tic; vishoals(run_tag, args{:}); el = toc(t0);
        fprintf('  sample_hours %-4s  %6.1f s\n', ...
                ternary_(isnan(sh), 'none', sprintf('%d', sh)), el);
    end
end
end

% ================== helpers ==================

function out = ternary_(c, a, b)
if c
    out = a;
else
    out = b;
end
end

function A = safe_(fn, m, n)
try, A = fn(); catch, A = nan(m, n); end
end

function v = struct2array_(s)
f = fieldnames(s); v = zeros(1, numel(f));
for i = 1:numel(f), v(i) = s.(f{i}); end
end

function data = try_nc_(fname, names)
for i = 1:numel(names)
    try, data = ncread(fname, names{i}); return; catch, end
end
info = ncinfo(fname);
error('No coordinate among {%s}. Available: %s', strjoin(names, ', '), ...
      strjoin({info.Variables.Name}, ', '));
end
