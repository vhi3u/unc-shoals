function results = bench_render(run_tag, varargin)
% BENCH_RENDER  Why is print() 85% of a frame, and what makes it cheaper?
%
%   bench_render('sweep_wind_m0p020', 'cluster', 'cluster_weakwinds')
%
% bench_animation established that print(fig,'-RGBImage') costs ~3.1 s/frame
% and that the cost barely moves with dpi (-r50 is only 14% cheaper than
% -r100). A raster cost would scale with pixels, so the time is upstream of
% rasterization. The suspect is the PAINTERS renderer, which emits one vector
% primitive per DATA CELL:
%
%   6 XY panels x 200x360 = 432,000
%   6 XZ panels x 200x50  =  60,000
%                            492,000 quads per frame, at any dpi
%
% This script crosses three plot constructions with three renderer choices
% and times print for each, so the cause is identified rather than assumed.
%
%   pcolor + shading flat     one quad per cell
%   pcolor + shading interp   what vishoals.m:459 actually does for XY --
%                             per-cell Gouraud, the worst case for painters
%   imagesc                   ONE image object per panel instead of N quads
%
%   default / -painters / -opengl
%
% imagesc is only a fair substitute because the grid is uniform (dx = dy =
% 500 m). It is not pixel-identical to pcolor: pcolor+flat draws cells
% between grid POINTS and so drops the last row and column, while imagesc
% centres cells ON the points, a half-cell offset -- 250 m on a 100 km axis.
% The script measures the resulting image difference instead of waving it
% away, so the trade is visible before anything is changed.
%
% Options: 'cluster', 'datadir', 'sweep_root', 'reps' (default 5), 'clipped'.

p = inputParser;
addParameter(p, 'cluster',    '', @ischar);
addParameter(p, 'datadir',    '', @ischar);
addParameter(p, 'sweep_root', '/glade/work/nguyen/unc-shoals/sweep_output', @ischar);
addParameter(p, 'reps',       5, @(x)isnumeric(x)&&isscalar(x)&&x>=1);
addParameter(p, 'clipped',    true, @(x)islogical(x)&&isscalar(x));
addParameter(p, 'save_png',   '', @ischar);
parse(p, varargin{:});
o = p.Results;

% ── locate the run ────────────────────────────────────────────────────────
datadir = o.datadir;
if isempty(datadir)
    d = dir(fullfile(o.sweep_root, o.cluster, '*'));
    d = d([d.isdir] & ~ismember({d.name}, {'.','..'}));
    for k = 1:numel(d)
        cand = fullfile(o.sweep_root, o.cluster, d(k).name);
        if isfile(fullfile(cand, ['top_' run_tag '.nc'])), datadir = cand; break; end
    end
    assert(~isempty(datadir), 'No top_%s.nc under %s/%s', run_tag, o.sweep_root, o.cluster);
end
f_top  = fullfile(datadir, ['top_'  run_tag '.nc']);
f_midy = fullfile(datadir, ['midy_' run_tag '.nc']);

xc = try_nc_(f_top,  {'xC','x_caa','xc','x'});
yc = try_nc_(f_top,  {'yC','y_aca','yc','y'});
zc = try_nc_(f_midy, {'zC','z_aac','zc','z'});
Nt = numel(ncread(f_top, 'time'));
ix = 1:numel(xc); iy = 1:numel(yc);
if o.clipped
    ix = find(xc/1e3 >= 0 & xc/1e3 <= 100);
    iy = find(yc/1e3 >= 20 & yc/1e3 <= 200);
end
Nx = numel(ix); Ny = numel(iy); Nz = numel(zc);
xk = xc(ix)/1e3; yk = yc(iy)/1e3;

% One frame of real data is enough; this measures rendering, not reading.
rd_xy = @(v) squeeze(ncread(f_top,  v, [ix(1) iy(1) 1 Nt], [Nx Ny 1 1]))';
rd_xz = @(v) squeeze(ncread(f_midy, v, [ix(1) 1 1 Nt], [Nx 1 Nz 1]))';
F = {'T','S','u_c','v_c','Ro','KE'};

% No NaN-fill fallback here, deliberately. Painters does not emit quads for
% NaN cells, so a silently empty panel would make the render look cheap and
% invalidate the measurement. If a field is missing, say so and stop.
have = {ncinfo(f_top).Variables.Name};
miss = F(~ismember(F, have));
if ~isempty(miss)
    error('bench_render:missingFields', ...
          'Not in %s: %s\nAvailable: %s', f_top, strjoin(miss, ', '), ...
          strjoin(have, ', '));
end

XY = cell(1,6); XZ = cell(1,6);
for k = 1:6, XY{k} = rd_xy(F{k}); XZ{k} = rd_xz(F{k}); end

fprintf('\n╔══════════════════════════════════════════════════════════════════╗\n');
fprintf('║ bench_render: %-50s ║\n', run_tag);
fprintf('╚══════════════════════════════════════════════════════════════════╝\n');
fprintf('  XY %d x %d, XZ %d x %d  ->  %s quads/frame for pcolor\n', ...
        Nx, Ny, Nx, Nz, addcommas_(6*(Nx-1)*(Ny-1) + 6*(Nx-1)*(Nz-1)));
fprintf('  %d reps per cell, median reported\n', o.reps);
fin_xy = mean(cellfun(@(A) mean(isfinite(A(:)) & A(:) ~= 0), XY));
fin_xz = mean(cellfun(@(A) mean(isfinite(A(:)) & A(:) ~= 0), XZ));
fprintf('  live cells: XY %.0f%%, XZ %.0f%%  (painters skips NaN, so near-0%% would\n', ...
        100*fin_xy, 100*fin_xz);
fprintf('              mean the render cost below is not representative)\n');

% ── is OpenGL even available in this MATLAB? ──────────────────────────────
fprintf('\n── renderer availability ─────────────────────────────────────────\n');
try
    gl = opengl('data');
    fprintf('  OpenGL vendor   : %s\n', gl.Vendor);
    fprintf('  OpenGL version  : %s\n', gl.Version);
    fprintf('  software        : %d\n', gl.Software);
catch ME
    fprintf('  opengl(''data'') failed: %s\n', ME.message);
    fprintf('  -> almost certainly -nodisplay with no OpenGL, so painters is forced.\n');
end

% Probe -opengl ONCE. Under -nodisplay MATLAB refuses it and warns on every
% single print call, so 3 variants x reps buries the table in identical
% warnings. Detect it here, drop the column, and say why.
% Empty on purpose: the refusal comes from print's inputcheck during
% argument validation, before anything is rendered, so no content is needed
% and the probe stays instant.
probe = figure('Visible','off');
% Refusal can arrive as a warning OR as an error depending on release, and a
% swallowed error must not read as success -- so check both paths.
gl_usable = true;
lastwarn('');
ws = warning('off', 'all');
try
    print(probe, '-RGBImage', '-r50', '-opengl');
catch
    gl_usable = false;
end
warning(ws);
if contains(lower(lastwarn), 'opengl') || contains(lower(lastwarn), 'nodisplay')
    gl_usable = false;
end
close(probe);

% ── how many pixels does one data cell get? ───────────────────────────────
% interp vs flat is only visible when a cell spans more than a pixel or two.
% Measured from the real axes rather than estimated.
fprintf('\n── pixels per data cell, at -r100 ────────────────────────────────\n');
pf = build_('pcolor+flat', xk, yk, zc, XY, XZ);
drawnow;
axs_ = findobj(pf, 'Type', 'axes');
pos_xy = []; pos_xz = [];
for a = axs_'
    P = getpixelposition(a);
    if P(4) > 250, pos_xy = P; else, pos_xz = P; end
end
if ~isempty(pos_xy)
    fprintf('  XY panel %4.0f x %4.0f px  for %d x %d cells  ->  %.2f x %.2f px/cell\n', ...
            pos_xy(3), pos_xy(4), Nx, Ny, pos_xy(3)/Nx, pos_xy(4)/Ny);
end
if ~isempty(pos_xz)
    fprintf('  XZ panel %4.0f x %4.0f px  for %d x %d cells  ->  %.2f x %.2f px/cell\n', ...
            pos_xz(3), pos_xz(4), Nx, Nz, pos_xz(3)/Nx, pos_xz(4)/Nz);
end
fprintf('  (<~1 px/cell: interp cannot be seen. >~2 px/cell: flat shows steps.)\n');
close(pf);

variants  = {'pcolor+flat', 'pcolor+interp', 'imagesc', 'mixed'};
if gl_usable
    renderers = {'default', '-painters', '-opengl'};
else
    renderers = {'default', '-painters'};
    fprintf(['\n  -opengl SKIPPED: MATLAB refuses it under -nodisplay, so painters\n' ...
             '  is forced and no renderer flag can change that. To get an OpenGL\n' ...
             '  context on Casper you would have to run MATLAB under a virtual\n' ...
             '  framebuffer instead of -nodisplay, e.g.\n' ...
             '      xvfb-run -a matlab -nosplash -softwareopengl -r "..."\n' ...
             '  which is worth testing only if imagesc below is not enough.\n']);
end
T   = nan(numel(variants), numel(renderers));
img = cell(1, numel(variants));

for iv = 1:numel(variants)
    fig = build_(variants{iv}, xk, yk, zc, XY, XZ);
    drawnow;
    print(fig, '-RGBImage', '-r100');          % warm-up, discarded
    for ir = 1:numel(renderers)
        ts = nan(1, o.reps);
        for r = 1:o.reps
            try
                wss = warning('off', 'all');
                t0 = tic;
                if strcmp(renderers{ir}, 'default')
                    A = print(fig, '-RGBImage', '-r100');
                else
                    A = print(fig, '-RGBImage', '-r100', renderers{ir});
                end
                ts(r) = toc(t0);
                warning(wss);
            catch ME
                fprintf('  %s / %s failed: %s\n', variants{iv}, renderers{ir}, ME.message);
                break;
            end
        end
        T(iv, ir) = median(ts, 'omitnan');
        if ir == 1 && exist('A', 'var'), img{iv} = A; end
        % Keep the default-renderer frame at full size for eyeball comparison.
        if ir == 1 && ~isempty(o.save_png) && exist('A', 'var')
            if ~exist(o.save_png, 'dir'), mkdir(o.save_png); end
            imwrite(A, fullfile(o.save_png, sprintf('render_%s.png', ...
                    strrep(variants{iv}, '+', '_'))));
        end
    end
    close(fig);
end

% ── report ────────────────────────────────────────────────────────────────
fprintf('\n── print(-RGBImage, -r100), seconds per frame ────────────────────\n');
fprintf('  %-16s', 'construction');
fprintf('%12s', renderers{:}); fprintf('\n');
for iv = 1:numel(variants)
    fprintf('  %-16s', variants{iv});
    for ir = 1:numel(renderers)
        if isnan(T(iv,ir)), fprintf('%12s', 'failed');
        else,               fprintf('%12.3f', T(iv,ir)); end
    end
    fprintf('\n');
end

base = T(1,1);
fprintf('\n── speedup vs pcolor+flat/default ───────────────────────────────\n');
for iv = 1:numel(variants)
    for ir = 1:numel(renderers)
        if ~isnan(T(iv,ir)) && ~(iv==1 && ir==1)
            fprintf('  %-16s %-11s %6.1fx\n', variants{iv}, renderers{ir}, base/T(iv,ir));
        end
    end
end

% ── does imagesc actually look the same? ──────────────────────────────────
fprintf('\n── imagesc vs pcolor+flat, rendered image difference ────────────\n');
if ~isempty(img{1}) && ~isempty(img{3}) && isequal(size(img{1}), size(img{3}))
    d = abs(double(img{1}) - double(img{3}));
    fprintf('  identical pixels : %.2f%%\n', 100*mean(d(:) == 0));
    fprintf('  mean |difference|: %.2f / 255\n', mean(d(:)));
    fprintf('  max  |difference|: %.0f / 255\n', max(d(:)));
    fprintf('  (a half-cell offset is expected; large values mean something else moved)\n');
else
    fprintf('  frame sizes differ -- compare the two PNGs by eye instead.\n');
end

% Where the difference IS matters more than how big it is. Concentrated on
% cell edges means sub-pixel resampling and is invisible in motion;
% concentrated inside panels or on the colorbars means something moved.
if ~isempty(img{1}) && ~isempty(img{3}) && isequal(size(img{1}), size(img{3}))
    dg = max(abs(double(img{1}) - double(img{3})), [], 3);
    fprintf('\n  pixels differing by >32/255 : %.2f%%\n', 100*mean(dg(:) > 32));
    fprintf('  ... of those, how they cluster is what the PNGs show.\n');
    if ~isempty(o.save_png)
        imwrite(uint8(min(255, dg*4)), fullfile(o.save_png, 'render_diff_x4.png'));
        fprintf('\n  wrote, for side-by-side inspection:\n');
        fprintf('    %s/render_pcolor_flat.png\n', o.save_png);
        fprintf('    %s/render_pcolor_interp.png   <- what vishoals draws now\n', o.save_png);
        fprintf('    %s/render_imagesc.png         <- the 6.4x candidate\n', o.save_png);
        fprintf('    %s/render_diff_x4.png         <- |flat - imagesc|, x4 contrast\n', o.save_png);
    end
end

% ── what this means for a run ─────────────────────────────────────────────
best = min(T(~isnan(T)));
[bi, bj] = find(T == best, 1);
fprintf('\n── projection, 50-day run, print only ───────────────────────────\n');
fprintf('  now      201 frames x %.3f s = %5.1f min\n', base, 201*base/60);
fprintf('  best     201 frames x %.3f s = %5.1f min   (%s / %s)\n', ...
        best, 201*best/60, variants{bi}, renderers{bj});
fprintf('  best + sample_hours 24, 51 frames = %5.1f min\n\n', 51*best/60);

results = struct('variants', {variants}, 'renderers', {renderers}, ...
                 'seconds', T, 'datadir', datadir);
end

% ================== helpers ==================

function fig = build_(kind, xk, yk, zc, XY, XZ)
% The same 3x6 full_panel figure as vishoals, built three ways.
cm = {cmocean('thermal'), cmocean('haline'), cmocean('balance'), ...
      cmocean('balance'), cmocean('balance'), pmkmp(100,'swtth')};
fig = figure('Color','w','Position',[80 50 1350 700]);
tl  = tiledlayout(fig, 3, 6, 'TileSpacing','compact','Padding','compact');
[Xc, Yc] = meshgrid(xk, yk);
[Xz, Zz] = meshgrid(xk, zc);
for k = 1:6
    ax = nexttile(tl, k, [2 1]);
    draw_(ax, kind, xk, yk, Xc, Yc, XY{k}, true);
    colormap(ax, cm{k}); colorbar(ax); set(ax,'YDir','normal'); axis(ax,'tight');
end
for k = 1:6
    ax = nexttile(tl, 12+k);
    draw_(ax, kind, xk, zc, Xz, Zz, XZ{k}, false);
    colormap(ax, cm{k}); colorbar(ax); set(ax,'YDir','normal'); axis(ax,'tight');
end
end

function draw_(ax, kind, xv, yv, XX, YY, C, is_xy)
% 'mixed' is the construction the pixel arithmetic implies: imagesc on the XY
% maps, where cells render at roughly one pixel so interp smoothing cannot be
% seen and 88% of the quads live, and pcolor+interp on the XZ transects, where
% Nz = 50 over the panel height puts a cell at ~3 px and flat shows steps.
% Note that is ALSO a change from what vishoals draws today, which has interp
% on XY and flat on XZ -- the smoothing on the row that cannot show it.
switch kind
    case 'pcolor+flat'
        pcolor(ax, XX, YY, C); shading(ax, 'flat');
    case 'pcolor+interp'
        pcolor(ax, XX, YY, C); shading(ax, 'interp');
    case 'imagesc'
        imagesc(ax, xv, yv, C);
    case 'mixed'
        if is_xy
            imagesc(ax, xv, yv, C);
        else
            pcolor(ax, XX, YY, C); shading(ax, 'interp');
        end
end
end

function data = try_nc_(fname, names)
for i = 1:numel(names)
    try, data = ncread(fname, names{i}); return; catch, end
end
error('No coordinate among {%s} in %s. Available: %s', strjoin(names, ', '), ...
      fname, strjoin({ncinfo(fname).Variables.Name}, ', '));
end

function s = addcommas_(n)
s = regexprep(sprintf('%d', n), '(\d)(?=(\d{3})+$)', '$1,');
end
