function R = ape_wedge(cluster_name, varargin)
% APE_WEDGE  Measure APE directly, instead of assuming the wedge geometry.
%
%   ape_wedge('cluster_geometry_120')
%   R = ape_wedge('cluster_res250_120', 'window', 10);
%
% The theory (Zhao et al. Eq. 8, via Brink 2024) ASSUMES the available
% potential energy of the current is
%
%     APE_D = C_geom * g' * H0^2 * Ld        [m^4/s^2, per unit alongshore m]
%
% with C_geom = 1/6 for a triangular wedge (depth H0 at the coast tapering to
% zero at Ld) or 1/2 if the anomaly fills the whole H0 x Ld box. That shape is
% a guess about OUR geometry, not a measurement, and the factor of 3 between
% those two choices is the smallest of the uncertainties in it -- Ld is ~1 km
% here against a 40 km shoal, so the WIDTH may be wrong by a factor of tens.
%
% This script measures APE_D from the model instead, so C_geom is diagnosed:
%
%     C_geom,measured = APE_D / (g' H0^2 Ld)
%
% If that clusters near a constant across runs, the closed form is usable and
% you can quote the constant. If it drifts systematically, the wedge is not
% self-similar in this geometry and the Eq. 7 test should use measured APE.
%
% TWO ESTIMATES, deliberately. They fail in different ways, so agreement is
% evidence and disagreement is itself a result:
%
%   APE_qg      = 1/2 \int b'^2 / N_ref^2 dV,  b' = b - b_ref(z)
%                 The quasi-geostrophic form, valid for SMALL displacements.
%                 Matches the assumption behind Eq. 8, but divides by N_ref^2
%                 and so is fragile wherever the reference column is weakly
%                 stratified.
%   APE_sorted  = PE - PE_min, with PE_min from adiabatically sorting every
%                 wet parcel into the actual basin (Lorenz). No reference
%                 profile, no division, valid at any displacement -- which
%                 matters here, because a 5 m shoal truncates the column and
%                 the displacements are NOT small.
%
% Both are divided by the alongshore length of the box, giving m^4/s^2 to
% match APE_D (a cross-sectional quantity -- Zhao et al. Sec. 3d(2)).
%
% g', H0 and Ld use the SAME definitions as plot_theory_vs_model.m, so the
% diagnosed C_geom drops straight into that script.
%
% Options:
%   'sweep_root'      default /glade/work/nguyen/unc-shoals/sweep_output
%   'window'          which 10-day average; [] = last (default)
%   'y_box'           km, alongshore range                 default [20 Inf]
%   'x_ambient_width' km, offshore strip taken as ambient  default 10
%   'x_g_range'       km, shelf columns used for g'        default [5 50]
%   'x_break'         km, shelf-break fallback             default 50
%   'outdir'          where the PNGs go

p = inputParser;
addParameter(p, 'sweep_root', '/glade/work/nguyen/unc-shoals/sweep_output', @ischar);
addParameter(p, 'window', [], @(x)isempty(x) || (isnumeric(x)&&isscalar(x)));
addParameter(p, 'y_box', [20 Inf], @(v)isnumeric(v)&&numel(v)==2);
addParameter(p, 'x_ambient_width', 10, @isnumeric);
addParameter(p, 'x_g_range', [5 50], @(v)isnumeric(v)&&numel(v)==2);
addParameter(p, 'x_break', 50, @isnumeric);
addParameter(p, 'outdir', '/glade/work/nguyen/unc-shoals/animations', @ischar);
parse(p, varargin{:});
o = p.Results;

% Model constants, matching plot_theory_vs_model.m
g    = 9.81;
f    = 2 * (2*pi/86400) * sin(deg2rad(35.2480));
a_T  = 1.67e-4;      % thermal expansion  [1/K]
b_S  = 7.80e-4;      % haline contraction [1/psu]

cluster_dir = fullfile(o.sweep_root, cluster_name);
[labels, tags] = discover_sweeps(cluster_dir);
n = numel(labels);

fprintf('\n╔══════════════════════════════════════════════════════════════════╗\n');
fprintf('║ APE wedge: %-53s ║\n', cluster_name);
fprintf('╚══════════════════════════════════════════════════════════════════╝\n');
fprintf('  APE measured per unit alongshore length (m^4/s^2), over the shelf\n');
fprintf('  box x = 0 to the shelf break, y = %g km to the end of the domain.\n\n', o.y_box(1));

[APE_qg, APE_srt, gp, H0, Ld, Wmeas, WE, dxv, nfloor] = deal(nan(1, n));
mixed = repmat({''}, 1, n);
BP = cell(1, n); XK = cell(1, n); ZK = cell(1, n);

for k = 1:n
    fn = find_tavg_(cluster_dir, labels{k}, tags{k});
    if isempty(fn)
        fprintf('  %-16s no time_avg_3d file -- skipped\n', labels{k});
        continue;
    end

    xc = try_nc_(fn, {'xC','x_caa','xc'});  x_km = xc(:)/1e3;
    yc = try_nc_(fn, {'yC','y_aca','yc'});  y_km = yc(:)/1e3;
    zc = try_nc_(fn, {'zC','z_aac','zc'});  zc   = zc(:);
    nt = numel(ncread(fn, 'time'));
    it = o.window; if isempty(it), it = nt; end
    it = max(1, min(nt, it));

    dx = cell_width_(fn, xc(:), {'xF','x_faa','xf'});
    dy = cell_width_(fn, yc(:), {'yF','y_afa','yf'});
    dz = cell_width_(fn, zc,    {'zF','z_aaf','zf'});
    dxv(k) = median(dx);

    iy = find(y_km >= o.y_box(1) & y_km <= o.y_box(2));
    Ly = sum(dy(iy));                                   % alongshore length [m]

    % ── buoyancy, from the model's linear equation of state ──────────────
    T = ncread(fn, 'T', [1 iy(1) 1 it], [Inf numel(iy) Inf 1]);
    S = ncread(fn, 'S', [1 iy(1) 1 it], [Inf numel(iy) Inf 1]);
    wet = (T ~= 0) & isfinite(T);                       % immersed cells are 0
    b = g * (a_T * T - b_S * S);
    b(~wet) = NaN;

    % ── reference profile: the offshore ambient column ───────────────────
    % Same strip plot_theory_vs_model.m uses for N, so b' means "departure
    % from the open-ocean column" in both places.
    ix_amb = find(x_km >= max(x_km) - o.x_ambient_width);
    b_ref  = squeeze(median(median(b(ix_amb,:,:), 1, 'omitnan'), 2, 'omitnan'));
    N2_ref = gradient(b_ref(:), zc(:));

    % N2_ref -> 0 makes b'^2/N2 blow up. Floor it at 1% of the median stable
    % value rather than dropping those levels, and report how many were hit,
    % because a large count means the QG estimate should not be trusted.
    good = N2_ref(N2_ref > 0);
    if isempty(good)
        fprintf('  %-16s reference column is not stably stratified -- skipped\n', labels{k});
        continue;
    end
    N2_floor   = 0.01 * median(good);
    nfloor(k)  = sum(N2_ref < N2_floor);
    N2_ref     = max(N2_ref, N2_floor);

    bp = b - reshape(b_ref, 1, 1, []);                  % b'

    % ── shelf box: coast out to the shelf break ──────────────────────────
    % Zsh lives in sweep_metadata_<tag>.nc, NOT in time_avg_3d, so reading it
    % off the averages file silently returned the -25 m default for every run
    % -- including Zsh_m10 and Zsh_m50, the only two runs in the sweep that
    % give H0 any dynamic range. Worse, H0 also sets the shelf-break search,
    % so those two had their integration box cut at the wrong x: 8.7 km for
    % Zsh_m50 (a sliver) and 63 km for Zsh_m10 (past the break). Resolved the
    % same way plot_theory_vs_model.m does it: run name first, metadata on top.
    H0(k)   = abs(zsh_for_(cluster_dir, labels{k}, tags{k}));
    depth_x = median(sum(wet .* reshape(dz, 1, 1, []), 3), 2);
    ib = shelf_break_(depth_x, H0(k));
    if isempty(ib), [~, ib] = min(abs(x_km - o.x_break)); end
    ix = 1:ib;

    V = reshape(dx(ix), [], 1) .* reshape(dy(iy), 1, []) .* reshape(dz, 1, 1, []);
    V = V .* wet(ix,:,:);
    bpx = bp(ix,:,:);  bx = b(ix,:,:);  wx = wet(ix,:,:);

    % ── APE, estimate 1: quasi-geostrophic ───────────────────────────────
    e = 0.5 * bpx.^2 ./ reshape(N2_ref, 1, 1, []);
    e(~wx | ~isfinite(e)) = 0;
    APE_qg(k) = sum(e .* V, 'all') / Ly;

    % ── APE, estimate 2: Lorenz, by adiabatic sorting ────────────────────
    % PE is proportional to -\int b z dV. The minimum-PE state puts the
    % lightest water (largest b) at the smallest |z|. Sorting is done against
    % CUMULATIVE VOLUME rather than cell index, so it stays correct if the
    % grid is ever stretched, and so the reference state is rearranged into
    % the actual basin shape rather than a rectangle.
    zz = repmat(reshape(zc, 1, 1, []), size(bx,1), size(bx,2), 1);
    sel = wx & isfinite(bx);
    bw = bx(sel);  zw = zz(sel);  Vw = V(sel);

    [bs, is] = sort(bw, 'descend');                     % lightest first
    Vs = Vw(is);
    cumP = cumsum(Vs) - Vs/2;                           % mid-volume of each parcel

    [zs, iz] = sort(zw, 'descend');                     % shallowest first
    Vz = Vw(iz);
    cumC = cumsum(Vz) - Vz/2;                           % the container

    z_ref = interp1(cumC, zs, cumP, 'linear', 'extrap');
    E_act = -sum(bw .* zw .* Vw);
    E_ref = -sum(bs .* z_ref .* Vs);
    APE_srt(k) = (E_act - E_ref) / Ly;
    % These are the largest arrays in the loop (one full 3D field each at
    % 250 m), and the wedge figure below holds one cross-section per run.
    clear zz sel bw zw Vw bs Vs cumP zs Vz cumC z_ref e;

    % ── g' and Ld, defined exactly as in plot_theory_vs_model.m ──────────
    ixg = find(x_km >= o.x_g_range(1) & x_km <= o.x_g_range(2));
    gp(k) = column_contrast_(b(ixg,:,:), wet(ixg,:,:), zc);

    % A forced run mixes the shelf column vertically, and then the SURFACE-
    % MINUS-BOTTOM contrast -- which is how g' is defined here and in
    % plot_theory_vs_model.m -- goes to zero. Ld = sqrt(g' H0)/f goes with it,
    % and C = APE/(g' H0^2 Ld) divides by something approaching zero. Flooring
    % g' at 1e-3 (what this used to do) hid that behind a plausible-looking
    % number and produced C values of 1e5.
    %
    % It is not a numerical problem to paper over: it says the predictor is
    % UNDEFINED for those runs. The APE is still there -- it has just moved
    % into the horizontal structure, which a vertical contrast cannot see.
    % So g' is left as NaN and the run drops out of the fit, while its
    % measured APE is still reported.
    gp_amb = ambient_contrast_(b_ref);
    if ~(gp(k) > 0)
        gp(k) = NaN;
        mixed{k} = 'vertically mixed: no surface-to-bottom contrast, g'' undefined';
    elseif isfinite(gp_amb) && gp_amb > 0 && gp(k) < 0.1 * gp_amb
        mixed{k} = sprintf('nearly mixed: g'' is %.1f%% of the ambient contrast', ...
                           100*gp(k)/gp_amb);
    end
    Ld(k) = sqrt(gp(k) * H0(k)) / f;

    % ── how wide is the anomaly, really? ─────────────────────────────────
    % Integral width: (area under the depth-integrated |b'| profile) divided
    % by its peak. Threshold-free, and it answers the question Ld is being
    % ASSUMED to answer. If this comes out at tens of km while Ld is ~1 km,
    % the error in the assumed APE is a factor of tens, not the factor of 3
    % between C_geom = 1/2 and 1/6.
    absbp = abs(bpx); absbp(~wx) = 0;
    P = squeeze(sum(sum(absbp .* V, 2), 3));            % per x, [m^3 * m/s^2]
    if max(P) > 0
        Wmeas(k) = sum(P .* dx(ix)) / max(P);
    end

    % ...and the same width weighted by b'^2 rather than |b'|. APE goes as the
    % SQUARE of the anomaly, so the long weak offshore tail that dominates W
    % contributes almost nothing to the energy. W_E is therefore the width
    % that actually matters, and it is the honest comparison against Ld.
    b2 = bpx.^2; b2(~wx) = 0;
    P2 = squeeze(sum(sum(b2 .* V, 2), 3));
    if max(P2) > 0
        WE(k) = sum(P2 .* dx(ix)) / max(P2);
    end

    % Keep the alongshore-mean cross-section for the wedge figure.
    bpm = bpx; bpm(~wx) = NaN;
    BP{k} = squeeze(mean(bpm, 2, 'omitnan'))';          % (Nz x Nx)
    XK{k} = x_km(ix);  ZK{k} = zc;

    % g' as a FRACTION of the ambient contrast is printed on every run, not
    % only the flagged ones. It is the single most diagnostic number for a
    % forced run: wind_0p03 and wind_0p05 clear the 10% flag but still sit at
    % 34% and 19% of ambient, which is enough to put their C_geom off by more
    % than an order of magnitude.
    fprintf(['  %-16s g'' = %.3e (%5.1f%% of ambient)  H0 = %4.1f m  Ld = %6.0f m ' ...
             '(Ld/dx = %4.1f)  W_E = %6.0f m  APE_srt = %.3e\n'], ...
            labels{k}, gp(k), 100*gp(k)/gp_amb, H0(k), Ld(k), Ld(k)/dxv(k), ...
            WE(k), APE_srt(k));
    if ~isempty(mixed{k})
        fprintf('                   ** %s\n', mixed{k});
    end
    if nfloor(k) > 0
        fprintf('                   (%d of %d reference levels floored -- APE_qg is soft here)\n', ...
                nfloor(k), numel(N2_ref));
    end
end

% ── the number this script exists to produce ─────────────────────────────
base_Ld = gp .* H0.^2 .* Ld;
base_W  = gp .* H0.^2 .* Wmeas;
base_WE = gp .* H0.^2 .* WE;
C_Ld_qg  = APE_qg  ./ base_Ld;
C_Ld_srt = APE_srt ./ base_Ld;
C_W_srt  = APE_srt ./ base_W;
C_WE_srt = APE_srt ./ base_WE;

fprintf('\n── C_geom = APE_sorted / (g'' H0^2 L) ────────────────────────────\n');
fprintf('  %-16s %12s %12s %12s   %s\n', 'run', 'L = Ld', 'L = W', 'L = W_E', '');
for k = 1:n
    fprintf('  %-16s %12.4f %12.4f %12.4f   %s\n', labels{k}, ...
            C_Ld_srt(k), C_W_srt(k), C_WE_srt(k), mixed{k});
end

% A run whose column is mixed has no usable g', whether that shows up as a
% NaN or as a tiny positive residual. Both must leave the statistics: a
% "nearly mixed" run with g' at 0.1% of ambient still divides by almost
% nothing and drags the spread to six figures.
usable = cellfun(@isempty, mixed) & isfinite(gp) & gp > 0;
nmix = sum(~usable);
if nmix > 0
    fprintf(['\n  %d run(s) excluded from the statistics below: the shelf column is\n' ...
             '  vertically mixed, so the surface-to-bottom g'' -- and every scale built\n' ...
             '  on it -- is undefined or numerically meaningless. Their measured APE is\n' ...
             '  still in the table above and is NOT small, which is the point: the\n' ...
             '  energy has moved into horizontal structure, where a vertical contrast\n' ...
             '  cannot see it.\n'], nmix);
end

report_spread_('with L = Ld  (as the theory assumes)', C_Ld_srt, usable);
report_spread_('with L = measured width W',            C_W_srt,  usable);
report_spread_('with L = energy-weighted width W_E',   C_WE_srt, usable);

% ── and the same thing grouped by shelf depth ────────────────────────────
% Eq. 8 puts the entire depth dependence in H0^2, so whether C_geom survives
% a change in H0 is the question the closed form lives or dies on. Pooling
% the depths hides it: C can be tight at each H0 separately and still vary by
% a large factor between them.
Hs = unique(H0(usable));
if numel(Hs) > 1
    fprintf('\n── C_geom (L = Ld) grouped by shelf depth ───────────────────────\n');
    fprintf('  %8s %5s %10s %10s\n', 'H0 (m)', 'n', 'median', 'spread');
    for hh = Hs(:)'
        m = usable & H0 == hh;
        c = C_Ld_srt(m);
        if numel(c) == 1
            fprintf('  %8.1f %5d %10.4f %10s\n', hh, 1, c, '--');
        else
            fprintf('  %8.1f %5d %10.4f %9.0f%%\n', hh, numel(c), median(c), ...
                    100*(max(c)-min(c))/median(c));
        end
    end
    cg = arrayfun(@(hh) median(C_Ld_srt(usable & H0 == hh)), Hs);
    fprintf(['  C_geom changes by a factor of %.1f across these depths. Eq. 8 asks it\n' ...
             '  to be constant, so a factor well above the within-depth spread means\n' ...
             '  the H0^2 form does not hold here and the Eq. 7 test should use the\n' ...
             '  MEASURED APE rather than the closed form.\n'], max(cg)/min(cg));
end

fprintf(['\n  Reference values: 1/6 = %.3f (triangular wedge, Zhao et al. Eq. 8)\n' ...
         '                    1/2 = %.3f (anomaly fills the H0 x L box)\n'], 1/6, 1/2);
fprintf(['  A tight cluster means the closed form works and its constant can be\n' ...
         '  quoted. A drift with Z_S or shelf depth means the wedge is not\n' ...
         '  self-similar here, and the Eq. 7 test should use measured APE.\n']);

qa = APE_qg ./ APE_srt;
qa = qa(isfinite(qa));
if isempty(qa), qa = NaN; end
fprintf(['\n  APE_qg / APE_sorted: %.2f to %.2f. Near 1 means displacements are\n' ...
         '  small enough for the QG form, which is the assumption behind Eq. 8.\n' ...
         '  Far from 1 means they are not, and the sorted value is the one to use.\n'], ...
        min(qa), max(qa));

if ~exist(o.outdir, 'dir'), mkdir(o.outdir); end

% ── figure 1: the wedge itself ───────────────────────────────────────────
% The point of looking rather than assuming: this shows the shape AND the
% width in one picture, which is what C_geom is standing in for.
ok = find(isfinite(APE_srt));
if ~isempty(ok)
    nc  = min(3, numel(ok));
    nr  = ceil(numel(ok)/nc);
    f1  = figure('Color','w','Position',[60 60 min(1700, 520*nc) 340*nr]);
    t1  = tiledlayout(f1, nr, nc, 'TileSpacing','compact','Padding','compact');
    lim = max(cellfun(@(c) max(abs(c(:))), BP(ok)));
    if ~(lim > 0), lim = 1e-6; end   % all-NaN or flat: keep clim valid
    for q = ok
        ax = nexttile(t1);
        set(ax, 'Color', [0.65 0.65 0.65]);   % immersed cells show through as grey
        h = imagesc(ax, XK{q}, ZK{q}, BP{q});
        set(h, 'AlphaData', ~isnan(BP{q}));
        set(ax, 'YDir', 'normal');
        xlim(ax, [min(XK{q}) max(XK{q})]);  ylim(ax, [min(ZK{q}) 0]);
        clim(ax, [-lim lim]);
        colormap(ax, redblue_());
        hold(ax, 'on');
        % Ld and the measured width, drawn from the coast for comparison
        plot(ax, [0 Ld(q)/1e3],    [-2 -2], 'k-',  'LineWidth', 3);
        plot(ax, [0 WE(q)/1e3],    [-4 -4], 'k-.', 'LineWidth', 2);
        plot(ax, [0 Wmeas(q)/1e3], [-6 -6], 'k--', 'LineWidth', 2);
        title(ax, sprintf('%s   L_d = %.1f,  W_E = %.1f,  W = %.1f km', ...
              strrep(labels{q},'_','\_'), Ld(q)/1e3, WE(q)/1e3, Wmeas(q)/1e3), 'FontWeight','bold');
        xlabel(ax, 'x (km)');  ylabel(ax, 'z (m)');
        set(ax, 'FontSize', 10);
    end
    cb = colorbar(nexttile(t1, numel(ok)));
    cb.Label.String = 'b'' = b - b_{ref}(z)   (m s^{-2})';
    title(t1, sprintf(['%s  —  the buoyancy anomaly whose shape C_{geom} stands in for' ...
          '   (solid = L_d, dash-dot = W_E, dashed = W)'], strrep(cluster_name,'_','\_')), ...
          'FontWeight','bold', 'FontSize', 12);
    out1 = fullfile(o.outdir, sprintf('%s_ape_wedge.png', cluster_name));
    exportgraphics(f1, out1, 'Resolution', 200);
    fprintf('\nSaved: %s\n', out1);
    close(f1);
end

% ── figure 2: does APE scale with g' H0^2 L? ─────────────────────────────
f2 = figure('Color','w','Position',[60 60 1250 470]);
t2 = tiledlayout(f2, 1, 2, 'TileSpacing','compact','Padding','compact');

APE_fit = APE_srt; APE_fit(~usable) = NaN;    % mixed columns do not constrain C
scatter_panel_(nexttile(t2), base_Ld, APE_fit, labels, ...
    'g'' H_0^2 L_d   (m^4 s^{-2})', 'measured APE_\Delta   (m^4 s^{-2})', ...
    'L = L_d  (what the theory assumes)');
scatter_panel_(nexttile(t2), base_WE, APE_fit, labels, ...
    'g'' H_0^2 W_E   (m^4 s^{-2})', '', ...
    'L = energy-weighted width  (APE \propto b''^2)');

title(t2, sprintf(['%s  —  a constant C_{geom} means the points sit on one line ' ...
      'through the origin'], strrep(cluster_name,'_','\_')), 'FontWeight','bold', 'FontSize', 12);
out2 = fullfile(o.outdir, sprintf('%s_ape_scaling.png', cluster_name));
exportgraphics(f2, out2, 'Resolution', 200);
fprintf('Saved: %s\n', out2);
close(f2);

R = struct('labels', {labels}, 'APE_qg', APE_qg, 'APE_sorted', APE_srt, ...
           'g_prime', gp, 'H0', H0, 'Ld', Ld, 'W', Wmeas, 'W_E', WE, 'dx', dxv, ...
           'mixed', {mixed}, 'C_Ld_qg', C_Ld_qg, 'C_Ld_sorted', C_Ld_srt, ...
           'C_W_sorted', C_W_srt, 'C_WE_sorted', C_WE_srt);
end

% ================== helpers ==================

function report_spread_(what, C, usable)
if nargin < 3, usable = true(size(C)); end
c = C(isfinite(C) & C > 0 & usable);
if numel(c) < 2, return; end
fprintf('\n  %s\n', what);
fprintf('    median %.4f, range %.4f to %.4f, spread %.0f%% of the median\n', ...
        median(c), min(c), max(c), 100*(max(c)-min(c))/median(c));
end

function scatter_panel_(ax, x, y, labels, xl, yl, ttl)
ok = isfinite(x) & isfinite(y) & x > 0 & y > 0;
hold(ax,'on'); grid(ax,'on');
cc = lines(numel(labels));
for k = find(ok)
    plot(ax, x(k), y(k), 'o', 'MarkerSize', 10, 'MarkerFaceColor', cc(k,:), ...
         'MarkerEdgeColor', 'k', 'DisplayName', strrep(labels{k},'_','\_'));
end
if any(ok)
    xr = [min(x(ok))/2, max(x(ok))*2];
    % Best constant through the origin, which IS C_geom if the form holds
    Cfit = sum(x(ok).*y(ok)) / sum(x(ok).^2);
    plot(ax, xr, Cfit*xr, 'k-',  'LineWidth', 1.6, 'DisplayName', sprintf('fit C = %.3f', Cfit));
    plot(ax, xr, xr/2,    'k--', 'LineWidth', 1.0, 'DisplayName', 'C = 1/2');
    plot(ax, xr, xr/6,    'k:',  'LineWidth', 1.4, 'DisplayName', 'C = 1/6');
    xlim(ax, xr);
end
set(ax, 'XScale','log', 'YScale','log', 'FontSize', 11);
xlabel(ax, xl, 'FontWeight','bold');
if ~isempty(yl), ylabel(ax, yl, 'FontWeight','bold'); end
title(ax, ttl, 'FontWeight','bold');
legend(ax, 'Location','southeast', 'FontSize', 8);
end

function gpr = column_contrast_(b, wet, zc)
% Surface-minus-bottom buoyancy contrast, median over columns. Same
% definition as plot_theory_vs_model.m: the density structure here is
% vertical, so Zhao et al.'s horizontal plume-vs-ambient contrast does not
% apply.
[~, kup] = sort(zc, 'ascend');
b = b(:,:,kup);  wet = wet(:,:,kup);
nz = size(b,3);  kk = reshape(1:nz, 1, 1, []);
k_top = max(wet .* kk, [], 3);
k_bot = min(wet .* kk + ~wet * (nz+1), [], 3);
okc = (k_top > k_bot);
if ~any(okc(:)), gpr = NaN; return; end
[ii, jj] = ndgrid(1:size(b,1), 1:size(b,2));
gpr = median(b(sub2ind(size(b), ii(okc), jj(okc), k_top(okc))) ...
           - b(sub2ind(size(b), ii(okc), jj(okc), k_bot(okc))), 'omitnan');
end

function zsh = zsh_for_(cluster_dir, label, tag)
% Shelf depth: from the run label if it names one, then from sweep_metadata
% if that file is present. Mirrors params_from_name + read_att in
% plot_theory_vs_model.m, so both scripts agree on H0.
zsh = -25;
tok = regexp(label, 'Zsh_m(\d+)', 'tokens', 'once');
if ~isempty(tok), zsh = -str2double(tok{1}); end
d = dir(fullfile(cluster_dir, 'sweep_*'));
d = d([d.isdir]);
for k = 1:numel(d)
    if ~isempty(regexp(d(k).name, [regexptranslate('escape', label) '$'], 'once'))
        for cand = {fullfile(cluster_dir, d(k).name, ['sweep_metadata_' tag '.nc']), ...
                    fullfile(cluster_dir, d(k).name, ['sweep_metadata_sweep_' tag '.nc'])}
            if isfile(cand{1})
                try, zsh = double(ncreadatt(cand{1}, '/', 'Zsh')); catch, end
                return;
            end
        end
    end
end
end

function gpa = ambient_contrast_(b_ref)
% Surface-to-bottom contrast of the offshore reference column itself, as the
% yardstick for deciding that a shelf column has been mixed.
ok = isfinite(b_ref);
if sum(ok) < 2, gpa = NaN; return; end
bb = b_ref(ok);
gpa = bb(end) - bb(1);
end

function ib = shelf_break_(depth_x, H0)
ib = [];
i0 = find(depth_x >= 0.9*H0, 1, 'first');
if isempty(i0), return; end
k = find(depth_x(i0:end) > 1.1*H0, 1, 'first');
if ~isempty(k), ib = i0 + k - 1; end
end

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

function v = read_att_(fname, att, default)
try
    v = double(ncreadatt(fname, '/', att));
catch
    v = default;
end
end

function d = cell_width_(fname, c, face_names)
for nm = face_names
    try
        fc = ncread(fname, nm{1});
        if numel(fc) == numel(c) + 1
            d = abs(diff(fc(:)));  return;
        end
    catch
    end
end
if numel(c) == 1, d = 1; return; end
mid   = (c(1:end-1) + c(2:end)) / 2;
edges = [c(1) - (mid(1) - c(1)); mid; c(end) + (c(end) - mid(end))];
d     = abs(diff(edges));
end

function data = try_nc_(fname, names)
for i = 1:numel(names)
    try, data = ncread(fname, names{i}); return; catch, end
end
error('No coordinate among {%s} in %s.', strjoin(names, ', '), fname);
end

function cm = redblue_()
n = 256; h = floor(n/2);
r = [linspace(0.13,1,h)'; linspace(1,0.70,n-h)'];
g = [linspace(0.33,1,h)'; linspace(1,0.09,n-h)'];
b = [linspace(0.60,1,h)'; linspace(1,0.17,n-h)'];
cm = [r g b];
end
