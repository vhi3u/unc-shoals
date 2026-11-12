function check_eke_theory(cluster_name)
% check_eke_theory(cluster_name)
%
% Tests the EKE scaling of Zhao et al. (Sec. 3d, Eq. 7-8; after Brink 2024)
%     EKE = a (1 + c s)^n / (1 + b R) * APE / (H0 Ld),   APE = C_geom g' H0^2 Ld
% against the diagnosed EKE from our runs, and fits alternatives.
%
% Run plot_theory_vs_model(cluster_name) first: it saves the per-run inputs to
%   <animations>/<cluster>_theory_inputs.mat
%
% Models compared (all predict specific EKE, m^2 s^-2; base = g' H0):
%   M0  paper coefficients (regime), rectangle  C_geom = 1/2
%   M0t paper coefficients (regime), triangle   C_geom = 1/6 (as fitted in the paper)
%   M1  a * g'H0                                 (1 parameter; = any C_geom, refit a)
%   M2  a (1+cs)^n / (1+bR) * g'H0               (paper form, refit a,b,c,n)
%   M3  a * g'H0 * (h_s L_s / (H0 Ld))^p         (shoal cross-sectional area)
%   M4  a * g'H0 * (h_s/H0)^p * (L_s/Ld)^q       (shoal height and length separately)
% with h_s = H0 + Z_s the shoal height above the shelf.
%
% Skill is measured in log space (EKE spans >1 decade):
%   R2_log  = 1 - SS_res / SS_tot of log(EKE)
%   bias    = median(theory / simulation)
%   R2_loo  = leave-one-out R2_log (guards against over-fitting with few runs)
%
% Output: table in the log, <cluster>_eke_theory_check.png and .csv

if nargin < 1 || isempty(cluster_name)
    cluster_name = 'cluster_Fnnie_1_19';
end
outdir = '/glade/work/nguyen/unc-shoals/animations';
if ~exist(outdir, 'dir'), outdir = pwd; end
in_file = fullfile(outdir, sprintf('%s_theory_inputs.mat', cluster_name));
if ~isfile(in_file)
    error('Inputs not found: %s\nRun plot_theory_vs_model(''%s'') first.', in_file, cluster_name);
end
S = load(in_file);  D = S.theory_inputs;

n    = numel(D.sweeps);
y    = D.EKE_diag(:);                     % simulated specific EKE (m^2/s^2)
base = D.g_prime(:) .* D.H0(:);           % g' H0 (m^2/s^2)
hs   = D.H0(:) + D.Zs(:);                 % shoal height above shelf (m)
s    = D.s(:);  R = D.R(:);

% ---------- per-run table ----------
fprintf('\n=== Inputs (%d runs) ===\n', n);
fprintf('%-12s %6s %6s %5s %5s %9s %9s %7s %7s %8s %10s %9s\n', ...
    'run', 'Zs', 'Ls_km', 'H0', 'h_s', 'g''', 'N', 'Ld_km', 's', 'R', 'EKE_sim', 'sqrtEKE');
for i = 1:n
    fprintf('%-12s %6.1f %6.1f %5.1f %5.1f %9.3e %9.3e %7.2f %7.3f %8.4f %10.3e %9.3e\n', ...
        D.sweeps{i}, D.Zs(i), D.Ls(i)/1e3, D.H0(i), hs(i), D.g_prime(i), D.N(i), ...
        D.Ld(i)/1e3, s(i), R(i), y(i), sqrt(y(i)));
end

% ---------- direction check: does EKE vary with s the same way in every sweep? ----------
fprintf('\n=== Sign check: d log(EKE) / d log(s) within each sweep ===\n');
groups = { 'Z_s sweep', {'Zs15_Ls40k','Zs10_Ls40k','Zs5_Ls40k'}; ...
           'L_s sweep', {'Zs5_Ls20k','Zs5_Ls30k','Zs5_Ls40k'} };
for g = 1:size(groups, 1)
    idx = find(ismember(D.sweeps, groups{g, 2}));
    if numel(idx) < 2, continue; end
    pfit = polyfit(log(s(idx)), log(y(idx)), 1);
    fprintf('  %-10s  s = [%s]   EKE = [%s]   slope = %+.2f\n', groups{g, 1}, ...
        sprintf('%.3f ', s(idx)), sprintf('%.2e ', y(idx)), pfit(1));
end
fprintf(['  -> The paper form needs ONE monotonic dependence on s. Opposite signs in the two\n' ...
         '     sweeps mean no choice of (a,b,c,n) can fit both: s = (h_s/L_s) N/f rises with\n' ...
         '     shoal height AND with shorter shoals, but EKE may respond oppositely to the two.\n']);

% ---------- models ----------
reg = regime_coeffs(D.tau(:));
M = struct('name', {}, 'pred', {}, 'np', {}, 'p0', {});
M(end+1) = struct('name', 'M0  paper, rect 1/2', 'pred', @(p) 0.5 * base .* reg.a .* (1 + reg.c .* s).^reg.n ./ (1 + reg.b .* R), 'np', 0, 'p0', []);
M(end+1) = struct('name', 'M0t paper, tri 1/6',  'pred', @(p) (1/6) * base .* reg.a .* (1 + reg.c .* s).^reg.n ./ (1 + reg.b .* R), 'np', 0, 'p0', []);
M(end+1) = struct('name', 'M1  a g''H0',          'pred', @(p) exp(p(1)) * base, 'np', 1, 'p0', log(1e-3));
% M2 parameters are bounded so the fit stays physical: 0 < b < 10, 0 < c < 100, |n| < 3
M(end+1) = struct('name', 'M2  paper form refit', 'pred', @(p) m2_pred(p, base, s, R), 'np', 4, 'p0', [log(1e-3), 0, 0, 0.3]);
M(end+1) = struct('name', 'M3  shoal area',       'pred', @(p) exp(p(1)) * base .* (hs .* D.Ls(:) ./ (D.H0(:) .* D.Ld(:))).^p(2), 'np', 2, 'p0', [log(1e-3), 1]);
M(end+1) = struct('name', 'M4  h_s, L_s separate','pred', @(p) exp(p(1)) * base .* (hs ./ D.H0(:)).^p(2) .* (D.Ls(:) ./ D.Ld(:)).^p(3), 'np', 3, 'p0', [log(1e-3), 1, 1]);

fprintf('\n=== Model skill (log space) ===\n');
fprintf('%-24s %4s %8s %8s %8s   %s\n', 'model', 'k', 'R2_log', 'R2_loo', 'bias', 'parameters');
res = struct('name', {}, 'yhat', {}, 'R2', {}, 'R2loo', {}, 'bias', {}, 'p', {});
for m = 1:numel(M)
    mask = true(n, 1);
    p = fit_log(M(m), y, mask);
    yhat = M(m).pred(p);
    R2 = r2_log(y, yhat);
    % leave-one-out
    if M(m).np > 0 && n > M(m).np + 1
        yloo = nan(n, 1);
        for i = 1:n
            mk = mask; mk(i) = false;
            pi_ = fit_log(M(m), y, mk);
            yi = M(m).pred(pi_);
            yloo(i) = yi(i);
        end
        R2loo = r2_log(y, yloo);
    else
        R2loo = NaN;
    end
    bias = median(yhat ./ y, 'omitnan');
    fprintf('%-24s %4d %8.2f %8.2f %8.2g   %s\n', M(m).name, M(m).np, R2, R2loo, bias, param_str(M(m), p));
    res(end+1) = struct('name', M(m).name, 'yhat', yhat, 'R2', R2, 'R2loo', R2loo, 'bias', bias, 'p', p); %#ok<AGROW>
end
fprintf(['\nNotes:\n' ...
         '  * C_geom (rectangle 1/2 vs triangle 1/6) only rescales APE by a constant, so it\n' ...
         '    changes the bias but never R2: compare M0 vs M0t, and M1 (any C_geom, refit a).\n' ...
         '  * The paper''s a, b, c, n were fitted to a 180 m deep Greenland shelf with f = 1.26e-4;\n' ...
         '    our shelf is 10-50 m deep with f = 8.4e-5, so they are not expected to transfer.\n' ...
         '  * With ~9 runs, trust R2_loo over R2_log for the multi-parameter fits.\n']);

% ---------- figure ----------
show = [1 3 4 5];          % M0, M1, M2, M3
fig = figure('Color', 'w', 'Position', [100 100 1350 400]);
tl = tiledlayout(fig, 1, numel(show), 'TileSpacing', 'compact', 'Padding', 'compact');
[cols, mks, szs, filled] = run_styles(D.sweeps);
lims = [min([y; vertcat(res(show).yhat)]) / 2, max([y; vertcat(res(show).yhat)]) * 2];
lims = sqrt(lims);
for k = 1:numel(show)
    r = res(show(k));
    ax = nexttile(tl); hold(ax, 'on');
    plot(ax, lims, lims, 'k--', 'LineWidth', 1.2);
    for i = 1:n
        if filled(i)
            scatter(ax, sqrt(r.yhat(i)), sqrt(y(i)), szs(i), mks{i}, 'MarkerEdgeColor', cols(i,:), 'MarkerFaceColor', cols(i,:), 'LineWidth', 1.5);
        else
            scatter(ax, sqrt(r.yhat(i)), sqrt(y(i)), szs(i), mks{i}, 'MarkerEdgeColor', cols(i,:), 'LineWidth', 1.5);
        end
    end
    set(ax, 'XScale', 'log', 'YScale', 'log', 'FontSize', 14, 'Box', 'on', 'XGrid', 'on', 'YGrid', 'on', ...
        'LineWidth', 1, 'TickDir', 'out', 'XColor', 'k', 'YColor', 'k');
    xlim(ax, lims); ylim(ax, lims); pbaspect(ax, [1 1 1]);
    title(ax, sprintf('(%s) %s', char('a' + k - 1), strtrim(r.name(4:end))), 'FontSize', 16, 'FontWeight', 'bold');
    text(ax, 0.05, 0.92, sprintf('R^2_{log} = %.2f', r.R2), 'Units', 'normalized', 'FontSize', 12);
    xlabel(ax, 'Theory $\sqrt{EKE}$ (m s$^{-1}$)', 'Interpreter', 'latex', 'FontSize', 16);
    if k == 1
        ylabel(ax, 'Simulation $\sqrt{EKE}$ (m s$^{-1}$)', 'Interpreter', 'latex', 'FontSize', 16);
    end
end
outpng = fullfile(outdir, sprintf('%s_eke_theory_check.png', cluster_name));
exportgraphics(fig, outpng, 'Resolution', 300);
fprintf('\nSaved: %s\n', outpng);

% ---------- CSV ----------
outcsv = fullfile(outdir, sprintf('%s_eke_theory_check.csv', cluster_name));
fid = fopen(outcsv, 'w');
fprintf(fid, 'run,Zs,Ls_km,H0,h_s,g_prime,N,Ld_km,s,R,EKE_sim');
for m = 1:numel(res), fprintf(fid, ',%s', matlab.lang.makeValidName(res(m).name)); end
fprintf(fid, '\n');
for i = 1:n
    fprintf(fid, '%s,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g', D.sweeps{i}, D.Zs(i), D.Ls(i)/1e3, D.H0(i), hs(i), ...
        D.g_prime(i), D.N(i), D.Ld(i)/1e3, s(i), R(i), y(i));
    for m = 1:numel(res), fprintf(fid, ',%g', res(m).yhat(i)); end
    fprintf(fid, '\n');
end
fclose(fid);
fprintf('Saved: %s\n', outcsv);
end

% ================== helpers ==================

function p = fit_log(Mi, y, mask)
% Least squares in log space over the runs in mask
if Mi.np == 0, p = []; return; end
cost = @(p) log_cost(Mi.pred(p), y, mask);
p = fminsearch(cost, Mi.p0, optimset('Display', 'off', 'MaxIter', 2e4, 'MaxFunEvals', 4e4, 'TolX', 1e-8, 'TolFun', 1e-10));
end

function J = log_cost(yhat, y, mask)
v = yhat(mask);
if any(~isfinite(v) | v <= 0 | imag(v) ~= 0)
    J = 1e6;                 % reject parameter sets giving non-physical EKE
else
    J = sum((log(v) - log(y(mask))).^2);
end
end

function R2 = r2_log(y, yhat)
ok = isfinite(yhat) & yhat > 0 & y > 0;
ly = log(y(ok));  lh = log(yhat(ok));
R2 = 1 - sum((ly - lh).^2) / sum((ly - mean(ly)).^2);
end

function yhat = m2_pred(p, base, s, R)
[a, b, c, n] = m2_params(p);
yhat = a * base .* (1 + c .* s).^n ./ (1 + b .* R);
end

function [a, b, c, n] = m2_params(p)
% Bounded parameters for M2: 0 < b < 10, 0 < c < 100, |n| < 3
a = exp(p(1));
b = 10  / (1 + exp(-p(2)));
c = 100 / (1 + exp(-p(3)));
n = 3 * tanh(p(4));
end

function c = regime_coeffs(tau)
% Regime-dependent coefficients used in plot_theory_vs_model ('regime')
n = numel(tau);
c.a = 0.086 * ones(n,1); c.n = ones(n,1); c.c = 8 * ones(n,1); c.b = zeros(n,1);
up = tau > 0;  dn = tau < 0;
c.a(up) = 0.490; c.n(up) = -1; c.c(up) = 2; c.b(up) = 0;
c.a(dn) = 0.110; c.n(dn) =  1; c.c(dn) = 2; c.b(dn) = 0.2;
end

function str = param_str(Mi, p)
switch Mi.np
    case 0, str = '(fixed: paper)';
    case 1, str = sprintf('a = %.3g', exp(p(1)));
    case 2, str = sprintf('a = %.3g, p = %.2f', exp(p(1)), p(2));
    case 3, str = sprintf('a = %.3g, p_hs = %.2f, q_Ls = %.2f', exp(p(1)), p(2), p(3));
    case 4
        [a, b, c, nn] = m2_params(p);
        str = sprintf('a = %.3g, b = %.3g, c = %.3g, n = %.2f', a, b, c, nn);
end
end

function [cols, mks, szs, filled] = run_styles(sweeps)
% Same colors / markers as plot_theory_vs_model
n = numel(sweeps);
cols = zeros(n, 3); mks = cell(n, 1); szs = 85 * ones(n, 1); filled = false(n, 1);
for i = 1:n
    sw = sweeps{i};
    if strcmp(sw, 'Zs5_Ls40k')
        cols(i,:) = [0 0 0]; mks{i} = 'p'; szs(i) = 160; filled(i) = true;
    elseif startsWith(sw, 'Zs') && contains(sw, 'Ls40k')
        cols(i,:) = [0 0 0.7]; mks{i} = 'o'; szs(i) = 40 + 45 * contains(sw, 'Zs10');
    elseif startsWith(sw, 'Zs5_Ls')
        cols(i,:) = [0.7 0 0]; mks{i} = 'd'; szs(i) = 40 + 45 * contains(sw, 'Ls30k');
    elseif startsWith(sw, 'Zsh')
        cols(i,:) = [0 0.5 0.2]; mks{i} = '^'; szs(i) = 40 + 100 * contains(sw, 'm50');
    elseif startsWith(sw, 'v0')
        cols(i,:) = [0.49 0.18 0.56]; mks{i} = 's'; szs(i) = 40 + 45 * contains(sw, '0p15');
    else
        cols(i,:) = [0.4 0.4 0.4]; mks{i} = 'x';
    end
end
end
