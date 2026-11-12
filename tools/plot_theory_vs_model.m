function plot_theory_vs_model(cluster_name, coeff_mode, C_geom, include_wind)
% plot_theory_vs_model(cluster_name, coeff_mode, C_geom, include_wind)
%
% Replicates the theory vs model framework for offshore transport and EKE.
% Computes g', N, H0, Ld dynamically per simulation, using the model's linear
% equation of state (b = g(alpha T - beta S)):
%   g' = b(surface) - b(bottom) across the shelf water column (median over
%        shelf columns); the density structure here is vertical, so Zhao et
%        al.'s plume-vs-ambient contrast (g alpha_S (S_ref - S_0)) is too small
%   N  = bulk background stratification of the ambient offshore column
% Uses theoretical / empirical coefficients from offshore_fwf_EKE_sim_vs_theory.m:
%   - Regime-dependent coefficients (default: coeff_mode = 'regime'):
%       no wind (tau = 0):       a = 0.086, n =  1, c = 8, b = 0.0
%       upwelling (tau > 0):      a = 0.490, n = -1, c = 2, b = 0.0
%       downwelling (tau < 0):    a = 0.110, n =  1, c = 2, b = 0.2
%   - Geometry factor C_geom for APE:
%       C_geom = 1/2 (default for rectangular shelf / layer geometry)
%       C_geom = 1/6 (for triangular wedge geometry)
%   - include_wind: also plot the wind-stress sweep (default: false)
%
% Diagnosed quantities (following Zhao et al. Sec. 3c-d as closely as our
% closed, alongshore-integrated geometry allows):
%   Q   = maximum over the shelf (coast to x_b) of the offshore-directed,
%         alongshore-integrated transport  int max(int_y u dy, 0) dz.
%         Equals the upper branch of the overturning cell when the net
%         cross-shelf transport is zero (it is not, here: see the log)
%   EKE = volume-weighted mean EKE over the shelf cross-section (coast to x_b),
%         full depth, over the same alongshore range
%   Both use y_range_Q (whole domain outside the alongshore nudging zone).
%   x_b is detected per run as the start of the slope beyond the shelf (H > 1.1 H0).
% Theory: Q_Ek = tau L / (rho0 f) is signed (onshore for downwelling winds),
% and L = length of y_range_Q.

if nargin < 1 || isempty(cluster_name)
    cluster_name = 'cluster_Fnnie_1_19';
end
if nargin < 2 || isempty(coeff_mode)
    coeff_mode = 'regime'; % 'regime', 'baseline', or 'fit'
end
if nargin < 3 || isempty(C_geom)
    C_geom = 1/2; % Default rectangular shelf layer geometry
end
if nargin < 4 || isempty(include_wind)
    include_wind = false;
end

sweep_output_root = '/glade/work/nguyen/unc-shoals/sweep_output';
cluster_dir       = fullfile(sweep_output_root, cluster_name);

% Define Sweep Groups
group_Zs     = { 'Zs15_Ls40k', 'Zs10_Ls40k', 'Zs5_Ls40k' };
group_Ls     = { 'Zs5_Ls20k', 'Zs5_Ls30k', 'Zs5_Ls40k' };
group_ShelfH = { 'Zsh_m10', 'Zsh_m50' };
group_v0     = { 'v0_0p10', 'v0_0p15', 'Zs5_Ls40k' };
group_wind   = { 'wind_m0p15', 'wind_m0p10', 'wind_m0p08', 'wind_m0p05', 'wind_m0p03', ...
                 'wind_0p03', 'wind_0p05', 'wind_0p08', 'wind_0p10', 'wind_0p15' };

if include_wind
    all_sweeps = unique([group_Zs, group_Ls, group_ShelfH, group_v0, group_wind], 'stable');
else
    all_sweeps = unique([group_Zs, group_Ls, group_ShelfH, group_v0], 'stable');
end

% Constants
f = 2 * (2*pi/86400) * sin(deg2rad(35.2480)); % Coriolis parameter (~ 8.44e-5 s^-1)
g = 9.81;
rho0 = 1024.0;
Cd = 2.5e-3;
Ub = 0.2;
r_drag = Cd * Ub;
% =========================================================================
% 🟢 DIAGNOSTIC SETTINGS
%   y_range_Q          alongshore range (km) for Q and EKE: whole domain
%                      outside the alongshore nudging zone (check its width)
%   x_break_fallback   shelf-break position (km) used if detection fails
%   eos_alpha/beta     linear equation of state of the model
%   x_ambient_width    width of the offshore strip used as ambient water (N)
%   x_g_range          cross-shelf range of the shelf columns used for g'
% =========================================================================
y_range_Q        = [20, Inf];
x_break_fallback = 50;
% Linear equation of state used by the model:  b = g (alpha T - beta S)
% (Oceananigans LinearEquationOfState defaults; change if the runs used others)
eos_alpha        = 1.67e-4;   % thermal expansion   [1/K]
eos_beta         = 7.80e-4;   % haline contraction  [1/psu]
x_ambient_width  = 10;        % km: offshore strip (at the outer edge) taken as ambient water (for N)
x_g_range        = [5, 50];   % km: shelf columns used for g' (avoid the coastal wall and the slope)

% Data collection arrays
num_sw = numel(all_sweeps);
EKE_diag    = nan(num_sw, 1);
Q_diag      = nan(num_sw, 1);
R_val       = zeros(num_sw, 1);
s_val       = zeros(num_sw, 1);
APE_val     = zeros(num_sw, 1);
QEk_val     = zeros(num_sw, 1);
Ld_val      = zeros(num_sw, 1);
g_prime_val = zeros(num_sw, 1);
H0_val      = zeros(num_sw, 1);
tau_val     = zeros(num_sw, 1);
Zs_val      = nan(num_sw, 1);
Ls_val      = nan(num_sw, 1);
v0_val      = nan(num_sw, 1);
N_val       = nan(num_sw, 1);
L_val       = nan(num_sw, 1);
xb_val      = nan(num_sw, 1);
Qnet_val    = nan(num_sw, 1);

fprintf('=== Extracting data from %d sweeps ===\n', num_sw);

for idx = 1:num_sw
    sw = all_sweeps{idx};
    sw_core = strrep(sw, 'sweep_', '');
    d = dir(fullfile(cluster_dir, ['*' sw_core '*']));
    d = d([d.isdir]);
    d = d(~ismember({d.name}, {'.', '..'}));
    if isempty(d)
        warning('Folder not found for %s', sw);
        continue;
    end
    datadir = fullfile(cluster_dir, d(1).name);
    % Output files carry a 'sweep_' prefix (e.g. time_avg_3d_sweep_Zs5_Ls40k.nc)
    fname      = first_existing({fullfile(datadir, ['time_avg_3d_sweep_' sw_core '.nc']), ...
                                 fullfile(datadir, ['time_avg_3d_' sw_core '.nc'])});
    meta_fname = first_existing({fullfile(datadir, ['sweep_metadata_sweep_' sw_core '.nc']), ...
                                 fullfile(datadir, ['sweep_metadata_' sw_core '.nc'])});
    
    if isempty(fname)
        warning('No time_avg_3d file for %s in %s', sw, datadir);
        continue;
    end
    fprintf('  %s -> %s\n', sw, fname);
    
    % Parameters parsed from the run name (defaults = reference case), then
    % overridden by the metadata file where available
    [Zs, Ls, tau, v0, Zsh] = params_from_name(sw_core);
    if ~isempty(meta_fname)
        Zs  = read_att(meta_fname, 'Zs',           Zs);
        Ls  = read_att(meta_fname, 'shoal_length', Ls);
        tau = read_att(meta_fname, 'wind_stress',  tau);
        v0  = read_att(meta_fname, 'v0',           v0);
        Zsh = read_att(meta_fname, 'Zsh',          Zsh);
    else
        fprintf('  [Fallback Activated] Metadata file missing for %s. Using parameters parsed from run name.\n', sw);
    end
    
    try
        % 1. Compute g', N, and depth layers
        xc = try_ncread_(fname, {'xC','xc','x','x_caa'}); xc_km = xc(:) / 1e3;
        yc = try_ncread_(fname, {'yC','yc','y','y_aca'}); yc_km = yc(:) / 1e3;
        zc = try_ncread_(fname, {'zC','zc','z','z_aac'}); zc = zc(:);
        time = ncread(fname, 'time');
        dz = cell_thickness(fname, zc);   % per-level thickness (handles stretched grids)
        
        t_idx = numel(time);
        H0    = abs(Zsh);                 % shelf depth (from metadata / run name, e.g. Zsh_m10 -> 10 m)
        
        % ---- Buoyancy from the model's linear equation of state ----
        %   b = g (alpha T - beta S)   (consistent with the simulations)
        buoy = @(T, S) g * (eos_alpha * T - eos_beta * S);
        iyP  = find(yc_km >= y_range_Q(1) & yc_km <= y_range_Q(2));
        
        % Ambient (offshore) water: outermost x_ambient_width km of the domain
        ix_amb = find(xc_km >= max(xc_km) - x_ambient_width);
        
        % g': buoyancy contrast across the shelf water column (surface minus
        % bottom), median over shelf columns (x_g_range). In these runs the
        % density structure is set by vertical stratification; the horizontal
        % plume-vs-ambient contrast used by Zhao et al. is ~10x smaller and noisy.
        % Then Ld = sqrt(g' H0)/f ~ N H0 / f (first baroclinic deformation radius).
        ix_g = find(xc_km >= x_g_range(1) & xc_km <= x_g_range(2));
        Tg = ncread(fname, 'T', [ix_g(1) iyP(1) 1 t_idx], [numel(ix_g) numel(iyP) Inf 1]);
        Sg = ncread(fname, 'S', [ix_g(1) iyP(1) 1 t_idx], [numel(ix_g) numel(iyP) Inf 1]);
        [~, kup] = sort(zc, 'ascend');                       % bottom -> top along dim 3
        Tg = Tg(:, :, kup);  Sg = Sg(:, :, kup);
        wetg = (Tg ~= 0) & isfinite(Tg);
        bg = buoy(Tg, Sg);
        Nzz = size(bg, 3);
        kk  = reshape(1:Nzz, 1, 1, []);
        k_top = max(wetg .* kk, [], 3);                      % highest wet level
        k_bot = min(wetg .* kk + ~wetg * (Nzz + 1), [], 3);  % lowest wet level
        okc = (k_top > k_bot);                               % columns with >= 2 wet levels
        [ii, jj] = ndgrid(1:size(bg, 1), 1:size(bg, 2));
        b_top = bg(sub2ind(size(bg), ii(okc), jj(okc), k_top(okc)));
        b_bot = bg(sub2ind(size(bg), ii(okc), jj(okc), k_bot(okc)));
        g_prime = median(b_top - b_bot, 'omitnan');
        if ~(g_prime > 0)
            g_prime = 1e-3;
            fprintf('  [Fallback Activated] Non-positive vertical buoyancy contrast for %s. Using g'' = 1e-3 m/s^2.\n', sw);
        end
        
        % N: background (ambient) stratification, bulk over the offshore column
        % (Zhao et al. use the prescribed background N)
        Tc = ncread(fname, 'T', [ix_amb(1) iyP(1) 1 t_idx], [numel(ix_amb) numel(iyP) Inf 1]);
        Sc = ncread(fname, 'S', [ix_amb(1) iyP(1) 1 t_idx], [numel(ix_amb) numel(iyP) Inf 1]);
        bc = buoy(Tc, Sc);  bc(Tc == 0 | ~isfinite(Tc)) = NaN;
        b_prof = squeeze(median(median(bc, 1, 'omitnan'), 2, 'omitnan'));    % (Nz x 1)
        okz = isfinite(b_prof);
        if sum(okz) >= 2
            [z_hi, i_hi] = max(zc(okz));  [z_lo, i_lo] = min(zc(okz));
            bz = b_prof(okz);
            N2 = (bz(i_hi) - bz(i_lo)) / (z_hi - z_lo);
        else
            N2 = NaN;
        end
        if ~(N2 > 0)
            N2 = g_prime / H0;
            fprintf('  [Fallback Activated] Background N not resolved for %s. Using N^2 = g''/H0.\n', sw);
        end
        N = sqrt(N2);
        
        % Continuous Stratification metrics
        Ld = sqrt(g_prime * H0) / f; 
        
        % Bottom slope alpha
        if Ls > 0
            alpha = (H0 + Zs) / Ls;
        else
            alpha = 0;
        end
        
        % Nondimensional numbers
        R = r_drag / (f * H0);
        s = alpha * N / f;
        APE = C_geom * g_prime * H0^2 * Ld;
        
        % ---- Diagnostic section: alongshore range and shelf break ----
        dx  = cell_width(fname, xc(:), {'xF','x_faa','xf'});
        dy  = cell_width(fname, yc(:), {'yF','y_afa','yf'});
        iyQ = find(yc_km >= y_range_Q(1) & yc_km <= y_range_Q(2));
        L_Q = sum(dy(iyQ));                                   % alongshore length (m)
        
        % Local water depth from wet cells, median over (every few) y rows
        ystep  = max(1, floor(numel(yc) / 40));
        Tsub   = ncread(fname, 'T', [1 1 1 t_idx], [Inf Inf Inf 1], [1 ystep 1 1]);
        wetsub = (Tsub ~= 0) & isfinite(Tsub);
        depth_x = median(squeeze(sum(wetsub .* reshape(dz, 1, 1, []), 3)), 2);
        ix_b = shelf_break_index(depth_x, H0);
        if isempty(ix_b)
            [~, ix_b] = min(abs(xc_km - x_break_fallback));
            fprintf('  [Fallback Activated] Shelf break not detected for %s. Using x = %.0f km.\n', sw, x_break_fallback);
        end
        x_b = xc_km(ix_b);
        
        % Ekman transport (signed: onshore for downwelling-favorable winds)
        QEk = tau * L_Q / (rho0 * f);
        
        % Store metrics
        R_val(idx)       = R;
        s_val(idx)       = s;
        APE_val(idx)     = APE;
        Ld_val(idx)      = Ld;
        g_prime_val(idx) = g_prime;
        H0_val(idx)      = H0;
        tau_val(idx)     = tau;
        QEk_val(idx)     = QEk;
        Zs_val(idx)      = Zs;
        Ls_val(idx)      = Ls;
        v0_val(idx)      = v0;
        N_val(idx)       = N;
        L_val(idx)       = L_Q;
        xb_val(idx)      = x_b;
        
        % ---- 2. Diagnosed EKE: volume mean over the shelf cross-section ----
        try
            rd = @(v) ncread(fname, v, [1 iyQ(1) 1 t_idx], [ix_b numel(iyQ) Inf 1]);
            Tb  = rd('T');
            wet = (Tb ~= 0) & isfinite(Tb);
            eke = 0.5 * ((rd('uu') - rd('u_c').^2) + (rd('vv') - rd('v_c').^2) + (rd('ww') - rd('w_c').^2));
            Wv  = dx(1:ix_b) .* reshape(dy(iyQ), 1, []) .* reshape(dz, 1, 1, []) .* wet;   % cell volumes
            eke(~wet | ~isfinite(eke)) = 0;
            EKE_diag(idx) = sum(eke .* Wv, 'all') / sum(Wv, 'all');
        catch ME
            fprintf('  [Fallback Activated] EKE calculation failed for %s (%s).\n', sw, ME.message);
            EKE_diag(idx) = NaN;
        end
        
        % ---- 3. Diagnosed Q: offshore transport across the shelf ----
        % Alongshore-integrated cross-shelf velocity at every x on the shelf,
        %   U(x,z) = int_y u dy                       [m^2/s]
        % offshore-directed transport at each x (full depth),
        %   Q(x) = int max(U(x,z), 0) dz              [m^3/s]
        % and Q = max over the shelf (coast to x_b).
        % Integrating alongshore first cancels eddy/recirculation cells in y;
        % keeping only U > 0 makes Q independent of any net cross-shelf flow
        % (which is not zero here: mass can pass through the excluded
        % nudging strip). When the net is zero, Q equals the upper branch of
        % the overturning cell.
        try
            ush = ncread(fname, 'u_c', [1 iyQ(1) 1 t_idx], [ix_b numel(iyQ) Inf 1]);   % (Nx, Ny, Nz)
            ush(~isfinite(ush)) = 0;
            U_y = squeeze(sum(ush .* reshape(dy(iyQ), 1, []), 2));                     % (Nx, Nz)
            Qx  = sum(max(U_y, 0) .* dz(:)', 2);                                      % (Nx, 1)
            [Qmax, ixm] = max(Qx);
            Q_diag(idx)   = Qmax;
            Qnet_val(idx) = sum(U_y(ixm, :) .* dz(:)');       % full-depth net at that x
            fprintf('    max offshore transport at x = %.1f km\n', xc_km(ixm));
        catch ME
            fprintf('  [Fallback Activated] Q_diag calculation failed for %s (%s).\n', sw, ME.message);
            Q_diag(idx) = NaN;
        end
        
        Ld_dx = Ld / median(dx);
        fprintf('  %s: g'' = %.3e m/s^2, N = %.2e 1/s, H0 = %.1f m, Ld = %.1f km (Ld/dx = %.1f), x_b = %.1f km, EKE = %.2e m^2/s^2, Q = %.2e m^3/s (net %.1e)\n', ...
            sw, g_prime, N, H0, Ld/1000, Ld_dx, x_b, EKE_diag(idx), Q_diag(idx), Qnet_val(idx));
        if Ld_dx < 5
            fprintf('    [Resolution] Ld/dx = %.1f < 5: eddies from baroclinic instability are only marginally resolved.\n', Ld_dx);
        end
            
    catch ME
        warning('Error processing %s: %s', sw, ME.message);
    end
end

% Filter out invalid points
valid = ~isnan(EKE_diag) & ~isnan(Q_diag) & (APE_val > 0);
EKE_diag     = EKE_diag(valid);
Q_diag       = Q_diag(valid);
R_val        = R_val(valid);
s_val        = s_val(valid);
APE_val      = APE_val(valid);
QEk_val      = QEk_val(valid);
Ld_val       = Ld_val(valid);
g_prime_val  = g_prime_val(valid);
H0_val       = H0_val(valid);
tau_val      = tau_val(valid);
Zs_val       = Zs_val(valid);
Ls_val       = Ls_val(valid);
v0_val       = v0_val(valid);
N_val        = N_val(valid);
L_val        = L_val(valid);
xb_val       = xb_val(valid);
sweeps_valid = all_sweeps(valid);
num_valid    = numel(sweeps_valid);

%% Compute Theoretical EKE based on offshore_fwf_EKE_sim_vs_theory.m
% EKE_integrated (EKE_Delta): cross-section integrated EKE (m^4/s^2)
% EKE_spec_theory: specific EKE (m^2/s^2) = EKE_Delta / (H0 * Ld), since
%   EKE_Delta ~ EKE * H0 * Ld  (Zhao et al., Sec. 3d)
EKE_integrated  = zeros(num_valid, 1);
EKE_spec_theory = zeros(num_valid, 1);

for i = 1:num_valid
    t_i = tau_val(i);
    if strcmpi(coeff_mode, 'baseline')
        % Baseline no-wind parameters everywhere
        a_i = 0.086; n_i = 1; c_i = 8; b_i = 0.0;
    elseif strcmpi(coeff_mode, 'fit')
        % Default fallback if fitting requested
        a_i = 0.086; n_i = 1; c_i = 8; b_i = 0.0;
    else
        % Regime-dependent coefficients from offshore_fwf_EKE_sim_vs_theory.m
        if t_i == 0
            a_i = 0.086; n_i =  1; c_i = 8; b_i = 0.0; % no wind
        elseif t_i > 0
            a_i = 0.490; n_i = -1; c_i = 2; b_i = 0.0; % upwelling (+tau)
        else
            a_i = 0.110; n_i =  1; c_i = 2; b_i = 0.2; % downwelling (-tau)
        end
    end
    
    EKE_integrated(i)  = APE_val(i) * a_i * (1 + c_i * s_val(i))^n_i / (1 + b_i * R_val(i));
    EKE_spec_theory(i) = EKE_integrated(i) / (H0_val(i) * Ld_val(i));
end

if strcmpi(coeff_mode, 'fit')
    % Attempt empirical fit if explicitly requested
    % Fit a, b, c, n to the diagnosed SPECIFIC EKE (m^2/s^2), in log space
    try
        base = APE_val ./ (H0_val .* Ld_val);                      % APE per unit area (m^2/s^2)
        pred = @(p) exp(p(1)) .* (1 + exp(p(3)) .* s_val).^p(4) .* base ./ (1 + exp(p(2)) .* R_val);
        cost = @(p) sum((log(pred(p)) - log(EKE_diag)).^2);
        p = fminsearch(cost, [log(0.1), log(0.1), log(2), 1], optimset('Display','off','MaxIter',5e3,'MaxFunEvals',1e4));
        EKE_spec_theory = pred(p);
        EKE_integrated  = EKE_spec_theory .* H0_val .* Ld_val;
        a_f = exp(p(1)); b_f = exp(p(2)); c_f = exp(p(3)); n_f = p(4);
        fprintf('Fitted empirical coefficients: a=%.3g, b=%.3g, c=%.3g, n=%.3g\n', a_f, b_f, c_f, n_f);
        % Limits: b*R >> 1  ->  1/(1+bR) ~ 1/(bR) = f H0 / (b r)
        %         c*s << 1 with |n| large  ->  (1+cs)^n ~ exp(n c s)
        if min(b_f .* R_val) > 100 && max(c_f .* s_val) < 0.01
            A_eff = a_f / b_f * f / r_drag;            % EKE ~ A_eff * (APE/(H0 Ld)) * H0 * exp(k s)
            k_eff = n_f * c_f;
            fprintf(['  -> fit reached its limits; effective law:\n' ...
                     '     EKE = %.3g * C_geom g'' H0^2 * exp(%.3g s)   [i.e. a/(bR) * exp(nc s)]\n'], A_eff, k_eff);
        end
    catch ME
        fprintf('  [Fallback Activated] Empirical fit failed (%s). Falling back to regime-dependent coefficients.\n', ME.message);
    end
end

%% Compute Theoretical Offshore Volume Transport (Q_tot)
% L = alongshore length of the diagnostic section (y_range_Q), per run.
% Ekman transport is signed (Zhao et al. Eq. 5b): onshore for downwelling winds.
QEk_val = (tau_val .* L_val) ./ (rho0 * f);

% Q_eddy = C_eddy * f * L * Ld * EKE / g'   (Zhao et al. Eq. 5a)
Q_eddy_basis = (f * L_val .* Ld_val .* EKE_spec_theory) ./ g_prime_val;

% Fit C_eddy (or default to 1.1 from offshore_fwf_EKE_sim_vs_theory.m)
Q_diff = Q_diag - QEk_val;
C_eddy = Q_eddy_basis \ Q_diff;
if isnan(C_eddy) || C_eddy <= 0
    C_eddy = 1.1;
    fprintf('  [Fallback Activated] Fitted C_eddy invalid or <= 0. Using default C_eddy = 1.1.\n');
end

fprintf('Fitted eddy transport coefficient C_eddy: %.3f\n', C_eddy);

Q_eddy_theory = C_eddy .* Q_eddy_basis;
Q_tot_theory  = QEk_val + Q_eddy_theory;

% Save the per-run inputs so check_eke_theory.m can test / refit the scaling
outdir = '/glade/work/nguyen/unc-shoals/animations';
if ~exist(outdir, 'dir'), outdir = pwd; end
theory_inputs = struct('sweeps', {sweeps_valid}, 'EKE_diag', EKE_diag, 'Q_diag', Q_diag, ...
    'R', R_val, 's', s_val, 'APE', APE_val, 'C_geom', C_geom, 'Ld', Ld_val, 'g_prime', g_prime_val, ...
    'H0', H0_val, 'N', N_val, 'tau', tau_val, 'Zs', Zs_val, 'Ls', Ls_val, 'v0', v0_val, ...
    'f', f, 'L', L_val, 'x_break', xb_val, 'cluster', cluster_name);
save(fullfile(outdir, sprintf('%s_theory_inputs.mat', cluster_name)), 'theory_inputs');
fprintf('Saved theory inputs: %s\n', fullfile(outdir, sprintf('%s_theory_inputs.mat', cluster_name)));

% Convert Q to mSv (1 mSv = 10^3 m^3/s) for plotting
Q_diag_mSv       = Q_diag / 1e3;
Q_tot_theory_mSv = Q_tot_theory / 1e3;

% Specific EKE (m/s) for Panel (a)
sqrt_EKE_theory = sqrt(EKE_spec_theory);
sqrt_EKE_diag   = sqrt(EKE_diag);

%% Plotting (Theory vs Model)
% Same width / compact spacing as the 4-column figures (1350 * 4/6 = 900 px),
% same fonts as the other figures.
fig = figure('Color','w', 'Position', [100, 100, 1350 * 4/6, 480]);
tl = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Group colors / markers (blue / red / green as in the other sweep plots)
col_Zs   = [0.0, 0.0, 0.7];     mk_Zs   = 'o';
col_Ls   = [0.7, 0.0, 0.0];     mk_Ls   = 'd';
col_Zsh  = [0.0, 0.5, 0.2];     mk_Zsh  = '^';
col_v0   = [0.49, 0.18, 0.56];  mk_v0   = 's';
col_wind = [0.4, 0.4, 0.4];     mk_wind = 'x';
sz_small = 40; sz_mid = 85; sz_large = 140;

% Panel (a): sqrt(EKE)
ax1 = nexttile(tl); hold(ax1, 'on');
% Panel (b): Q_tot
ax2 = nexttile(tl); hold(ax2, 'on');

for i = 1:num_valid
    sw_i = sweeps_valid{i};
    lw = 1.5;
    if strcmp(sw_i, 'Zs5_Ls40k')
        c = [0, 0, 0]; m = 'p'; sz = 160; lw = 2;   % Pentagram for reference
    elseif ismember(sw_i, group_Zs)
        c = col_Zs; m = mk_Zs;                      % marker size grows with Z_s (shallower crest)
        if contains(sw_i, 'Zs15'), sz = sz_small;
        elseif contains(sw_i, 'Zs10'), sz = sz_mid;
        else, sz = sz_large; end
    elseif ismember(sw_i, group_Ls)
        c = col_Ls; m = mk_Ls;
        if contains(sw_i, 'Ls20k'), sz = sz_small;
        elseif contains(sw_i, 'Ls30k'), sz = sz_mid;
        else, sz = sz_large; end
    elseif ismember(sw_i, group_ShelfH)
        c = col_Zsh; m = mk_Zsh;
        if contains(sw_i, 'Zsh_m10'), sz = sz_small;
        elseif contains(sw_i, 'Zsh_m50'), sz = sz_large;
        else, sz = sz_mid; end
    elseif ismember(sw_i, group_v0)
        c = col_v0; m = mk_v0;
        if contains(sw_i, 'v0_0p10'), sz = sz_small;
        elseif contains(sw_i, 'v0_0p15'), sz = sz_mid;
        else, sz = sz_large; end
    else
        c = col_wind; m = mk_wind; sz = 80;
    end
    
    if strcmp(sw_i, 'Zs5_Ls40k')
        scatter(ax1, sqrt_EKE_theory(i), sqrt_EKE_diag(i), sz, m, 'MarkerEdgeColor', c, 'MarkerFaceColor', c, 'LineWidth', lw);
        scatter(ax2, Q_tot_theory_mSv(i), Q_diag_mSv(i), sz, m, 'MarkerEdgeColor', c, 'MarkerFaceColor', c, 'LineWidth', lw);
    else
        scatter(ax1, sqrt_EKE_theory(i), sqrt_EKE_diag(i), sz, m, 'MarkerEdgeColor', c, 'LineWidth', lw);
        scatter(ax2, Q_tot_theory_mSv(i), Q_diag_mSv(i), sz, m, 'MarkerEdgeColor', c, 'LineWidth', lw);
    end
end

% Panel (a) 1:1 line & styling
max_val1 = max([sqrt_EKE_theory; sqrt_EKE_diag], [], 'omitnan') * 1.15;
plot(ax1, [0, max_val1], [0, max_val1], 'k--', 'LineWidth', 1.2);
xlabel(ax1, 'Theory $\sqrt{EKE}$ (m s$^{-1}$)', 'Interpreter', 'latex', 'FontSize', 18);
ylabel(ax1, 'Simulation $\sqrt{EKE}$ (m s$^{-1}$)', 'Interpreter', 'latex', 'FontSize', 18);
title(ax1, '(a) Eddy Kinetic Energy', 'FontSize', 18, 'FontWeight', 'bold');
xlim(ax1, [0, max_val1]); ylim(ax1, [0, max_val1]);
pbaspect(ax1, [1 1 1]);
set(ax1, 'FontSize', 14, 'Box', 'on', 'XGrid', 'on', 'YGrid', 'on', ...
    'LineWidth', 1, 'TickDir', 'out', 'XColor', 'k', 'YColor', 'k');

% Panel (b) 1:1 line & styling
min_val2 = min([Q_tot_theory_mSv; Q_diag_mSv], [], 'omitnan');
max_val2 = max([Q_tot_theory_mSv; Q_diag_mSv], [], 'omitnan');
min_val2 = min(0, min_val2 * 1.15);
max_val2 = max_val2 * 1.15;
plot(ax2, [min_val2, max_val2], [min_val2, max_val2], 'k--', 'LineWidth', 1.2);
xlabel(ax2, 'Theory $Q$ (mSv)', 'Interpreter', 'latex', 'FontSize', 18);
ylabel(ax2, 'Simulation $Q$ (mSv)', 'Interpreter', 'latex', 'FontSize', 18);
title(ax2, '(b) Offshore Volume Transport', 'FontSize', 18, 'FontWeight', 'bold');
xlim(ax2, [min_val2, max_val2]); ylim(ax2, [min_val2, max_val2]);
pbaspect(ax2, [1 1 1]);
set(ax2, 'FontSize', 14, 'Box', 'on', 'XGrid', 'on', 'YGrid', 'on', ...
    'LineWidth', 1, 'TickDir', 'out', 'XColor', 'k', 'YColor', 'k');

% Legend (same parameter symbols as the other figures: Z_s, L_s, Z_sh, v_0)
h_ref   = plot(ax2, NaN, NaN, 'p', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'k', 'LineWidth', 2, 'MarkerSize', 10);
h_zs    = plot(ax2, NaN, NaN, mk_Zs,  'MarkerEdgeColor', col_Zs,  'LineWidth', 1.5, 'MarkerSize', 8);
h_ls    = plot(ax2, NaN, NaN, mk_Ls,  'MarkerEdgeColor', col_Ls,  'LineWidth', 1.5, 'MarkerSize', 8);
h_shelf = plot(ax2, NaN, NaN, mk_Zsh, 'MarkerEdgeColor', col_Zsh, 'LineWidth', 1.5, 'MarkerSize', 8);
h_v0    = plot(ax2, NaN, NaN, mk_v0,  'MarkerEdgeColor', col_v0,  'LineWidth', 1.5, 'MarkerSize', 8);
leg_h = [h_ref, h_zs, h_ls, h_shelf, h_v0];
leg_s = {'Reference', '$Z_s$', '$L_s$', '$Z_{sh}$', '$v_0$'};
if include_wind
    leg_h(end+1) = plot(ax2, NaN, NaN, mk_wind, 'MarkerEdgeColor', col_wind, 'LineWidth', 1.5, 'MarkerSize', 8);
    leg_s{end+1} = '$\tau$';
end
legend(ax2, leg_h, leg_s, 'Location', 'northwest', 'FontSize', 12, 'Interpreter', 'latex');

drawnow;

% Save Output Figure
outfile = fullfile(outdir, sprintf('%s_theory_vs_model.png', cluster_name));
exportgraphics(fig, outfile, 'Resolution', 300);
fprintf('Saved theory vs model plot: %s\n', outfile);

end

function ib = shelf_break_index(depth_x, H0)
% Index of the shelf break: first x, beyond where the shelf depth H0 is
% reached, at which the (alongshore-median) depth exceeds 1.1 H0
ib = [];
i0 = find(depth_x >= 0.9 * H0, 1, 'first');
if isempty(i0), return; end
k  = find(depth_x(i0:end) > 1.1 * H0, 1, 'first');
if ~isempty(k), ib = i0 + k - 1; end
end

function d = cell_width(fname, c, face_names)
% Cell widths from face coordinates if present, else from center midpoints
for nm = face_names
    try
        fc = ncread(fname, nm{1});
        if numel(fc) == numel(c) + 1
            d = abs(diff(fc(:)));
            return;
        end
    catch
    end
end
if numel(c) == 1, d = 1; return; end
mid   = (c(1:end-1) + c(2:end)) / 2;
edges = [c(1) - (mid(1) - c(1)); mid; c(end) + (c(end) - mid(end))];
d     = abs(diff(edges));
end

function f = first_existing(candidates)
% Return the first file in the list that exists, or '' if none do
f = '';
for c = 1:numel(candidates)
    if isfile(candidates{c}), f = candidates{c}; return; end
end
end

function val = read_att(fname, att, default)
% Read a global attribute, keeping the default if it is missing
try
    val = double(ncreadatt(fname, '/', att));
catch
    val = default;
end
end

function [Zs, Ls, tau, v0, Zsh] = params_from_name(sw_core)
% Parse run parameters from the run label; defaults = reference case
Zs = -5; Ls = 40e3; tau = 0; v0 = 0.2; Zsh = -25;
tok = regexp(sw_core, 'Zs(\d+)_', 'tokens', 'once');      if ~isempty(tok), Zs  = -str2double(tok{1}); end
tok = regexp(sw_core, 'Ls(\d+)k', 'tokens', 'once');      if ~isempty(tok), Ls  = str2double(tok{1}) * 1e3; end
tok = regexp(sw_core, 'Zsh_m(\d+)', 'tokens', 'once');    if ~isempty(tok), Zsh = -str2double(tok{1}); end
tok = regexp(sw_core, 'v0_(\d+)p(\d+)', 'tokens', 'once');
if ~isempty(tok), v0 = str2double([tok{1} '.' tok{2}]); end
tok = regexp(sw_core, 'wind_(m?)(\d+)p(\d+)', 'tokens', 'once');
if ~isempty(tok)
    tau = str2double([tok{2} '.' tok{3}]);
    if ~isempty(tok{1}), tau = -tau; end
end
end

function dz = cell_thickness(fname, zc)
% Cell thicknesses from face coordinates if present, else from center midpoints
for nm = {'zF', 'z_aaf', 'zf'}
    try
        zf = ncread(fname, nm{1});
        if numel(zf) == numel(zc) + 1
            dz = abs(diff(zf(:)));
            return;
        end
    catch
    end
end
mid   = (zc(1:end-1) + zc(2:end)) / 2;
edges = [zc(1) - (mid(1) - zc(1)); mid; zc(end) + (zc(end) - mid(end))];
dz    = abs(diff(edges));
end

function data = try_ncread_(fname, names)
for i = 1:numel(names)
    try
        data = ncread(fname, names{i});
        return;
    catch
    end
end
error('Coordinate variable not found.');
end


