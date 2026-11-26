function plot_theory_vs_model(cluster_name, coeff_mode, C_geom, include_wind, refresh_ape)
% plot_theory_vs_model(cluster_name, coeff_mode, C_geom, include_wind, refresh_ape)
%
% Theory against model for shelf EKE and offshore transport Q (Zhao et al.
% Eqs. 5, 7, 8). Two figures, because the chain APE -> EKE -> Q can be
% tested two different ways and they answer different questions:
%
%   <cluster>_theory_vs_model_stepwise.png   (three panels)
%     Each step fed the MEASURED input it needs, so a miss is attributable:
%       (a) Eq. 7:  EKE = a(1+cs)^n (APE/A_box) / (1+bR), with APE the sorted
%           (Lorenz) APE from ape_wedge.m and A_box the shelf box's wet
%           cross-section, i.e. APE per unit volume -- the same average as the
%           diagnosed shelf-mean EKE. Tests the conversion only.
%       (b) Eq. 5 eddy term, c f L Ld_h EKE / g'_h, with the MEASURED EKE and
%           the horizontal g'_h (rms of b - b_init(z) over the shelf
%           columns), Ld_h = sqrt(g'_h H0)/f, against the MEASURED eddy
%           transport: the transient eddy buoyancy flux <u'b'>, integrated
%           alongshore and over depth at the shelf x where it peaks, / g'_h,
%           times sign(tilt): positive = DOWN-gradient exchange of shelf water
%           (Zhao's plume is light inshore; our unforced and upwelling shelves
%           are dense inshore, so the same exchange moves light water onshore).
%           c is fitted here. Tests the transport step only.
%       (c) total Q = Q_Ek + that eddy term, against the diagnosed Q. Q is
%           built from the time-mean u, so it holds no eddy flux: (b) is the
%           real test of Eq. 5, (c) shows whether eddies matter for Q at all.
%           Runs whose offshore water has drifted from the initial profile by
%           more than g'_h (downwelling) are circled in (b) and (c) and left
%           out of the fit for c: g'_h does not mean shelf-against-ocean there.
%   <cluster>_flux_vs_x.png
%     Where the exchange happens: eddy, standing-meander and mean-overturning
%     transport against x for every run, with the shoal crest, its flanks and
%     the shelf break marked.
%
%     Also printed: the STANDING-eddy flux (u - [u])(b - [b]) of the time-
%     mean fields -- steady meanders around the shoal -- which carries
%     buoyancy without appearing in <u'b'>.
%
%   <cluster>_theory_vs_model_predictive.png
%     Only what is known before the run: g'_0 = surface-minus-bottom contrast
%     of the INITIAL profile over the shelf depth, Ld_0 = sqrt(g'_0 H0)/f,
%     APE = C_geom g'_0 H0^2 Ld_0 (Eq. 8), divided by the same A_box (fixed by
%     the bathymetry), then Eqs. 7 and 5 with the constants calibrated in the
%     stepwise figure. Nothing is refitted, so the EKE panel differs from the
%     stepwise one only through the Eq. 8 error.
%     Expected to hold for unforced runs and to miss under wind, because the
%     background stratification does not set the APE once wind acts.
%
% The old g' (surface-minus-bottom on the EVOLVING shelf columns) is still
% computed and printed for comparison, and saved to theory_inputs for
% check_eke_theory.m, but no figure uses it: it is diagnosed, so it cannot
% predict, and it collapses to zero when wind mixes the column while the APE
% grows. See the vault note "Why switch to a horizontal g' (g'_h)".
%
% Arguments:
%   coeff_mode    Eq. 7 coefficients (Zhao et al. / Brink 2024 eq. 4.1a)
%                   'regime' (default)  by sign of tau:
%                       no wind (tau = 0):      a = 0.086, n =  1, c = 8, b = 0.0
%                       upwelling (tau > 0):    a = 0.490, n = -1, c = 2, b = 0.0
%                       downwelling (tau < 0):  a = 0.110, n =  1, c = 2, b = 0.2
%                   'baseline'  the no-wind set everywhere
%                   'fit'       one set fitted to the diagnosed EKE (4 parameters:
%                               do not trust it on a handful of runs)
%   C_geom        Eq. 8 shape factor, predictive figure only. 1/2 (default,
%                 anomaly fills H0 x Ld) or 1/6 (triangular wedge). The script
%                 prints the value the unforced runs would need.
%   include_wind  also plot the wind-stress sweep (default: false)
%   refresh_ape   rerun ape_wedge even if <cluster>_ape_wedge.mat is current
%                 (default: false). It is rerun anyway when the file is
%                 missing, older than these fields, made with a different box,
%                 or lacks a run now on disk.
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
if nargin < 5 || isempty(refresh_ape)
    refresh_ape = false;
end

sweep_output_root = '/glade/work/nguyen/unc-shoals/sweep_output';
cluster_dir       = fullfile(sweep_output_root, cluster_name);
outdir            = '/glade/work/nguyen/unc-shoals/animations';
if ~exist(outdir, 'dir'), outdir = pwd; end

% Sweep groups. These set MARKER STYLE only, for the Annie-Fnnie clusters
% whose labels they hold. Which runs are processed comes from the disk
% (discover_sweeps), so a cluster with other labels -- Zs05_edge,
% wind_m0p020 -- is no longer silently empty; such runs are styled by
% their label prefix instead (see the plotting loop).
group_Zs     = { 'Zs15_Ls40k', 'Zs10_Ls40k', 'Zs5_Ls40k' };
group_Ls     = { 'Zs5_Ls20k', 'Zs5_Ls30k', 'Zs5_Ls40k' };
group_ShelfH = { 'Zsh_m10', 'Zsh_m50' };
group_v0     = { 'v0_0p10', 'v0_0p15', 'Zs5_Ls40k' };

% Every run on disk; scatter order carries no meaning, so discovery is safe.
[all_sweeps, ~] = discover_sweeps(cluster_dir);
is_wind = startsWith(all_sweeps, 'wind_');
if ~include_wind && any(is_wind)
    if all(is_wind)
        error('plot_theory_vs_model:onlyWind', ...
              ['Every run in %s is a wind run, and include_wind is false.\n' ...
               'Call plot_theory_vs_model(''%s'', mode, C_geom, true).'], cluster_name, cluster_name);
    end
    fprintf('  include_wind = false: leaving out %d wind run(s).\n', sum(is_wind));
    all_sweeps = all_sweeps(~is_wind);
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
[Ft_val, Fs_val, xFt_val, xFs_val] = deal(nan(num_sw, 1));   % eddy buoyancy fluxes (see section 4)
[XP, FTX, FSX, QXP] = deal(cell(num_sw, 1));                 % their cross-shelf profiles
[sst_val, siw_val] = deal(zeros(num_sw, 1));                 % shoal_start, shoal_inner_width (m)

fprintf('=== Extracting data from %d sweeps ===\n', num_sw);

for idx = 1:num_sw
    sw = all_sweeps{idx};
    sw_core = strrep(sw, 'sweep_', '');
    % Exact match on the label: a '*label*' wildcard picks whichever of
    % two runs sorts first when one label is a substring of another.
    d = dir(fullfile(cluster_dir, 'sweep_*'));
    d = d([d.isdir]);
    d = d(~cellfun(@isempty, regexp({d.name}, ['^sweep_(\d+_)?' regexptranslate('escape', sw_core) '$'], 'once')));
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
        sst_val(idx) = read_att(meta_fname, 'shoal_start', 0);
        siw_val(idx) = read_att(meta_fname, 'shoal_inner_width', 0);
        tau = read_att(meta_fname, 'wind_stress',  tau);
        % Pulsed / reversing wind: 'wind_stress' is the PEAK. The Ekman
        % transport and the regime both follow the TIME-MEAN stress.
        tau = read_att(meta_fname, 'wind_stress_mean', tau);
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
        
        % Old g': buoyancy contrast across the EVOLVING shelf water column
        % (surface minus bottom), median over shelf columns (x_g_range).
        % Kept for the log and for check_eke_theory.m only; the figures use
        % g'_h and g'_0 from ape_wedge.m (see the header).
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
            QXP{idx} = Qx;
            [Qmax, ixm] = max(Qx);
            Q_diag(idx)   = Qmax;
            Qnet_val(idx) = sum(U_y(ixm, :) .* dz(:)');       % full-depth net at that x
            fprintf('    max offshore transport at x = %.1f km\n', xc_km(ixm));
        catch ME
            fprintf('  [Fallback Activated] Q_diag calculation failed for %s (%s).\n', sw, ME.message);
            Q_diag(idx) = NaN;
        end

        % ---- 4. Eddy buoyancy flux across the shelf ----
        % Q above is built from the time-mean u, so it holds the Eulerian-
        % mean overturning and NO eddy flux. Eq. 5 predicts an eddy flux, so
        % it is measured here directly, alongshore- and depth-integrated at
        % every shelf x:
        %   transient  <u'b'> = g [alpha (<uT> - <u><T>) - beta (<uS> - <u><S>)]
        %              (time-varying eddies; the model averages every step)
        %   standing   (u - [u]) (b - [b]), [.] = alongshore mean of the
        %              time-mean fields: steady meanders around the shoal,
        %              which carry buoyancy without appearing in <u'b'>
        % Both are independent of the buoyancy reference. Each is reported at
        % the x where its magnitude peaks, signed: > 0 means light water
        % offshore (or dense onshore), the sense of Zhao et al.'s offshore
        % freshwater flux. Divided by g'_h below, they become volume
        % transports [m^3/s], which is what Eq. 5 predicts.
        try
            rdq = @(v) ncread(fname, v, [1 iyQ(1) 1 t_idx], [ix_b numel(iyQ) Inf 1]);
            um = rdq('u_c');  Tm = rdq('T');  Sm = rdq('S');
            wq = (Tm ~= 0) & isfinite(Tm);
            % covariances in T and S first, then combined: keeps the
            % cancellation in <uT> - <u><T> at T, S magnitudes, not b's
            cuT = rdq('uT') - um .* Tm;
            cuS = rdq('uS') - um .* Sm;
            ubp = g * (eos_alpha * cuT - eos_beta * cuS);
            ubp(~wq | ~isfinite(ubp)) = 0;
            clear cuT cuS;
            wdy = reshape(dy(iyQ), 1, []) .* wq;                       % wet widths
            wy  = max(sum(wdy, 2), eps);
            um(~wq) = 0;  bm = buoy(Tm, Sm);  bm(~wq) = 0;
            ust = um - sum(um .* wdy, 2) ./ wy;
            bst = bm - sum(bm .* wdy, 2) ./ wy;
            sfl = ust .* bst;  sfl(~wq | ~isfinite(sfl)) = 0;
            dzr = reshape(dz, 1, 1, []);
            Ftx = squeeze(sum(sum(ubp .* reshape(dy(iyQ), 1, []), 2) .* dzr, 3));   % (Nx,1) [m^4/s^3]
            Fsx = squeeze(sum(sum(sfl .* reshape(dy(iyQ), 1, []), 2) .* dzr, 3));
            [~, it_] = max(abs(Ftx));  [~, is_] = max(abs(Fsx));
            Ft_val(idx) = Ftx(it_);  xFt_val(idx) = xc_km(it_);
            Fs_val(idx) = Fsx(is_);  xFs_val(idx) = xc_km(is_);
            XP{idx} = xc_km(1:ix_b);  FTX{idx} = Ftx;  FSX{idx} = Fsx;
            clear um Tm Sm wq ubp wdy ust bst sfl bm;
        catch ME
            fprintf('  [Fallback Activated] Eddy flux calculation failed for %s (%s).\n', sw, ME.message);
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

%% Measured APE, g'_h and the initial-profile g'_0, from ape_wedge.m
% Same shelf box (coast to the break, y_range_Q), same shelf columns for g'
% (x_g_range), same averaging window (the last), so the numbers line up with
% the EKE and Q diagnosed above.
A = load_ape_(cluster_name, sweep_output_root, outdir, y_range_Q, x_g_range, all_sweeps, refresh_ape);
lab_ape = strrep(A.labels, 'sweep_', '');
[APEm_val, Abox_val, gph_val, drift_val, gp0_val, tilt_val] = deal(nan(num_sw, 1));
for idx = 1:num_sw
    j = find(strcmp(lab_ape, strrep(all_sweeps{idx}, 'sweep_', '')), 1);
    if isempty(j), continue; end
    APEm_val(idx)  = A.APE_sorted(j);
    Abox_val(idx)  = A.A_box(j);
    gph_val(idx)   = A.g_prime_h(j);
    drift_val(idx) = A.ref_drift(j);
    gp0_val(idx)   = A.g_prime_0(j);
    tilt_val(idx)  = A.tilt(j);
end

% Filter out runs without a diagnosed EKE and Q
valid = isfinite(EKE_diag) & isfinite(Q_diag);
EKE_diag     = EKE_diag(valid);
Q_diag       = Q_diag(valid);
R_val        = R_val(valid);
s_val        = s_val(valid);
APE_val      = APE_val(valid);
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
APEm_val     = APEm_val(valid);
Abox_val     = Abox_val(valid);
gph_val      = gph_val(valid);
drift_val    = drift_val(valid);
gp0_val      = gp0_val(valid);
tilt_val     = tilt_val(valid);
XP = XP(valid);  FTX = FTX(valid);  FSX = FSX(valid);  QXP = QXP(valid);
sst_val      = sst_val(valid);
siw_val      = siw_val(valid);
Ft_val       = Ft_val(valid);
Fs_val       = Fs_val(valid);
xFt_val      = xFt_val(valid);
xFs_val      = xFs_val(valid);
sweeps_valid = all_sweeps(valid);
num_valid    = numel(sweeps_valid);

% Ekman transport, signed (Zhao et al. Eq. 5b): onshore for downwelling winds
QEk_val = (tau_val .* L_val) ./ (rho0 * f);

% g'_h is measured against the INITIAL profile. Where the offshore water has
% itself drifted from that profile by more than g'_h (downwelling), g'_h no
% longer means "shelf against the ocean" and Eq. 5 is not tested there.
h_unreliable = ~(drift_val < gph_val);

%% ===== Figure 1: STEPWISE -- each step tested with measured inputs =====
% Eq. 7 with MEASURED (sorted) APE. APE per unit volume of the shelf box,
%   APE_sorted / A_box   [m^2/s^2],
% is the same average as the diagnosed shelf-mean EKE, so the conversion
% factor a(1+cs)^n/(1+bR) is compared directly, with no assumed L_d or shape.
base_step = APEm_val ./ Abox_val;
[EKE_step, coef] = eke_theory_(base_step, s_val, R_val, tau_val, coeff_mode, EKE_diag);

% Eq. 5 with MEASURED EKE and the horizontal g'_h:
%   Q_eddy = c f L Ld_h EKE / g'_h,   Ld_h = sqrt(g'_h H0)/f
%          = c L EKE sqrt(H0 / g'_h)
%
% The constant c is fitted against the MEASURED eddy transport (transient
% eddy buoyancy flux / g'_h), not against Q - Q_Ek: Q is built from the
% time-mean u and holds no eddy flux, so fitting c to it calibrated Eq. 5
% against the mean overturning (unforced) or Ekman residuals (wind).
Ldh_val    = sqrt(gph_val .* H0_val) / f;
basis_step = f * L_val .* Ldh_val .* EKE_diag ./ gph_val;
% Eq. 5 describes eddies EXCHANGING shelf water with offshore water by a
% down-gradient buoyancy flux. Which way that flux points depends on which
% side is light: Zhao et al.'s plume is light inshore, so it goes offshore;
% here the unforced and upwelling shelves are DENSE inshore (tilt < 0), so
% the same exchange carries light water onshore. Multiplying by sign(tilt)
% makes the measured eddy transport positive for down-gradient exchange in
% either geometry. Runs with |tilt| < 5% of g'_h have no clear cross-shelf
% gradient, so the sign is ambiguous: they are circled and left out of the fit.
tilt_amb   = ~(abs(tilt_val) >= 0.05 * gph_val);
Qe_t       = sign(tilt_val) .* Ft_val ./ gph_val;   % measured eddy exchange of shelf water [m^3/s]
counter    = Qe_t < 0 & ~tilt_amb;                  % up-gradient: not baroclinic instability
Qe_s       = Fs_val ./ gph_val;                % standing-meander transport        [m^3/s]
fit_ok     = isfinite(basis_step) & isfinite(Qe_t) & ~h_unreliable & ~tilt_amb;
c_raw      = NaN;
if any(fit_ok), c_raw = basis_step(fit_ok) \ Qe_t(fit_ok); end
if c_raw > 0
    c_eddy = c_raw;
else
    c_eddy = 1.1;
    fprintf(['  ** The measured eddy flux points the wrong way for Eq. 5 (fitted c = %.3g):\n' ...
             '     the eddy flux is mostly UP the cross-shelf gradient. Using c = 1.1 for the Q panels.\n'], c_raw);
end
Q_step     = QEk_val + c_eddy * basis_step;

%% ===== Figure 2: PREDICTIVE -- only what is known before the run =====
% g'_0 = surface-minus-bottom contrast of the INITIAL profile over the shelf
% depth: the stratification the wind and the jet have to work with. Eq. 8,
% Eq. 7 and Eq. 5 chained, with the constants calibrated in figure 1 (c_eddy,
% and a, b, c, n if coeff_mode = 'fit'). Nothing is refitted here.
Ld0_val   = sqrt(gp0_val .* H0_val) / f;
APE_pred  = C_geom * gp0_val .* H0_val.^2 .* Ld0_val;         % Eq. 8 [m^4/s^2]
% Per unit volume over the SAME shelf box as figure 1 and as the diagnosed
% EKE. (Zhao et al. divide by H0 Ld instead -- the anomaly's assumed area --
% but our EKE is a shelf-box mean, so that would slip in a factor W/Ld.)
% A_box is set by the bathymetry alone, so this is still a prediction, and
% predictive/stepwise differ ONLY by APE_pred vs APE_sorted: the Eq. 8 error.
base_pred = APE_pred ./ Abox_val;
EKE_pred  = eke_from_coef_(base_pred, s_val, R_val, coef);
Q_pred    = QEk_val + c_eddy * (f * L_val .* Ld0_val .* EKE_pred ./ gp0_val);
C_meas_0  = APEm_val ./ (gp0_val .* H0_val.^2 .* Ld0_val);    % what C_geom would have to be

%% Report
fprintf('\n=== Theory vs model, per run ===\n');
fprintf(['  %-14s %9s %9s %9s %6s %9s | %9s %9s %9s | %9s %9s %9s %9s\n'], 'run', ...
        'APE_srt', 'g''_h', 'g''_0', 'drift', 'C(g''_0)', ...
        'EKE_mod', 'EKE_step', 'EKE_pred', 'Q_mod', 'Q_Ek', 'Q_step', 'Q_pred');
for i = 1:num_valid
    flag = ''; if h_unreliable(i), flag = '  * g''_h unreliable'; end
    fprintf(['  %-14s %9.3e %9.3e %9.3e %5.0f%% %9.4f | %9.3e %9.3e %9.3e | %9.3e %9.3e %9.3e %9.3e%s\n'], ...
            sweeps_valid{i}, APEm_val(i), gph_val(i), gp0_val(i), 100*drift_val(i)/gph_val(i), C_meas_0(i), ...
            EKE_diag(i), EKE_step(i), EKE_pred(i), Q_diag(i), QEk_val(i), Q_step(i), Q_pred(i), flag);
end
fprintf(['  (EKE in m^2/s^2, Q in m^3/s; drift = offshore-strip drift from the initial profile, as %% of g''_h)\n' ...
         '  Q_Ek is the Ekman part of both theory columns; Q_step - Q_Ek is the eddy term. Q_mod counts\n' ...
         '  offshore flow only, so it cannot go negative, while theory Q is signed: for downwelling\n' ...
         '  (tau < 0) the two are not the same quantity.\n']);
fprintf('\n=== Eddy buoyancy flux across the shelf (alongshore + depth integrated, / g''_h) ===\n');
fprintf('  %-14s %11s %7s %11s %7s %10s | %11s %11s %11s %11s\n', 'run', 'transient', 'x (km)', 'standing', 'x (km)', ...
        'tilt', 'eddy exch', 'Eq.5 c=1', 'Q_mod-Q_Ek', 'Q_mod');
for i = 1:num_valid
    note = '';
    if tilt_amb(i), note = '  * no clear gradient: sign ambiguous'; elseif counter(i), note = '  * UP-gradient'; end
    fprintf('  %-14s %11.3e %7.1f %11.3e %7.1f %+10.2e | %11.3e %11.3e %11.3e %11.3e%s\n', sweeps_valid{i}, ...
            Ft_val(i)/gph_val(i), xFt_val(i), Qe_s(i), xFs_val(i), tilt_val(i), Qe_t(i), basis_step(i), ...
            Q_diag(i) - QEk_val(i), Q_diag(i), note);
end
fprintf(['  (m^3/s; > 0 = light water offshore, the sense of Zhao''s offshore freshwater flux.\n' ...
         '   transient = <u''b''>, time-varying eddies; standing = steady alongshore meanders.\n' ...
         '   eddy exch = transient x sign(tilt): > 0 = down-gradient exchange of shelf water, what\n' ...
         '   Eq. 5 predicts (tilt < 0 = dense inshore, so light water moves ONSHORE).\n' ...
         '   Compare its size with Q_mod - Q_Ek: if it is far\n' ...
         '   smaller, Q is not eddy-driven and Eq. 5 cannot explain it whatever c is.)\n']);
fprintf('  Eq. 5 constant c (fitted to the measured transient eddy transport, %d of %d runs): %.3f\n', ...
        sum(fit_ok), num_valid, c_eddy);
if sum(fit_ok) < 3
    fprintf(['  ** c rests on %d run(s): the stepwise Q panel then sits on the 1:1 line\n' ...
             '     by construction and tests nothing. Pool clusters before reading it.\n'], sum(fit_ok));
end
cu = sprintf('%.3f ', C_meas_0(tau_val == 0));  if isempty(cu), cu = 'none in this cluster'; end
fprintf('  C_geom used in the predictive figure: %.4f. Measured APE / (g''_0 H0^2 Ld_0), unforced runs: %s\n', ...
        C_geom, cu);
report_skill_('stepwise   EKE', EKE_step, EKE_diag);
report_skill_('stepwise   eddy Q', c_eddy * basis_step(fit_ok), Qe_t(fit_ok));
report_skill_('stepwise   Q  ', Q_step(~h_unreliable), Q_diag(~h_unreliable));
report_skill_('predictive EKE', EKE_pred, EKE_diag);
report_skill_('predictive Q  ', Q_pred, Q_diag);

% Save the per-run inputs so check_eke_theory.m can test / refit the scaling
theory_inputs = struct('sweeps', {sweeps_valid}, 'EKE_diag', EKE_diag, 'Q_diag', Q_diag, ...
    'R', R_val, 's', s_val, 'APE', APE_val, 'C_geom', C_geom, 'Ld', Ld_val, 'g_prime', g_prime_val, ...
    'H0', H0_val, 'N', N_val, 'tau', tau_val, 'Zs', Zs_val, 'Ls', Ls_val, 'v0', v0_val, ...
    'f', f, 'L', L_val, 'x_break', xb_val, 'cluster', cluster_name, ...
    'APE_sorted', APEm_val, 'A_box', Abox_val, 'g_prime_h', gph_val, 'Ld_h', Ldh_val, ...
    'ref_drift', drift_val, 'g_prime_0', gp0_val, 'Ld_0', Ld0_val, 'c_eddy', c_eddy, ...
    'EKE_step', EKE_step, 'Q_step', Q_step, 'EKE_pred', EKE_pred, 'Q_pred', Q_pred, ...
    'Q_eddy_transient', Qe_t, 'Q_eddy_standing', Qe_s, 'x_transient_km', xFt_val, 'x_standing_km', xFs_val, ...
    'F_transient', Ft_val, 'F_standing', Fs_val, 'tilt', tilt_val, 'tilt_ambiguous', tilt_amb);
save(fullfile(outdir, sprintf('%s_theory_inputs.mat', cluster_name)), 'theory_inputs');
fprintf('Saved theory inputs: %s\n', fullfile(outdir, sprintf('%s_theory_inputs.mat', cluster_name)));

%% Marker styles, shared by both figures
col_Zs   = [0.0, 0.0, 0.7];     mk_Zs   = 'o';
col_Ls   = [0.7, 0.0, 0.0];     mk_Ls   = 'd';
col_Zsh  = [0.0, 0.5, 0.2];     mk_Zsh  = '^';
col_v0   = [0.49, 0.18, 0.56];  mk_v0   = 's';
col_wind = [0.4, 0.4, 0.4];     mk_wind = 'x';
sz_small = 40; sz_mid = 85; sz_large = 140;
sty = repmat(struct('c', [0 0 0], 'm', 'o', 'sz', 80, 'lw', 1.5, 'filled', false), num_valid, 1);
for i = 1:num_valid
    sw_i = sweeps_valid{i};
    lw = 1.5; filled = false;
    if strcmp(sw_i, 'Zs5_Ls40k')
        c = [0, 0, 0]; m = 'p'; sz = 160; lw = 2; filled = true;   % reference
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
    elseif startsWith(sw_i, 'wind_')
        % Up- and downwelling are different regimes in the theory, so keep
        % them apart; size grows with |tau| within the cluster.
        c = col_wind;
        if tau_val(i) > 0, m = '^'; elseif tau_val(i) < 0, m = 'v'; else, m = mk_wind; end
        sz = rank_size(abs(tau_val(i)), abs(tau_val(startsWith(sweeps_valid, 'wind_'))), sz_small, sz_large);
    elseif startsWith(sw_i, 'Zs')
        % Z_s runs outside the Annie-Fnnie labels (e.g. depthaxis Zs05_edge):
        % same colour as the Z_s group, size grows as the crest shoals.
        c = col_Zs; m = mk_Zs;
        sz = rank_size(-abs(Zs_val(i)), -abs(Zs_val(startsWith(sweeps_valid, 'Zs'))), sz_small, sz_large);
    else
        c = [0.4, 0.4, 0.4]; m = '*'; sz = 80;      % unrecognised label
    end
    sty(i) = struct('c', c, 'm', m, 'sz', sz, 'lw', lw, 'filled', filled);
end

% Legend entries: only groups actually plotted, so a wind-only or Z_s-only
% cluster does not advertise four empty categories.
has    = @(pred) any(cellfun(pred, sweeps_valid));
is_ref = @(s) strcmp(s, 'Zs5_Ls40k');
in_grp = @(G) @(s) ismember(s, G) && ~is_ref(s);
iw     = startsWith(sweeps_valid, 'wind_');
leg = struct('m', {}, 'c', {}, 'filled', {}, 'txt', {});
if has(is_ref), leg(end+1) = struct('m', 'p', 'c', [0 0 0], 'filled', true, 'txt', 'Reference'); end
if has(@(s) (ismember(s, group_Zs) || (startsWith(s, 'Zs') && ~ismember(s, [group_Ls, group_v0, group_ShelfH]) && ~startsWith(s, 'Zsh'))) && ~is_ref(s))
    leg(end+1) = struct('m', mk_Zs, 'c', col_Zs, 'filled', false, 'txt', '$Z_s$');
end
if has(in_grp(group_Ls)),     leg(end+1) = struct('m', mk_Ls,  'c', col_Ls,  'filled', false, 'txt', '$L_s$');    end
if has(in_grp(group_ShelfH)), leg(end+1) = struct('m', mk_Zsh, 'c', col_Zsh, 'filled', false, 'txt', '$Z_{sh}$'); end
if has(in_grp(group_v0)),     leg(end+1) = struct('m', mk_v0,  'c', col_v0,  'filled', false, 'txt', '$v_0$');    end
if any(iw & tau_val(:)' > 0), leg(end+1) = struct('m', '^', 'c', col_wind, 'filled', false, 'txt', '$\tau > 0$ (upwelling)');   end
if any(iw & tau_val(:)' < 0), leg(end+1) = struct('m', 'v', 'c', col_wind, 'filled', false, 'txt', '$\tau < 0$ (downwelling)'); end

%% Plotting
lbl = strrep(cluster_name, '_', '\_');
lab_eke = {'Theory $\sqrt{EKE}$ (m s$^{-1}$)', 'Simulation $\sqrt{EKE}$ (m s$^{-1}$)'};
lab_qe  = {'Theory eddy $Q$ (mSv)', 'Measured eddy exchange, $\pm\langle u''b''\rangle / g''_h$ (mSv)'};
lab_q   = {'Theory $Q$ (mSv)', 'Simulation $Q$ (mSv)'};
nof = false(num_valid, 1);
draw_theory_fig_({sqrt(EKE_step), c_eddy * basis_step / 1e3, Q_step / 1e3}, ...
                 {sqrt(EKE_diag), Qe_t / 1e3, Q_diag / 1e3}, ...
                 {nof, h_unreliable | tilt_amb, h_unreliable}, {lab_eke, lab_qe, lab_q}, [true false false], sty, leg, ...
    {'(a) EKE: Eq. 7 with measured APE', ...
     sprintf('(b) eddy $Q$: Eq. 5 term ($c = %.2f$)', c_eddy), ...
     '(c) total $Q$: $Q_{Ek}$ + eddy term'}, ...
    sprintf('%s  --  stepwise: each step fed measured inputs (coeff\\_mode = %s)', lbl, coeff_mode), ...
    fullfile(outdir, sprintf('%s_theory_vs_model_stepwise.png', cluster_name)));
draw_theory_fig_({sqrt(EKE_pred), Q_pred / 1e3}, {sqrt(EKE_diag), Q_diag / 1e3}, {nof, nof}, ...
                 {lab_eke, lab_q}, [true false], sty, leg, ...
    {sprintf('(a) EKE: Eqs. 8 + 7, $C_{geom} = %.3g$', C_geom), ...
     '(b) $Q$: Eqs. 8 + 7 + 5, initial $g''_0$'}, ...
    sprintf('%s  --  predictive: initial stratification only, constants from the stepwise test', lbl), ...
    fullfile(outdir, sprintf('%s_theory_vs_model_predictive.png', cluster_name)));

%% Where on the shelf does the exchange happen?
% The table above compares each pathway at its own peak x, i.e. across
% different sections. This plots all three against x, with the shoal marked,
% so they can be compared at the same section and placed relative to the
% topography that drives them.
plot_flux_profiles_(XP, FTX, FSX, QXP, gph_val, tilt_val, tilt_amb, QEk_val, Ls_val, sst_val, siw_val, ...
    xb_val, sweeps_valid, cluster_name, fullfile(outdir, sprintf('%s_flux_vs_x.png', cluster_name)));

end

function A = load_ape_(cluster_name, sweep_root, outdir, y_box, x_g_range, runs, refresh)
% ape_wedge results for this cluster: from <cluster>_ape_wedge.mat when it
% matches this script's box and holds every run, otherwise by running
% ape_wedge (which writes that file).
fmat = fullfile(outdir, sprintf('%s_ape_wedge.mat', cluster_name));
A = [];
if ~refresh && isfile(fmat)
    S = load(fmat);  A = S.R;
    need = {'APE_sorted', 'A_box', 'g_prime_h', 'ref_drift', 'g_prime_0', 'tilt', 'y_box', 'x_g_range', 'window'};
    why = '';
    if ~all(isfield(A, need))
        why = 'written by an older ape_wedge';
    elseif ~isequal(A.y_box(:)', y_box(:)') || ~isequal(A.x_g_range(:)', x_g_range(:)') || ~isempty(A.window)
        why = 'made with a different box or window';
    elseif ~all(ismember(strrep(runs, 'sweep_', ''), strrep(A.labels, 'sweep_', '')))
        why = 'missing some of the runs now on disk';
    end
    if isempty(why)
        fprintf('Using measured APE from %s (made %s)\n', fmat, A.created);
    else
        fprintf('  %s is %s: rerunning ape_wedge.\n', fmat, why);
        A = [];
    end
end
if isempty(A)
    A = ape_wedge(cluster_name, 'sweep_root', sweep_root, 'y_box', y_box, ...
                  'x_g_range', x_g_range, 'outdir', outdir);
end
end

function [EKE, coef] = eke_theory_(base, s, R, tau, coeff_mode, EKE_diag)
% Eq. 7 per unit volume: EKE = a (1 + c s)^n base / (1 + b R), base = APE per
% unit volume. coef is [a n c b] per run, so the predictive figure can reuse
% exactly the coefficients chosen (or fitted) here.
n = numel(base);
coef = zeros(n, 4);
for i = 1:n
    if strcmpi(coeff_mode, 'baseline') || strcmpi(coeff_mode, 'fit')
        coef(i, :) = [0.086, 1, 8, 0.0];                 % no-wind (also the fit's fallback)
    elseif tau(i) == 0
        coef(i, :) = [0.086, 1, 8, 0.0];                 % no wind
    elseif tau(i) > 0
        coef(i, :) = [0.490, -1, 2, 0.0];                % upwelling
    else
        coef(i, :) = [0.110, 1, 2, 0.2];                 % downwelling
    end
end
if strcmpi(coeff_mode, 'fit')
    % One set of a, b, c, n for every run, fitted in log space to the
    % diagnosed EKE. Four parameters: do not trust it on a handful of runs.
    ok = isfinite(base) & base > 0 & isfinite(EKE_diag) & EKE_diag > 0;
    try
        pred = @(p, k) exp(p(1)) .* (1 + exp(p(3)) .* s(k)).^p(4) .* base(k) ./ (1 + exp(p(2)) .* R(k));
        cost = @(p) sum((log(pred(p, ok)) - log(EKE_diag(ok))).^2);
        p = fminsearch(cost, [log(0.1), log(0.1), log(2), 1], ...
                       optimset('Display', 'off', 'MaxIter', 5e3, 'MaxFunEvals', 1e4));
        coef = repmat([exp(p(1)), p(4), exp(p(3)), exp(p(2))], n, 1);
        fprintf('Fitted Eq. 7 coefficients (on %d runs): a=%.3g, n=%.3g, c=%.3g, b=%.3g\n', ...
                sum(ok), coef(1,1), coef(1,2), coef(1,3), coef(1,4));
    catch ME
        fprintf('  [Fallback Activated] Eq. 7 fit failed (%s). Using the no-wind coefficients.\n', ME.message);
    end
end
EKE = eke_from_coef_(base, s, R, coef);
end

function EKE = eke_from_coef_(base, s, R, coef)
EKE = coef(:,1) .* (1 + coef(:,3) .* s).^coef(:,2) .* base ./ (1 + coef(:,4) .* R);
end

function report_skill_(what, th, md)
ok = isfinite(th) & isfinite(md) & th > 0 & md > 0;
if sum(ok) < 2, return; end
r = log10(th(ok) ./ md(ok));
fprintf('  %s: theory/model median %.2f, rms log10 error %.2f (factor %.1f), %d runs\n', ...
        what, 10^median(r), sqrt(mean(r.^2)), 10^sqrt(mean(r.^2)), sum(ok));
end

function draw_theory_fig_(X, Y, F, labs, nonneg, sty, leg, ttl, suptl, outfile)
% One panel per cell of X / Y: theory (x) against model (y), with a 1:1 line.
% F{p} marks runs to circle in panel p (g'_h unreliable); labs{p} = {xlabel,
% ylabel}; nonneg(p) starts both axes at 0. Legend goes on the last panel.
np  = numel(X);
fig = figure('Color', 'w', 'Position', [100, 100, 450 * np, 520]);
tl  = tiledlayout(1, np, 'TileSpacing', 'compact', 'Padding', 'compact');
for p = 1:np
    ax = nexttile(tl); hold(ax, 'on');
    x = X{p}; y = Y{p};
    for i = 1:numel(x)
        st = sty(i);
        fc = 'none'; if st.filled, fc = st.c; end
        scatter(ax, x(i), y(i), st.sz, st.m, 'MarkerEdgeColor', st.c, 'MarkerFaceColor', fc, 'LineWidth', st.lw);
        if F{p}(i)
            scatter(ax, x(i), y(i), st.sz * 1.8, 'o', 'MarkerEdgeColor', [0.85 0.4 0], 'LineWidth', 1.2);
        end
    end
    v  = [x(:); y(:)];
    hi = max(v, [], 'omitnan') * 1.15;
    if nonneg(p), lo = 0; else, lo = min(0, min(v, [], 'omitnan') * 1.15); end
    if ~(hi > lo), hi = lo + 1; end
    plot(ax, [lo, hi], [lo, hi], 'k--', 'LineWidth', 1.2);
    xlim(ax, [lo, hi]); ylim(ax, [lo, hi]);
    xlabel(ax, labs{p}{1}, 'Interpreter', 'latex', 'FontSize', 15);
    ylabel(ax, labs{p}{2}, 'Interpreter', 'latex', 'FontSize', 15);
    title(ax, ttl{p}, 'Interpreter', 'latex', 'FontSize', 14);
    pbaspect(ax, [1 1 1]);
    set(ax, 'FontSize', 13, 'Box', 'on', 'XGrid', 'on', 'YGrid', 'on', ...
        'LineWidth', 1, 'TickDir', 'out', 'XColor', 'k', 'YColor', 'k');
end
hl = gobjects(0); tx = {};
for e = leg
    fc = 'none'; if e.filled, fc = e.c; end
    hl(end+1) = plot(ax, NaN, NaN, e.m, 'MarkerEdgeColor', e.c, 'MarkerFaceColor', fc, 'LineWidth', 1.5, 'MarkerSize', 8); %#ok<AGROW>
    tx{end+1} = e.txt; %#ok<AGROW>
end
if any(cellfun(@any, F))
    hl(end+1) = plot(ax, NaN, NaN, 'o', 'MarkerEdgeColor', [0.85 0.4 0], 'LineWidth', 1.2, 'MarkerSize', 12);
    tx{end+1} = '$g''_h$ unreliable, not in fit';
end
if ~isempty(hl)
    legend(ax, hl, tx, 'Location', 'northwest', 'FontSize', 11, 'Interpreter', 'latex');
end
title(tl, suptl, 'FontWeight', 'bold', 'FontSize', 12);
drawnow;
exportgraphics(fig, outfile, 'Resolution', 300);
fprintf('Saved: %s\n', outfile);
end

function plot_flux_profiles_(XP, FTX, FSX, QXP, gph, tilt, amb, QEk, Ls, sst, siw, xb, runs, cluster_name, outfile)
% One panel per run: cross-shelf profiles (alongshore + depth integrated) of
%   eddy     transient <u'b'> / g'_h
%   standing (u-[u])(b-[b]) / g'_h     (steady meanders)
%   mean     offshore branch of the alongshore-integrated time-mean u
% in mSv, with Q_Ek for reference. Both buoyancy fluxes are multiplied by
% sign(tilt), so > 0 is DOWN-gradient exchange of shelf water (the sense Eq. 5
% predicts) whichever side of the shelf is dense; where the tilt is too weak
% to give a sign, they are plotted raw (> 0 = light water offshore).
% Shading: the shoal crest (dark) and its inshore flank / 15 km offshore
% taper (light), from sweep_metadata; dotted line: shelf break.
n  = numel(runs);
nc = min(3, n);  nr = ceil(n / nc);
fig = figure('Color', 'w', 'Position', [60, 60, 560 * nc, 380 * nr]);
tl  = tiledlayout(fig, nr, nc, 'TileSpacing', 'compact', 'Padding', 'compact');
for i = 1:n
    ax = nexttile(tl); hold(ax, 'on'); grid(ax, 'on');
    if isempty(XP{i}) || ~isfinite(gph(i)) || ~(gph(i) > 0)
        title(ax, sprintf('%s: no flux profile', strrep(runs{i}, '_', '\_')));
        continue;
    end
    x = XP{i};
    sg = sign(tilt(i));  if amb(i) || sg == 0, sg = 1; end
    ft = sg * FTX{i} / gph(i) / 1e3;
    fs = sg * FSX{i} / gph(i) / 1e3;
    qm = QXP{i} / 1e3;
    yl = [min([ft; fs; 0]), max([ft; fs; qm; QEk(i)/1e3; 0])];
    yl = yl + [-0.08 0.08] * max(diff(yl), eps);
    % shoal: crest [crest_in, Ls], flank [crest_in - inner_width, crest_in] if
    % the shoal sits on the shelf, taper [Ls, Ls + 15 km]
    crest_in = max(sst(i), 0) / 1e3;
    patch(ax, [crest_in Ls(i)/1e3 Ls(i)/1e3 crest_in], yl([1 1 2 2]), [0.75 0.75 0.75], ...
          'EdgeColor', 'none', 'FaceAlpha', 0.45, 'HandleVisibility', 'off');
    if sst(i) > 0 && siw(i) > 0
        patch(ax, [crest_in - siw(i)/1e3, crest_in, crest_in, crest_in - siw(i)/1e3], yl([1 1 2 2]), ...
              [0.88 0.88 0.88], 'EdgeColor', 'none', 'FaceAlpha', 0.45, 'HandleVisibility', 'off');
    end
    patch(ax, [Ls(i)/1e3, Ls(i)/1e3 + 15, Ls(i)/1e3 + 15, Ls(i)/1e3], yl([1 1 2 2]), ...
          [0.88 0.88 0.88], 'EdgeColor', 'none', 'FaceAlpha', 0.45, 'HandleVisibility', 'off');
    xline(ax, xb(i), 'k:', 'LineWidth', 1.2, 'HandleVisibility', 'off');
    yline(ax, 0, 'k-', 'LineWidth', 0.6, 'HandleVisibility', 'off');
    plot(ax, x, ft, '-', 'Color', [0.80 0.20 0.10], 'LineWidth', 1.8, 'DisplayName', 'eddy (transient)');
    plot(ax, x, fs, '-', 'Color', [0.10 0.35 0.75], 'LineWidth', 1.8, 'DisplayName', 'standing meanders');
    plot(ax, x, qm, '-', 'Color', [0.35 0.35 0.35], 'LineWidth', 1.2, 'DisplayName', 'mean overturning (offshore)');
    if QEk(i) ~= 0
        yline(ax, QEk(i)/1e3, '--', 'Color', [0.35 0.35 0.35], 'LineWidth', 1.0, 'DisplayName', 'Q_{Ek}');
    end
    xlim(ax, [0, max(x)]);  ylim(ax, yl);
    if amb(i), sgn_txt = 'raw sign: tilt too weak'; elseif tilt(i) < 0, sgn_txt = 'dense inshore'; else, sgn_txt = 'light inshore'; end
    title(ax, sprintf('%s   (%s)', strrep(runs{i}, '_', '\_'), sgn_txt), 'FontWeight', 'bold');
    xlabel(ax, 'x (km from coast)');  ylabel(ax, 'transport (mSv)');
    set(ax, 'FontSize', 10);
    if i == 1, legend(ax, 'Location', 'best', 'FontSize', 8); end
end
title(tl, sprintf(['%s  --  cross-shelf exchange by pathway (> 0 = down-gradient for the eddy and standing ' ...
      'curves). Dark: shoal crest; light: flank / taper; dotted: shelf break'], strrep(cluster_name, '_', '\_')), ...
      'FontWeight', 'bold', 'FontSize', 11);
drawnow;
exportgraphics(fig, outfile, 'Resolution', 200);
fprintf('Saved: %s\n', outfile);
close(fig);
end

function sz = rank_size(v, all_v, sz_lo, sz_hi)
% Marker size from v's position within all_v: smallest value -> sz_lo,
% largest -> sz_hi, linear in rank so uneven spacing does not crowd sizes.
u = unique(all_v(isfinite(all_v)));
if numel(u) <= 1, sz = (sz_lo + sz_hi) / 2; return; end
sz = sz_lo + (sz_hi - sz_lo) * (find(u == v, 1) - 1) / (numel(u) - 1);
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


