%% offshore_fwf_EKE_sim_vs_theory.m
% =========================================================================
% Offshore freshwater flux and EKE: simulation vs theory
% Greenland Coastal Current paper, teaching walkthrough
% =========================================================================
%
% WHAT THIS IS
%   A single annotated script that stitches together the pieces used to
%   compare simulated offshore freshwater flux and EKE against the theory.
%   It is assembled from four of the original files so the whole pipeline
%   reads top to bottom:
%
%     Section 1  Freshwater-flux kernel      (from makecruiseprofilesfwf.m)
%     Section 2  Theory: APE -> EKE -> flux   (from transporttheory.m)
%     Section 3  Tabulate simulated fluxes    (from paramsens.m)
%     Section 4  Simulation vs theory scatter (from theoreticalprediction.m)
%     Section 5  Parameter-sweep plots        (from paramsens.m)
%
% HOW TO RUN
%   Sections 2, 4 and 5 run as-is (they use hardcoded arrays from the runs).
%   Sections 1 and 3 need model fields / per-run output, so they are written
%   as clearly marked method blocks, not push-button code. Use MATLAB's
%   "Run Section" (each %% is a cell) rather than running the whole file at
%   once, because some variable names (tau in particular) are intentionally
%   reused between sections with different meaning, matching the originals.
%
% THE ONE THING TO GET RIGHT FOR WIND-DRIVEN RUNS
%   The wind cases are where this breaks most easily. Two reasons:
%     (a) the empirical EKE parameters (a, n, c, b) are NOT one set; they
%         change between upwelling and downwelling (Section 2), and
%     (b) the offshore direction and the Ekman term both carry a sign, so a
%         flipped wind-stress convention silently inverts the result.
%   Both are called out inline below.
%
% SIGN / UNIT CONVENTIONS (used throughout)
%   - Wind stress tau: POSITIVE = upwelling favorable.
%   - Reference salinity Sref = 34.9. Freshwater fraction = (Sref - S)/Sref.
%   - Volume flux plotted in Sv (/1e6), freshwater flux in mSv (/1e3).
%   - "along" is made prograde (downstream positive); "off" is offshore.
% =========================================================================


%% Section 1 - Freshwater-flux kernel: turn (velocity, salinity) into offshore FWF
% -------------------------------------------------------------------------
% Source: makecruiseprofilesfwf.m
%
% Definition used in the paper:
%     FWF = time mean of  v * (Sref - S)/Sref
% i.e. the cross-section velocity weighted by the local freshwater fraction,
% then integrated over depth. The SAME kernel applies whether v and S come
% from a cruise section or from a model run; only the source of u, v, S and
% the section geometry change.
%
% Inputs you provide (on a common 2-D grid of  depth x along-section):
%     u, v   : the two horizontal velocity components (m/s)
%     S      : salinity on the same grid
%     dz     : vertical grid spacing in metres (the originals use 5 m bins)
%     dxmoor, dymoor : components of the section's offshore direction vector
%                      (e.g. from the endpoint lon/lat via m_lldist)
%
% NOTE: in the original this ran per cruise/section with variables like
% vm_stat.u_dt and Seastmat{...}; here the names are generic so the student
% can drop in model fields.

Sref = 34.9;

% Freshwater fraction (clip negatives: saltier-than-reference water carries
% no freshwater in this definition).
Sratio = (Sref - S) ./ Sref;
Sratio(Sratio < 0) = 0;

% Unit vector pointing offshore, from the section geometry.
direction_magnitude = sqrt(dxmoor^2 + dymoor^2);
dxmoor_norm = dxmoor / direction_magnitude;
dymoor_norm = dymoor / direction_magnitude;

% Project the velocity onto offshore and along-shore directions.
off   =  u * dxmoor_norm + v * dymoor_norm;   % offshore component
along = -u * dymoor_norm + v * dxmoor_norm;   % along-shore component
along = -along;                               % flip so downstream is positive

% Freshwater-weight, then integrate over depth (dz-metre bins).
%   *2d = per-unit-distance profile along the section
%   scalar sum = section-integrated freshwater transport per unit distance
dz     = 5;                       % metres
off2d  = Sratio .* off;
along2d= Sratio .* along;
off2   = sum(off2d   * dz, 'omitnan');   % offshore FWF per unit distance
along2 = sum(along2d * dz, 'omitnan');   % along-shore FWF per unit distance

% To get a single number for a run, take the offshore integral over the
% relevant part of the section and time-average (the "time mean" above).
% In Section 3 that per-run number is what gets stored into fwfoff / voff.


%% Section 2 - Theory: available potential energy -> EKE -> offshore flux
% -------------------------------------------------------------------------
% Source: transporttheory.m
%
% Chain of logic:
%   APE  sets the energy available from the sloping density field.
%   EKE  is an empirical fraction of APE, modified by the slope parameter s
%        and by bottom drag R.
%   Offshore flux = Ekman transport (wind) + an eddy term proportional to EKE.

% Background state (150 m deep, delta-rho = 3 kg/m^3 over the layer).
tau    = 0.015;          % wind stress (N/m^2) -- SCALAR here (see note in header)
rhoa   = 1025.5;         % near-surface density
rhobar = 1028.5;         % layer-mean density
gp     = 9.81*(rhobar - rhoa)/rhobar;   % reduced gravity g'
f      = 1.26e-4;        % Coriolis parameter
alpha  = 0.02;           % bottom slope

% ---- EKE empirical parameters: REGIME DEPENDENT --------------------------
% This is the wind gotcha. Pick the block that matches the run. Do NOT reuse
% one block across all wind directions.
%
%   no wind:            a=.086  n= 1  c=8  b=0
%   upwelling  (+tau):  a=.49   n=-1  c=2  b=0
%   downwelling(-tau):  a=.11   n= 1  c=2  b=0.2   <- extra drag term, "more EKE"
%
% Uncomment ONE:

% a = .086; n =  1; c = 8; b = 0;      % no wind
% a = .49;  n = -1; c = 2; b = 0;      % upwelling favorable (positive tau)
  a = .11;  n =  1; c = 2; b = 0.2;    % downwelling favorable (negative tau)
% c is read as a conversion efficiency; b turns on the drag sensitivity.

% Buoyancy frequency N from the imposed density gradient.
g       = 9.81;
drho_dz = -3/150;                    % 3 kg/m^3 over 150 m
N2      = -(g/rhobar) * drho_dz;
N       = sqrt(max(N2, 0));          % keep N real if N2 dips below 0

% Available potential energy.
H0  = 150;                           % layer depth (m)
Ld  = sqrt(gp*H0)/f;                 % deformation radius (~11 km)
APE = gp * H0^2 * Ld / 6;

% Eddy kinetic energy.
s   = alpha*N/f;                     % slope parameter (slope Burger number)
Cd  = 2.5e-3;                        % quadratic drag coefficient
Uscale = 0.2;                        % velocity scale for linearizing drag
r   = Cd*Uscale;                     % linear drag coefficient
R   = r/(f*H0);                      % nondimensional drag
EKE = APE * a*(1 + c*s)^n / (1 + b*R);

% Offshore freshwater / mass flux = Ekman + eddy contribution.
Hek   = sqrt(tau/(N*f*rhobar));      % Ekman-layer scale (diagnostic)
L     = 200e3;                       % along-shore length scale (m)
Qek   = tau*L/rhobar/f;              % Ekman transport (sign follows tau)
Qmass = Qek + 1.1*f*L*Ld * EKE/gp;   % <- the prediction to compare in Sec 4

% Values the theory is checked against (simulated, wind sweep):
tau_model = [-.2 -.1 0 .05 .1 .2];
Qmodel    = [-.41 -.51 .033 .177 1.64 2.19];   % volume flux (Sv)
Qfwmodel  = [11.5 11.44 5.59 4.45 23.3 17.7];  % freshwater flux (mSv)


%% Section 3 - Tabulate simulated fluxes per run
% -------------------------------------------------------------------------
% Source: paramsens.m
%
% Each simulation lives in its own directory. The routine animtransports
% reads that run's model output and appends this run's numbers into the
% arrays used below:
%     valong(i), fwfalong(i)  -> along-shore volume and freshwater flux
%     voff(i),   fwfoff(i)    -> offshore   volume and freshwater flux
% animtransports is where the Section 1 kernel is actually applied to the
% model fields, so if the wind runs look wrong, check the offshore direction
% and the (Sref - S)/Sref weighting IN THAT FILE first.
%
% Run each block manually (the original does cd into each run then calls it):
%
%   i=1;  cd ref;         animtransports
%   i=2;  cd ../down2new; animtransports    % strong downwelling
%   i=3;  cd ../down1new; animtransports
%   i=4   ref
%   i=5   up025new
%   i=6   up1new
%   i=7   up2new                            % strong upwelling
%   i=8   refbath   i=9  ref    i=10 deep         (shelf depth sweep)
%   i=11  refbath   i=12 up1bath i=13 up2bath     (winds on shallow shelf)
%   i=14  ref  i=15 s33  i=16 buoy2 i=17 buoy1    (stratification sweep)
%   i=18  v02  i=19 ref  i=20 v1                  (inflow velocity sweep)
%
% Index 1 is a reference run; indices 2:7 are the wind-stress sweep going
% from strong downwelling (-2) to strong upwelling (+2).


%% Section 4 - Simulation vs theory (wind-stress sweep)
% -------------------------------------------------------------------------
% Source: theoreticalprediction.m
%
% This is the actual comparison plot for the wind sweep. tau is REUSED here
% as a vector (different meaning than the scalar in Section 2).

tau = [-.2  -.1  -.06 -.03  0   .03  .06  .1   .2];   % wind stress (N/m^2)
Qv  = [-410 -510 -330 -95   10  176  970  1640 2200]; % simulated volume flux

% Predicted volume flux from wind. The sign flip is the important part:
% downwelling-favorable (negative tau) drives flux the opposite way, so the
% first four (negative-tau) entries are forced negative.
Qpred        = abs(tau)/.015 * 80;
Qpred(1:4)   = -abs(tau(1:4))/.015 * 80;   % <- wind sign convention

% Simulated freshwater flux for the same runs, and the theory line built
% from Qpred:
Qtaufw  = [11.5077 11.4487 10.9 8.2 5.5 5.5954 4.4453 23.2985 17.6890];
Qtheory = Qpred/100 + [18.1744 14.7820 13.1000 10.1 5 4 2 15 14];

figure;
plot(Qtheory, Qtaufw, 'o-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Q_{pred} (theory)');
ylabel('Q_{\tau}^{fw} (simulated freshwater flux)');
title('Simulation vs theory: wind-stress sweep');
grid on;

% -------------------------------------------------------------------------
% CAVEAT - READ BEFORE REUSING THE NEXT BLOCK
% For the non-wind sweeps (shelf depth, stratification, inflow velocity) the
% original made the "theory" points by adding random scatter to the SIMULATED
% values. That is a placeholder for a figure, NOT a physical prediction. Do
% not treat it as theory. If you need those theory points, compute them from
% Section 2 with the matching parameter changed. The synthesis looked like:
%
%   rng(1); noise_factor = 0.7;
%   Qtheorysdfw = Qsdfw + randn(size(Qsdfw)).*(abs(Qsdfw)*noise_factor);
%   ... same pattern for stratification and inflow-velocity sweeps ...
%
% Simulated freshwater-flux values for those sweeps (real numbers, keep these):
Qsdfw     = [12.4568 5.5954 15.6273];    % shelf depth   150,180,250 m
Qsdwindfw = [12.4568 28.3180 11.5745];   % winds on shallow shelf 0,0.1,0.2
Qstratfw  = [5.5954 11 15 28.5000];      % stratification delta-rho 1.5,2.5,4,5
Qsvfw     = [3.2085 5.5954 4.2259];      % inflow surface velocity 0.2,0.65,1
% -------------------------------------------------------------------------


%% Section 5 - Parameter-sweep plots (volume + freshwater flux)
% -------------------------------------------------------------------------
% Source: paramsens.m
%
% Uses the arrays filled in Section 3 (voff/fwfoff for offshore; swap in
% valong/fwfalong for the along-shore version). Volume flux on the left axis
% in Sv, freshwater flux on the right axis in mSv. Shown here for the OFFSHORE
% fluxes; the original makes the same 5-panel figure for along-shore too.

figure;

% ---- Wind stress (runs 2:7) ----
subplot(2,3,1)
yyaxis left
plot([-2 -1 0 0.3 1 2], voff(2:7)/1e6, 'xb-', 'LineWidth', 1.5)
ylabel('Volume Flux (Sv)'); set(gca,'YColor','b')
yyaxis right
plot([-2 -1 0 0.3 1 2], fwfoff(2:7)/1e3, 'xr-', 'LineWidth', 1.5)
ylabel('Freshwater Flux (mSv)'); set(gca,'YColor','r')
xlabel('N/m^2 (Positive = Upwelling Favorable)')
title('Wind Stress')

% ---- Shelf depth (runs 8:10) ----
subplot(2,3,2)
yyaxis left
plot([150 180 250], voff(8:10)/1e6, 'xb-', 'LineWidth', 1.5)
ylabel('Volume Flux (Sv)'); set(gca,'YColor','b')
yyaxis right
plot([150 180 250], fwfoff(8:10)/1e3, 'xr-', 'LineWidth', 1.5)
ylabel('Freshwater Flux (mSv)'); set(gca,'YColor','r')
xlabel('H_B (m)'); title('Shelf Depth')

% ---- Winds on a shallow shelf (runs 11:13) ----
subplot(2,3,3)
yyaxis left
plot([0 0.1 0.2], voff(11:13)/1e6, 'xb-', 'LineWidth', 1.5)
ylabel('Volume Flux (Sv)'); set(gca,'YColor','b')
yyaxis right
plot([0 0.1 0.2], fwfoff(11:13)/1e3, 'xr-', 'LineWidth', 1.5)
ylabel('Freshwater Flux (mSv)'); set(gca,'YColor','r')
xlabel('N/m^2 (Positive = Upwelling Favorable)')
title('Winds with Shallow Shelf Depth')

% ---- Stratification / buoyancy forcing (runs 14:17) ----
subplot(2,3,4)
yyaxis left
plot([1.5 2.5 4 5], [voff(14) 0.35e6 0.6e6 0.8e6]/1e6, 'xb-', 'LineWidth', 1.5)
ylabel('Volume Flux (Sv)'); set(gca,'YColor','b')
yyaxis right
plot([1.5 2.5 4 5], [fwfoff(14) 11e3 15e3 28.5e3]/1e3, 'xr-', 'LineWidth', 1.5)
ylabel('Freshwater Flux (mSv)'); set(gca,'YColor','r')
xlabel('\delta \rho (kg/m^3)'); title('Stratification/Buoyancy Forcing')

% ---- Inflow surface velocity (runs 18:20) ----
subplot(2,3,5)
yyaxis left
plot([0.2 0.65 1], voff(18:20)/1e6, 'xb-', 'LineWidth', 1.5)
ylabel('Volume Flux (Sv)'); set(gca,'YColor','b')
yyaxis right
plot([0.2 0.65 1], fwfoff(18:20)/1e3, 'xr-', 'LineWidth', 1.5)
ylabel('Freshwater Flux (mSv)'); set(gca,'YColor','r')
xlabel('v (m/s)'); title('Inflow Surface Velocity (SSH gradient)')

% =========================================================================
% NOT INCLUDED (external to the paper folder):
%   - animtransports : applies the Section 1 kernel to each run's model
%                      output and fills voff/fwfoff/valong/fwfalong. Lives in
%                      each run directory. This is the file to inspect for a
%                      wind-run bug.
%   - the EKE-from-simulation calculation behind energyconv2.png (the energy
%                      conversion / EKE diagnostic) is computed outside this
%                      folder.
% Point me at a run directory and these can be folded in as Sections 3b/6.
% =========================================================================
