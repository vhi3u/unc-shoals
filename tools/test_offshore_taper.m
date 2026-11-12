%% Offshore sponge: effective nudge along x
% Shows how each field is pushed from shelf→offshore values as a function of x

clear; clc;

%--- Grid ---
x    = linspace(0, 100e3, 500);
x0   = 60e3;   sig = 20e3;
mask = 0.5 * (1 + tanh((x - x0) / sig));

% Depths to plot
z_pts  = [0, -15, -30, -45];
colors = [0.1 0.4 0.8;   0.2 0.7 0.4;   0.85 0.4 0.1;   0.6 0.2 0.7];

delta = 2.5;  % smoothing (same as Julia)

%--- Smooth-step helper ---
smooth_step = @(z, z0) 0.5 * (1 - tanh((z - z0) / delta));

%--- PWL profile evaluator ---
function val = pwl(z, z1, z2, z3, v1, v2, v3, delta)
    smooth_step = @(zz, z0) 0.5*(1 - tanh((zz - z0)/delta));
    m12 = (v2-v1)/(z2-z1);  m23 = (v3-v2)/(z3-z2);
    w1 = smooth_step(z,z1); w2 = smooth_step(z,z2); w3 = smooth_step(z,z3);
    val = v1*(1-w1) + (v1+m12*(z-z1)).*(w1-w2) + (v2+m23*(z-z2)).*(w2-w3) + v3*w3;
end

%--- Evaluate T and S shelf + offshore targets at each depth ---
T_shelf  = arrayfun(@(z) pwl(z, -5,-15,-30, 24.5378,24.3073,23.4116, delta), z_pts);
S_shelf  = arrayfun(@(z) pwl(z, -5,-15,-30, 35.5830,35.9986,36.1776, delta), z_pts);
T_offshr = arrayfun(@(z) pwl(z, -5,-25,-45, 25.0,23.0,21.0,           delta), z_pts);
S_offshr = arrayfun(@(z) pwl(z, -5,-25,-45, 36.0,35.9,35.8,           delta), z_pts);

v0 = 0.1;   % background along-shore velocity (m/s)

%--- Plot ---
figure('Position',[100 100 1200 380]);
t = tiledlayout(1,3,'TileSpacing','compact','Padding','compact');

% Panel 1 — velocity (nudged → 0)
nexttile; hold on;
v_profile = v0 * (1 - mask);   % equilibrium: v0 on shelf, 0 offshore
plot(x/1e3, v_profile*100, '-', 'Color',[0.15 0.45 0.8], 'LineWidth',2.5);
xline(x0/1e3,':','Color',[0.5 0.5 0.5],'LineWidth',1.2,'Label','shelf break');
xlabel('x  [km]'); ylabel('v  [cm s^{-1}]');
title('Velocity nudge  (v_0 \rightarrow 0)');
ylim([-0.5 11]); grid on;

% Panel 2 — temperature
nexttile; hold on;
for i = 1:length(z_pts)
    T_x = T_shelf(i)*(1-mask) + T_offshr(i)*mask;   % blended target along x
    plot(x/1e3, T_x, '-', 'Color', colors(i,:), 'LineWidth', 2, ...
         'DisplayName', sprintf('z = %d m', z_pts(i)));
end
xline(x0/1e3,':','Color',[0.5 0.5 0.5],'LineWidth',1.2);
xlabel('x  [km]'); ylabel('T  [°C]');
title('Temperature nudge  (T_{shelf} \rightarrow T_{east})');
legend('Location','best','FontSize',9); grid on;

% Panel 3 — salinity
nexttile; hold on;
for i = 1:length(z_pts)
    S_x = S_shelf(i)*(1-mask) + S_offshr(i)*mask;
    plot(x/1e3, S_x, '-', 'Color', colors(i,:), 'LineWidth', 2, ...
         'DisplayName', sprintf('z = %d m', z_pts(i)));
end
xline(x0/1e3,':','Color',[0.5 0.5 0.5],'LineWidth',1.2);
xlabel('x  [km]'); ylabel('S  [psu]');
title('Salinity nudge  (S_{shelf} \rightarrow S_{east})');
legend('Location','best','FontSize',9); grid on;

sgtitle('Offshore sponge equilibrium targets along x  (\sigma = 20 km)', ...
        'FontSize',13,'FontWeight','bold');
