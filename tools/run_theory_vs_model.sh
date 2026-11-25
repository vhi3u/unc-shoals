#!/bin/bash
# run_theory_vs_model.sh
#
# Generates the theory vs model comparison (sqrt(EKE) and offshore Q) on Casper.
# Two figures per cluster, in /glade/work/nguyen/unc-shoals/animations/:
#   <cluster>_theory_vs_model_stepwise.png    each step fed measured inputs
#                                             (Eq. 7 <- sorted APE; Eq. 5 <- EKE, g'_h)
#   <cluster>_theory_vs_model_predictive.png  initial stratification only (g'_0)
# Measured APE and g'_h come from <cluster>_ape_wedge.mat, written by
# ape_wedge.m; if that file is missing or stale, ape_wedge is run first.
#
# Usage:
#   ./run_theory_vs_model.sh [cluster_name] [coeff_mode] [C_geom] [include_wind] [refresh_ape]
#
#   ./run_theory_vs_model.sh cluster_depthaxis_120
#   ./run_theory_vs_model.sh cluster_weakwind250_120 regime 1/2 true
#
# Defaults: regime, 1/2, false, false. C_geom is a MATLAB expression, so 1/6
# works; it only affects the predictive figure. coeff_mode 'fit' refits the
# four Eq. 7 coefficients and should not be trusted on a handful of runs.
# A cluster holding only wind runs needs include_wind = true, or the script
# stops and says so.

CLUSTER_NAME="${1:-cluster_Fnnie_1_19}"
COEFF_MODE="${2:-regime}"
C_GEOM="${3:-1/2}"
INCLUDE_WIND="${4:-false}"
REFRESH_APE="${5:-false}"

# Load NCAR Casper environment
module purge
module load ncarenv-basic/23.10
module load ncarenv/23.10
module load matlab/R2024a

echo "=== Running Theory vs Model Comparison ==="
echo "    Cluster:          ${CLUSTER_NAME}"
echo "    coeff_mode:       ${COEFF_MODE}   C_geom: ${C_GEOM}   include_wind: ${INCLUDE_WIND}   refresh_ape: ${REFRESH_APE}"
echo "    Output Directory: /glade/work/nguyen/unc-shoals/animations/"

# Run from the tools directory so the toolbox paths resolve
cd "$(dirname "$0")"

# Run MATLAB in non-interactive mode
#   plot_theory_vs_model(cluster, coeff_mode, C_geom, include_wind, refresh_ape)
#     coeff_mode:   'regime' | 'baseline' | 'fit'
#     C_geom:       1/2 (rectangular layer) or 1/6 (triangular wedge)
#     include_wind: true / false
#     refresh_ape:  true / false (rerun ape_wedge even if its .mat is current)
matlab -nodisplay -nosplash -r "\
    addpath(genpath('gsw_matlab_v3_06_16')); \
    try, \
        fprintf('--- Generating theory vs model for ${CLUSTER_NAME} ---\n'); \
        plot_theory_vs_model('${CLUSTER_NAME}', '${COEFF_MODE}', ${C_GEOM}, ${INCLUDE_WIND}, ${REFRESH_APE}); \
    catch ME, \
        fprintf('Error: %s\n', ME.message); \
        fprintf('%s\n', getReport(ME, 'extended')); \
    end; \
    exit"

echo "=== Finished processing ${CLUSTER_NAME} ==="

