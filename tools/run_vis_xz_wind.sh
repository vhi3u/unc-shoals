#!/bin/bash
# run_vis_u_xz_wind.sh
#
# 6-panel x-z time-mean u with linear-EOS isopycnals:
# reference (tau = 0) + upwelling winds (0.03-0.15 N/m^2), on Casper.
#
# Usage:
#   ./run_vis_u_xz_wind.sh [cluster_name]

CLUSTER_NAME="${1:-cluster_Fnnie_1_19}"

# Load NCAR Casper environment
module purge
module load ncarenv-basic/23.10
module load ncarenv/23.10
module load matlab/R2024a

echo "=== Running x-z u (wind sweep) ==="
echo "    Cluster:          ${CLUSTER_NAME}"
echo "    Output Directory: /glade/work/nguyen/unc-shoals/animations/"

# Run from the tools directory so the toolbox paths resolve
cd "$(dirname "$0")"

# Run MATLAB in non-interactive mode
#   'rho_step' : isopycnal spacing in kg/m^3 (0.1 matches vishoals_still.m; 0.05 for more lines)
matlab -nodisplay -nosplash -r "\
    addpath(genpath('cmocean')); \
    try, \
        fprintf('--- x-z u, wind sweep, for ${CLUSTER_NAME} ---\n'); \
        vis_u_xz_wind('${CLUSTER_NAME}', 'clipped', true, 'rho_step', 0.1); \
    catch ME, \
        fprintf('Error: %s\n', ME.message); \
    end; \
    exit"

echo "=== Finished processing ${CLUSTER_NAME} ==="

