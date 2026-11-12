#!/bin/bash
# run_vis_eke_sweeps.sh
#
# Surface EKE maps + eddy fraction for the shoal height and shoal length sweeps
# (last time-averaging window) on Casper.
#
# Usage:
#   ./run_vis_eke_sweeps.sh [cluster_name]

CLUSTER_NAME="${1:-cluster_Fnnie_1_19}"

# Load NCAR Casper environment
module purge
module load ncarenv-basic/23.10
module load ncarenv/23.10
module load matlab/R2024a

echo "=== Running Surface EKE Sweep Figure ==="
echo "    Cluster:          ${CLUSTER_NAME}"
echo "    Output Directory: /glade/work/nguyen/unc-shoals/animations/"

# Run from the tools directory so the toolbox paths resolve
cd "$(dirname "$0")"

# Run MATLAB in non-interactive mode
#   'scale': 'log' or 'linear'  (both are made below)
matlab -nodisplay -nosplash -r "\
    addpath(genpath('cmocean')); \
    addpath(genpath('pmkmp')); \
    try, \
        fprintf('--- Generating surface EKE sweeps (log) for ${CLUSTER_NAME} ---\n'); \
        vis_eke_sweeps('${CLUSTER_NAME}', 'scale', 'log', 'clipped', true, 'level', 'depth_avg'); \
        fprintf('--- Generating surface EKE sweeps (linear) for ${CLUSTER_NAME} ---\n'); \
        vis_eke_sweeps('${CLUSTER_NAME}', 'scale', 'linear', 'clipped', true, 'level', 'depth_avg'); \
    catch ME, \
        fprintf('Error: %s\n', ME.message); \
    end; \
    exit"

echo "=== Finished processing ${CLUSTER_NAME} ==="

