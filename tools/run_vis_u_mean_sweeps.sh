#!/bin/bash
# run_vis_u_mean_sweeps.sh
#
# Generates the time-mean cross-shelf velocity figure (shoal height & length
# sweeps: 6 maps + 2 line plots) on Casper.
#
# Usage:
#   ./run_vis_u_mean_sweeps.sh [cluster_name]

CLUSTER_NAME="${1:-cluster_Fnnie_1_19}"

# Load NCAR Casper environment
module purge
module load ncarenv-basic/23.10
module load ncarenv/23.10
module load matlab/R2024a

echo "=== Running Time-Mean U Sweep Figure ==="
echo "    Cluster:          ${CLUSTER_NAME}"
echo "    Output Directory: /glade/work/nguyen/unc-shoals/animations/"

# Run from the tools directory so the toolbox paths resolve
cd "$(dirname "$0")"

# Run MATLAB in non-interactive mode
matlab -nodisplay -nosplash -r "\
    addpath(genpath('gsw_matlab_v3_06_16')); \
    addpath(genpath('cmocean')); \
    try, \
        fprintf('--- Generating time-mean u sweeps for ${CLUSTER_NAME} ---\n'); \
        vis_u_mean_sweeps('${CLUSTER_NAME}', 'clipped', true, 'x_line', 40, 'climu', [-0.1, 0.1]); \
    catch ME, \
        fprintf('Error: %s\n', ME.message); \
    end; \
    exit"

echo "=== Finished processing ${CLUSTER_NAME} ==="

