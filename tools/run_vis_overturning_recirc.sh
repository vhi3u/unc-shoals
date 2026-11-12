#!/bin/bash
# run_vis_overturning_recirc.sh
#
# Generates the overturning (x-z, on density) and recirculation (x-y)
# streamfunction figure for the reference case on Casper.
#
# Usage:
#   ./run_vis_overturning_recirc.sh [cluster_name]

CLUSTER_NAME="${1:-cluster_Fnnie_1_19}"

# Load NCAR Casper environment
module purge
module load ncarenv-basic/23.10
module load ncarenv/23.10
module load matlab/R2024a

echo "=== Running Overturning / Recirculation Figure ==="
echo "    Cluster:          ${CLUSTER_NAME}"
echo "    Output Directory: /glade/work/nguyen/unc-shoals/animations/"

# Run from the tools directory so the toolbox paths resolve
cd "$(dirname "$0")"

# Run MATLAB in non-interactive mode
matlab -nodisplay -nosplash -r "\
    addpath(genpath('gsw_matlab_v3_06_16')); \
    addpath(genpath('cmocean')); \
    try, \
        fprintf('--- Generating overturning/recirculation for ${CLUSTER_NAME} ---\n'); \
        vis_overturning_recirc('${CLUSTER_NAME}', 'run', 'Zs5_Ls40k', 'clipped', true); \
    catch ME, \
        fprintf('Error: %s\n', ME.message); \
    end; \
    exit"

echo "=== Finished processing ${CLUSTER_NAME} ==="

