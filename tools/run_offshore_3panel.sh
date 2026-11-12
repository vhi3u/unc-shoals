#!/bin/bash
# run_vis_offshore_3panel.sh
#
# Runs the 3-panel offshore flux plot diagnostic on Casper.
#
# Usage:
#   ./run_vis_offshore_3panel.sh [cluster_name]

# Default to cluster13_111_117 if no argument provided
CLUSTER_NAME="${1:-cluster_Cnnie_1_15}"

# Load NCAR Casper environment
module purge
module load ncarenv-basic/23.10
module load ncarenv/23.10
module load matlab/R2024a

echo "=== Running 3-Panel Offshore Flux Diagnostic ==="
echo "    Cluster:          ${CLUSTER_NAME}"
echo "    Output Directory: /glade/work/nguyen/unc-shoals/animations/"

# Run MATLAB in non-interactive mode
matlab -nodisplay -nosplash -r "\
    addpath(genpath('cmocean')); \
    addpath(genpath('pmkmp')); \
    try, \
        fprintf('--- Generating 3-Panel Offshore Flux Plots for ${CLUSTER_NAME} ---\n'); \
        vis_offshore_3panel('${CLUSTER_NAME}', 'full_domain', true); \
    catch ME, \
        fprintf('Error: %s\n', ME.message); \
    end; \
    exit"

echo "=== Finished processing ${CLUSTER_NAME} ==="

