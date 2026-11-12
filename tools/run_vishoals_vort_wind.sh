#!/bin/bash
# run_vishoals_vort_wind.sh
#
# Generates the 6-panel surface vorticity wind-sweep still (last frame) on Casper.
#
# Usage:
#   ./run_vishoals_vort_wind.sh [cluster_name]

CLUSTER_NAME="${1:-cluster_Fnnie_1_19}"

# Load NCAR Casper environment
module purge
module load ncarenv-basic/23.10
module load ncarenv/23.10
module load matlab/R2024a

echo "=== Running Wind-Sweep Vorticity Still ==="
echo "    Cluster:          ${CLUSTER_NAME}"
echo "    Output Directory: /glade/work/nguyen/unc-shoals/animations/"

# Run from the tools directory so the toolbox paths resolve
cd "$(dirname "$0")"

# Run MATLAB in non-interactive mode
matlab -nodisplay -nosplash -r "\
    addpath(genpath('gsw_matlab_v3_06_16')); \
    addpath(genpath('cmocean')); \
    try, \
        fprintf('--- Generating wind-sweep vorticity for ${CLUSTER_NAME} ---\n'); \
        vishoals_vort_wind('${CLUSTER_NAME}', 'wind_sign', 'negative', 'clipped', true, 'climRo', [-1.0, 1.0]); \
    catch ME, \
        fprintf('Error: %s\n', ME.message); \
    end; \
    exit"

echo "=== Finished processing ${CLUSTER_NAME} ==="

