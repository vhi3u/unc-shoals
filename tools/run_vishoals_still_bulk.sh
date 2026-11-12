#!/bin/bash
# run_vishoals_still_bulk.sh
#
# Runs vishoals_still_bulk in MATLAB on Casper/HPC.
# Generates 8-panel still images at day 100 for all sweeps.
#
# Usage:
#   ./run_vishoals_still_bulk.sh [cluster_name]

CLUSTER_NAME=${1:-"cluster13_111_117"}

# Module environment for Casper (NCAR)
module purge
module load ncarenv-basic/23.10
module load ncarenv/23.10
module load matlab/R2024a

echo "=== Running vishoals_still_bulk ==="
echo "    Cluster: $CLUSTER_NAME"
echo "    Output:  /glade/work/nguyen/unc-shoals/animations/"

# Ensure we are in the directory containing the script
cd "$(dirname "$0")"

# Execute MATLAB
matlab -nodisplay -nosplash -r " \
    addpath(genpath('gsw_matlab_v3_06_16')); \
    addpath(genpath('cmocean')); \
    addpath(genpath('pmkmp')); \
    try, \
        vishoals_still_bulk('$CLUSTER_NAME'); \
    catch ME, \
        fprintf('Fatal Error: %s\n', ME.message); \
    end, \
    exit"

echo "=== Done ==="
echo "Check /glade/work/nguyen/unc-shoals/animations/ for resulting .png files."
