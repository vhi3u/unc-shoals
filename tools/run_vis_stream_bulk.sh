#!/bin/bash
# run_vis_streamfunction_xz_bulk.sh
#
# Runs vis_streamfunction_xz_bulk in MATLAB on Casper/HPC.
# Generates streamfunction plots overlaid on density for all sweeps.
#
# Usage:
#   ./run_vis_streamfunction_xz_bulk.sh [cluster_name]

CLUSTER_NAME=${1:-"cluster13_111_117"}

# Module environment for Casper (NCAR)
module purge
module load ncarenv-basic/23.10
module load ncarenv/23.10
module load matlab/R2024a

echo "=== Running vis_streamfunction_xz_bulk ==="
echo "    Cluster: $CLUSTER_NAME"
echo "    Output:  /glade/work/nguyen/unc-shoals/animations/"

# Ensure we are in the directory containing the script
cd "$(dirname "$0")"

# Execute MATLAB
matlab -nodisplay -nosplash -r " \
    addpath(genpath('gsw_matlab_v3_06_16')); \
    addpath(genpath('cmocean')); \
    try, \
        vis_streamfunction_xz_bulk('$CLUSTER_NAME'); \
    catch ME, \
        fprintf('Fatal Error: %s\n', ME.message); \
    end, \
    exit"

echo "=== Done ==="
echo "Check /glade/work/nguyen/unc-shoals/animations/ for resulting .png files."

