#!/bin/bash
# run_offshore_velocity_comparison.sh
#
# Runs the MATLAB comparison script to generate depth & time averaged 
# offshore velocity profiles (u at x=50km) for selected sweeps.
#
# Usage:
#   ./run_offshore_velocity_comparison.sh [cluster_name]

CLUSTER_NAME="${1:-cluster13_111_117}"
TOOLDIR="/glade/work/nguyen/unc-shoals/tools/" # Adjust if scripts are elsewhere

# Module environment (standard for NCAR/Casper/Derecho)
module purge
module load ncarenv-basic/23.10
module load ncarenv/23.10
module load matlab/R2024a

echo "=== Running Offshore Velocity Comparison ==="
echo "    Cluster:    ${CLUSTER_NAME}"
echo "    Script:     plot_offshore_velocity_sweeps.m"

# Ensure we are in the directory with the MATLAB scripts
# or add the path explicitly inside MATLAB.
cd "$(dirname "$0")"

matlab -nodisplay -nosplash -r "\
    addpath(genpath('cmocean')); \
    try, \
        plot_offshore_velocity_sweeps('${CLUSTER_NAME}'); \
    catch ME, \
        fprintf('Error: %s\n', ME.message); \
    end, \
    exit"

echo "=== Done ==="
echo "Output should be in: /glade/work/nguyen/unc-shoals/animations/offshore_velocity_comparison.png"
