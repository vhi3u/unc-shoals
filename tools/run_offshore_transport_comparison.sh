#!/bin/bash
# run_offshore_transport_comparison.sh
#
# Runs the MATLAB comparison script to generate depth-integrated 
# offshore transport profiles (u integrated over depth at x=50km).
#
# Usage:
#   ./run_offshore_transport_comparison.sh [cluster_name]

CLUSTER_NAME="${1:-cluster13_111_117}"

# Module environment (standard for NCAR/Casper/Derecho)
module purge
module load ncarenv-basic/23.10
module load ncarenv/23.10
module load matlab/R2024a

echo "=== Running Offshore Transport Comparison ==="
echo "    Cluster:    ${CLUSTER_NAME}"
echo "    Script:     plot_offshore_transport_sweeps.m"

# Ensure we are in the directory with the MATLAB scripts
cd "$(dirname "$0")"

matlab -nodisplay -nosplash -r "\
    try, \
        plot_offshore_transport_sweeps('${CLUSTER_NAME}'); \
    catch ME, \
        fprintf('Error: %s\n', ME.message); \
    end, \
    exit"

echo "=== Done ==="
echo "Outputs:"
echo "  1. Profiles: /glade/work/nguyen/unc-shoals/animations/offshore_transport_profiles.png"
echo "  2. Summary:  /glade/work/nguyen/unc-shoals/animations/offshore_transport_summary.png"
