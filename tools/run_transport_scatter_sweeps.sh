#!/bin/bash
# run_transport_scatter_sweeps.sh
#
# Runs the transport scatter plot diagnostic on Casper.
#
# Usage:
#   ./run_transport_scatter_sweeps.sh [cluster_name]

# Default to cluster13_111_117 if no argument provided
CLUSTER_NAME="${1:-cluster13_111_117}"

# Load NCAR Casper environment
module purge
module load ncarenv-basic/23.10
module load ncarenv/23.10
module load matlab/R2024a

echo "=== Running Volume Transport Scatter Diagnostic ==="
echo "    Cluster:          ${CLUSTER_NAME}"
echo "    Output Directory: /glade/work/nguyen/unc-shoals/animations/"

# Run MATLAB in non-interactive mode
matlab -nodisplay -nosplash -r "\
    try, \
        fprintf('--- Generating Transport Scatter for ${CLUSTER_NAME} ---\n'); \
        plot_transport_scatter_sweeps('${CLUSTER_NAME}'); \
    catch ME, \
        fprintf('Error: %s\n', ME.message); \
    end; \
    exit"

echo "=== Finished processing ${CLUSTER_NAME} ==="
