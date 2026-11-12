#!/bin/bash
# run_ke_vs_mean_bulk.sh
#
# Runs the ke_vs_eke_bulk MATLAB script on Casper to compare 
# Total KE and Mean KE for simulations in a cluster.
#
# Usage:
#   ./run_ke_vs_mean_bulk.sh [cluster_name]

CLUSTER_NAME="${1:-cluster13_111_117}"
OUTDIR="/glade/work/nguyen/unc-shoals/animations/"
TOOLDIR="/glade/work/nguyen/unc-shoals/tools/"

# Load environment
module purge
module load ncarenv-basic/23.10
module load ncarenv/23.10
module load matlab/R2024a

echo "=== Running KE vs Mean Comparison Bulk ==="
echo "    Cluster:    ${CLUSTER_NAME}"
echo "    Output dir: ${OUTDIR}"

# Run MATLAB
# Note: Assuming this script is in the same directory as the MATLAB files
# or added to the path.
matlab -nodisplay -nosplash -r "\
    fprintf('--- Processing KE Comparison for ${CLUSTER_NAME} ---\n'); \
    ke_vs_mke_bulk('${CLUSTER_NAME}', '${OUTDIR}'); \
    exit"

echo "=== Finished processing ${CLUSTER_NAME} ==="
