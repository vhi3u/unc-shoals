#!/bin/bash
# run_theory_vs_model.sh
#
# Generates the theory vs model comparison (sqrt(EKE) and offshore Q) on Casper.
#
# Usage:
#   ./run_theory_vs_model.sh [cluster_name]

CLUSTER_NAME="${1:-cluster_Fnnie_1_19}"

# Load NCAR Casper environment
module purge
module load ncarenv-basic/23.10
module load ncarenv/23.10
module load matlab/R2024a

echo "=== Running Theory vs Model Comparison ==="
echo "    Cluster:          ${CLUSTER_NAME}"
echo "    Output Directory: /glade/work/nguyen/unc-shoals/animations/"

# Run from the tools directory so the toolbox paths resolve
cd "$(dirname "$0")"

# Run MATLAB in non-interactive mode
#   plot_theory_vs_model(cluster, coeff_mode, C_geom, include_wind)
#     coeff_mode:   'regime' | 'baseline' | 'fit'
#     C_geom:       1/2 (rectangular layer) or 1/6 (triangular wedge)
#     include_wind: true / false
matlab -nodisplay -nosplash -r "\
    addpath(genpath('gsw_matlab_v3_06_16')); \
    try, \
        fprintf('--- Generating theory vs model for ${CLUSTER_NAME} ---\n'); \
        plot_theory_vs_model('${CLUSTER_NAME}', 'fit', 1/2, false); \
    catch ME, \
        fprintf('Error: %s\n', ME.message); \
    end; \
    exit"

echo "=== Finished processing ${CLUSTER_NAME} ==="

