#!/bin/bash
# run_vishoals_reference.sh
#
# Full-panel animation of the sweep reference case (no wind, Zs = -5 m, Ls = 40 km).
#
# Usage:
#   ./run_vishoals_reference.sh [cluster_name]

CLUSTER_NAME="${1:-cluster_Fnnie_1_19}"

module purge
module load ncarenv-basic/23.10
module load ncarenv/23.10
module load matlab/R2024a

echo "=== Animating reference case (Zs5_Ls40k) ==="
echo "    Cluster:          ${CLUSTER_NAME}"
echo "    Output Directory: /glade/work/nguyen/unc-shoals/animations/"

cd "$(dirname "$0")"

matlab -nodisplay -nosplash -r "\
    addpath(genpath('gsw_matlab_v3_06_16')); \
    addpath(genpath('cmocean')); \
    addpath(genpath('pmkmp')); \
    try, \
        vishoals_reference('${CLUSTER_NAME}', 'mp4', true, 'fps', 15, 'sample_hours', 24); \
    catch ME, \
        fprintf('Error: %s\n', ME.message); \
    end; \
    exit"

echo "=== Finished processing ${CLUSTER_NAME} ==="

