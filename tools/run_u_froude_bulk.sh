#!/bin/bash
# run_u_froude_bulk.sh — Generate diagnostic and visualization maps for all sweeps
#
# Usage:
#   ./run_u_froude_bulk.sh [cluster_name]

CLUSTER_NAME="${1:-cluster13_111_117}"

SWEEP_OUTPUT="/glade/work/nguyen/unc-shoals/sweep_output"
DATADIR="${SWEEP_OUTPUT}/${CLUSTER_NAME}"
TOOLDIR="/glade/work/nguyen/unc-shoals/tools/"
# NOTE: a VISDIR/addpath pointing at /glade/work/nguyen/vnmain/matlabvis
# was removed here. That path does not exist on Casper (matlabvis is
# laptop-only), so the addpath was always a silent no-op. All MATLAB
# code now lives alongside this script in tools/.

if [ ! -d "${DATADIR}" ]; then
    echo "ERROR: cluster directory not found: ${DATADIR}"
    exit 1
fi

module purge
module load ncarenv-basic/23.10
module load ncarenv/23.10
module load matlab/R2024a

echo "=== Running Bulk Visualization Pipeline ==="
echo "    Cluster:    ${CLUSTER_NAME}"
echo "    Data dir:   ${DATADIR}"

cd "${TOOLDIR}"

matlab -nodisplay -nosplash -r "\
    addpath(genpath('gsw_matlab_v3_06_16')); \
    addpath(genpath('cmocean')); \
    fprintf('--- Starting Density Diagnostics ---\n'); \
    vis_density_diagnostics_bulk('${CLUSTER_NAME}'); \
    fprintf('--- Starting Mean U (Surface) ---\n'); \
    vis_u_mean_bulk('${CLUSTER_NAME}', 0); \
    fprintf('--- Starting Mean U (z = -20m) ---\n'); \
    vis_u_mean_bulk('${CLUSTER_NAME}', -20); \
    fprintf('--- Starting Froude Bulk ---\n'); \
    vis_froude_bulk('${CLUSTER_NAME}'); \
    exit"

echo "=== Bulk Visualization Pipeline finished ==="

