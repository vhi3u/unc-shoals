#!/bin/bash
# run_offshore_flux_bulk.sh — Generate clipped heat and salt flux maps for all sweeps
#
# Usage:
#   ./run_offshore_flux_bulk.sh [cluster_name]
#
# Examples:
#   ./run_offshore_flux_bulk.sh                    # defaults to cluster10_81_89
#   ./run_offshore_flux_bulk.sh cluster10_81_89
#   ./run_offshore_flux_bulk.sh cluster07_70_80
CLUSTER_NAME="${1:-cluster10_81_89}"
SWEEP_OUTPUT="/glade/work/nguyen/unc-shoals/sweep_output"
DATADIR="${SWEEP_OUTPUT}/${CLUSTER_NAME}"
TOOLDIR="/glade/work/nguyen/unc-shoals/tools/"
if [ ! -d "${DATADIR}" ]; then
    echo "ERROR: cluster directory not found: ${DATADIR}"
    exit 1
fi
module purge
module load ncarenv-basic/23.10
module load ncarenv/23.10
module load matlab/R2024a
echo "=== Running offshore_flux_bulk ==="
echo "    Cluster:    ${CLUSTER_NAME}"
echo "    Data dir:   ${DATADIR}"
echo "    Output dir: /glade/work/nguyen/unc-shoals/animations/"
cd "${TOOLDIR}"
matlab -nodisplay -nosplash -r "\
    addpath(genpath('gsw_matlab_v3_06_16')); \
    addpath(genpath('cmocean')); \
    offshore_flux_bulk('${CLUSTER_NAME}'); \
    exit"
echo "=== offshore_flux_bulk finished ==="
