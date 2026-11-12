#!/bin/bash
# run_vishoals_8panel_bulk.sh — Generate 8-panel (T,S,u,Ro) animations for all sweeps
#
# Usage:
#   ./run_vishoals_8panel_bulk.sh [cluster_name]
#
# Examples:
#   ./run_vishoals_8panel_bulk.sh                    # defaults to cluster10_81_89
#   ./run_vishoals_8panel_bulk.sh cluster10_81_89
#   ./run_vishoals_8panel_bulk.sh cluster07_70_80

CLUSTER_NAME="${1:-cluster13_111_117}"

SWEEP_OUTPUT="/glade/work/nguyen/unc-shoals/sweep_output"
DATADIR="${SWEEP_OUTPUT}/${CLUSTER_NAME}"
TOOLDIR="/glade/work/nguyen/unc-shoals/tools/"

if [ ! -d "${DATADIR}" ]; then
    echo "ERROR: cluster directory not found: ${DATADIR}"
    exit 1
fi

module --force purge
module load ncarenv-basic/23.10
module load ncarenv/23.10
module load matlab/R2024a

echo "=== Running vishoals_8panel_bulk ==="
echo "    Cluster:    ${CLUSTER_NAME}"
echo "    Data dir:   ${DATADIR}"
echo "    Output dir: /glade/work/nguyen/unc-shoals/animations/"

cd "${TOOLDIR}"

matlab -nodisplay -nosplash -r "\
    addpath(genpath('gsw_matlab_v3_06_16')); \
    addpath(genpath('cmocean')); \
    addpath(genpath('pmkmp')); \
    vishoals_8panel_bulk('${CLUSTER_NAME}'); \
    exit"

echo "=== vishoals_8panel_bulk finished ==="

