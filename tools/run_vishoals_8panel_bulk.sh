#!/bin/bash
# run_vishoals_8panel_bulk.sh — animate every sweep in a cluster.
#
# Twelve panels, not eight: six columns (T, S, u cross-shore, v along-shore,
# Ro, KE), each as a surface XY panel and a mid-y XZ panel. The '8panel' in
# the name and the '_14panel' in the output filename are both historical;
# see the note in vishoals_8panel_bulk.m. This is the bulk counterpart of
# run_vishoals.sh.
#
# Usage:
#   ./run_vishoals_8panel_bulk.sh [cluster_name]
#
# Examples:
#   ./run_vishoals_8panel_bulk.sh cluster_weakwinds
#   ./run_vishoals_8panel_bulk.sh cluster_Fnnie_1_19
#
# ALWAYS pass the cluster. The default below is cluster13_111_117, which is
# long dead -- a bare call plots months-old data and reports no error. (The
# comment here used to say the default was cluster10_81_89, which did not
# match the code either.)

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

