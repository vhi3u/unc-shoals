#!/bin/bash
# run_eke_map.sh — Generate EKE map for a single run
#
# Usage:
#   ./run_eke_map.sh <run_tag>
#
# Examples:
#   ./run_eke_map.sh run52
#   ./run_eke_map.sh run53

if [ $# -lt 1 ]; then
    echo "Usage: $0 <run_tag>"
    exit 1
fi

RUN_TAG="$1"
CLUSTER_NAME="${2:-cluster13_111_117}"

SWEEP_OUTPUT="/glade/work/nguyen/unc-shoals/nc/"
DATADIR="${SWEEP_OUTPUT}"
TOOLDIR="/glade/work/nguyen/unc-shoals/tools/"

if [ ! -d "${DATADIR}" ]; then
    echo "ERROR: run directory not found: ${DATADIR}"
    exit 1
fi

module purge
module load ncarenv-basic/23.10
module load ncarenv/23.10
module load matlab/R2024a

echo "=== Running eke_map ==="
echo "    Run tag:    ${RUN_TAG}"
echo "    Data dir:   ${DATADIR}"
echo "    Output dir: /glade/work/nguyen/unc-shoals/animations/"

cd "${TOOLDIR}"

matlab -nodisplay -nosplash -r "\
    addpath(genpath('gsw_matlab_v3_06_16')); \
    addpath(genpath('cmocean')); \
    eke_map('${RUN_TAG}', 'datadir', '${DATADIR}'); \
    exit"

echo "=== eke_map finished ==="

