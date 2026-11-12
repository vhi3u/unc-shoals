#!/bin/bash
# run_density_bulk.sh — Generate density and gradient maps for all sweeps
#
# Usage:
#   ./run_density_bulk.sh [cluster_name]

CLUSTER_NAME="${1:-cluster13_111_117}"

SWEEP_OUTPUT="/glade/work/nguyen/unc-shoals/sweep_output"
DATADIR="${SWEEP_OUTPUT}/${CLUSTER_NAME}"
TOOLDIR="/glade/work/nguyen/unc-shoals/tools/"
# NOTE: there was a VISDIR here pointing at /glade/work/nguyen/vnmain/matlabvis.
# That path does not exist on Casper -- matlabvis is a laptop-only directory --
# so the addpath below silently did nothing and this script has been running on
# whatever happened to be in TOOLDIR. All MATLAB code now lives in TOOLDIR.
OUTDIR="/glade/work/nguyen/unc-shoals/animations/"

if [ ! -d "${DATADIR}" ]; then
    echo "ERROR: cluster directory not found: ${DATADIR}"
    exit 1
fi

module purge
module load ncarenv-basic/23.10
module load ncarenv/23.10
module load matlab/R2024a

echo "=== Running Density & Gradient Visualization Pipeline ==="
echo "    Cluster:    ${CLUSTER_NAME}"
echo "    Data dir:   ${DATADIR}"

cd "${TOOLDIR}"

matlab -nodisplay -nosplash -r "\
    addpath(genpath('gsw_matlab_v3_06_16')); \
    addpath(genpath('cmocean')); \
    fprintf('--- Starting Density Diagnostics (4-panel) ---\n'); \
    vis_density_diagnostics_bulk('${CLUSTER_NAME}'); \
    fprintf('--- Starting Density Gradient Mapping (Max drho/dz & g-prime) ---\n'); \
    vis_density_gradients_bulk('${CLUSTER_NAME}', '${OUTDIR}'); \
    exit"

echo "=== Density Visualization Pipeline finished ==="
