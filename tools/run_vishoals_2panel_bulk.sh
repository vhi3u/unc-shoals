#!/bin/bash
# run_vishoals_2panel_bulk.sh — Generate 2-panel animations for all sweeps
#
# Usage:
#   ./run_vishoals_2panel_bulk.sh [cluster_name]
#
# Examples:
#   ./run_vishoals_2panel_bulk.sh                    # defaults to cluster10_81_89
#   ./run_vishoals_2panel_bulk.sh cluster10_81_89
#   ./run_vishoals_2panel_bulk.sh cluster07_70_80

CLUSTER_NAME="${1:-cluster10_81_89}"

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

# ffmpeg must be on PATH BEFORE MATLAB starts. MATLAB's VideoWriter has no
# MPEG-4 encoder on Linux, so vishoals writes Motion-JPEG AVI and then calls
# ffmpeg to transcode it to H.264. Without this line the transcode is skipped
# and a much larger .avi is left behind -- which is what was happening.
module load ffmpeg 2>/dev/null
command -v ffmpeg >/dev/null 2>&1 || echo \
    "WARNING: ffmpeg not found; animations will be left as .avi. Try 'module avail ffmpeg'." 

echo "=== Running vishoals_2panel_bulk ==="
echo "    Cluster:    ${CLUSTER_NAME}"
echo "    Data dir:   ${DATADIR}"
echo "    Output dir: /glade/work/nguyen/unc-shoals/tools/"

cd "${TOOLDIR}"

matlab -nodisplay -nosplash -r "\
    addpath(genpath('gsw_matlab_v3_06_16')); \
    addpath(genpath('cmocean')); \
    addpath(genpath('pmkmp')); \
    vishoals_2panel_bulk('${CLUSTER_NAME}'); \
    exit"

echo "=== vishoals_2panel_bulk finished ==="

