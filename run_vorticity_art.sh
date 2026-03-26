#!/bin/bash
# run_vorticity_art.sh — Generate artistic vorticity animation on Casper
#
# Usage:
#   ./run_vorticity_art.sh periodic_hpc_shoals1

RUN_TAG="${1:?Usage: $0 <run_tag> <colormap_command>}"
CMAP="${2:-cmocean('balance')}"
DATADIR="/glade/work/nguyen/unc-shoals/nc/"
OUTDIR="/glade/work/nguyen/unc-shoals/animations/"
TOOLDIR="/glade/work/nguyen/unc-shoals/tools/"

# Load modules
module purge
module load ncarenv-basic/23.10
module load ncarenv/23.10
module load matlab/R2024a

echo "=== Running artistic vorticity visualization for run_tag: ${RUN_TAG} ==="
echo "    Color Map: ${CMAP}"
echo "    Data dir:  ${DATADIR}"
echo "    Output dir: ${OUTDIR}"

cd "${TOOLDIR}"

# Run the artistic visualization script
# Use ${CMAP} without internal quotes so MATLAB evaluates it
matlab -nodisplay -nosplash -r "\
    addpath(genpath('cmocean')); \
    vis_vortex_abstract('filename', 'vortex_abstract_highres.nc', \
        'datadir', '${DATADIR}', \
        'outdir', '${OUTDIR}', \
        'cmap', ${CMAP}, \
        'var', 'c', 'fps', 15); \
    exit"

echo "=== Done ==="
