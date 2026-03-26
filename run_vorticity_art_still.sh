#!/bin/bash
# run_vorticity_art_still.sh — Generate a high-res still image on Casper
#
# Usage:
#   ./run_vorticity_art_still.sh periodic_hpc_shoals1

RUN_TAG="${1:?Usage: $0 <run_tag>  (e.g. periodic_hpc_shoals1)}"
DATADIR="/glade/work/nguyen/unc-shoals/nc/"
OUTDIR="/glade/work/nguyen/unc-shoals/figures/"
TOOLDIR="/glade/work/nguyen/unc-shoals/tools/"

# Load modules
module purge
module load ncarenv-basic/23.10
module load ncarenv/23.10
module load matlab/R2024a

echo "=== Capturing high-res still for run_tag: ${RUN_TAG} ==="
echo "    Data dir:  ${DATADIR}"
echo "    Output dir: ${OUTDIR}"

cd "${TOOLDIR}"

# Run the high-res capture script
# var 'c' is the dye (best for stills)
matlab -nodisplay -nosplash -r "\
    addpath(genpath('cmocean')); \
    vis_vortex_abstract_still('filename', 'vortex_abstract_highres.nc', \
        'datadir', '${DATADIR}', \
        'outdir', '${OUTDIR}', \
        'var', 'c'); \
    exit"

echo "=== Done ==="
