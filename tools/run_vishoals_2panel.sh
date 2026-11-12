#!/bin/bash
# run_vishoals_2panel.sh — Load MATLAB and generate 2-panel shoals visualizations on Casper
#
# Usage:
#   ./run_vishoals_2panel.sh periodic_hpc_shoals1
#   ./run_vishoals_2panel.sh bounded_hpc_shoals2

RUN_TAG="${1:?Usage: $0 <run_tag>  (e.g. periodic_hpc_shoals1)}"
DATADIR="/glade/work/nguyen/unc-shoals/nc/"
OUTDIR="/glade/work/nguyen/unc-shoals/animations/"
TOOLDIR="/glade/work/nguyen/unc-shoals/tools/"

# Load modules
module purge
module load ncarenv-basic/23.10
module load ncarenv/23.10
module load matlab/R2024a

echo "=== Running vishoals_2panel for run_tag: ${RUN_TAG} ==="
echo "    Data dir:  ${DATADIR}"
echo "    Output dir: ${OUTDIR}"

cd "${TOOLDIR}"

matlab -nodisplay -nosplash -r "\
    addpath(genpath('gsw_matlab_v3_06_16')); \
    addpath(genpath('cmocean')); \
    vishoals_2panel('${RUN_TAG}', \
        'datadir', '${DATADIR}', \
        'outdir', '${OUTDIR}', \
	'velocity_field', 'v'); \
    exit"

echo "=== Done ==="

