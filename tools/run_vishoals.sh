#!/bin/bash
# run_vishoals.sh — Load MATLAB and generate shoals visualizations on Casper
#
# Usage:
#   ./run_vishoals.sh periodic_hpc_shoals1
#   ./run_vishoals.sh bounded_hpc_shoals2

RUN_TAG="${1:?Usage: $0 <run_tag>  (e.g. periodic_hpc_shoals1)}"
DATADIR="/glade/work/nguyen/unc-shoals/nc/"
OUTDIR="/glade/work/nguyen/unc-shoals/animations/"
TOOLDIR="/glade/work/nguyen/unc-shoals/tools/"

# Load modules
module purge
module load ncarenv-basic/23.10
module load ncarenv/23.10
module load matlab/R2024a

echo "=== Running vishoals for run_tag: ${RUN_TAG} ==="
echo "    Data dir:  ${DATADIR}"
echo "    Output dir: ${OUTDIR}"

cd "${TOOLDIR}"

matlab -nodisplay -nosplash -r "\
    addpath(genpath('gsw_matlab_v3_06_16')); \
    addpath(genpath('cmocean')); \
    addpath(genpath('pmkmp')); \
    vishoals('${RUN_TAG}', \
        'datadir', '${DATADIR}', \
        'outdir', '${OUTDIR}', \
	'vel', 'Ro', \
	'full_panel', true, \
	'fps', 15, \
	'clipped', true); \
    exit"

echo "=== Done ==="
