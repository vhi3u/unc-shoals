#!/bin/bash
# run_calc_Q_EKE.sh — Load MATLAB and calculate Q & EKE metrics on Casper
#
# Usage:
#   ./run_calc_Q_EKE.sh periodic_hpc_shoals1
#   ./run_calc_Q_EKE.sh sweep_Hs20_Ls40k

RUN_TAG="${1:?Usage: $0 <run_tag>  (e.g. periodic_hpc_shoals1)}"
DATADIR="/glade/work/nguyen/unc-shoals/nc/"
OUTDIR="/glade/work/nguyen/unc-shoals/animations/"
TOOLDIR="/glade/work/nguyen/unc-shoals/tools/"

# Load modules
module purge
module load ncarenv-basic/23.10
module load ncarenv/23.10
module load matlab/R2024a

echo "=== Running calc_Q_EKE for run_tag: ${RUN_TAG} ==="
echo "    Data dir: ${DATADIR}"

cd "${TOOLDIR}"

matlab -nodisplay -nosplash -r "\
    addpath(genpath('gsw_matlab_v3_06_16')); \
    addpath(genpath('cmocean')); \
    addpath(genpath('pmkmp')); \
    calc_Q_EKE('${RUN_TAG}', \
        'datadir', '${DATADIR}'); \
    exit"

echo "=== Done ==="

