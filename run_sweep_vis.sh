#!/bin/bash
# run_sweep_vis.sh — Generate shoals visualizations for one sweep run
#   Runs: vishoals (8-panel animation) + offshore_flux + vis_ke_eke
#
# Usage:
#   ./run_sweep_vis.sh sweep_001_Hs05_Ls20k

set -euo pipefail

RUN_TAG="${1:?Usage: $0 <run_tag>  (e.g. sweep_001_Hs05_Ls20k)}"
DATADIR="/glade/work/nguyen/unc-shoals/sweep_output/${RUN_TAG}"
OUTDIR="/glade/work/nguyen/unc-shoals/animations"
TOOLDIR="/glade/work/nguyen/unc-shoals/tools"

if [ ! -d "${DATADIR}" ]; then
    echo "ERROR: Data directory not found: ${DATADIR}"
    exit 1
fi

# ---------- module setup ----------
module purge
module load ncarenv-basic/23.10
module load ncarenv/23.10
module load matlab/R2024a

echo "═══════════════════════════════════════════════════════"
echo "  Run tag:    ${RUN_TAG}"
echo "  Data dir:   ${DATADIR}"
echo "  Output dir: ${OUTDIR}"
echo "═══════════════════════════════════════════════════════"

mkdir -p "${OUTDIR}"
cd "${TOOLDIR}"

matlab -nodisplay -nosplash -r "\
    addpath(genpath('gsw_matlab_v3_06_16')); \
    addpath(genpath('cmocean')); \
    try; \
        vishoals('${RUN_TAG}', \
            'datadir', '${DATADIR}', \
            'outdir', '${OUTDIR}', \
            'isopycnals', true, \
            'rho_levels', 22.5:0.1:25.0, 'fps', 5); \
    catch e; \
        fprintf('vishoals failed: %s\n', e.message); \
    end; \
    try; \
        offshore_flux('${RUN_TAG}', \
            'datadir', '${DATADIR}', \
            'outdir', '${OUTDIR}'); \
    catch e; \
        fprintf('offshore_flux failed: %s\n', e.message); \
    end; \
    try; \
        vis_ke_eke('${RUN_TAG}', \
            'datadir', '${DATADIR}', \
            'outdir', '${OUTDIR}'); \
    catch e; \
        fprintf('vis_ke_eke failed: %s\n', e.message); \
    end; \
    exit"

echo "=== Done: ${RUN_TAG} ==="

