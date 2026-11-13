#!/bin/bash
# run_eke_geometry_controls.sh
#
# EKE contrasts for the Fig. 4b geometry controls (params_geomcontrol.toml).
#
# Usage:
#   ./run_eke_geometry_controls.sh [cluster_name] [window]
#
#   ./run_eke_geometry_controls.sh cluster_geometry        # last 10-day average
#   ./run_eke_geometry_controls.sh cluster_geometry 4      # the 4th window
#
# Reduces each run to one EKE number and forms the pairwise contrasts that
# separate shoal DEPTH from the two things that covary with it -- crest
# length and inshore flank slope. Reads the ONLINE 10-day averages in
# time_avg_3d_<tag>.nc, so the answer does not depend on snapshot sampling.
#
# EKE is defined exactly as in vis_eke_sweeps.m, so the numbers are directly
# comparable with Fig. 4b.

set -u

CLUSTER="${1:-cluster_geometry}"
WINDOW="${2:-}"

module purge
module load ncarenv-basic/23.10
module load ncarenv/23.10
module load matlab/R2024a

echo "=== EKE geometry controls ==="
echo "    Cluster: ${CLUSTER}"
echo "    Window:  ${WINDOW:-last}"

cd "$(dirname "$0")"

if [ -n "${WINDOW}" ]; then WARG=", 'window', ${WINDOW}"; else WARG=""; fi

matlab -nodisplay -nosplash -r "\
    try, \
        eke_geometry_controls('${CLUSTER}'${WARG}); \
    catch ME, \
        fprintf('Error: %s\n', ME.message); \
        fprintf('%s\n', getReport(ME, 'extended')); \
    end; \
    exit"

echo "=== done ==="
