#!/bin/bash
# run_ape_wedge.sh
#
# Measure APE directly instead of assuming the wedge geometry, so that
# C_geom = APE / (g' H0^2 Ld) is diagnosed rather than chosen.
#
# Usage:
#   ./run_ape_wedge.sh [cluster_name] [window]
#
#   ./run_ape_wedge.sh cluster_geometry_120       # last 10-day average
#   ./run_ape_wedge.sh cluster_res250_120 10      # the 10th window
#
# Zhao et al. Eq. 8 assumes APE_D = C_geom g' H0^2 Ld with C_geom = 1/6 (a
# triangular plume wedge). That shape is a guess about OUR geometry. This
# computes APE two independent ways from time_avg_3d_<tag>.nc -- the
# quasi-geostrophic form and the Lorenz sorted form -- and reports the
# C_geom each one implies, with Ld and with the measured anomaly width.
#
# Reads only T and S, so it needs no rerun.
#
# Prints a table and writes two figures to animations/:
#   <cluster>_ape_wedge.png     the b' cross-section, i.e. the actual shape
#   <cluster>_ape_scaling.png   APE vs g' H0^2 L, where a constant C_geom
#                               shows up as one line through the origin

set -u

CLUSTER="${1:-cluster_geometry_120}"
WINDOW="${2:-}"

module purge
module load ncarenv-basic/23.10
module load ncarenv/23.10
module load matlab/R2024a

echo "=== APE wedge ==="
echo "    Cluster: ${CLUSTER}"
echo "    Window:  ${WINDOW:-last}"

cd "$(dirname "$0")"

if [ -n "${WINDOW}" ]; then WARG=", 'window', ${WINDOW}"; else WARG=""; fi

matlab -nodisplay -nosplash -r "\
    try, \
        ape_wedge('${CLUSTER}'${WARG}); \
    catch ME, \
        fprintf('Error: %s\n', ME.message); \
        fprintf('%s\n', getReport(ME, 'extended')); \
    end; \
    exit"

echo "=== done ==="
