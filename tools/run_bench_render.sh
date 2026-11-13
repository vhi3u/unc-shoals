#!/bin/bash
# run_bench_render.sh
#
# Follow-up to run_bench_animation.sh. That run showed print() is 85% of a
# frame AND that the cost barely moves with dpi (-r50 only 14% cheaper than
# -r100), which rules out rasterization and points at the painters renderer
# emitting one vector quad per data cell -- ~492,000 of them per frame here,
# at any resolution.
#
# This crosses three plot constructions (pcolor+flat, pcolor+interp as
# vishoals.m:459 actually uses, and imagesc) with three renderer choices
# (default, -painters, -opengl) and times print for each, so the cause is
# measured rather than assumed. It also reports whether OpenGL is available
# at all under -nodisplay, and how different an imagesc frame really looks
# from a pcolor one.
#
# Usage:
#   ./run_bench_render.sh <run_tag> [cluster_name] [reps]
#
#   ./run_bench_render.sh sweep_wind_m0p020 cluster_weakwinds
#   ./run_bench_render.sh sweep_Zs5_Ls40k cluster_Fnnie_1_19 9
#
# Renders only; writes no video and no figure. One frame of data is read and
# re-rendered, so numbers are not polluted by NetCDF I/O.

set -u

RUN_TAG="${1:?Usage: $0 <run_tag> [cluster_name] [reps]}"
CLUSTER="${2:-cluster_weakwinds}"
REPS="${3:-5}"

module purge
module load ncarenv-basic/23.10
module load ncarenv/23.10
module load matlab/R2024a

echo "=== bench_render ==="
echo "    Run:     ${RUN_TAG}"
echo "    Cluster: ${CLUSTER}"
echo "    Reps:    ${REPS}"

cd "$(dirname "$0")"

matlab -nodisplay -nosplash -r "\
    addpath(genpath('cmocean')); \
    addpath(genpath('pmkmp')); \
    try, \
        bench_render('${RUN_TAG}', 'cluster', '${CLUSTER}', 'reps', ${REPS}); \
    catch ME, \
        fprintf('Error: %s\n', ME.message); \
        fprintf('%s\n', getReport(ME, 'extended')); \
    end; \
    exit"

echo "=== done ==="
