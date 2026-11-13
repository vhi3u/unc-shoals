#!/bin/bash
# run_bench_animation.sh
#
# Measure where vishoals spends its animation time, before changing anything.
#
# Usage:
#   ./run_bench_animation.sh <run_tag> [cluster_name] [n_frames]
#
#   ./run_bench_animation.sh sweep_wind_m0p020 cluster_weakwinds
#   ./run_bench_animation.sh sweep_Zs5_Ls40k cluster_Fnnie_1_19 20
#
# Prints a per-stage breakdown (NetCDF read / set CData / drawnow / print /
# crop / writeVideo), the cost of the separate colour-limit pass, how print
# scales with dpi, and a projection to a full 50-day run at 6-hourly vs
# daily frame sampling.
#
# Reads nothing but the run's top_/midy_ NetCDF and writes only a throwaway
# AVI under out/bench, so it is safe to run alongside a bulk animation job.
# It will CONTEND FOR CPU with one, though -- numbers from a busy node read
# slow across the board. Prefer a quiet node, or at least compare runs taken
# under the same load.
#
# ENDTOEND=1 additionally times two real vishoals calls over the last 5 days,
# with and without 'sample_hours', 24 -- slower, but it measures the actual
# function rather than a reconstruction of its inner loop.

set -u

RUN_TAG="${1:?Usage: $0 <run_tag> [cluster_name] [n_frames]}"
CLUSTER="${2:-cluster_weakwinds}"
NFRAMES="${3:-12}"
ENDTOEND="${ENDTOEND:-0}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

module purge
module load ncarenv-basic/23.10
module load ncarenv/23.10
module load matlab/R2024a

# ffmpeg is not needed by the benchmark itself (it writes AVI and deletes it),
# but ENDTOEND calls vishoals, which transcodes.
module load ffmpeg 2>/dev/null

echo "=== bench_animation ==="
echo "    Run:      ${RUN_TAG}"
echo "    Cluster:  ${CLUSTER}"
echo "    Frames:   ${NFRAMES}"
echo "    Endtoend: ${ENDTOEND}"

cd "$(dirname "$0")"

if [ "${ENDTOEND}" = "1" ]; then E2E="true"; else E2E="false"; fi

matlab -nodisplay -nosplash -r "\
    addpath(genpath('cmocean')); \
    addpath(genpath('pmkmp')); \
    addpath(genpath('gsw_matlab_v3_06_16')); \
    try, \
        bench_animation('${RUN_TAG}', 'cluster', '${CLUSTER}', \
                        'n_frames', ${NFRAMES}, 'endtoend', ${E2E}); \
    catch ME, \
        fprintf('Error: %s\n', ME.message); \
        fprintf('%s\n', getReport(ME, 'extended')); \
    end; \
    exit"

echo "=== done ==="
