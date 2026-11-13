#!/bin/bash
# run_vishoals.sh — full-panel animation for one run, on Casper.
#
# Usage:
#   ./run_vishoals.sh <run_tag> [cluster_name]
#
#   ./run_vishoals.sh periodic_hpc_shoals1              # a run in nc/
#   ./run_vishoals.sh sweep_Zs5_Ls40k cluster_Fnnie_1_19
#   ./run_vishoals.sh sweep_wind_0p010 cluster_weakwinds
#   ./run_vishoals.sh sweep_Zs15_Lsfix                  # found by search
#
# NOTE this takes a RUN TAG (one run), not a cluster -- unlike the bulk
# runners, which take a cluster and loop. The optional second argument only
# says WHERE to look for that run.
#
# Where the data is found, in order:
#   1. the named cluster:   sweep_output/<cluster>/sweep_NNN_<label>/
#   2. nc/                  (hand-placed runs, the historical default)
#   3. a search of          sweep_output/*/<run>/ and sweep_output/*/*/<run>/
#
# Step 3 is what makes a freshly finished sweep work with no extra argument:
# sweep_driver.jl writes to sweep_output/sweep_NNN_<label>/ and nothing moves
# it into a cluster directory, so uncollected runs are found in place. If the
# tag matches in more than one cluster the script lists them and stops rather
# than guessing.
#
# Panels: 'full_panel' true is the 6-column layout -- T, S, u, v, Ro, KE, each
# as surface XY and mid-y XZ. That is every panel vishoals.m can draw. The
# function's own default is 4 columns, so this must be passed explicitly.
# A run whose top_*.nc has no T/S drops to 4 columns (u, v, Ro, KE) on its
# own and says so; see the no_TS branch in vishoals.m.

set -u

RUN_TAG="${1:?Usage: $0 <run_tag> [cluster_name]  (e.g. sweep_Zs5_Ls40k cluster_Fnnie_1_19)}"
CLUSTER="${2:-}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SWEEP_ROOT="${ROOT}/sweep_output"
OUTDIR="${ROOT}/animations/"
TOOLDIR="${ROOT}/tools"

shopt -s nullglob

# A directory holds this run if BOTH slices are there -- vishoals.m reads
# top_ for the XY panels and midy_ for the XZ row and hard-errors on either.
has_run() {
    [ -f "$1/top_${RUN_TAG}.nc" ] && [ -f "$1/midy_${RUN_TAG}.nc" ]
}

# "not found" and "found but incomplete" are different problems with the same
# symptom. A run that is still writing, or that blew up mid-sweep, leaves one
# slice behind -- say which one is missing instead of reporting the run absent.
report_partial() {
    for d in "$1"/*/; do
        d="${d%/}"
        t="$d/top_${RUN_TAG}.nc"
        m="$d/midy_${RUN_TAG}.nc"
        if { [ -f "$t" ] || [ -f "$m" ]; } && ! has_run "$d"; then
            echo "  Incomplete run in $(basename "$d"):" >&2
            [ -f "$t" ] || echo "    missing top_${RUN_TAG}.nc  (XY panels)" >&2
            [ -f "$m" ] || echo "    missing midy_${RUN_TAG}.nc (XZ row)" >&2
            echo "  Still running, or it stopped before the writer flushed." >&2
        fi
    done
}

DATADIR=""
if [ -n "${CLUSTER}" ]; then
    for d in "${SWEEP_ROOT}/${CLUSTER}"/*/; do
        if has_run "${d%/}"; then DATADIR="${d%/}"; break; fi
    done
    if [ -z "${DATADIR}" ]; then
        echo "ERROR: ${RUN_TAG} not found in ${SWEEP_ROOT}/${CLUSTER}" >&2
        if [ -d "${SWEEP_ROOT}/${CLUSTER}" ]; then
            report_partial "${SWEEP_ROOT}/${CLUSTER}"
            echo "Runs in that cluster:" >&2
            for d in "${SWEEP_ROOT}/${CLUSTER}"/sweep_*/; do
                echo "    $(basename "${d%/}")" >&2
            done
        else
            echo "That cluster directory does not exist." >&2
        fi
        exit 1
    fi
elif has_run "${ROOT}/nc"; then
    DATADIR="${ROOT}/nc"
else
    # Both layouts: collected into a cluster, and straight out of the driver.
    MATCHES=()
    for d in "${SWEEP_ROOT}"/*/ "${SWEEP_ROOT}"/*/*/; do
        if has_run "${d%/}"; then MATCHES+=("${d%/}"); fi
    done
    if [ ${#MATCHES[@]} -eq 0 ]; then
        echo "ERROR: no top_${RUN_TAG}.nc + midy_${RUN_TAG}.nc anywhere under" >&2
        echo "    ${ROOT}/nc" >&2
        echo "    ${SWEEP_ROOT}" >&2
        report_partial "${SWEEP_ROOT}"
        for c in "${SWEEP_ROOT}"/*/; do report_partial "${c%/}"; done
        echo "Check the tag. Sweep runs are tagged sweep_<label>, e.g. sweep_Zs5_Ls40k." >&2
        exit 1
    elif [ ${#MATCHES[@]} -gt 1 ]; then
        echo "ERROR: ${RUN_TAG} exists in ${#MATCHES[@]} places -- name the cluster:" >&2
        for m in "${MATCHES[@]}"; do echo "    ${m}" >&2; done
        exit 1
    fi
    DATADIR="${MATCHES[0]}"
fi

# Load modules
module purge
module load ncarenv-basic/23.10
module load ncarenv/23.10
module load matlab/R2024a

echo "=== Running vishoals for run_tag: ${RUN_TAG} ==="
echo "    Data dir:   ${DATADIR}"
echo "    Output dir: ${OUTDIR}"
echo "    Panels:     full_panel (6 columns: T, S, u, v, Ro, KE)"

cd "${TOOLDIR}"

# 'vel' and 'vort' are NOT passed. They pick the single velocity and
# vorticity column of the 4-column layout, and are read only in the
# non-full_panel branch (vishoals.m:810, :837; the climit scan at :309 and
# :314 is gated on ~full_panel). Under full_panel the columns are fixed at
# u_c, v_c, Ro, KE, so the old 'vel', 'Ro' here did nothing -- Ro is drawn
# as column 5 regardless.
matlab -nodisplay -nosplash -r "\
    addpath(genpath('gsw_matlab_v3_06_16')); \
    addpath(genpath('cmocean')); \
    addpath(genpath('pmkmp')); \
    try, \
        vishoals('${RUN_TAG}', \
            'datadir', '${DATADIR}', \
            'outdir', '${OUTDIR}', \
            'full_panel', true, \
            'fps', 15, \
            'clipped', true); \
    catch ME, \
        fprintf('Error: %s\n', ME.message); \
        fprintf('%s\n', getReport(ME, 'extended')); \
    end; \
    exit"

echo "=== Done ==="
