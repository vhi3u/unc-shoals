#!/bin/bash
# run_vishoals_vort_wind.sh
#
# Surface vorticity (Ro = zeta/f) stills across a wind-stress sweep, last
# saved frame, one panel per run.
#
# Usage:
#   ./run_vishoals_vort_wind.sh [cluster_name] [wind_sign ...]
#
#   ./run_vishoals_vort_wind.sh cluster_Fnnie_1_19              # negative + positive
#   ./run_vishoals_vort_wind.sh cluster_weakwinds both          # one signed row
#   ./run_vishoals_vort_wind.sh cluster_weakwinds negative positive both
#
# wind_sign is 'negative', 'positive' or 'both'; one figure is written per
# sign given. Which RUNS go in is no longer fixed in the script -- every
# wind_* run in the cluster is used and the panels are ordered by tau, so a
# new sweep needs no code change. See vishoals_vort_wind.m.
#
# cluster_weakwinds has no unforced (tau = 0) run, so 'ref_run' is passed
# empty there and the row starts at the weakest forced case. Any cluster
# containing Zs5_Ls40k gets that as its leading tau = 0 panel automatically.

CLUSTER_NAME="${1:-cluster_Fnnie_1_19}"
shift || true
SIGNS=("$@")
if [ ${#SIGNS[@]} -eq 0 ]; then
    SIGNS=(negative positive)
fi

# Load NCAR Casper environment
module purge
module load ncarenv-basic/23.10
module load ncarenv/23.10
module load matlab/R2024a

echo "=== Running Wind-Sweep Vorticity Still ==="
echo "    Cluster:          ${CLUSTER_NAME}"
echo "    Wind signs:       ${SIGNS[*]}"
echo "    Output Directory: /glade/work/nguyen/unc-shoals/animations/"

# Run from the tools directory so the toolbox paths resolve
cd "$(dirname "$0")"

# A cluster with no unforced run would warn on every call; pass ref_run ''
# there so the warning is opted out of rather than repeated per figure.
REF="'Zs5_Ls40k'"
if ! compgen -G "/glade/work/nguyen/unc-shoals/sweep_output/${CLUSTER_NAME}/*Zs5_Ls40k" > /dev/null; then
    echo "    No Zs5_Ls40k in this cluster -- plotting forced runs only."
    REF="''"
fi

CALLS=""
for s in "${SIGNS[@]}"; do
    CALLS="${CALLS} fprintf('--- vorticity sweep: ${CLUSTER_NAME} (${s}) ---\n'); \
        vishoals_vort_wind('${CLUSTER_NAME}', 'wind_sign', '${s}', 'ref_run', ${REF}, \
                           'clipped', true, 'climRo', [-1.0, 1.0]);"
done

matlab -nodisplay -nosplash -r "\
    addpath(genpath('gsw_matlab_v3_06_16')); \
    addpath(genpath('cmocean')); \
    try, \
        ${CALLS} \
    catch ME, \
        fprintf('Error: %s\n', ME.message); \
        fprintf('%s\n', getReport(ME, 'extended')); \
    end; \
    exit"

echo "=== Finished processing ${CLUSTER_NAME} ==="
