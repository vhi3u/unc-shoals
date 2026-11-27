#!/bin/bash
# run_inflow_diagnostics.sh
#
# What flow reaches the shoal, and how disturbed is it, in each run given.
# Compares periodic and open-y runs: the transport through a section upstream
# of the shoal against the pinned target Q_in, and the size of the incoming
# disturbance (RMS of surface u over the shelf), along y and over time.
#
# Usage:
#   ./run_inflow_diagnostics.sh RUN [RUN ...]
#
# A RUN is a label (B_narrow_open_500), a folder (sweep_002_B_narrow_open_500),
# cluster/folder, or a full path; labels are searched for under
# /glade/derecho/scratch/nguyen/unc-shoals/sweep_output at any depth.
#
#   ./run_inflow_diagnostics.sh B_narrow_open B_narrow_periodic
#   ./run_inflow_diagnostics.sh B_narrow_open_500 B_narrow_periodic_500 B_narrow_open_seed1_500 B_narrow_open_seed5_500
#
# Reads top_*.nc and time_avg_3d_*.nc; needs no rerun. Prints a table and
# writes animations/inflow_<labels>.png.

set -u
if [ "$#" -lt 1 ]; then
    sed -n '2,20p' "$0"; exit 1
fi

module purge
module load ncarenv-basic/23.10
module load ncarenv/23.10
module load matlab/R2024a

cd "$(dirname "$0")"

LIST=""
for r in "$@"; do LIST="${LIST}'${r}', "; done
LIST="{${LIST%, }}"

echo "=== inflow diagnostics: $* ==="
matlab -nodisplay -nosplash -r "\
    try, \
        inflow_diagnostics(${LIST}); \
    catch ME, \
        fprintf('Error: %s\n', ME.message); \
        fprintf('%s\n', getReport(ME, 'extended')); \
    end; \
    exit"
echo "=== done ==="
