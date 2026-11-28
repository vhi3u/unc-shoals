#!/bin/bash
# run_locate_extremes.sh -- where do T and S leave the initial range? (tools/locate_extremes.m)
#
#   ./run_locate_extremes.sh RUN [RUN ...]
#   ./run_locate_extremes.sh DSN_cape_v020_N2x38 DSN_cape_v010_N2x38 DSN_cape_v020_N2x6
#
# Prints, per run, how many surface cells are out of range by zone (north
# outflow, south inflow, east, coast, interior), the worst values and when they
# first appear; writes animations/extremes_<label>.png.
set -u
[ "$#" -lt 1 ] && { sed -n '2,8p' "$0"; exit 1; }
module purge
module load ncarenv-basic/23.10
module load ncarenv/23.10
module load matlab/R2024a
cd "$(dirname "$0")"
for r in "$@"; do
    matlab -nodisplay -nosplash -r "try, locate_extremes('${r}'); catch ME, fprintf('Error: %s\n', getReport(ME, 'extended')); end; exit" \
        | sed -n '/initial T/,$p'
done
