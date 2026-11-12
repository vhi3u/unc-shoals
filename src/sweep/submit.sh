#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# submit.sh — submit a sweep without hardcoding your account in a tracked file
# ═══════════════════════════════════════════════════════════════════════════
#
#   ./src/sweep/submit.sh res250              # whole array
#   ./src/sweep/submit.sh res250 1            # just index 1 (test first)
#   ./src/sweep/submit.sh weakwind
#   ./src/sweep/submit.sh sweep
#
# Reads src/sweep/job_env.sh for the account, email and depot, and passes
# them to qsub on the command line, where they override the #PBS directives
# baked into the .pbs file. Nothing machine-specific has to live in git.
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${HERE}/job_env.sh"

if [ ! -f "${ENV_FILE}" ]; then
    echo "ERROR: ${ENV_FILE} not found."
    echo
    echo "  cp ${HERE}/job_env.sh.example ${ENV_FILE}"
    echo "  \$EDITOR ${ENV_FILE}"
    echo
    echo "It is gitignored, so it stays out of the repo and survives pulls."
    exit 1
fi
# shellcheck source=/dev/null
source "${ENV_FILE}"

: "${PBS_ACCOUNT:?set PBS_ACCOUNT in job_env.sh}"
: "${JULIA_DEPOT:?set JULIA_DEPOT in job_env.sh}"

case "${JULIA_DEPOT}" in
  *:) ;;
  *) echo "WARNING: JULIA_DEPOT does not end in ':' — this replaces the depot"
     echo "         list instead of prepending, and will break Pkg. Fix job_env.sh." ;;
esac

NAME="${1:-}"
INDEX="${2:-}"
if [ -z "${NAME}" ]; then
    echo "usage: $0 <res250|weakwind|sweep|wind_sweep|wind_fork> [array_index]"
    exit 1
fi

SCRIPT="${HERE}/submit_${NAME}.pbs"
[ -f "${SCRIPT}" ] || { echo "ERROR: no such script: ${SCRIPT}"; exit 1; }

ARGS=(-A "${PBS_ACCOUNT}" -v "JULIA_DEPOT_PATH=${JULIA_DEPOT}")
[ -n "${PBS_EMAIL:-}" ] && ARGS+=(-M "${PBS_EMAIL}")
[ -n "${INDEX}" ] && ARGS+=(-J "${INDEX}-${INDEX}")

echo "qsub ${ARGS[*]} ${SCRIPT}"
qsub "${ARGS[@]}" "${SCRIPT}"
