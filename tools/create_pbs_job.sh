#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# create_pbs_job.sh — PBS Job Script Generator for Casper GPU Runs
# ═══════════════════════════════════════════════════════════════════════════
# Usage:
#   ./create_pbs_job.sh <run_name> [walltime_hours] [gpu_type] [memory] [jl_script]
#
# Examples:
#   ./create_pbs_job.sh 100d500m_sponge 12 v100 100gb
#   ./create_pbs_job.sh test_run 6 a100 120gb flow_over_shoals_turb.jl
# ═══════════════════════════════════════════════════════════════════════════

set -e

# Default values
DEFAULT_WALLTIME=12
DEFAULT_GPU="v100"
DEFAULT_MEM="100gb"
DEFAULT_SCRIPT="flow_over_shoals.jl"

# Help message
if [ "$1" == "-h" ] || [ "$1" == "--help" ] || [ $# -lt 1 ]; then
    echo "Usage: $0 <run_name> [walltime_hours] [gpu_type] [memory] [jl_script]"
    echo ""
    echo "Arguments:"
    echo "  run_name        : Name of the run (e.g., 100d500m_sponge)"
    echo "  walltime_hours  : Walltime limit in hours (default: ${DEFAULT_WALLTIME})"
    echo "  gpu_type        : GPU type (e.g., v100, a100, h100; default: ${DEFAULT_GPU})"
    echo "  memory          : Total memory (e.g., 100gb, 120gb; default: ${DEFAULT_MEM})"
    echo "  jl_script       : Julia script to execute (default: ${DEFAULT_SCRIPT})"
    echo ""
    echo "Example:"
    echo "  $0 100d500m_sponge 12 v100 100gb flow_over_shoals.jl"
    exit 0
fi

RUN_NAME="$1"
WALLTIME_HRS="${2:-$DEFAULT_WALLTIME}"
GPU_TYPE="${3:-$DEFAULT_GPU}"
MEM="${4:-$DEFAULT_MEM}"
JL_SCRIPT="${5:-$DEFAULT_SCRIPT}"

# Format walltime as HH:MM:SS
if [[ "$WALLTIME_HRS" =~ ^[0-9]+$ ]]; then
    WALLTIME=$(printf "%02d:00:00" "$WALLTIME_HRS")
else
    WALLTIME="$WALLTIME_HRS"
fi

OUTPUT_FILE="pbs_${RUN_NAME}.sh"

cat <<EOF > "$OUTPUT_FILE"
#!/bin/bash -l
#PBS -A uosc0035
#PBS -N ${RUN_NAME}
#PBS -k eod
#PBS -o logs/${RUN_NAME}.out
#PBS -e logs/${RUN_NAME}.err
#PBS -l walltime=${WALLTIME}
#PBS -q casper
#PBS -l select=1:ncpus=4:ngpus=1:gpu_type=${GPU_TYPE}:mem=${MEM}
#PBS -M vhieu@unc.edu
#PBS -m abe

# Clear the environment from any previously loaded modules
module purge
module load ncarenv-basic/24.12
module load ncarenv/24.12
module load julia/1.11.2 cuda

module load peak-memusage
module li

#/glade/u/apps/ch/opt/usr/bin/dumpenv # Dumps environment (for debugging with CISL support)

#export JULIA_DEPOT_PATH="/glade/work/tomasc/.julia_bkp"
export JULIA_DEPOT_PATH="/glade/work/nguyen/.julia"

mkdir -p logs out

peak_memusage julia --project ${JL_SCRIPT} --simname=${RUN_NAME} --factor=2 2>&1 | tee out/${RUN_NAME}
EOF

chmod +x "$OUTPUT_FILE"

echo "Successfully generated PBS script: ${OUTPUT_FILE}"
echo "  Run Name : ${RUN_NAME}"
echo "  Walltime : ${WALLTIME}"
echo "  GPU Type : ${GPU_TYPE}"
echo "  Memory   : ${MEM}"
echo "  Script   : ${JL_SCRIPT}"
echo ""
echo "To submit to Casper, run:"
echo "  qsub ${OUTPUT_FILE}"
