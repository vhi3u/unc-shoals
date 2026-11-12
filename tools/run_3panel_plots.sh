#!/bin/bash
# run_3panel_plots.sh
#
# Runs the MATLAB 3-panel visualization script for U-velocity, Heat transport,
# and Salt transport comparisons on Casper.
#
# Usage:
#   ./run_3panel_plots.sh

# Module environment for Casper (NCAR)
module purge
module load ncarenv-basic/23.10
module load ncarenv/23.10
module load matlab/R2024a

echo "=== Running 3-Panel Transport Comparison Plots ==="
echo "    Target Directory: /glade/work/nguyen/unc-shoals/animations/"

# Ensure we are in the directory containing the scripts
cd "$(dirname "$0")"

# Execute MATLAB
matlab -nodisplay -nosplash -r "\
    addpath(genpath('cmocean')); \
    try, \
        generate_all_3panel_plots; \
    catch ME, \
        fprintf('Error: %s\n', ME.message); \
    end, \
    exit"

echo "=== Done ==="
echo "Check /glade/work/nguyen/unc-shoals/animations/ for resulting .png files."
