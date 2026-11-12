#!/bin/bash
# load MATLAB environment on NCAR HPC (Casper)

module purge
module load ncarenv-basic/23.10
module load ncarenv/23.10
module load matlab/R2024a

echo "MATLAB environment loaded"
