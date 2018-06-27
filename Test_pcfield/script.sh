#!/bin/bash
#SBATCH --time=1200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50000
#SBATCH --cpus-per-task=2
#SBATCH --wckey=P10WB:ASTER
make test1

