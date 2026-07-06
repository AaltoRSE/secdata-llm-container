#!/bin/bash
#SBATCH --job-name=build_container
#SBATCH --time=02:00:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=4

apptainer build sec_llm_Jul2026.sif sec_llm.def