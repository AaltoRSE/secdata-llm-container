#!/bin/bash
#SBATCH --job-name=build_container
#SBATCH --time=01:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4

# Ensure env.yml is in the current directory
if [ ! -f env.yml ]; then
  echo "Error: env.yml not found in current directory"
  exit 1
fi

apptainer build --bind /scratch/rse/secdata-llm-container/env.yml:/tmp/env.yml sec_llm.sif sec_llm.def
