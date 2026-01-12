#!/bin/bash
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --partition=gpu-a100-80g
#SBATCH --time=02:00:00


SIF=sec_llm.sif

# Iterate over all .py files in the tests/ directory
for test_script in tests/*.py; do
    echo "Testing ${test_script}................"
    singularity run --nv --network=none \
    --bind /scratch/shareddata/dldata/huggingface-hub-cache:/models/huggingface-hub \
    $SIF \
    ${test_script}
done