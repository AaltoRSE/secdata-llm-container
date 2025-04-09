#!/bin/bash
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --partition=gpu-a100-80g
#SBATCH --time=02:00:00

SIF=sec_llm.sif

echo "Testing basic environment................"
singularity run --nv --net --network none \
--bind /scratch/shareddata/dldata/huggingface-hub-cache:/models/huggingface-hub \
$SIF \
basic_env.py

echo "Testing HuggingFace Langchain................"
singularity run --nv --net --network none \
--bind /scratch/shareddata/dldata/huggingface-hub-cache:/models/huggingface-hub \
$SIF \
hug_langchain.py

echo "Testing RAG................"
singularity run --nv --net --network none \
--bind /scratch/shareddata/dldata/huggingface-hub-cache:/models/huggingface-hub \
$SIF \
rag.py

echo "Testing LlamaIndex Agent................"
singularity run --nv --net --network none \
--bind /scratch/shareddata/dldata/huggingface-hub-cache:/models/huggingface-hub \
$SIF \
llama_index_agent.py

echo "Testing LlamaIndex Structured LLM................"
singularity run --nv --net --network none \
--bind /scratch/shareddata/dldata/huggingface-hub-cache:/models/huggingface-hub \
$SIF \
llama_index_structured_llm.py