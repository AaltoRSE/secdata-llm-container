#!/bin/bash
## to submit: bash gpu-queue-submit --sif absolute/path/to/your.sif --job absolute/path/to/my_job.sh --name my_job
set -euo pipefail

echo "Job started: $(date)"
echo "User: ${USER:-unknown}"
echo "Hostname: $(hostname)"
echo "Working dir: $(pwd)"

# --- quick GPU check ---
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    x = torch.randn(1000, 1000, device="cuda")
    print("GPU matmul OK, result shape:", (x @ x).shape)
PY

# --- your real work (example: run a Python script) ---
# conda env is already active in sec_llm.sif (sec-llm-env)
python /nfs/home/${USER}/my_train.py

echo "Job finished: $(date)"