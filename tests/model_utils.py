"""Utility functions for loading HuggingFace models in offline mode."""
import os


def get_local_model_path(model_name: str) -> str:
    """
    Get the local filesystem path for a HuggingFace model to avoid API calls.
    
    This works around a bug in transformers where it still tries to check 
    Mistral models via API even when local_files_only=True.
    
    Args:
        model_name: HuggingFace model identifier (e.g., "Qwen/Qwen2.5-3B-Instruct")
        
    Returns:
        Local filesystem path to the model, or the original model_name if not found
    """
    # The cache structure is: HF_HOME/hub/models--<org>--<model>/snapshots/<hash>/
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    model_cache_name = model_name.replace("/", "--")
    model_cache_dir = os.path.join(hf_home, "hub", f"models--{model_cache_name}")

    # Find the latest snapshot
    if os.path.exists(model_cache_dir):
        snapshots_dir = os.path.join(model_cache_dir, "snapshots")
        if os.path.exists(snapshots_dir):
            snapshots = [d for d in os.listdir(snapshots_dir) 
                        if os.path.isdir(os.path.join(snapshots_dir, d))]
            if snapshots:
                # Use the first snapshot (or could sort by modification time)
                model_path = os.path.join(snapshots_dir, snapshots[0])
                print(f"Using local model path: {model_path}")
                return model_path
    
    # Fallback to model name if local path not found
    print(f"Warning: Model cache directory not found at {model_cache_dir}, using model name")
    return model_name

