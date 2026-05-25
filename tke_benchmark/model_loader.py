import os
import sys

def setup_and_get_model(model_id="openai-community/gpt2-xl", base_dir="E:/cache/huggingface"):
    """
    Get the local path to the model. Return it directly if downloaded, otherwise pull from Hugging Face.
    Prioritizes checking the legacy flat path format: base_dir/models/model_name.

    Args:
        model_id (str): Model ID on Hugging Face.
        base_dir (str): Base directory for local model caching.

    Returns:
        str: Absolute path to the model directory.
    """
    # Parse the model's short name, e.g., openai-community/gpt2-xl -> gpt2-xl
    model_name_short = model_id.split("/")[-1]
    
    # Check legacy storage structure
    legacy_path = os.path.join(base_dir, "models", model_name_short)
    legacy_path = legacy_path.replace("\\", "/")
    
    # If config.json already exists in the legacy path, the model is fully ready.
    if os.path.exists(os.path.join(legacy_path, "config.json")):
        print(f"✅ Local intact model files detected, skipping download!")
        print(f"📂 Offline model directory found: {legacy_path}")
        print(f"✅ Model {model_id} is ready at: {legacy_path}")
        return legacy_path

    # Attempt to import huggingface_hub
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("huggingface_hub is not detected. Please run: pip install huggingface_hub")

    print(f"⬇️ Intact model not found locally, pulling from Hugging Face: {model_id}")
    os.environ["HF_HOME"] = base_dir
    os.environ["HF_HUB_CACHE"] = base_dir
    os.makedirs(base_dir, exist_ok=True)

    try:
        model_path = snapshot_download(
            repo_id=model_id,
            cache_dir=base_dir,
        )
        print(f"✅ Model {model_id} has been downloaded and is ready at: {model_path}")
        return model_path
    except Exception as e:
        raise RuntimeError(f"Model download failed. Please check your network or manually place the model in {legacy_path}. Error: {e}")
