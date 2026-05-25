from huggingface_hub import snapshot_download

# Download via Hugging Face official hub
model_dir = snapshot_download(
   repo_id='Qwen/Qwen2.5-7B',
   local_dir='./huggingface_cache/models/Qwen2.5-7B',
   ignore_patterns=['*.pth', '*.pt', 'original/*'],  # Ignore redundant original weights
   max_workers=8  # 🌟 Enable native safe multi-threading acceleration to maximize bandwidth and completely avoid OS Error 32
)
print(f"Model has been downloaded to: {model_dir}")