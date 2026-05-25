from modelscope import snapshot_download

target_dir = "./huggingface_cache/models/gpt-j-6B"
print(f"🚀 Starting to download model using ModelScope to {target_dir} ...")

# 使用魔塔社区下载
snapshot_download(
    model_id="AI-ModelScope/gpt-j-6B",  # 魔塔上的仓库 ID
    local_dir=target_dir,
    ignore_file_pattern=[r".*\.h5$", r".*\.msgpack$", r".*\.ot$"] # 使用正则忽略 TensorFlow 和 Flax 的权重
)

print(f"✅ Model downloaded successfully via ModelScope to: {target_dir}")