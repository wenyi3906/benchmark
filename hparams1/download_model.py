# 如果没有安装 modelscope，请先在终端执行：pip install modelscope
from modelscope.hub.snapshot_download import snapshot_download

# 指定你想保存模型的本地主目录，比如 D盘的 models 文件夹
# 建议找一个空间大于 20GB 的磁盘
base_cache_dir = 'E:/cache/huggingface/modules'

print("开始下载 Qwen2.5-7B 模型权重...")
# 下载基础模型
model_dir = snapshot_download('qwen/Qwen2.5-7B', cache_dir=base_cache_dir)

print("\n下载完成！")
print(f"请将 yaml 文件中的 model_name 修改为以下绝对路径：\n{model_dir}")