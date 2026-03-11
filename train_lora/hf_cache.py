"""
解析 Hugging Face 缓存中的 LTX-2 权重路径。
run.py 用 from_pretrained("Lightricks/LTX-2") 时会把权重缓存在 ~/.cache/huggingface/hub/ 下，
训练脚本可通过 --model-path @cache 使用该缓存，无需再指定本地 .safetensors 路径。
"""
import os


def _hub_cache_dir():
    return os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface/hub")


def get_ltx2_model_path_from_cache():
    """
    返回 HF 缓存中 LTX-2 模型权重文件路径。
    优先使用 ltx-2-19b-dev.safetensors，若不存在则用 ltx-2-19b-dev-fp4.safetensors（diffusers 常用）。
    若未找到则返回 None。
    """
    root = os.path.join(_hub_cache_dir(), "models--Lightricks--LTX-2", "snapshots")
    if not os.path.isdir(root):
        return None
    # 按目录 mtime 取最新 snapshot（或可读 refs/main 指向的 commit）
    snapshots = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if not snapshots:
        return None
    snapshots.sort(key=lambda s: os.path.getmtime(os.path.join(root, s)), reverse=True)
    for snap in snapshots:
        base = os.path.join(root, snap)
        for name in ("ltx-2-19b-dev.safetensors", "ltx-2-19b-dev-fp4.safetensors"):
            path = os.path.join(base, name)
            if os.path.isfile(path):
                return path
    return None


# 你本机已确认存在的路径（仅作默认值，若 @cache 解析失败可回退）
DEFAULT_LTX2_CACHE_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Lightricks--LTX-2/snapshots/47da56e2ad66ce4125a9922b4a8826bf407f9d0a/ltx-2-19b-dev-fp4.safetensors"
)
