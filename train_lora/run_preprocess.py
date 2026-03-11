#!/usr/bin/env python3
"""
调用 LTX-2 官方的 process_dataset.py 做预处理（算 latent + 文本 embedding，并加触发词）。
需先克隆 LTX-2 并安装依赖：git clone https://github.com/Lightricks/LTX-2 && cd LTX-2 && uv sync
--model-path 可传 @cache 使用 Hugging Face 缓存中的 LTX-2 权重（与 run.py 共用缓存）。
"""
import argparse
import os
import subprocess
import sys

# 本包内解析 HF 缓存
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from hf_cache import get_ltx2_model_path_from_cache, DEFAULT_LTX2_CACHE_PATH


def _resolve_model_path(path: str) -> str:
    if not path or str(path).strip().lower() in ("@cache", "hf_cache", "hf"):
        resolved = get_ltx2_model_path_from_cache()
        if resolved is None and os.path.isfile(DEFAULT_LTX2_CACHE_PATH):
            resolved = DEFAULT_LTX2_CACHE_PATH
        if resolved is None:
            raise FileNotFoundError(
                "未找到 LTX-2 缓存。请先运行 run.py 加载一次模型，或指定 model_path 为本地 .safetensors 路径。"
            )
        return os.path.abspath(resolved)
    return os.path.abspath(path)


def run_preprocess(
    ltx2_repo,
    text_encoder_path,
    dataset_json,
    model_path="@cache",
    resolution_buckets="960x544x49",
    lora_trigger="MYCHAR_001",
    with_audio=False,
):
    """运行 LTX-2 数据预处理。返回 .precomputed 目录路径。"""
    repo = os.path.abspath(ltx2_repo)
    scripts_dir = os.path.join(repo, "packages", "ltx-trainer", "scripts")
    process_script = os.path.join(scripts_dir, "process_dataset.py")
    if not os.path.isfile(process_script):
        raise FileNotFoundError(f"未找到脚本: {process_script}，请先克隆 LTX-2 并 uv sync")

    dataset_json = os.path.abspath(dataset_json)
    if not os.path.isfile(dataset_json):
        raise FileNotFoundError(f"dataset.json 不存在: {dataset_json}")

    model_path = _resolve_model_path(model_path)
    print("使用模型权重:", model_path)

    cmd = [
        "uv", "run", "python", process_script, dataset_json,
        "--resolution-buckets", resolution_buckets,
        "--model-path", model_path,
        "--text-encoder-path", os.path.abspath(text_encoder_path),
        "--lora-trigger", lora_trigger,
    ]
    if with_audio:
        cmd.append("--with-audio")

    print("执行:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=repo)
    if r.returncode != 0:
        raise SystemExit(r.returncode)

    precomputed = os.path.join(os.path.dirname(dataset_json), ".precomputed")
    print(f"预处理完成。训练时 preprocessed_data_root: {precomputed}")
    return precomputed


def main():
    ap = argparse.ArgumentParser(description="运行 LTX-2 数据预处理（需在 LTX-2 环境中）")
    ap.add_argument("--ltx2-repo", required=True, help="LTX-2 仓库根目录")
    ap.add_argument("--model-path", default="@cache", help="ltx-2-19b-dev.safetensors 路径；传 @cache 使用 HF 缓存")
    ap.add_argument("--text-encoder-path", required=True, help="Gemma 文本编码器目录路径")
    ap.add_argument("--dataset-json", required=True, help="dataset.json 路径（prepare_dataset.py 生成）")
    ap.add_argument("--resolution-buckets", default="960x544x49", help="分辨率桶，如 960x544x49")
    ap.add_argument("--lora-trigger", default="MYCHAR_001", help="LoRA 触发词，推理时 prompt 里写此词即激活")
    ap.add_argument("--with-audio", action="store_true", help="同时预处理音频（音视频联合训练时用）")
    args = ap.parse_args()
    run_preprocess(
        ltx2_repo=args.ltx2_repo,
        text_encoder_path=args.text_encoder_path,
        dataset_json=args.dataset_json,
        model_path=args.model_path,
        resolution_buckets=args.resolution_buckets,
        lora_trigger=args.lora_trigger,
        with_audio=args.with_audio,
    )


if __name__ == "__main__":
    main()
