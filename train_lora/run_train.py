#!/usr/bin/env python3
"""
用 4×4090 启动 LTX-2 固定角色 LoRA 训练。
会替换 config 中的路径占位符并调用 LTX-2 的 train.py（需先克隆并 uv sync）。
--model-path 可传 @cache 使用 Hugging Face 缓存中的 LTX-2 权重。
"""
import argparse
import os
import subprocess
import sys

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


def run_train(
    ltx2_repo,
    text_encoder_path,
    preprocessed_root,
    model_path="@cache",
    output_dir=None,
    trigger="MYCHAR_001",
    config_template=None,
    num_processes=4,
):
    """启动 LTX-2 LoRA 训练。返回训练输出目录。"""
    repo = os.path.abspath(ltx2_repo)
    train_script_abs = os.path.join(repo, "packages", "ltx-trainer", "scripts", "train.py")
    if not os.path.isfile(train_script_abs):
        raise FileNotFoundError(f"未找到: {train_script_abs}，请先克隆 LTX-2 并 uv sync")
    train_script_rel = os.path.join("packages", "ltx-trainer", "scripts", "train.py")

    this_dir = os.path.dirname(os.path.abspath(__file__))
    config_template = config_template or os.path.join(this_dir, "configs", "ltx2_char_lora_4x4090.yaml")
    with open(config_template, "r", encoding="utf-8") as f:
        yaml_content = f.read()

    model_path = _resolve_model_path(model_path)
    print("使用模型权重:", model_path)
    text_encoder_path = os.path.abspath(text_encoder_path)
    preprocessed_root = os.path.abspath(preprocessed_root)
    output_dir = output_dir or os.path.join(this_dir, "outputs", "char_lora")
    output_dir = os.path.abspath(output_dir)

    yaml_content = (
        yaml_content.replace("MODEL_PATH", model_path)
        .replace("TEXT_ENCODER_PATH", text_encoder_path)
        .replace("PREPROCESSED_DATA_ROOT", preprocessed_root)
        .replace("TRIGGER_TOKEN", trigger)
        .replace("OUTPUT_DIR", output_dir)
    )

    generated_config = os.path.join(this_dir, "configs", "_generated_char_lora.yaml")
    with open(generated_config, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"已生成配置: {generated_config}")

    cmd = [
        "uv", "run", "accelerate", "launch",
        "--num_processes", str(num_processes),
        "--mixed_precision", "bf16",
        train_script_rel,
        generated_config,
    ]
    print("执行:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=repo)
    if r.returncode != 0:
        raise SystemExit(r.returncode)
    print(f"训练完成。LoRA 权重在: {output_dir}")
    return output_dir


def main():
    ap = argparse.ArgumentParser(description="启动 LTX-2 LoRA 训练（4 卡）")
    ap.add_argument("--ltx2-repo", required=True, help="LTX-2 仓库根目录")
    ap.add_argument("--model-path", default="@cache", help="ltx-2-19b-dev.safetensors 路径；传 @cache 使用 HF 缓存")
    ap.add_argument("--text-encoder-path", required=True, help="Gemma 文本编码器目录路径")
    ap.add_argument("--preprocessed-root", required=True, help="预处理输出目录（即 dataset 目录下的 .precomputed）")
    ap.add_argument("--output-dir", default=None, help="训练输出目录（默认：train_lora/outputs/char_lora）")
    ap.add_argument("--trigger", default="MYCHAR_001", help="与预处理时一致的触发词")
    ap.add_argument("--config", default=None, help="YAML 模板（默认用 configs/ltx2_char_lora_4x4090.yaml）")
    ap.add_argument("--num-processes", type=int, default=4, help="GPU 数量")
    args = ap.parse_args()
    run_train(
        ltx2_repo=args.ltx2_repo,
        text_encoder_path=args.text_encoder_path,
        preprocessed_root=args.preprocessed_root,
        model_path=args.model_path,
        output_dir=args.output_dir,
        trigger=args.trigger,
        config_template=args.config,
        num_processes=args.num_processes,
    )


if __name__ == "__main__":
    main()
