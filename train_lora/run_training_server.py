#!/usr/bin/env python3
"""
服务器训练脚本（只在有多卡 GPU 的训练服务器上运行）。

前提：
- 本地机已跑完预处理，将 data/ 目录（包含 dataset.json 和 .precomputed/）同步到服务器相同结构路径下。
- 服务器上已克隆并安装 LTX-2（uv sync），并下载好 Gemma 文本编码器。

用法（示例）：
  cd /path/to/video_generate/train_lora
  python run_training_server.py
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from run_train import run_train

# ====================== 在这里改「服务器」配置 ======================
# 服务器上的 LTX-2 仓库根目录
LTX2_REPO = "/workspace/LTX-2"

# 服务器上的 LTX-2 权重：可用 "@cache" 或绝对路径
MODEL_PATH = "@cache"

# 服务器上的 Gemma 文本编码器目录
TEXT_ENCODER_PATH = "/workspace/models/gemma-3-12b-it-qat-q4_0-unquantized"

# 服务器上的预处理输出目录（即包含 .precomputed 的目录）
# 通常是你同步到服务器的 train_lora/data 目录
PREPROCESSED_ROOT = "/workspace/video_generate/train_lora/data/.precomputed"

# LoRA 触发词（需与本地预处理时一致）
TRIGGER = "yuna"

# 训练输出目录（服务器本地）
OUTPUT_DIR = "/workspace/video_generate/train_lora/outputs/char_lora"

# GPU 数量（服务器上可用的卡数）
NUM_PROCESSES = 4

# 可选：自定义 YAML 模板路径（默认用 configs/ltx2_char_lora_4x4090.yaml）
CONFIG_TEMPLATE = None

# ====================== 以下为执行逻辑，一般无需改 ======================


def main():
    pre_root = os.path.abspath(PREPROCESSED_ROOT)
    if not os.path.isdir(pre_root):
        raise FileNotFoundError(f"找不到预处理输出目录 PREPROCESSED_ROOT: {pre_root}")

    out_dir = os.path.abspath(OUTPUT_DIR) if OUTPUT_DIR else None

    run_train(
        ltx2_repo=LTX2_REPO,
        text_encoder_path=TEXT_ENCODER_PATH,
        preprocessed_root=pre_root,
        model_path=MODEL_PATH,
        output_dir=out_dir,
        trigger=TRIGGER,
        config_template=CONFIG_TEMPLATE,
        num_processes=NUM_PROCESSES,
    )


if __name__ == "__main__":
    main()

