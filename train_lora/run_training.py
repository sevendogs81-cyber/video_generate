#!/usr/bin/env python3
"""
本地预处理脚本（只在「有原始视频」的机器上运行）。

功能：
- 从本地视频目录生成 dataset.json
- 调用 LTX-2 官方 trainer 的 process_dataset.py 生成 .precomputed（latent + 文本 embedding）

不在本脚本里做训练。训练步骤请在服务器上运行单独的 server 训练脚本。

用法（示例）：
  cd /home/wenhanxiao/code/video_generate/train_lora
  python run_training.py
"""
import os
import sys

# 当前脚本所在目录（train_lora），保证同目录模块可被导入
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# ====================== 在这里改「本地机」配置 ======================
# LTX-2 仓库根目录（克隆后路径）
LTX2_REPO = "/home/wenhanxiao/code/LTX-2"

# LTX-2 权重：用 "@cache" 表示使用 HF 缓存；或写本地 .safetensors 路径
MODEL_PATH = "@cache"

# Gemma 文本编码器目录（必须本地路径）
TEXT_ENCODER_PATH = "/path/to/gemma-3-12b-it-qat-q4_0-unquantized"

# 同一角色的视频所在目录（将用来生成 dataset.json）
VIDEOS_DIR = "/path/to/我的角色视频目录"

# LoRA 触发词（推理时在 prompt 里写这个词即激活该角色）
TRIGGER = "yuna"

# 预处理分辨率桶（宽x高x帧数）
RESOLUTION_BUCKETS = "960x544x49"

# 是否预处理时包含音频（音视频联合训练）
WITH_AUDIO = True

# 是否跳过预处理（已预处理过可设为 True，仅检查路径）
SKIP_PREPROCESS = False

# 默认 caption（所有视频共用，固定角色时可保持默认）
DEFAULT_CAPTION = "a person, natural motion, various angles and expressions, high quality"

# ====================== 以下为执行逻辑，一般无需改 ======================


def main():
    from prepare_dataset import prepare_dataset
    from run_preprocess import run_preprocess

    data_dir = os.path.join(_THIS_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    dataset_json = os.path.join(data_dir, "dataset.json")

    # Step 1: 生成 dataset.json
    if not os.path.isfile(dataset_json) or not SKIP_PREPROCESS:
        prepare_dataset(
            VIDEOS_DIR,
            output=dataset_json,
            default_caption=DEFAULT_CAPTION,
        )
    else:
        print("使用已有 dataset.json:", dataset_json)

    # Step 2: 预处理
    precomputed = os.path.join(data_dir, ".precomputed")
    if not SKIP_PREPROCESS:
        run_preprocess(
            ltx2_repo=LTX2_REPO,
            text_encoder_path=TEXT_ENCODER_PATH,
            dataset_json=dataset_json,
            model_path=MODEL_PATH,
            resolution_buckets=RESOLUTION_BUCKETS,
            lora_trigger=TRIGGER,
            with_audio=WITH_AUDIO,
        )

    if not os.path.isdir(precomputed):
        raise FileNotFoundError(f"预处理输出不存在: {precomputed}")

    print("✅ 本地预处理完成。请将以下目录同步到服务器：")
    print(f"  - dataset.json: {dataset_json}")
    print(f"  - .precomputed: {precomputed}")
    print("然后在服务器上运行 server 训练脚本进行 LoRA 训练。")


if __name__ == "__main__":
    main()
