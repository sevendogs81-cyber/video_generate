#!/usr/bin/env python3
"""
命令行入口：从视频目录 → 生成 dataset.json → 预处理 → 4 卡训练。
若不想用命令行，请直接改 run_training.py 顶部配置后运行：python run_training.py
"""
import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


def main():
    ap = argparse.ArgumentParser(description="一键完成数据准备 + 预处理 + LoRA 训练")
    ap.add_argument("--ltx2-repo", required=True, help="LTX-2 仓库根目录")
    ap.add_argument("--model-path", default="@cache", help="LTX-2 权重路径；@cache 使用 HF 缓存")
    ap.add_argument("--text-encoder-path", required=True, help="Gemma 文本编码器目录路径")
    ap.add_argument("--videos-dir", required=True, help="同一角色的视频所在目录")
    ap.add_argument("--trigger", default="MYCHAR_001", help="LoRA 触发词")
    ap.add_argument("--output-dir", default=None, help="训练输出目录")
    ap.add_argument("--resolution-buckets", default="960x544x49", help="预处理分辨率桶")
    ap.add_argument("--num-processes", type=int, default=4, help="GPU 数量")
    ap.add_argument("--with-audio", action="store_true", help="预处理时包含音频")
    ap.add_argument("--skip-preprocess", action="store_true", help="已预处理过则跳过")
    ap.add_argument("--skip-train", action="store_true", help="只做数据与预处理，不训练")
    args = ap.parse_args()

    from prepare_dataset import prepare_dataset
    from run_preprocess import run_preprocess
    from run_train import run_train

    data_dir = os.path.join(_THIS_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    dataset_json = os.path.join(data_dir, "dataset.json")

    if not os.path.isfile(dataset_json) or not args.skip_preprocess:
        prepare_dataset(args.videos_dir, output=dataset_json)
    else:
        print("使用已有 dataset.json:", dataset_json)

    precomputed = os.path.join(data_dir, ".precomputed")
    if not args.skip_preprocess:
        run_preprocess(
            ltx2_repo=args.ltx2_repo,
            text_encoder_path=args.text_encoder_path,
            dataset_json=dataset_json,
            model_path=args.model_path,
            resolution_buckets=args.resolution_buckets,
            lora_trigger=args.trigger,
            with_audio=args.with_audio,
        )
    if not os.path.isdir(precomputed):
        print("预处理输出不存在:", precomputed, file=sys.stderr)
        sys.exit(1)

    if args.skip_train:
        print("已跳过训练（--skip-train）")
        return

    run_train(
        ltx2_repo=args.ltx2_repo,
        text_encoder_path=args.text_encoder_path,
        preprocessed_root=precomputed,
        model_path=args.model_path,
        output_dir=args.output_dir or os.path.join(_THIS_DIR, "outputs", "char_lora"),
        trigger=args.trigger,
        num_processes=args.num_processes,
    )


if __name__ == "__main__":
    main()
