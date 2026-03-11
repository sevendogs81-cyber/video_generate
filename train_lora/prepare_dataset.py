#!/usr/bin/env python3
"""
从视频目录生成 dataset.json，供 LTX-2 预处理使用。
固定角色：所有视频应为同一人，caption 只需描述场景/动作，触发词在预处理时用 --lora-trigger 统一加。
"""
import argparse
import json
import os


def prepare_dataset(
    videos_dir,
    output="dataset.json",
    default_caption="a person, natural motion, various angles and expressions, high quality",
    extensions=(".mp4", ".mov", ".avi", ".mkv", ".webm"),
    relative=False,
):
    """从视频目录生成 dataset.json，返回输出路径。"""
    videos_dir = os.path.abspath(videos_dir)
    if not os.path.isdir(videos_dir):
        raise FileNotFoundError(f"目录不存在: {videos_dir}")

    entries = []
    for name in sorted(os.listdir(videos_dir)):
        ext = os.path.splitext(name)[1].lower()
        if ext not in extensions:
            continue
        path = os.path.join(videos_dir, name)
        if not os.path.isfile(path):
            continue
        media_path = name if relative else path
        entries.append({"caption": default_caption, "media_path": media_path})

    if not entries:
        raise FileNotFoundError(f"未在 {videos_dir} 下找到任何视频文件（扩展名: {extensions}）")

    out_path = os.path.abspath(output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    print(f"已写入 {len(entries)} 条记录到 {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="生成 LTX-2 LoRA 训练用 dataset.json")
    ap.add_argument("videos_dir", help="视频所在目录（或已切好的片段目录）")
    ap.add_argument("--output", "-o", default="dataset.json", help="输出 dataset.json 路径")
    ap.add_argument("--default-caption", default="a person, natural motion, various angles and expressions, high quality")
    ap.add_argument("--extensions", nargs="+", default=[".mp4", ".mov", ".avi", ".mkv", ".webm"])
    ap.add_argument("--relative", action="store_true")
    args = ap.parse_args()
    prepare_dataset(
        args.videos_dir,
        output=args.output,
        default_caption=args.default_caption,
        extensions=tuple(args.extensions),
        relative=args.relative,
    )


if __name__ == "__main__":
    main()
