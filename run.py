import os

# 不在脚本里做多卡或物理卡映射，只使用进程内的 cuda:0。
# 若需要限制可见 GPU，请在命令行用 CUDA_VISIBLE_DEVICES 控制。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video
from PIL import Image

# ====================== 基础配置 ======================
# 设了 CUDA_VISIBLE_DEVICES 后进程内只有一张卡，主设备用 cuda:0
DEVICE = "cuda:0"
# 使用官方 Wan2.2 I2V Diffusers 模型
MODEL_ID = "lopho/Wan2.2-I2V-A14B-Diffusers_nf4"

# 输出视频路径
OUTPUT_VIDEO_PATH = "wan_i2v_output.mp4"

# ====================== 条件输入配置 ======================
# 一张或多张图片路径（Wan2.2 I2V 仅支持 image-to-video）
IMAGE_PATHS = [
    "start_frame.png",
]

# ==================== LoRA 配置（可选） ====================
# Wan2.2 I2V 兼容的 LoRA（如 DR34ML4Y_I2V_14B_HIGH_V2）；None 则不加载
#CUSTOM_LORA_PATH = None
CUSTOM_LORA_PATH = os.path.join(os.path.dirname(__file__), "train_lora", "DR34ML4Y_I2V_14B_HIGH_V2.safetensors")
CUSTOM_LORA_SCALE = 0.6

# ==================== 生成参数 ====================
# 统一目标输出分辨率（你想要的视频尺寸，所有输入图像都会被等比例缩放+填充到这个画布上）
# 例如竖屏 480x832；注意实际送入模型时会按 Wan 的 mod_value 向下对齐到合法分辨率
TARGET_WIDTH = 480
TARGET_HEIGHT = 832
# 是否允许根据模型约束自动把分辨率向下对齐到最近的合法值（建议 True，避免 OOM 或尺寸报错）
ALLOW_AUTO_ADJUST_RESOLUTION = True
# 等比例缩放后用于填充剩余区域的背景色（RGB）
LETTERBOX_BACKGROUND_COLOR = (0, 0, 0)

# 24GB 显存 + 14B 模型非常吃紧，可以通过调低分辨率和帧数保证能跑完
NUM_FRAMES = 81
FRAME_RATE = 20
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 1
SEED = 42

# ====================== 提示词 ======================
BASE_PROMPT = (
        "same girl as in the reference image, keep her face and hairstyle the same, "   
        "very small natural movements, no big pose or camera change."
)
PROMPT = BASE_PROMPT

NEGATIVE_PROMPT = (
    "different person from the reference, face not matching the reference image, changing hairstyle, changing outfit, changing background,Facial distortion, face collapse, facial features inconsistent with reference image, limb deformity, messed up fingers, blurry footage, excessive noise, vulgar exposure, pornographic content, exaggerated distorted movements, messy inconsistent background, severe shaking, low frame rate, goofs, extra characters, wrong hair color or outfit, horror elements, unnatural lighting, picture breakdown, stiff unnatural movements."
)

def _load_pipeline():
    """加载 Wan2.2 I2V pipeline。

    - NF4 量化版（如 lopho/Wan2.2-I2V-A14B-Diffusers_nf4）需要 expand_timesteps=True，走 16 通道分支；
    - 官方全精度版（Wan-AI/Wan2.2-I2V-A14B-Diffusers）保持默认配置。
    单卡 + CPU offload，避免多卡/CPU 混用导致的设备不一致错误。
    """
    print(f"正在加载模型: {MODEL_ID} ...")
    load_kwargs = {"torch_dtype": torch.bfloat16}
    pipe = WanImageToVideoPipeline.from_pretrained(MODEL_ID, **load_kwargs)
    # NF4 版权重的 patch_embedding 期望 16 通道输入，而默认 I2V 会堆成 36 通道；需开启 expand_timesteps 走 16 通道路径
    if "Diffusers_nf4" in MODEL_ID or getattr(pipe.config, "expand_timesteps", None) is not None:
        setattr(pipe.config, "expand_timesteps", True)
    # 单卡：整模型逻辑上放到 DEVICE，并开启 CPU offload 以撑住大模型
    pipe.enable_model_cpu_offload(device=DEVICE)

    # 可选：加载自定义 LoRA（需为 Wan I2V 兼容格式）
    if CUSTOM_LORA_PATH and os.path.exists(CUSTOM_LORA_PATH):
        try:
            print(f"加载自定义 LoRA: {CUSTOM_LORA_PATH}")
            pipe.load_lora_weights(CUSTOM_LORA_PATH)
            adapters = pipe.get_list_adapters()
            names = [a for adapters_list in adapters.values() for a in adapters_list]
            names = list(dict.fromkeys(names))
            if names:
                pipe.set_adapters(names, adapter_weights=[CUSTOM_LORA_SCALE] * len(names))
        except Exception as e:
            print(f"LoRA 加载失败（将仅用基座生成）: {e}")

    return pipe


def _get_mod_value(pipe) -> int:
    """获取 Wan I2V 在空间维度上的步长约束（宽高必须是该值的整数倍）。"""
    return pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]


def _normalize_resolution(pipe, target_width: int, target_height: int) -> tuple[int, int]:
    """根据模型的 mod_value，把用户指定的分辨率向下对齐到合法分辨率。"""
    mod_value = _get_mod_value(pipe)
    if not ALLOW_AUTO_ADJUST_RESOLUTION:
        # 不自动调整时，仅保证不小于 mod_value；超出部分可能会报错或 OOM，由用户自己控制
        width = max(mod_value, target_width)
        height = max(mod_value, target_height)
        return width, height

    width = max(mod_value, (target_width // mod_value) * mod_value)
    height = max(mod_value, (target_height // mod_value) * mod_value)
    return width, height


def _resize_with_letterbox(
    image: Image.Image,
    canvas_width: int,
    canvas_height: int,
    background_color: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """等比例缩放图片并在指定画布上居中填充，避免拉伸原图比例。"""
    src_width, src_height = image.size
    # 等比例缩放到“完全装进画布”为止（可能上下或左右留边）
    scale = min(canvas_width / src_width, canvas_height / src_height)
    new_width = max(1, int(round(src_width * scale)))
    new_height = max(1, int(round(src_height * scale)))

    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (canvas_width, canvas_height), background_color)
    offset_x = (canvas_width - new_width) // 2
    offset_y = (canvas_height - new_height) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas


def run_generation():
    pipe = _load_pipeline()
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)

    if not IMAGE_PATHS:
        raise ValueError("IMAGE_PATHS 为空，请至少指定一张输入图片")

    base, ext = os.path.splitext(OUTPUT_VIDEO_PATH)
    multiple = len(IMAGE_PATHS) > 1

    # 统一计算一次实际送入模型的分辨率
    target_width, target_height = _normalize_resolution(pipe, TARGET_WIDTH, TARGET_HEIGHT)
    mod_value = _get_mod_value(pipe)
    print(
        f"目标输出尺寸（用户设定）：{TARGET_WIDTH}×{TARGET_HEIGHT}，"
        f"实际送入模型尺寸：{target_width}×{target_height}（mod_value={mod_value}）"
    )

    for idx, img_path in enumerate(IMAGE_PATHS, start=1):
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"找不到图片：{img_path}")

        print(f"基于图片生成视频（{idx}/{len(IMAGE_PATHS)}）：{img_path}")
        image = Image.open(img_path).convert("RGB")

        # 等比例缩放 + 居中填充到统一尺寸，避免拉伸原图比例
        image = _resize_with_letterbox(
            image,
            canvas_width=target_width,
            canvas_height=target_height,
            background_color=LETTERBOX_BACKGROUND_COLOR,
        )
        width, height = image.size
        print(f"  生成尺寸（等比例+填充后）：{width}×{height}")

        output = pipe(
            image=image,
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            height=height,
            width=width,
            num_frames=NUM_FRAMES,
            guidance_scale=GUIDANCE_SCALE,
            num_inference_steps=NUM_INFERENCE_STEPS,
            generator=generator,
        )

        frames = output.frames[0]
        out_path = OUTPUT_VIDEO_PATH if not multiple else f"{base}_{idx}{ext}"
        export_to_video(frames, out_path, fps=FRAME_RATE)
        print(f"✅ 生成完成！输出文件：{out_path}")


if __name__ == "__main__":
    run_generation()
