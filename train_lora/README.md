# LTX-2 固定角色 LoRA 训练（4×4090）

在 `video_generate` 库内完成：数据准备 → 预处理 → 多卡训练。训练得到的 `.safetensors` 可直接用于 `run.py` 的 `CUSTOM_LORA_PATH`。

**现在已将「本地预处理」和「服务器训练」脚本分开：**

- 本地机（有原始视频）：使用 `run_training.py` 只做 **dataset.json + .precomputed**。
- 服务器（有多卡 GPU）：使用 `run_training_server.py` 只做 **LoRA 训练**。

## 前置条件

1. **安装 uv**（LTX-2 官方用 uv 管理依赖，若未安装可任选其一）：
   ```bash
   # 方式 A：官方安装脚本（推荐，无需 sudo，会装到 ~/.local/bin）
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # 安装后若找不到 uv，把 ~/.local/bin 加入 PATH 或执行：
   source $HOME/.local/bin/env  # 若安装脚本有创建

   # 方式 B：用 pip 安装
   pip install uv
   ```
   安装完成后执行 `uv --version` 确认可用。

2. **克隆并安装 LTX-2（本地机 & 服务器各自执行一次）**：
   ```bash
   cd /home/wenhanxiao/code   # 本地机示例路径
   git clone https://github.com/Lightricks/LTX-2
   cd LTX-2
   uv sync
   ```
3. **准备模型文件**（本地路径）：
   - **LTX-2**：若已用 `run.py` 或 `from_pretrained("Lightricks/LTX-2")` 拉过模型，权重会在 HF 缓存里，训练时可直接用 **`--model-path @cache`**（无需再写路径）。缓存目录一般为 `~/.cache/huggingface/hub/models--Lightricks--LTX-2/snapshots/.../`，脚本会优先用其中的 `ltx-2-19b-dev.safetensors`，若只有 `ltx-2-19b-dev-fp4.safetensors` 也会用该文件。
   - **Gemma 文本编码器**：需单独下载到本地目录（如 [gemma-3-12b-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized)），用 `--text-encoder-path` 指定。
4. **同一角色的视频（仅存放在本地机）**：放到一个目录下，建议多角度、多表情、多场景，总时长 5–20 分钟以上。

## 流程一：本地机（只做预处理，不训练）

1. 用编辑器打开 `train_lora/run_training.py`。
2. 在文件顶部修改配置（路径、触发词等）：
   - `LTX2_REPO`：LTX-2 仓库根目录
   - `MODEL_PATH`：`"@cache"` 或本地 .safetensors 路径
   - `TEXT_ENCODER_PATH`：Gemma 目录
   - `VIDEOS_DIR`：角色视频目录
   - `TRIGGER`：触发词（如 `MYCHAR_001`）
   - 其余如 `RESOLUTION_BUCKETS`、`WITH_AUDIO`、`SKIP_PREPROCESS` 等按需改。
3. 保存后在本地机运行：
```bash
cd /home/wenhanxiao/code/video_generate/train_lora
python run_training.py
```

运行完成后，本地机会得到：

- `train_lora/data/dataset.json`
- `train_lora/data/.precomputed/`

然后你可以用 `scp` / `rsync` 等工具将这两个文件/目录同步到服务器上对应位置。

## 流程二：服务器（只做训练，不接触原视频）

1. 确保服务器上有：
   - `video_generate/train_lora/data/dataset.json`
   - `video_generate/train_lora/data/.precomputed/`
2. 用编辑器打开 `train_lora/run_training_server.py`，修改顶部配置：
   - `LTX2_REPO`：服务器上的 LTX-2 仓库根目录
   - `MODEL_PATH`：`"@cache"` 或服务器上 LTX-2 权重路径
   - `TEXT_ENCODER_PATH`：服务器上的 Gemma 目录
   - `PREPROCESSED_ROOT`：服务器上的 `.precomputed` 路径
   - `TRIGGER`：需与本地预处理时一致
   - `OUTPUT_DIR`：训练输出目录
   - `NUM_PROCESSES`：服务器 GPU 数量
3. 在服务器上运行：

```bash
cd /path/to/video_generate/train_lora
python run_training_server.py
```

训练结束后，LoRA 权重会出现在 `OUTPUT_DIR`（例如 `outputs/char_lora/checkpoints/step-3000-lora.safetensors`）。

## 流程三：在单台机器上全流程（命令行一键，可选）

如果你不区分本地 / 服务器，所有步骤都在同一台多卡机器上完成，也可以继续使用命令行一键脚本：

```bash
cd /home/wenhanxiao/code/video_generate/train_lora

python run_all.py \
  --ltx2-repo /path/to/LTX-2 \
  --model-path @cache \
  --text-encoder-path /path/to/gemma-3-12b-it-qat-q4_0-unquantized \
  --videos-dir /path/to/我的角色视频目录 \
  --trigger MYCHAR_001
```

- 默认 4 卡；少于 4 卡时加 `--num-processes 2` 等。
- 已预处理过、只想重新训练时：先准备好 `data/dataset.json` 和 `data/.precomputed`，再单独运行 `run_train.py`（见下方分步说明）。

## 流程四：分步执行（命令行）

### 1. 生成 dataset.json

```bash
python prepare_dataset.py /path/to/视频目录 --output data/dataset.json
```

可按需加 `--default-caption "你的默认描述"`。如需为每个视频写不同 caption，可手动编辑生成的 `data/dataset.json`。

### 2. 预处理（算 latent + 触发词）

```bash
python run_preprocess.py \
  --ltx2-repo /path/to/LTX-2 \
  --model-path @cache \
  --text-encoder-path /path/to/gemma \
  --dataset-json data/dataset.json \
  --lora-trigger MYCHAR_001
```

预处理结果在 `data/.precomputed`。

### 3. 启动 4 卡训练

```bash
python run_train.py \
  --ltx2-repo /path/to/LTX-2 \
  --model-path @cache \
  --text-encoder-path /path/to/gemma \
  --preprocessed-root data/.precomputed \
  --trigger MYCHAR_001 \
  --num-processes 4
```

训练输出默认在 `train_lora/outputs/char_lora`，其中会有 checkpoint 与最终 LoRA 权重（如 `step-3000-lora.safetensors`）。

## 在 run.py 里使用训练好的 LoRA

1. 在 `video_generate/run.py` 中设置：
   ```python
   CUSTOM_LORA_PATH = "/home/wenhanxiao/code/video_generate/train_lora/outputs/char_lora/checkpoints/step-3000-lora.safetensors"
   CUSTOM_LORA_NAME = "my_char_lora"
   CUSTOM_LORA_SCALE = 0.8
   ```
2. 在 `BASE_PROMPT` 开头加上与训练时一致的触发词：
   ```python
   BASE_PROMPT = "MYCHAR_001, " + "你原来的描述..."
   ```
3. 运行 `python run.py` 即可。

## 文件说明

| 文件 | 说明 |
|------|------|
| `configs/ltx2_char_lora_4x4090.yaml` | 训练配置模板（路径由脚本替换） |
| `prepare_dataset.py` | 从视频目录生成 dataset.json |
| `run_preprocess.py` | 调用 LTX-2 的 process_dataset.py |
| `run_train.py` | 生成最终配置并启动 accelerate 训练 |
| `run_training.py` | **本地预处理入口**：改顶部配置后在本地机运行，只做数据准备和预处理 |
| `run_training_server.py` | **服务器训练入口**：改顶部配置后在服务器上运行，只做 LoRA 训练 |
| `run_all.py` | 命令行一键：数据 → 预处理 → 训练（适合本地=服务器的场景） |
| `data/` | 默认放 dataset.json 与 .precomputed |
| `outputs/` | 默认训练输出目录 |

## 常见问题

- **用 HF 缓存（@cache）**：若你已用 `run.py` 或 diffusers 加载过 `Lightricks/LTX-2`，权重会在 `~/.cache/huggingface/hub/models--Lightricks--LTX-2/snapshots/<commit>/` 下。训练时 `--model-path @cache` 会自动用该目录下的 `.safetensors`（优先完整版，若无则用 `ltx-2-19b-dev-fp4.safetensors`）。若 LTX-2 官方 trainer 要求完整 19B 而缓存里只有 fp4，需再单独下载 `ltx-2-19b-dev.safetensors` 并传本地路径。
- **显存不足**：在 LTX-2 仓库中可改用 `configs/ltx2_av_lora_low_vram.yaml` 思路（减小 batch、启用 8bit 等），或在本库的 config 中调小 `resolution_buckets` / `batch_size`。
- **触发词不生效**：确保 `run.py` 里 `BASE_PROMPT` 开头的触发词与 `--lora-trigger` / `--trigger` 完全一致（如 `MYCHAR_001`）。
- **多卡报错**：首次运行可先 `cd /path/to/LTX-2 && uv run accelerate config` 按提示配置多 GPU。
