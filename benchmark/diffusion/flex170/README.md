# 双 Intel Flex 170 上的 Qwen 与 MOSS 生图

这里包含 Qwen-Image-2.1 Base / Turbo 的 ConvRot W8A8 导出、MOSS-Image-SFT 的四种量化导出、双卡生图验收和 Xe-Forge 注意力回放。九组历史实测的整段 DiT、每卡吞吐、HTTP 时间与 SSIM 见[精简结果表](RESULTS.md)。

## 环境

使用 Linux、双 Flex 170 和启用 P2P 的驱动，激活装有 XPU 版 PyTorch 的 Python 3.12 虚拟环境。原 Qwen 环境使用 PyTorch `2.14.0+xpu`、Transformers `5.19.0`、SGLang `0.5.23+xpu.qwen21.18` / `.19` 和 `sglang-kernel-xpu 0.7.0+xe.hpg.7`。量化用到 `safetensors`、`auto-round==0.16.1` 和 `llmcompressor==0.15.0`，图像指标用到 Pillow、NumPy、scikit-image。校准可以使用单独的环境，保留相同版本的 XPU PyTorch 和 Transformers。原机在校准进程的 `$VIRTUAL_ENV/opt/llmcompressor` 中单独放置 Pillow 13.0.0，服务使用 Pillow 11.3.0。

三个仓库分工如下。

| 仓库 | 使用内容 |
|---|---|
| 本仓库的 `xpu-int8-w8a8-optimized` | SGLang 模型加载、推理、W8A8 算子、Flex 双卡 P2P 与本目录脚本。[运行时代码](https://github.com/gesong2077/sglang/commit/fba9d76716dabae41cac8235161352ac88c2185f) |
| `sgl-kernel-xpu/xe-hpg-alchemist` | Flex 的专用 FA 和 Qwen packed KV 内核。原 wheel 使用与 PyTorch SYCL 运行时匹配的 oneAPI 编译器，`USE_SYCL_JIT=1`，CMake 目标 `DPCPP_SYCL_TARGET=dg2-g10`。[内核源码](https://github.com/gesong2077/sgl-kernel-xpu/tree/e7a897c12eb85e11c826cf3eb4541d3704f2a3a7) |
| `Xe-Forge/alchemist-backport` | Xe-Forge v0.4.0 计时、Flex OA 字段与逐派发 unitrace 适配器。[分析工具](https://github.com/gesong2077/Xe-Forge/tree/1c9f5c6435510d6f878799736b6d74db97ab4964) |

从本仓库根目录构建 SGLang wheel 前，先生成以下原生库。使用与 PyTorch 匹配的 oneAPI、oneDNN 和 Level Zero 开发环境，可通过 `ICPX`、`DNNL_ROOT`、`TBBROOT` 指定安装位置。SGLang 的 XPU 安装使用 `python/pyproject_xpu.toml`。

```bash
bash scripts/build_xpu_onednn_w8a8.sh
bash scripts/build_xpu_bf16_mlp.sh
bash python/sglang/srt/hardware_backend/xpu/kernels/build_flex_tp2_sum.sh
```

以下命令均从本目录运行。把模型放在同一文件系统中，原版组件和共享权重通过硬链接复用。校准工作区会保存逐层 Hessian、隐状态和 GPTQ 权重，需要数十 GB 可用空间。

```bash
export FLEX_MODELS_DIR="$HOME/models"
export FLEX_CALIBRATION_DIR="$PWD/work/calibration"
mkdir -p work
```

## 导出 Qwen Base 和 Turbo

输入目录为 `Qwen-Image-2.1`、`Qwen-Image-2.1-Turbo` 和 `ComfyUI-Qwen-Image-2.1`。ComfyUI 的 `diffusion_models/` 内应有 Base、Turbo 各自的 `*_int8_convrot.safetensors`，`text_encoders/` 内应有 `qwen3vl_8b_int8_convrot.safetensors`。

```bash
python qwen/export.py --manifest work/qwen-export.json
```

输出在模型根目录下，名称分别为 `Qwen-Image-2.1-int8-convrot-w8a8` 和 `Qwen-Image-2.1-Turbo-int8-convrot-w8a8`。脚本保留原版 VAE，拆分融合的 gate/up 投影，搬运 INT8 张量及逐行 FP32 scale，并记录 Turbo 的注意力配置。每个模型含 224 个量化 DiT 层和 254 个量化编码器层。可用 `--variant base` 或 `--variant turbo` 单独导出。原始输入对应的校验值见[导出清单](qwen/reference-export.json)。

## 导出 MOSS 四种量化模型

输入为原版 `MOSS-Image-SFT` 及 `MOSS-Image/{diffusion_models,text_encoders}` 中的 ComfyUI INT8 权重，每个子目录各一份。先导出直接使用 ComfyUI 张量的两个版本，再导出 AutoRound RTN。

```bash
python moss/export_moss.py comfy both rtn
```

`comfy` 只替换编码器，`both` 同时替换 DiT 与编码器。两者均保留 ComfyUI 实际量化的 309 个编码器权重，包括 108 个视觉投影和词嵌入。`both` 另含 252 个量化 DiT 投影。`rtn` 和 `gptq` 只量化相同的 200 个语言解码投影，保留 BF16 视觉层和词嵌入。四组都复用原始 VAE。输出目录后缀见[模型名称清单](cases.json)。

SmoothQuant + GPTQ 用 64 条英文、64 条中文和 128 条 LLaVA 图像描述文本组成 256 条长 2048 token 的样本。图像描述来自标注文本，校准走语言路径。以下命令下载并校验固定 ModelScope 修订，使用历史随机种子复现本表；省略 `--seed` 会生成新种子并写入 `packed.json`。每次变更校准种子或模型时使用新的 `FLEX_CALIBRATION_DIR`，以免沿用旧统计量。[数据来源与 SHA-256](moss/sources.json)

```bash
python moss/prepare_calibration.py --download --seed 1165361431
python moss/calibrate_gptq.py stats --device xpu:0
python moss/calibrate_gptq.py gptq --device cpu
python moss/export_moss.py gptq
python moss/verify_exports.py --output work/moss-verification.json
```

校验覆盖原始 VAE、ComfyUI 张量与 scale 的逐元素比较、QKV 拆分、目标层集合，以及 RTN/GPTQ 保留 BF16 的视觉层和词嵌入。对应的导出目录若已存在，导出器会报告冲突。

## 双卡生图与验证

```bash
python benchmark.py base-int8 --print-server
for case in base-bf16 base-int8 turbo-bf16 turbo-int8; do
    python benchmark.py "$case" --out "work/$case" || break
done
for case in moss-original moss-both moss-comfy moss-rtn moss-gptq; do
    python benchmark.py "$case" --out "work/$case" || break
done
python moss/compare_images.py work/moss-both/hot-1.png work/moss-original/hot-1.png
```

每次指定新的输出目录。程序启动当前虚拟环境的 `sglang serve`，向本机发送固定参数的 HTTP 请求并回收服务。目录中保存配置、PNG、逐卡去噪时间、独立整段诊断的算子形状及 `summary.json`。Qwen Base 有两个 40 步热请求，Turbo 有三个 8 步热请求。两组 INT8 都检查三组件与嵌入驻留、卸载管理器和首次真实编码调用，逐卡显存采样在 `resident-monitor-summary.json`。BF16 使用编码器逐层卸载。MOSS 采用 3 步预热，空闲 180 秒后连续两次 50 步，并读取两张 i915 卡的热限频标志。

复测会用归档的同权重 RGBA 摘要检查输出；Qwen 还比对每卡完整调度的对应精度算量。使用不同版本、模型或数值后端时，脚本会保留实际计数和图像，参考差异以非零退出码报告。原始参考用例与完整提示词保存在[参考清单](cases.json)和[请求参数](requests.json)。

当前运行时在支持的双 Flex 拓扑上默认启用 P2P。对照 BF16 无损指数打包时，可分别设置 `SGLANG_DIFFUSION_FLEX_TP2_PACKED_GATED=0` 和 `=1`，使用两个新的输出目录运行同一用例。

## 用 Xe-Forge 回放注意力

先从 Base BF16 的独立诊断捕获第 2 步第 5 层在两卡的真实 Q 和打包 KV。模型服务退出后，再交错计时专用 FA 与 oneDNN SDPA，并在采样位置对照 FP64。KV 解包在计时前完成。

```bash
python benchmark.py base-bf16 --out work/base-capture --capture-attention
python attention/compare.py work/base-capture 0
python attention/compare.py work/base-capture 1
```

硬件计数使用 Intel unitrace、可访问的 i915 OA 计数器及上述 Xe-Forge checkout。逐卡运行下例，`ZE_AFFINITY_MASK` 选中一张卡后，该进程内的设备索引为 0。`UNITRACE_REAL_BIN` 可指定实际的 unitrace 可执行文件。Xe-Forge 会输出实际派发的内核名称及采集目录。

```bash
export XE_FORGE_SRC=/path/to/Xe-Forge
export FLEX_CAPTURE_DIR="$PWD/work/base-capture"
for rank in 0 1; do
    for op in fa sdpa; do
        ZE_AFFINITY_MASK="$rank" FLEX_CAPTURE_RANK="$rank" FLEX_ATTENTION_BACKEND="$op" \
            xe-forge-skill profile "$PWD/attention/workload.py" --tool unitrace \
            --unitrace-bin "$XE_FORGE_SRC/scripts/unitrace-compute-query" \
            --metric-group ComputeBasic --warmup 3 --iters 12
    done
done
```

MOSS 的同输入对照使用原版 BF16 长非因果注意力。运行 `python benchmark.py moss-original --out work/moss-capture --capture-attention` 后，把上例的 `FLEX_CAPTURE_DIR` 改为 `$PWD/work/moss-capture`，工作负载改为 `attention/moss_workload.py`。两张卡分别捕获 2372 / 2314 个 token，回放保持每个 Q/K/V 张量的步幅及原 softmax scale。
