# Qwen3.8-27B ROCm/SGLang 本地运行工具

本目录保存针对 4× AMD Radeon AI PRO R9700（gfx1201）的启动脚本、基准工具和
TP4 RCCL tuner。当前阶段冻结的默认运行配置是 **TP2 + DFlash2**；脚本文件名中的
`tp4` 是历史名称，TP4 仍可通过环境变量显式启用。

## 目录依赖

- SGLang 源码：`/opt/apps/sglang-combo/sglang`
- AITER 源码：`/opt/apps/sglang-combo/aiter`
- Python 环境：`/opt/apps/sglang-combo/sglang-inference-env`
- 目标模型：`/mnt/models/Qwen3.8-27B-FP8`
- DFlash2 草稿模型：`/mnt/models/Qwen3.8-27B-DFlash2`

上述路径目前写在启动脚本顶部。迁移到其他机器时应先调整这些路径。

## 启动

先设置一个非占位 API Key：

```bash
cd /opt/apps/sglang-combo/sglang/benchmark/qwen38-tp4
export SGLANG_API_KEY='replace-with-a-private-value'
./start-sglang-qwen38-tp4.sh
```

默认配置使用 GPU 0、1，TP2，DFlash2，8 个 draft token，BF16 KV cache，
Mamba `extra_buffer`、memory ratio 6.08、BF16 SSM。主要切换方式：

```bash
# TP2，不使用投机解码
SGLANG_ENABLE_DFLASH=0 ./start-sglang-qwen38-tp4.sh

# TP2，使用模型内置 MTP/EAGLE；该模式自动使用 AITER target decode
SGLANG_ENABLE_DFLASH=0 SGLANG_ENABLE_MTP=1 ./start-sglang-qwen38-tp4.sh

# TP4，不默认启用 DFlash
SGLANG_TP_SIZE=4 ./start-sglang-qwen38-tp4.sh
```

调用者显式设置的 attention backend、设备列表和实验环境变量会覆盖脚本默认值。
不要同时启用 DFlash 和 MTP。

## 基准与检查

- `benchmark-openai-context-sweep.py`：构造应用 chat template 后精确长度的输入，测量
  TTFT/prefill 和 decode。
- `benchmark-prefill-fp8-irregular.py`：覆盖非规则 token shape 的 FP8 prefill 路径。
- `validate-greedy-semantics.py`：确定性 API 语义冒烟。
- `本地大模型推理速度测试工具v3.html`：浏览器端 OpenAI-compatible 测速工具。
- 其余 `benchmark-*.py`：attention、GDN、FP8 GEMM 和 QuickReduce 的专项微基准。

浏览器会在 `localStorage` 中保留 API Key 和采样设置；修改服务端 Key 后需要同步更新
页面设置。模型推荐采样默认值为 temperature 1.0、top-k 20、top-p 0.95。

## RCCL tuner

`rccl-tuner` 只用于显式 TP4。仓库不保存编译产物，首次使用前按照
`rccl-tuner/README.md` 重建 `libnccl-tuner-r9700.so`。

完整的阶段冻结状态、验证结果和已知限制见
[`STAGE_FREEZE_2026-08-25.md`](STAGE_FREEZE_2026-08-25.md)。
