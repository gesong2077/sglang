# 2026-08-25 阶段冻结清单

## 状态

这是阶段冻结候选，不是上游发布版本。冻结验证期间没有运行 vLLM。本工具快照已作为
`benchmark/qwen38-tp4/` 并入 SGLang 优化分支，目录内不包含嵌套 `.git`。

## 版本基线

| 组件 | 本地分支/环境 | 基线或当前 HEAD |
| --- | --- | --- |
| SGLang | `local/v0.5.18-rocm-gfx1201` | freeze commit `8fcb9e29c45b`; v0.5.18 commit `71de97b264b0` |
| AITER | `local/sglang-v0.5.18-gfx1201` | freeze commit `bf7692fa671b` |
| 部署/基准工具 | 导入 SGLang 分支 | 源快照 `bf0c0e0317f`; bundle tag `qwen38-rocm-sglang-bundle-20260825` |
| DFlash2 参考实现 | 已移植到当前 SGLang 工作树 | SGLang commit `1cf2b8c54d81802abc15dcf23a29b9cc687bc01e` |
| PyTorch | `sglang-inference-env` | `2.12.0+rocm7.14.0`, HIP `7.14.60850` |
| GPU | 4 卡可见 | 4× R9700, gfx1201, 34,208,743,424 bytes/card |

SGLang 冻结提交包含 41 个文件（+4069/-316），AITER 冻结提交包含 12 个文件
（+605/-12）。三个仓库在冻结后均为 clean worktree；运行缓存、profile trace、pickle、
日志、编译 `.so` 和可再生成的 in-place hipify 产物不属于冻结快照。

## 冻结配置

- 无环境变量时默认 TP2（GPU 0、1）并启用本地 DFlash2，draft token 数为 8。
- 主模型 attention/prefill/decode 使用 Triton；长 prefill 的既有 KV chunk 可路由到
  AITER Unified；DFlash draft attention 使用 Triton。
- MTP 是 opt-in。启用后 target decode 默认改为 AITER Unified，draft attention 使用
  AITER，并启用 ReplaySSM 与 FP8 lm-head shortlist。
- 多模态保持开启，视觉 attention 为 `triton_attn`，video 每请求限制为 0。
- TP2 开启针对小消息的 QuickReduce；TP4 保持已验证的 RCCL tuner 路径。
- KV cache 保持 BF16。FP8 KV cache 在本机测试中没有速度收益，因此未设为默认。
- 模型采样默认值由 generation config 提供：temperature 1.0、top-k 20、top-p 0.95。
- 启动入口已使用受支持的 `sglang serve`。

## 本轮最终验证

静态与单元测试：

- SGLang 与 AITER：`git diff --check` 通过。
- 启动脚本：`bash -n` 通过。
- 所有本轮修改的 Python 文件：`py_compile` 通过。
- SGLang DFlash、Qwen3.5 weight loader、GDN backend policy：30 passed，另有 22
  subtests passed。
- AITER fused SiLU/FP8 与 gated RMSNorm/FP8：142 passed。

默认 TP2+DFlash 真实启动成功，服务健康检查和 OpenAI chat 请求成功。精确输入、单并发、
输出 256 token、每点一次的冻结冒烟结果：

| prompt token | prefill token/s | decode token/s | DFlash accept len（日志） |
| ---: | ---: | ---: | ---: |
| 1024 | 3463.69 | 86.85 | 3.73 |
| 6503（非规则） | 3801.02 | 52.64 | 约 3.08–3.20 |

这组数据用于验证冻结配置没有功能或明显性能回退，不应当作多轮统计结论。服务已正常停止，
4 卡 VRAM 已释放到约 60 MB/card。

## 已知限制

- DFlash 接受率强依赖输入与采样。短上下文可到 80+ token/s，但当前不能保证平均 90+
  token/s；6503 token 的本次结果为 52.64 token/s。
- 默认 DFlash + Mamba ratio 6.08 启动时只分配到 28,864 个 full-KV token。虽然模型
  `context-length` 为 262,144，这个默认 profile 不等价于可直接承载 128K 单请求；长上下文
  应关闭 DFlash并重新选择 Mamba/KV 内存比例，或另建经过验证的 long-context profile。
- AITER/FlashInfer 的大词表 selector 当前不可用，启动日志会提示回退到 `torch.topk`；这是
  后续 DFlash decode 的主要候选优化点。
- MTP 的 AITER target verify 明显优于 Triton，但当前实测约 62.8–66.4 token/s，未达到
  vLLM 代码路径所暗示的 85+ 目标，所以未设为默认。
- 可选多模态 processor 会报告缺少 `cutlass`、`torchcodec` 以及部分非本模型 processor
  的导入告警；Qwen3.8 图像处理器和服务启动正常，音频输入没有依赖保障。
- HTML 内含用于公开排行榜的 Supabase anon key。公开仓库前必须确认对应 Supabase 表已启用
  正确的 RLS/限流策略；anon key 不是服务端 secret，但不能代替访问控制。

## GitHub 提交建议

当前采用两个源码仓库和一个 SGLang 内工具目录：

1. SGLang fork：从 `v0.5.18` 建发布分支，按 DFlash/Qwen、gfx1201 attention/GDN、
   FP8 GEMM/quant、TP2 QuickReduce、MTP/通用修复拆分提交。
2. AITER fork：按 gfx1201 Unified Attention、fused quant、tuned GEMM/custom AR 拆分提交。
3. 本目录：随 SGLang 优化分支发布 launcher、Python/HTML 工具、RCCL tuner 源码和配置；
   不提交 `__pycache__`、profile trace、pickle、日志或编译后的 `.so`。

SGLang 与 AITER 的 `origin` 仍指向官方上游，冻结分支发布到 `gesong2077` fork，不直接
推送上游 `origin`。SGLang 仓库许可证适用于并入其中的工具文件；第三方 RCCL tuner 源码
保留原始版权头。本地冻结提交使用 `Codex Migration <codex-migration@local>` 身份；
如需归属到仓库所有者，可在进一步开发分支中调整提交作者。
