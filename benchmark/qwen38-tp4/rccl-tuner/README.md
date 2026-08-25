# RCCL tuner for 4x R9700

This is the official RCCL CSV tuner example, pinned from ROCm/rccl commit
`57e58688f44c77076ad536ef1f6b68741fc6e694`, plus the local message-size
configuration in `r9700-tp4.conf`.

The launcher loads `libnccl-tuner-r9700.so`. Rebuild the binary after changing
the source or moving to an incompatible RCCL release:

```bash
cd /opt/apps/sglang-combo/sglang/benchmark/qwen38-tp4/rccl-tuner
cc -Isource/nccl -O2 -fPIC -shared \
  -Wl,-soname,libnccl-tuner-r9700.so \
  -o libnccl-tuner-r9700.so source/plugin.c
```

The plugin has no ROCm runtime dependency; it exposes the RCCL tuner v5 ABI and
only depends on libc. `NCCL_TUNER_PLUGIN` and `NCCL_TUNER_CONFIG_FILE` remain
overridable for focused experiments.
