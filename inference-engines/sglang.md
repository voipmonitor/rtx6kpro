# SGLang on RTX 6000 Pro Blackwell

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Configuration Flags Reference](#configuration-flags-reference)
- [Model-Specific Launch Commands](#model-specific-launch-commands)
  - [Qwen3.5-397B NVFP4 (4 GPUs)](#qwen35-397b-nvfp4-4-gpus)
  - [Qwen3.5-397B NVFP4 (8 GPUs)](#qwen35-397b-nvfp4-8-gpus)
  - [Qwen3.5-397B FP8 (8 GPUs)](#qwen35-397b-fp8-8-gpus)
  - [Kimi K2.5 INT4 (8 GPUs)](#kimi-k25-int4-8-gpus)
  - [Kimi K2.5 NVFP4 (8 GPUs)](#kimi-k25-nvfp4-8-gpus)
  - [GLM-5 NVFP4 with MTP (8 GPUs)](#glm-5-nvfp4-with-mtp-8-gpus)
  - [GLM-5 NVFP4 without MTP (8 GPUs)](#glm-5-nvfp4-without-mtp-8-gpus)
- [MTP in SGLang (SGLANG_ENABLE_SPEC_V2)](#mtp-in-sglang)
- [Decode Context Parallel (DCP)](#decode-context-parallel-dcp)
- [MOE Runner Backend Selection](#moe-runner-backend-selection)
- [EAGLE Speculative Decoding](#eagle-speculative-decoding)
- [FlashInfer Allreduce Fusion](#flashinfer-allreduce-fusion)
- [Performance Tuning Tips](#performance-tuning-tips)
- [Profiling](#profiling)
- [Relevant PRs](#relevant-prs)

---

## Overview

SGLang is the primary inference engine for GLM-5 on SM120 (it is the only engine that works) and a high-performance alternative to vLLM for Qwen3.5 and Kimi K2.5. It provides flexible backend selection, native MTP support via NEXTN speculative decoding, and FlashInfer-based allreduce fusion optimized for PCIe.

Key strengths on RTX 6000 Pro:

- Only engine that runs GLM-5 on SM120 (bypasses DSA backends with FlashInfer FA2)
- Flexible MOE runner backends (cutlass, flashinfer_cutlass, triton)
- FlashInfer allreduce fusion for PCIe-connected GPUs
- Expert parallel support with custom allreduce for switch topologies

---

## Installation

### Docker (recommended)

```bash
sudo docker pull lmsysorg/sglang:dev-cu13
sudo docker run -it --rm \
  -v /home/gpusvr/:/home/gpusvr/ \
  --ipc=host --shm-size=8g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus all --network host \
  lmsysorg/sglang:dev-cu13 bash
```

### Custom Docker (GLM-5 / SM120 patched)

For GLM-5 and general SM120 workloads, Festr's image is recommended:

```bash
docker pull voipmonitor/llm-pytorch-blackwell:nightly
docker run -it --rm \
    --entrypoint /bin/bash \
    --gpus all \
    --ipc=host --shm-size=8g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --network host \
    --cpuset-cpus "0-63" \
    -v /root/.cache/huggingface:/root/.cache/huggingface \
    -v /mnt:/mnt \
    -v vllm-nightly-jit:/cache/jit \
    voipmonitor/llm-pytorch-blackwell:nightly
```

This image includes:
- SGLang compiled from source with SM120 patches
- PyTorch 2.12 with CUTLASS 4.4.1, cuDNN 91901
- SM_120f compilation target enabled
- Pre-generated Triton MoE kernel configs for RTX PRO 6000 Blackwell Server Edition

### Patching the Official GLM-5 Image

If using `lmsysorg/sglang:glm5-blackwell` or `lmsysorg/sglang:dev-cu13`:

```dockerfile
FROM lmsysorg/sglang@sha256:426d1fa4b10722688678b99d817c2caa92a89eed4a8ee2927ab44a848bbe77df

RUN pip install --no-cache-dir transformers==5.2.0

# Fix DeepGemm scale format detection for NVFP4 models on Blackwell (SM120)
RUN sed -i "s/DEEPGEMM_SCALE_UE8M0 = DEEPGEMM_BLACKWELL/DEEPGEMM_SCALE_UE8M0 = False/" \
    /sgl-workspace/sglang/python/sglang/srt/layers/deep_gemm_wrapper/configurer.py
```

---

## Configuration Flags Reference

### Server Flags

| Flag | Description |
|------|-------------|
| `--model-path <path>` | HuggingFace model ID or local path. |
| `--tp N` / `--tensor-parallel-size N` | Number of GPUs for TP. Must be power-of-2 for most models. |
| `--pp N` | Pipeline parallel stages. Use for non-power-of-2 (e.g., `--tp 2 --pp 3` for 6 GPUs). |
| `--ep N` | Expert parallel size. Requires custom allreduce patches for PCIe. |
| `--quantization modelopt_fp4` | Required for NVFP4 checkpoints. |
| `--kv-cache-dtype fp8_e4m3` | FP8 KV cache. **Does not work for GLM-5 on SM120** (use `bf16`). |
| `--kv-cache-dtype bf16` | BF16 KV cache. Mandatory for GLM-5 on SM120. |
| `--attention-backend <backend>` | `flashinfer`, `triton`. `flashinfer` for GLM-5; `triton` for Qwen3.5/Kimi. |
| `--moe-runner-backend <backend>` | `cutlass`, `flashinfer_cutlass`, `triton`, `deep_gemm`. See [MOE Runner Backend](#moe-runner-backend-selection). |
| `--fp4-gemm-backend <backend>` | `flashinfer_cutlass` (default), `flashinfer_cudnn` (recommended). |
| `--fp8-gemm-backend triton` | Use Triton for FP8 GEMMs. |
| `--mem-fraction-static 0.85-0.94` | Static VRAM allocation fraction. |
| `--cuda-graph-max-bs N` | Max batch size for CUDA graph capture. 4-32 depending on model. |
| `--max-running-requests N` | Maximum concurrent requests. |
| `--context-length N` | Override maximum context length. |
| `--chunked-prefill-size N` | Chunk size for prefill. 4096-32768 typical. |
| `--disable-custom-all-reduce` | Required for PCIe-only setups (custom allreduce is NVLink-optimized). |
| `--enable-flashinfer-allreduce-fusion` | Fuse allreduce with attention. Measurable throughput gain. |
| `--page-size 64` | KV cache page size. |
| `--mamba-scheduler-strategy extra_buffer` | Scheduler strategy for hybrid models (Qwen3.5 GDN). |
| `--skip-server-warmup` | Skip warmup, saves ~3 min startup. |
| `--enable-mixed-chunk` | Enable mixed chunked prefill. |
| `--model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 16}'` | Multi-threaded model loading. |

### Tool Calling and Reasoning

| Flag | Description |
|------|-------------|
| `--tool-call-parser <parser>` | `qwen3_coder`, `qwen3`, `kimi_k2`, `glm47`. |
| `--reasoning-parser <parser>` | `qwen3`, `kimi_k2`, `glm45`. |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `SGLANG_ENABLE_SPEC_V2=True` | **Mandatory for MTP**. Without it, NEXTN falls back to EAGLE, loading model twice (OOM). |
| `SGLANG_ENABLE_JIT_DEEPGEMM=0` | Disable DeepGemm JIT. SM120 not supported by DeepGemm. |
| `SGLANG_ENABLE_DEEP_GEMM=0` | Fully disable DeepGemm fallback path. |
| `SGLANG_DISABLE_DEEP_GEMM=1` | Alternative flag to disable deep GEMM. |
| `SGLANG_SET_CPU_AFFINITY=1` | Bind to CPU cores for reduced jitter. |
| `SGLANG_DISABLE_CUDNN_CHECK=1` | Skip cuDNN version check. |
| `NCCL_P2P_LEVEL=SYS` | Enable P2P across system. |
| `NCCL_IB_DISABLE=1` | Disable InfiniBand. |
| `NCCL_MIN_NCHANNELS=8` | Increase NCCL channels (significant bandwidth improvement). |
| `NCCL_ALLOC_P2P_NET_LL_BUFFERS=1` | Allocate LL buffers for P2P. |
| `SAFETENSORS_FAST_GPU=1` | Faster weight loading. |
| `OMP_NUM_THREADS=8` | Limit OpenMP threads. |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Expandable CUDA memory segments. |
| `FLASHINFER_DISABLE_VERSION_CHECK=1` | Skip FlashInfer version check. |

---

## Model-Specific Launch Commands

### Qwen3.5-397B NVFP4 (4 GPUs)

Stable baseline without MTP (~42-85 tok/s):

```bash
SGLANG_DISABLE_DEEP_GEMM=1 \
NCCL_IB_DISABLE=1 \
NCCL_P2P_LEVEL=PHB \
OMP_NUM_THREADS=8 \
SAFETENSORS_FAST_GPU=1 \
python -m sglang.launch_server \
  --model-path nvidia/Qwen3.5-397B-A17B-NVFP4 \
  --tp-size 4 \
  --host 0.0.0.0 --port 8000 \
  --trust-remote-code \
  --mem-fraction-static 0.85 \
  --quantization modelopt_fp4 \
  --attention-backend triton \
  --moe-runner-backend flashinfer_cutlass \
  --fp4-gemm-backend flashinfer_cudnn \
  --context-length 262144 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --sleep-on-idle
```

With MTP and full tuning (~130 tok/s):

```bash
NCCL_P2P_LEVEL=4 \
SGLANG_ENABLE_SPEC_V2=1 \
python3 -m sglang.launch_server \
  --model-path nvidia/Qwen3.5-397B-A17B-NVFP4 \
  --tp 4 \
  --trust-remote-code \
  --attention-backend flashinfer \
  --moe-runner-backend flashinfer_cutlass \
  --kv-cache-dtype fp8_e4m3 \
  --tool-call-parser qwen3 \
  --reasoning-parser qwen3 \
  --quantization modelopt_fp4 \
  --disable-custom-all-reduce \
  --enable-flashinfer-allreduce-fusion \
  --mem-fraction-static 0.9 \
  --cuda-graph-max-bs 8 \
  --host 0.0.0.0 --port 5000 \
  --served-model-name qwen3.5 \
  --max-running-requests 8 \
  --fp8-gemm-backend triton \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 16}' \
  --sleep-on-idle
```

### Qwen3.5-397B NVFP4 (8 GPUs)

Fastest reported config (~350 tok/s by luke, 8x MaxQ with PCIe switches):

```bash
export SGLANG_ENABLE_SPEC_V2=True

python3 -m sglang.launch_server \
  --model nvidia/Qwen3.5-397B-A17B-NVFP4 \
  --served-model-name Qwen3.5 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --tensor-parallel-size 8 \
  --quantization modelopt_fp4 \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code \
  --attention-backend triton \
  --moe-runner-backend flashinfer_cutlass \
  --fp4-gemm-backend flashinfer_cudnn \
  --cuda-graph-max-bs 4 \
  --max-running-requests 4 \
  --context-length 262144 \
  --chunked-prefill-size 32768 \
  --speculative-algo NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --mamba-scheduler-strategy extra_buffer \
  --page-size 64 \
  --mem-fraction-static 0.85 \
  --sleep-on-idle \
  --host 0.0.0.0 --port 8000
```

### Qwen3.5-397B FP8 (8 GPUs)

```bash
python -m sglang.launch_server \
  --model-path /home/gpusvr/Qwen3.5-397B-A17B-FP8 \
  --host 0.0.0.0 --port 9501 \
  --tp-size 8 \
  --mem-fraction-static 0.8 \
  --context-length 262144 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --served-model-name llm_model \
  --speculative-algo NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --attention-backend triton \
  --fp8-gemm-backend triton \
  --moe-runner-backend triton \
  --sleep-on-idle
```

### Kimi K2.5 INT4 (8 GPUs)

Original BF16 KV cache (fastest single batch, limited context):

```bash
NCCL_P2P_LEVEL=4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
SGL_ENABLE_JIT_DEEPGEMM=0 \
python -m sglang.launch_server \
  --model moonshotai/Kimi-K2.5 \
  --tp 8 \
  --host 0.0.0.0 --port 5000 \
  --mem-fraction-static 0.94 \
  --enable-metrics \
  --sleep-on-idle \
  --attention-backend flashinfer \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2 \
  --served-model-name kimi-k2.5 \
  --chunked-prefill-size 8092 \
  --cuda-graph-max-bs 16 \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 16}' \
  --trust-remote-code \
  --enable-mixed-chunk
```

Result: 73-90 tok/s decode, BF16 KV cache, ~170K-232K context tokens.

**Warning**: Using `--kv-cache-dtype fp8` with the original Kimi checkpoint in SGLang drops speed to **16 tok/s** (unusable).

### Kimi K2.5 NVFP4 (8 GPUs)

```bash
NCCL_P2P_LEVEL=4 python -m sglang.launch_server \
  --model-path nvidia/Kimi-K2.5-NVFP4 \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --reasoning-parser kimi_k2 \
  --tool-call-parser kimi_k2 \
  --moe-runner-backend triton \
  --quantization modelopt_fp4 \
  --model-loader-extra-config '{"enable_multithread_load": true,"num_threads": 119}' \
  --mem-fraction-static 0.93 \
  --cuda-graph-max-bs 8 \
  --sleep-on-idle
```

Result: 53 tok/s, ~450K KV cache with FP8 cache. NVFP4 variant is slower than native INT4 with Marlin kernels.

### GLM-5 NVFP4 with MTP (8 GPUs)

Best known working command:

```bash
SGLANG_ENABLE_SPEC_V2=True \
SGLANG_ENABLE_JIT_DEEPGEMM=0 \
SGLANG_ENABLE_DEEP_GEMM=0 \
NCCL_GRAPH_FILE=/mnt/nccl_graph_opt.xml \
NCCL_IB_DISABLE=1 \
NCCL_P2P_LEVEL=SYS \
NCCL_ALLOC_P2P_NET_LL_BUFFERS=1 \
NCCL_MIN_NCHANNELS=8 \
OMP_NUM_THREADS=8 \
SAFETENSORS_FAST_GPU=1 \
python3 -m sglang.launch_server \
  --model-path /mnt/GLM-5-NVFP4-MTP \
  --tp 8 \
  --trust-remote-code \
  --attention-backend flashinfer \
  --moe-runner-backend cutlass \
  --kv-cache-dtype bf16 \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --quantization modelopt_fp4 \
  --disable-custom-all-reduce \
  --enable-flashinfer-allreduce-fusion \
  --mem-fraction-static 0.85 \
  --cuda-graph-max-bs 32 \
  --host 0.0.0.0 --port 5000 \
  --served-model-name glm-5 \
  --max-running-requests 64 \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}' \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-num-draft-tokens 4 \
  --speculative-eagle-topk 1 \
  --enable-metrics \
  --sleep-on-idle
```

Use `festr2/GLM-5-NVFP4-MTP` (includes MTP layer 78 in BF16). The `lukealonso/GLM-5-NVFP4` checkpoint does NOT include MTP weights.

**Critical**: `SGLANG_ENABLE_SPEC_V2=True` is mandatory. Without it, SGLang converts NEXTN to EAGLE and loads the model twice, causing instant OOM (57 GB x 2 = 114 GB per GPU on 96 GB cards).

**Critical**: `--kv-cache-dtype bf16` is mandatory on SM120. FP8 KV cache produces garbled output or stops after 1 token.

### GLM-5 NVFP4 without MTP (8 GPUs)

```bash
NCCL_GRAPH_FILE=/mnt/nccl_graph_opt.xml \
NCCL_IB_DISABLE=1 \
NCCL_P2P_LEVEL=SYS \
NCCL_ALLOC_P2P_NET_LL_BUFFERS=1 \
NCCL_MIN_NCHANNELS=8 \
OMP_NUM_THREADS=8 \
SAFETENSORS_FAST_GPU=1 \
python3 -m sglang.launch_server \
  --model-path lukealonso/GLM-5-NVFP4 \
  --tp 8 \
  --trust-remote-code \
  --attention-backend flashinfer \
  --moe-runner-backend flashinfer_cutlass \
  --kv-cache-dtype bf16 \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --quantization modelopt_fp4 \
  --disable-custom-all-reduce \
  --enable-flashinfer-allreduce-fusion \
  --mem-fraction-static 0.9 \
  --cuda-graph-max-bs 8 \
  --host 0.0.0.0 --port 5000 \
  --served-model-name glm-5 \
  --max-running-requests 8 \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}' \
  --sleep-on-idle
```

---

## MTP in SGLang

### Configuration

```bash
# Environment variable (MANDATORY):
SGLANG_ENABLE_SPEC_V2=True

# Launch flags:
--speculative-algorithm NEXTN
--speculative-num-steps 3
--speculative-num-draft-tokens 4
--speculative-eagle-topk 1
```

### Key Points

- `SGLANG_ENABLE_SPEC_V2=True` is **absolutely required**. Without it, SGLang silently converts NEXTN to EAGLE and tries to load a second copy of the model as a draft model.
- MTP models need checkpoints that include MTP weights (e.g., `festr2/GLM-5-NVFP4-MTP`, `nvidia/Qwen3.5-397B-A17B-NVFP4`).
- GLM-5 MTP accept rate: 0.55-0.94 (varies by context), accept length: 2.19-2.80 tokens, roughly **2x** speedup.
- Qwen3.5 with MTP=3 on SGLang: ~130 tok/s on 4 GPUs, ~350 tok/s on 8 GPUs (luke's switch setup).

---

## Decode Context Parallel (DCP)

DCP is available for Kimi K2.5 via SGLang's `--ep` flag combined with custom allreduce, or via the `SGLANG_DCP` environment variable:

```bash
export SGLANG_DCP=8
```

For GLM-4.7, there is a DCP branch: https://github.com/antgroup/sglang/tree/yjh/dcp-dev-main

DCP for GLM-5 via SGLang is **not yet available** as of 2026-03-08. This limits concurrent agentic workloads since all parallel requests share the same ~200K context window.

---

## MOE Runner Backend Selection

| Backend | Notes |
|---------|-------|
| `cutlass` | **Fastest on SM120** for MTP speculative decoding. Recommended for GLM-5 with MTP. |
| `flashinfer_cutlass` | Default backend. Slightly slower than `cutlass` on SM120. Safe fallback. |
| `deep_gemm` | Falls back to `cutlass` on SM120 (DeepGemm not supported). Misleading in logs. |
| `triton` | Alternative. Used for Kimi K2.5 NVFP4 and Qwen3.5 FP8. |

**Key finding**: `cutlass` is faster than `flashinfer_cutlass` on SM120. The `deep_gemm` backend does NOT actually use DeepGemm on SM120; the speed attributed to it comes from the underlying cutlass fallback.

---

## EAGLE Speculative Decoding

EAGLE speculative decoding (separate from native MTP) is available for Qwen3.5:

```bash
SGLANG_ENABLE_SPEC_V2=1 python3 -m sglang.launch_server \
  --model-path nvidia/Qwen3.5-397B-A17B-NVFP4 \
  --tp 4 \
  --trust-remote-code \
  --attention-backend triton \
  --moe-runner-backend cutlass \
  --kv-cache-dtype fp8_e4m3 \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --quantization modelopt_fp4 \
  --mamba-scheduler-strategy extra_buffer \
  --fp8-gemm-backend triton \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 16}' \
  --sleep-on-idle
```

EAGLE3 support for Kimi K2.5 is in development:
- SGLang PR: https://github.com/sgl-project/sglang/pull/19689
- Draft model: `AQ-MedAI/Kimi-K2-Instruct-eagle3`

---

## FlashInfer Allreduce Fusion

`--enable-flashinfer-allreduce-fusion` fuses allreduce with attention operations. This is gated to SM120 and uses TRT-LLM allreduce fusion with norm through FlashInfer, optimized for RTX 6000 Pro P2P over PCIe.

This provides a measurable throughput gain and should be used on all RTX 6000 Pro setups.

---

## Performance Tuning Tips

1. **Use `--disable-custom-all-reduce`** on PCIe-only setups. The default custom allreduce is optimized for NVLink.

2. **NCCL Graph XML on AMD Turin/Genoa**: Download from `https://www.voipmonitor.org/nccl_graph_opt.xml`. Measured +11% throughput improvement on Genoa with 2 NUMA nodes.

3. **Multi-threaded model loading**: Add `--model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 16}'` to significantly reduce load time.

4. **CPU performance tuning**:
   ```bash
   echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
   sysctl -w vm.swappiness=0
   sysctl -w kernel.numa_balancing=0
   sysctl -w kernel.sched_migration_cost_ns=50000
   export SGLANG_SET_CPU_AFFINITY=1
   ```

5. **GDDR7 memory overclocking** (Max-Q cards only): luke gained several percentage points by setting `nvmlDeviceSetMemClkVfOffset(handle, 4000)` via pynvml.

6. **Use `flashinfer_cudnn` for FP4 GEMM**: The default `flashinfer_cutlass` has a race condition bug (flashinfer#2708) silently corrupting memory. Use `--fp4-gemm-backend flashinfer_cudnn` instead. Requires: `pip install nvidia-cudnn-cu13==9.19.1.2`.

7. **Wipe JIT cache after upgrades**: `rm -rf /cache/jit/*` -- old cached kernels will not pick up bug fixes.

---

## Profiling

```bash
SGLANG_TORCH_PROFILER_DIR="./" \
SGLANG_PROFILE_RECORD_SHAPES=true \
SGLANG_PROFILE_WITH_STACK=true \
python3 -m sglang.bench_one_batch_server \
  --model baseten-admin/glm-4.7-fp8-attn-fp4-mlp \
  --base-url http://localhost:30000 \
  --batch-size 4 \
  --input-len 2048 \
  --output-len 1024 \
  --profile \
  --profile-steps 10 \
  --show-report \
  --profile-by-stage
```

View traces at https://magic-trace.org/

---

## Relevant PRs

| PR | Description |
|----|-------------|
| [SGLang #14194](https://github.com/sgl-project/sglang/pull/14194) | DCP implementation |
| [SGLang #18434](https://github.com/sgl-project/sglang/pull/18434) | Pipeline parallel for Kimi |
| [SGLang #18937](https://github.com/sgl-project/sglang/pull/18937) | NVFP4 support, merged into main |
| [SGLang #19428](https://github.com/sgl-project/sglang/pull/19428) | Performance improvement for GLM-5 |
| [SGLang #19689](https://github.com/sgl-project/sglang/pull/19689) | Kimi K2.5 EAGLE3 support |
| [SGLang #19897](https://github.com/sgl-project/sglang/pull/19897) | Fix for radix cache + speculative decoding crash |
| [SGLang #19948](https://github.com/sgl-project/sglang/pull/19948) | DeepGemm SCALE_UE8M0 fix for NVFP4 on SM120 |
| [SGLang #19951](https://github.com/sgl-project/sglang/pull/19951) | Fix for broken latest SGLang |
| [SGLang #19963](https://github.com/sgl-project/sglang/pull/19963) | Compilation fixes |
