# vLLM on RTX 6000 Pro Blackwell

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Configuration Flags Reference](#configuration-flags-reference)
- [Model-Specific Launch Commands](#model-specific-launch-commands)
  - [Qwen3.5-397B NVFP4 (4 GPUs)](#qwen35-397b-nvfp4-4-gpus)
  - [Qwen3.5-397B FP8 (8 GPUs)](#qwen35-397b-fp8-8-gpus)
  - [Qwen3.5-27B NVFP4 (1 GPU)](#qwen35-27b-nvfp4-1-gpu)
  - [Qwen3.5 CARVE with YaRN (up to 900K context)](#qwen35-carve-with-yarn-up-to-900k-context)
  - [Kimi K2.5 INT4 (8 GPUs)](#kimi-k25-int4-8-gpus)
  - [MiniMax-M2.5 FP8 (4 GPUs)](#minimax-m25-fp8-4-gpus)
- [MTP / Speculative Decoding](#mtp--speculative-decoding)
- [Decode Context Parallel (DCP)](#decode-context-parallel-dcp)
- [Performance Tuning Tips](#performance-tuning-tips)
- [Known Limitations](#known-limitations)
- [Relevant PRs](#relevant-prs)

---

## Overview

vLLM is one of the two primary inference engines used on RTX 6000 Pro Blackwell (SM120) rigs. It provides an OpenAI-compatible API server with support for tensor parallelism, speculative decoding (MTP), prefix caching, and tool calling.

Key strengths on RTX 6000 Pro:

- Stable MTP (Multi-Token Prediction) support for Qwen3.5 with ~50-70% throughput gains
- Decode Context Parallel (DCP) for Kimi K2.5, enabling 3M+ token KV cache on 8 GPUs
- FP8 KV cache with calibrated scales via TRITON_MLA attention backend
- Mature tool calling and reasoning parser integration

---

## Installation

### Docker (recommended)

```bash
docker pull vllm/vllm-openai:cu130-nightly
```

Run with:

```bash
docker run --rm --gpus all \
  --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:cu130-nightly \
  --model <model-path> \
  <additional flags>
```

For setups where the host has CUDA 13.1 and the container has CUDA 13.0, add:

```bash
--mount type=tmpfs,destination=/usr/local/cuda-13.0/compat
```

### Custom Docker (Qwen3.5 MTP patched)

The `orthozany/vllm-qwen35-mtp` image cherry-picks critical PRs for Qwen3.5 MTP:

- PR #35219: FlashInfer Blackwell accuracy fix, zeros freed SSM cache blocks
- PR #35421: Tool call streaming fix for speculative decoding
- PR #35675: Fix Qwen3.5-nvfp4 MTP fc layer shape mismatch

### Patched vLLM for Kimi K2.5

Festr's patched image includes FP8 KV cache + DCP support for MLA:

```bash
docker run -it --rm \
  --entrypoint /bin/bash \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v /mnt:/mnt/ \
  --ipc=host --shm-size=8g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus all --network host \
  --mount type=tmpfs,destination=/usr/local/cuda-13.0/compat \
  voipmonitor/vllm-openai:cu130-nightly-patched
```

Then install required dependency: `pip install fastsafetensors`

### Build from Source

```bash
wget https://www.voipmonitor.org/build_vllm_venv.sh.txt -O build_vllm_venv.sh
chmod +x build_vllm_venv.sh
./build_vllm_venv.sh /root/venvtest
source /root/venvtest/activate.sh
```

This builds vLLM with `ENABLE_SM120=1`, flash_attn 2.8.3 for SM120, and FlashMLA 1.0.0 with the CUTLASS TMEM patch for non-datacenter cards. Build time: 30-60 minutes.

### Tested Software Versions

| Component | Version |
|-----------|---------|
| CUDA Toolkit | 13.1.1 (nvcc 13.1.115) |
| NVIDIA Driver | 590.48.01 |
| PyTorch | 2.10.0+cu130 |
| FlashInfer | 0.6.4 |
| OS | Ubuntu 24.04.3 LTS |

---

## Configuration Flags Reference

### Server Flags

| Flag | Description |
|------|-------------|
| `--tensor-parallel-size N` | Number of GPUs for tensor parallelism. Must be power-of-2 (2, 4, 8, 16). |
| `--pipeline-parallel-size N` | Pipeline parallel stages. Use for non-power-of-2 GPU counts (e.g., TP2 PP3 = 6 GPUs). |
| `--gpu-memory-utilization 0.80-0.95` | Fraction of VRAM to use. 0.93 typical for Kimi, 0.80-0.89 for Qwen. |
| `--max-num-batched-tokens N` | Maximum tokens in a batch. 4096 is typical. |
| `--max-num-seqs N` | Maximum concurrent sequences. 128 for multi-user, 16 for single-user. |
| `--enable-prefix-caching` | Cache common prompt prefixes to reduce TTFT on repeated prompts. |
| `--enable-chunked-prefill` | Process long prompts in chunks rather than all at once. |
| `--load-format fastsafetensors` | Faster weight loading. Requires `pip install fastsafetensors`. |
| `--async-scheduling` | Async request scheduling for higher throughput. |
| `--language-model-only` | Disables vision encoder. Reduces TTFT from ~12s to <1s on first request. |
| `--attention-backend TRITON_MLA` | Required for MLA models (Kimi K2.5) with FP8 KV cache on SM120. |
| `--kv-cache-dtype fp8` | FP8 KV cache -- doubles context capacity vs BF16. |
| `--decode-context-parallel-size N` | Shares KV cache across N GPUs. Critical for long-context on Kimi K2.5. |

### Tool Calling and Reasoning

| Flag | Description |
|------|-------------|
| `--enable-auto-tool-choice` | Enable automatic tool call detection. |
| `--tool-call-parser <parser>` | Parser for tool calls: `qwen3_coder`, `kimi_k2`, `minimax_m2`. |
| `--reasoning-parser <parser>` | Parser for thinking mode: `qwen3`, `kimi_k2`. |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `NCCL_P2P_LEVEL=SYS` or `4` | Enable P2P across system. Use `SYS` for dual-CPU, `4` for single-CPU. |
| `NCCL_IB_DISABLE=1` | Disable InfiniBand (required for consumer/workstation setups). |
| `NCCL_PROTO=LL` | Force Low Latency protocol. Alternative to NCCL graph XML. |
| `NCCL_GRAPH_FILE=/path/to/nccl_graph_opt.xml` | Custom topology graph for AMD Turin. Download: `https://www.voipmonitor.org/nccl_graph_opt.xml` |
| `SAFETENSORS_FAST_GPU=1` | Faster weight loading. |
| `VLLM_WORKER_MULTIPROC_METHOD=spawn` | Required for vLLM multi-GPU. |
| `VLLM_LOG_STATS_INTERVAL=1` | Log throughput stats every second. |
| `VLLM_SLEEP_WHEN_IDLE=1` | Save power when no requests are active. |
| `VLLM_NVFP4_GEMM_BACKEND=cutlass` | Control FP4 GEMM backend for vLLM. |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` | Required for >262K context with YaRN. |
| `VLLM_TEST_FORCE_FP8_MARLIN=1` | Force FP8 Marlin kernels (Kimi K2.5). |
| `VLLM_MARLIN_USE_ATOMIC_ADD=1` | Enable atomic add in Marlin (Kimi K2.5). |
| `VLLM_MARLIN_INPUT_DTYPE=fp8` | FP8 input dtype for Marlin (Kimi K2.5). |

---

## Model-Specific Launch Commands

### Qwen3.5-397B NVFP4 (4 GPUs)

Best known working config with MTP=2, yielding ~130 tok/s single stream:

```bash
VLLM_LOG_STATS_INTERVAL=1 \
NCCL_P2P_LEVEL=4 \
SAFETENSORS_FAST_GPU=1 \
python3 -m vllm.entrypoints.openai.api_server \
  --model nvidia/Qwen3.5-397B-A17B-NVFP4 \
  --host 0.0.0.0 --port 5000 \
  --served-model-name Qwen3.5-397B-A17B-NVFP4 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.80 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 128 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-type shm \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --enable-prefix-caching
```

**Required config.json patch**: Add `"mtp.fc"` and `"model.language_model.layers..mlp.gate"` to `quantization_config.ignore` in the model's `config.json` and `hf_quant_config.json`.

### Qwen3.5-397B FP8 (8 GPUs)

```bash
vllm serve Qwen3.5-397B-A17B-FP8 \
  --port 9501 \
  --tensor-parallel-size 8 \
  --max-model-len -1 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --gpu-memory-utilization 0.9 \
  --served-model-name llm_model \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
```

### Qwen3.5-27B NVFP4 (1 GPU)

```bash
VLLM_SLEEP_WHEN_IDLE=1 VLLM_LOG_STATS_INTERVAL=1 \
vllm serve osoleve/Qwen3.5-27B-NVFP4-MTP \
  --served-model-name Qwen3.5-27B-NVFP4 \
  --trust-remote-code \
  --gpu-memory-utilization 0.85 \
  --max-model-len 128000 \
  --quantization modelopt \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --reasoning-parser qwen3 \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-type shm \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

### Qwen3.5 CARVE with YaRN (up to 900K context)

```bash
NCCL_P2P_LEVEL=4 \
NCCL_IB_DISABLE=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
SAFETENSORS_FAST_GPU=1 \
OMP_NUM_THREADS=8 \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
vllm serve vpyn/Qwen3.5-397B-A17B-CARVE-v1-NVFP4 \
  --served-model-name qwen3.5-carve \
  --tensor-parallel-size 4 \
  --max-model-len 921600 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --kv-cache-dtype fp8 \
  --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}'
```

Do NOT use MTP with CARVE -- MTP heads were trained on censored content.

### Kimi K2.5 INT4 (8 GPUs)

Production command with FP8 KV cache and DCP:

```bash
VLLM_LOG_STATS_INTERVAL=1 \
NCCL_P2P_LEVEL=SYS \
NCCL_GRAPH_FILE=/mnt/nccl_graph_opt.xml \
VLLM_VIDEO_LOADER_BACKEND=opencv_tempfile \
VLLM_TEST_FORCE_FP8_MARLIN=1 \
VLLM_MARLIN_USE_ATOMIC_ADD=1 \
VLLM_MARLIN_INPUT_DTYPE=fp8 \
vllm serve moonshotai/Kimi-K2.5 \
  --served-model-name Kimi-K2.5 \
  --trust-remote-code \
  --host 0.0.0.0 --port 5000 \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 1 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --load-format fastsafetensors \
  --tool-call-parser kimi_k2 \
  --enable-auto-tool-choice \
  --reasoning-parser kimi_k2 \
  --async-scheduling \
  --gpu-memory-utilization 0.95 \
  --max-num-batched-tokens 4096 \
  --attention-backend TRITON_MLA \
  --kv-cache-dtype fp8 \
  --mm-encoder-tp-mode data \
  --decode-context-parallel-size 8
```

**Required patches**: Cherry-pick vLLM PR #34597 (FP8 KV cache for Triton MLA decode) and PR #34795 (FP8 KV cache with DCP for MLA).

**Required packages**: `pip install fastsafetensors`

**NCCL graph file**: `wget https://www.voipmonitor.org/nccl_graph_opt.xml -O /mnt/nccl_graph_opt.xml`

### MiniMax-M2.5 FP8 (4 GPUs)

```bash
VLLM_LOG_STATS_INTERVAL=1 \
python3 -m vllm.entrypoints.openai.api_server \
  --model MiniMaxAI/MiniMax-M2.5 \
  --host 0.0.0.0 --port 5001 \
  --served-model-name MiniMaxAI/MiniMax-M2.5 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --attention-backend FLASHINFER \
  --gpu-memory-utilization 0.95 \
  --max-num-batched-tokens 4092 \
  --max-num-seqs 128 \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2 \
  --kv-cache-dtype bfloat16 \
  --load-format fastsafetensors
```

---

## MTP / Speculative Decoding

MTP (Multi-Token Prediction) is natively supported in Qwen3.5. Kimi K2.5 does **not** have native MTP.

### vLLM MTP Configuration

```
--speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

Or for older versions:
```
--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
```

### Key Findings

- **MTP=2 is the sweet spot** for Qwen3.5 NVFP4 on vLLM. Provides ~50-55% throughput improvement across all concurrency levels.
- **MTP=5** works for short context but is **unstable at long context** (illegal memory access crashes).
- **MTP>3** requires PR #35615 patch. Without it, set to `:2` max.
- MTP can cause **tool call format changes** (XML instead of JSON) when `tool_choice='required'` is set. Fix: PR #35936.
- Combining thinking mode + MTP exacerbates tool call issues. "If thinking is false, even with MTP there is no problem."

### Benchmark: MTP=2 on Qwen3.5-397B NVFP4 (4x RTX 6000 Pro)

| Concurrency | No MTP (tok/s) | MTP=2 (tok/s) | Improvement |
|:-----------:|:--------------:|:--------------:|:-----------:|
| 1 | 85.8 | 130.0 | +51.5% |
| 2 | 137.1 | 212.7 | +55.1% |
| 5 | 234.2 | 358.6 | +53.1% |
| 10 | 334.3 | 573.5 | +71.6% |
| 20 | 491.5 | 744.1 | +51.4% |
| 32 | 605.9 | 922.6 | +52.3% |

MTP=2 acceptance stats: 89.2% acceptance rate (165,550 drafted / 147,689 accepted).

---

## Decode Context Parallel (DCP)

DCP (`--decode-context-parallel-size N`) distributes KV cache across N GPUs, multiplying total context capacity:

| DCP | KV Cache (FP8) | KV Cache (BF16) |
|:---:|:--------------:|:---------------:|
| 1 | ~449K tokens | ~190K tokens |
| 8 | ~3.6M tokens | ~1.5M tokens |

DCP is critical for Kimi K2.5 long-context: without it, decode at 150K+ context drops to <10 tok/s. With DCP=8, you get 30+ tok/s.

**Requires**: vLLM PRs #34597 and #34795.

---

## Performance Tuning Tips

1. **NCCL Graph XML on AMD Turin**: Download `https://www.voipmonitor.org/nccl_graph_opt.xml` and set `NCCL_GRAPH_FILE`. This corrects NCCL's 16 GB/s bandwidth assumption for AMD CPUs (actual: 192-256 GB/s on Turin), yielding 1.5-1.9x speedup on small AllReduce messages.

2. **CPU performance governor**:
   ```bash
   echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
   sysctl -w vm.swappiness=0
   sysctl -w kernel.numa_balancing=0
   ```

3. **Do NOT use `--enable-expert-parallel`** on PCIe setups. It kills batch throughput due to massive inter-card traffic that saturates PCIe bandwidth.

4. **Power limit optimization**: 500W performs nearly identically to 600W. Going from 400W to 300W loses significant throughput at high concurrency (up to 30% loss at 64 concurrent users). At 4 concurrent users or below, 300W has almost no penalty.

5. **Docker vs bare metal**: ~3 tok/s difference (76 vs 79 tok/s). The gap washes away with context length.

6. **Disable vision encoder** with `--language-model-only` if you do not need multimodal. This eliminates a ~12s TTFT spike on the first request.

7. **fastsafetensors**: Use `--load-format fastsafetensors` for faster loading, but note it may allocate excess VRAM on GPU 0, reducing max context. If KV cache space is critical, use default safetensors with higher `--gpu-memory-utilization`.

---

## Known Limitations

- **GLM-5 does NOT run on vLLM for SM120** as of 2026-03-08. No attention backend supports MLA + sparse attention + SM120 simultaneously. Use SGLang for GLM-5.
- **Expert Parallel (EP)** is consistently slower or a wash on PCIe setups without NVLink.
- **MTP>3 is unstable** -- causes illegal memory access crashes under load.
- **FP8 KV cache** does not work for GLM-5 on SM120 (garbled output). Only BF16 KV cache works.
- **NVFP4 is 20-30 tok/s slower than FP8** due to slower fused MoE kernels and GEMM overhead.
- **`fla/ops/utils.py:113: UserWarning`** about tensor shape mismatch causes long TTFT delays on some builds.

---

## Relevant PRs

| PR | Description |
|----|-------------|
| [#34424](https://github.com/vllm-project/vllm/pull/34424) | +2 tok/s improvement on MiniMax M2.5 |
| [#34597](https://github.com/vllm-project/vllm/pull/34597) | FP8 KV cache support for Triton MLA decode (SM120) |
| [#34795](https://github.com/vllm-project/vllm/pull/34795) | FP8 KV cache with DCP for MLA |
| [#35156](https://github.com/vllm-project/vllm/pull/35156) | Hardcode mlp.gate as not quantizable (Qwen3.5) |
| [#35219](https://github.com/vllm-project/vllm/pull/35219) | FlashInfer Blackwell accuracy fix, zero freed SSM cache blocks |
| [#35347](https://github.com/vllm-project/vllm/pull/35347) | SM12.0 fixes (SymmMemCommunicator) |
| [#35421](https://github.com/vllm-project/vllm/pull/35421) | Tool call streaming fix for speculative decoding |
| [#35548](https://github.com/vllm-project/vllm/pull/35548) | MTP weight validation (GLM-5) |
| [#35581](https://github.com/vllm-project/vllm/pull/35581) | MTP fused kernel fix (~6% throughput boost) |
| [#35615](https://github.com/vllm-project/vllm/pull/35615) | Fix tool call streaming, allows MTP>1 |
| [#35675](https://github.com/vllm-project/vllm/pull/35675) | Fix Qwen3.5-nvfp4 MTP fc layer shape mismatch |
| [#35936](https://github.com/vllm-project/vllm/pull/35936) | Fix tool_choice='required' with MTP (XML->JSON) |
| [#35966](https://github.com/vllm-project/vllm/pull/35966) | Kimi K2/DeepSeek EAGLE3 support |
| [#36322](https://github.com/vllm-project/vllm/pull/36322) | FlashInfer FA2 MLA attention backend for SM120 |
