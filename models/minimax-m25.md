# MiniMax M2.5 — Setup & Benchmarks

MiniMax M2.5 is a 229B-parameter MoE model with native FP8 weights, strong coding/reasoning capabilities, and 196K context. It runs well on 2× or 4× RTX PRO 6000 GPUs.

## Table of Contents

- [Model Overview](#model-overview)
- [Available Quantizations](#available-quantizations)
- [Launch Commands — vLLM](#launch-commands--vllm)
- [Launch Commands — SGLang](#launch-commands--sglang)
- [REAP Variant (Single GPU)](#reap-variant-single-gpu)
- [AWQ Variant](#awq-variant)
- [Benchmarks](#benchmarks)
- [Quality: NVFP4 vs FP8](#quality-nvfp4-vs-fp8)
- [Reasoning Parser & Tool Calling](#reasoning-parser--tool-calling)
- [Known Issues](#known-issues)
- [Tips & Best Practices](#tips--best-practices)

---

## Model Overview

| Property | Value |
|----------|-------|
| **Full name** | MiniMaxAI/MiniMax-M2.5 |
| **Parameters** | 229B total (MoE) |
| **Native precision** | FP8 |
| **Context length** | 196,608 tokens |
| **Architecture** | MoE with custom reasoning |
| **Tool calling** | Native (minimax_m2 parser) |
| **Vision** | No (use separate vision model) |

---

## Available Quantizations

| Variant | Model ID | Min GPUs | VRAM Usage | Notes |
|---------|----------|----------|------------|-------|
| **FP8 (original)** | `MiniMaxAI/MiniMax-M2.5` | 4× | ~54 GB weights/GPU (TP4) | Best quality, official |
| **NVFP4** | `lukealonso/MiniMax-M2.5-NVFP4` | 2× | Fits on 2× 96GB | ~Same quality as FP8 per benchmarks |
| **REAP NVFP4** | `lukealonso/MiniMax-M2.5-REAP-139B-A10B-NVFP4` | 1× | Fits single 96GB | Pruned to 139B, ~81K max context |
| **AWQ (INT4)** | `mratsim/Minimax-M2.5-BF16-INT4-AWQ` | 2× | Fits on 2× | Faster at low context, slower at high |

---

## Launch Commands — vLLM

### FP8 on 4× GPUs (Official)

```bash
SAFETENSORS_FAST_GPU=1 VLLM_SLEEP_WHEN_IDLE=1 vllm serve \
    MiniMaxAI/MiniMax-M2.5 \
    --tensor-parallel-size 4 \
    --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2_append_think \
    --enable-auto-tool-choice \
    --trust-remote-code
```

Memory info (4× GPUs):
```
Model loading took 53.75 GiB memory
Available KV cache memory: 23.62 GiB
GPU KV cache size: 399,424 tokens
Maximum concurrency for 196,608 tokens per request: 2.03x
```

### FP8 on 8× GPUs (Expert Parallel)

```bash
vllm serve MiniMaxAI/MiniMax-M2.5 \
    --served-model-name llm_model \
    --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2_append_think \
    --enable-auto-tool-choice \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --gpu-memory-utilization 0.9 \
    --max-model-len -1 \
    --trust-remote-code \
    --port 9504
```

> Requires tuned MoE kernel config for SM120. Copy the tuned kernel JSON to:
> `vllm/model_executor/layers/fused_moe/configs/E=256,N=384,device_name=NVIDIA_RTX_PRO_6000_...,dtype=fp8_w8a8,block_shape=[128,128].json`

### NVFP4 on 2× GPUs (vLLM)

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  NCCL_P2P_LEVEL=4 \
  SAFETENSORS_FAST_GPU=1 \
  VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass \
  VLLM_USE_FLASHINFER_MOE_FP4=1 \
  VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  VLLM_FLASHINFER_MOE_BACKEND=latency \
  VLLM_SLEEP_WHEN_IDLE=1 \
  vllm serve lukealonso/MiniMax-M2.5-NVFP4 \
    --trust-remote-code \
    --served-model-name MiniMax-M2.5-NVFP4 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 256 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --max-num-batched-tokens 16384 \
    --enable-auto-tool-choice \
    --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2 \
    --quantization modelopt_fp4 \
    --kv-cache-dtype fp8 \
    --dtype auto \
    --attention-backend FLASHINFER \
    --load-format fastsafetensors \
    --tensor-parallel-size 2 \
    --port 30000
```

### Docker One-Liner (vLLM, FP8, 4× GPUs)

```bash
docker run --rm --gpus 4 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "SAFETENSORS_FAST_GPU=1" \
    --env "VLLM_SLEEP_WHEN_IDLE=1" \
    -p 5000:5000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model MiniMaxAI/MiniMax-M2.5 \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --served-model-name model \
    --enable-auto-tool-choice \
    --tool-call-parser minimax_m2 \
    --port 5000 \
    --reasoning-parser minimax_m2_append_think \
    --trust-remote-code
```

---

## Launch Commands — SGLang

### NVFP4 on 2× GPUs (Recommended)

```bash
docker run -it --rm \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  --ipc=host --shm-size=8g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus all --network host \
  lmsysorg/sglang:dev-cu13 bash
```

```bash
SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True \
NCCL_P2P_LEVEL=4 \
SGLANG_ENABLE_JIT_DEEPGEMM=0 \
python3 -m sglang.launch_server \
  --model lukealonso/MiniMax-M2.5-NVFP4 \
  --served-model-name MiniMax-M2.5 \
  --reasoning-parser minimax \
  --enable-torch-compile \
  --trust-remote-code \
  --tp 2 \
  --mem-fraction-static 0.93 \
  --max-running-requests 16 \
  --quantization modelopt_fp4 \
  --attention-backend flashinfer \
  --moe-runner-backend flashinfer_cutlass \
  --enable-pcie-oneshot-allreduce \
  --sleep-on-idle \
  --host 0.0.0.0 --port 5000
```

### FP8 on 8× GPUs (SGLang + EP)

```bash
NCCL_P2P_LEVEL=4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m sglang.launch_server \
  --model-path MiniMax-M2.5 \
  --tp-size 8 --ep-size 8 \
  --mem-fraction-static 0.8 \
  --tool-call-parser minimax-m2 \
  --reasoning-parser minimax-append-think \
  --served-model-name llm_model \
  --host 0.0.0.0 --port 9504 \
  --trust-remote-code \
  --fp8-gemm-backend triton \
  --moe-runner-backend triton \
  --sleep-on-idle
```

Optimize with tuned kernels:
```bash
python /sgl-workspace/sglang/benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py \
  --model MiniMax-M2.5 --tp-size 8 --ep-size 8 --dtype fp8_w8a8 --tune
```

### FP8 on 4× GPUs (SGLang)

```bash
python -m sglang.launch_server \
    --model-path MiniMax-M2.5 \
    --tp-size 4 \
    --tool-call-parser minimax-m2 \
    --reasoning-parser minimax-append-think \
    --host 0.0.0.0 \
    --trust-remote-code \
    --port 8000 \
    --mem-fraction-static 0.85 \
    --cuda-graph-max-bs 32 \
    --max-running-requests 32 \
    --enable-torch-compile \
    --torch-compile-max-bs 4 \
    --sleep-on-idle
```

### Concurrency Tip: TP2 + DP2 on 4 GPUs

Running `--tp 2` with data parallelism (DP=2) on 4 GPUs gives better concurrency than TP4:

> "Running it on TP=2 + DP=2 with SGLang currently, seems to be pretty good for concurrency. 2 devs using it full time with sub-agents, no slowdown like FP8 + TP=4" — mudaG

---

## REAP Variant (Single GPU)

The REAP (Reduce Expert And Prune) variant fits on a **single RTX PRO 6000** (96 GB):

- Model: `lukealonso/MiniMax-M2.5-REAP-139B-A10B-NVFP4`
- 139B params, 10B active
- Max context: ~81K tokens on 1× GPU
- ~70 tok/s single-stream on SGLang (luke's numbers)

### SGLang Launch (1× GPU)

```bash
python3 -m sglang.launch_server \
  --model lukealonso/MiniMax-M2.5-REAP-139B-A10B-NVFP4 \
  --served-model-name MiniMax-M2.5-NVFP4 \
  --reasoning-parser minimax \
  --tool-call-parser minimax-m2 \
  --trust-remote-code \
  --tp 1 \
  --mem-fraction-static 0.95 \
  --max-running-requests 32 \
  --quantization modelopt_fp4 \
  --attention-backend flashinfer \
  --moe-runner-backend flashinfer_cutlass \
  --kv-cache-dtype bf16 \
  --sleep-on-idle \
  --host 0.0.0.0 --port 8000
```

### REAP Quality

MMLU-Pro comparison (Qu's testing):

| Variant | GPUs | MMLU-Pro |
|---------|------|----------|
| FP8 (original) | 4× | Baseline |
| NVFP4 | 2× | ~Same as FP8 |
| REAP NVFP4 | 1× | Lower (pruned) |

> REAP may lose non-English language capability. Festr reported Czech language was "completely lost" in REAP versions of other models.

Also runs on NVIDIA DGX Spark: [forum post](https://forums.developer.nvidia.com/t/minimax-2-5-reap-nvfp4-on-single-dgx-spark/361248)

---

## AWQ Variant

AWQ quantization (`mratsim/Minimax-M2.5-BF16-INT4-AWQ`) runs on 2× GPUs via vLLM:

```bash
MODEL="mratsim/Minimax-M2.5-BF16-INT4-AWQ"

export VLLM_USE_FLASHINFER_MOE_FP4=1
export VLLM_USE_FLASHINFER_MOE_FP8=1
export VLLM_USE_FLASHINFER_MOE_FP16=1
export VLLM_FLASHINFER_MOE_BACKEND=throughput
export CUDA_VISIBLE_DEVICES=0,1
export SAFETENSORS_FAST_GPU=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_SLEEP_WHEN_IDLE=1

vllm serve "${MODEL}" \
  --tensor-parallel-size 2 \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think \
  --trust-remote-code \
  --host 0.0.0.0 --port 8080 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --attention-backend FLASHINFER \
  --max-num-seqs 64 \
  --max-model-len 196608 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.97
```

AWQ characteristics (Marky's testing):
- **114 tok/s** at low context
- **~50 tok/s** at 130K+ context
- Faster than NVFP4 at low context, slower at high context
- "Slightly dumber" than FP8/NVFP4

---

## Benchmarks

### Single-Stream Decode Speed

| Config | Engine | tok/s | Notes |
|--------|--------|-------|-------|
| FP8, 4× GPUs | SGLang | ~71 | Ixtrix, defaults |
| FP8, 8× GPUs (EP) | SGLang | ~86 | CyySky, tuned MoE kernels |
| NVFP4, 2× GPUs | SGLang | 85-89 | Festr, destroyed |
| NVFP4, 2× GPUs | vLLM | ~85 | Festr, TP2 |
| NVFP4, 4× GPUs | vLLM | ~81 (20K ctx) | chisleu |
| NVFP4, 4× GPUs | vLLM | ~61 (100K ctx) | chisleu |
| AWQ, 2× GPUs | vLLM | ~114 (low ctx) | Marky |
| AWQ, 2× GPUs | vLLM | ~50 (130K+ ctx) | Marky |
| REAP NVFP4, 1× GPU | SGLang | ~70 | luke |

### Concurrency Benchmarks

Marky's AWQ benchmark (2× GPUs, vLLM, 64 concurrent):
```
Output token throughput (tok/s):         930.12
Peak output token throughput (tok/s):    1551.00
Mean TTFT (ms):                          339.69
Mean TPOT (ms):                          56.32
```

### NVFP4 2× vs FP8 4× (Qu's Testing, 300W Power Limit)

At high concurrency, NVFP4 on 2× GPUs at 500W nearly matches FP8 on 4× GPUs at 300W — making 2× a strong value proposition.

Detailed wattage-performance analysis: [shihanqu.github.io/Blackwell-Wattage-Performance](https://shihanqu.github.io/Blackwell-Wattage-Performance/)

### Power Behavior

M2.5 utilizes power better than M2.1 — at 300W power limit, M2.1 rarely pushed past 200W per card, while M2.5 uses the full allocation.

---

## Quality: NVFP4 vs FP8

### MMLU-Pro Results (Lavd's Testing, temp=0.1)

| Model | Overall Accuracy | Notes |
|-------|-----------------|-------|
| NVFP4 (`lukealonso/MiniMax-M2.5-NVFP4`) | **86.2%** | Slightly higher |
| FP8 (`MiniMaxAI/MiniMax-M2.5`) | **85.8%** | Original |

Difference is within statistical noise. Festr confirmed with 120+ tests: "almost statistical noise differences."

### Coding Quality

Some users report NVFP4 produces lower-quality code output:

> "The difference between FP8 and NVFP4 on MiniMax M2.5 was noticeable for me. I went back to FP8." — chisleu (code generation use case)

luke's recommendation: try BF16 KV cache with NVFP4 if seeing quality issues.

---

## Reasoning Parser & Tool Calling

### vLLM

```
--tool-call-parser minimax_m2
--reasoning-parser minimax_m2_append_think
```

**Known issue:** `minimax_m2_append_think` can cause loops. Workarounds:
- Use `--reasoning-parser deepseek_r1` instead (chisleu confirmed this fixes tool calling with smolagents)
- Or use `--reasoning-parser minimax_m2` (different from append_think)

### SGLang

```
--tool-call-parser minimax-m2
--reasoning-parser minimax
```

> "Put 'minimax' as the reasoning parser. That properly separates the thinking into the reasoning_content field." — mudaG

---

## Known Issues

### Reasoning Loops

Model can enter repetitive loops ("I am now going to do xyz..." over and over). Causes:
- Temperature too high (1.0 can cause loops; try 0.7)
- Wrong reasoning parser
- KV cache dtype issues

**Fix:** Lower temperature to 0.7, use correct reasoning parser, try BF16 KV cache.

### NVFP4 Loading Issues on SGLang

Some users report crashes (`Exit code: -9`, scheduler dead) when loading NVFP4 on SGLang:
- Lower `--mem-fraction-static` (try 0.90 instead of 0.95)
- Ensure sufficient system RAM (128GB recommended)
- First run requires kernel autotuning (5-10 minutes), results are cached for subsequent runs

### Speculative Decoding

Not working with MiniMax M2.5 on either vLLM or SGLang as of March 2026.

### --trust-remote-code Security

MiniMax M2.5 requires `--trust-remote-code`. Only use with official model IDs from trusted sources (MiniMaxAI, lukealonso). Community members have reported malware incidents from running untrusted models with this flag.

---

## Tips & Best Practices

1. **Best value setup:** NVFP4 on 2× GPUs — nearly matches FP8 4× at high concurrency
2. **Best quality:** FP8 on 4× GPUs — official weights, most tested
3. **Single GPU:** REAP NVFP4 fits on 1× 96GB, ~70 tok/s, but may lose non-English quality
4. **KV cache:** Use BF16 if seeing quality issues; FP8 works but slightly slower on Blackwell
5. **Don't disable P2P** if GPUs are on same NUMA node — you'll go through DRAM with higher latency
6. **Recommended temperature:** 0.7 (1.0 causes loops)
7. **Generation config:** `{"temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.15}`
8. **SGLang vs vLLM:** SGLang better for single-stream; vLLM better for high concurrency
9. **TP2 + DP2 > TP4** for multi-user concurrent workloads on 4 GPUs
10. **CUDA 13** recommended — measurably faster than 12.9 for this model
