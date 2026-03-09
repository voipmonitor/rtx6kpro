# Benchmark Results -- RTX 6000 Pro Blackwell

## Table of Contents

- [Qwen3.5-397B Benchmarks](#qwen35-397b-benchmarks)
  - [Single-Batch Decode Speed](#qwen35-single-batch-decode-speed)
  - [MTP Scaling](#qwen35-mtp-scaling)
  - [Context Length Scaling](#qwen35-context-length-scaling)
  - [High Concurrency](#qwen35-high-concurrency)
- [Kimi K2.5 Benchmarks](#kimi-k25-benchmarks)
  - [Single-Batch Decode Speed](#kimi-k25-single-batch-decode-speed)
  - [Context Length Scaling](#kimi-k25-context-length-scaling)
  - [Attention Backend Comparison](#kimi-k25-attention-backend-comparison)
- [GLM-5 Benchmarks](#glm-5-benchmarks)
  - [Single-Batch Decode Speed](#glm-5-single-batch-decode-speed)
- [MiniMax-M2.5 Benchmarks](#minimax-m25-benchmarks)
- [Cross-Model Comparison](#cross-model-comparison)
- [Wattage-Performance Scaling](#wattage-performance-scaling)
- [NCCL AllReduce Benchmarks](#nccl-allreduce-benchmarks)
- [P2P Interconnect Benchmarks](#p2p-interconnect-benchmarks)
- [Benchmark Tools](#benchmark-tools)

---

## Qwen3.5-397B Benchmarks

### Qwen3.5 Single-Batch Decode Speed

All numbers are decode tok/s unless noted.

| GPUs | Quant | Engine | MTP | Decode tok/s | Notes |
|:----:|:-----:|:------:|:---:|:------------:|-------|
| 4x | NVFP4 | SGLang | No | 42-51 | kcramp, Ixtrix |
| 4x | NVFP4 | SGLang | Yes (3-step) | 85 | Festr (unstable) |
| 4x | NVFP4 | vLLM | No | 70-86 | Multiple users, stable |
| 4x | NVFP4 | vLLM | MTP=2 | 130 | malaiwah |
| 4x | NVFP4 | vLLM | MTP=5 | 150-250 | Festr, orangezed (peaks in code gen) |
| 8x | NVFP4 | SGLang | Yes | 350 | luke (EP=8, switches, heavily patched) |
| 8x | FP8 | SGLang | Yes | 75-125 | CyySky |
| 8x | FP8 | vLLM | MTP=2 | -- | Expected similar to SGLang |

### Qwen3.5 MTP Scaling

vLLM, nvidia NVFP4, 4x RTX 6000 Pro, MTP=2:

| Concurrency | No MTP (tok/s) | MTP=2 (tok/s) | Improvement |
|:-----------:|:--------------:|:--------------:|:-----------:|
| 1 | 85.8 | 130.0 | +51.5% |
| 2 | 137.1 | 212.7 | +55.1% |
| 5 | 234.2 | 358.6 | +53.1% |
| 10 | 334.3 | 573.5 | +71.6% |
| 20 | 491.5 | 744.1 | +51.4% |
| 32 | 605.9 | 922.6 | +52.3% |

Peak throughput: **1127.1 tok/s** at 50 users with 1K context.

MTP acceptance stats: 89.2% acceptance rate, mean acceptance length 2.73-3.69 tokens.

### Qwen3.5 Context Length Scaling

#### CARVE Model (no MTP, vLLM, 4x GPUs, YaRN enabled)

| Context | Decode tok/s |
|:-------:|:------------:|
| 10K | 77 |
| 100K | 75 |
| 300K | 73 |
| 500K | 67 |
| 900K | 56 |

#### CARVE vs NVIDIA Reference (no MTP, warm cache)

| Context | CARVE tok/s | NVIDIA REF tok/s | Winner |
|:-------:|:-----------:|:----------------:|:------:|
| 10K | 76.9 | 92.3 | REF +20% |
| 50K | 75.5 | 91.3 | REF +21% |
| 100K | 74.8 | 73.6 | ~tied |
| 200K | 74.3 | 95.5 | REF +29% |
| 300K | 73.3 | 43.8 | **CARVE +67%** |
| 400K | 67.9 | 42.3 | **CARVE +61%** |
| 500K | 67.0 | 42.2 | **CARVE +59%** |

Key finding: CARVE model maintains much better performance at >300K context than the NVIDIA reference NVFP4.

#### Qwen3.5 with MTP=2 (malaiwah, vLLM, 4x GPUs)

67 tok/s single stream at 256K context.

### Qwen3.5 High Concurrency

vLLM, MTP=2, nvidia NVFP4, 4x RTX 6000 Pro:

```
Peak Throughput:      1127.1 tok/s   50 users @ 1K context
Best Efficiency:      120.0 tok/s/user   1 user @ 1K context
Lowest Latency:       12.30s   1 user @ 1K context
```

At 32 concurrent requests:
```
Avg generation throughput: 1287.2 tokens/s
SpecDecoding metrics: Mean acceptance length: 2.82
Accepted throughput: 830.21 tokens/s
```

---

## Kimi K2.5 Benchmarks

### Kimi K2.5 Single-Batch Decode Speed

All on 8x RTX 6000 Pro unless noted.

| System | Engine | KV Cache | DCP | Decode tok/s (0K ctx) | Notes |
|--------|:------:|:--------:|:---:|:---------------------:|-------|
| luke (switches) | SGLang | BF16 | -- | **101** | INT4, EP=8, custom AR, overclocked GDDR7 |
| CyySky | SGLang | BF16 | -- | 90 | INT4, 232K context |
| Festr Turin | vLLM | BF16 | 1 | 90 | INT4, FA2, best single batch on vLLM |
| Festr Turin | vLLM | FP8 | 1 | 79 | INT4, Triton MLA |
| Festr Turin | vLLM | FP8 | 8 | 65 | INT4, Triton MLA, 3.6M context |
| Grimulkan (switches) | vLLM | FP8 | 8 | 62 | INT4, normal NCCL |
| nvidia checkpoint | SGLang | FP8 | -- | 53-55 | NVFP4, ~450K context |
| Festr Turin | vLLM | FP8 | 8 (no P2P) | 44 | INT4, NCCL_P2P_DISABLE=1 |
| orangezed | vLLM | FP8 | 8 | 32-35 | INT4, 5-channel DIMM bottleneck |

### Kimi K2.5 Context Length Scaling

vLLM, INT4, FP8 KV, DCP=8, 8x RTX 6000 Pro:

| System | 0K Context | 100K Context | 200K Context |
|--------|:----------:|:------------:|:------------:|
| Festr Turin (P2P) | 65 tok/s | 36 tok/s | 27 tok/s |
| Festr Turin (no P2P) | 44 tok/s | 29 tok/s | 23 tok/s |
| Festr Genoa | ~32 tok/s | ~32 tok/s | -- |
| Grimulkan (switches) | 62 tok/s | 32 tok/s | 21 tok/s |
| orangezed (5-ch DIMM) | 32-35 tok/s | 8.6-10.2 tok/s | 19-20 tok/s |

**Without DCP at 150K context: 6-7 tok/s (unusable). With DCP=8: 28-35 tok/s.**

### Kimi K2.5 Attention Backend Comparison

vLLM, 8x RTX 6000 Pro:

| TP | DCP | KV Cache | KV Cache Space | Triton MLA tok/s | FA2 tok/s | XQA tok/s |
|:--:|:---:|:--------:|:--------------:|:----------------:|:---------:|:---------:|
| 8 | 1 | FP8 | 380K tok | 79 | N/A | N/A |
| 8 | 8 | FP8 | 3M tok | 68 | N/A | N/A |
| 8 | 1 | BF16 | 190K tok | 78 | **90** | WIP |
| 8 | 8 | BF16 | 1.5M tok | 67 | 72 | N/A |

### Kimi K2.5 KV Cache Capacity

| Config | Total KV Cache Tokens |
|--------|:---------------------:|
| FP8 KV, DCP=1 | ~449,600 |
| FP8 KV, DCP=8 | ~3,621,504 |
| BF16 KV, DCP=1 | ~190,000 |
| BF16 KV, DCP=8 | ~1,500,000 |

### Kimi K2.5 High Concurrency

Festr, 100 concurrent requests at 40K context each:
- **900 tok/s total** with vLLM, FP8 KV, DCP=8, TP=8

P2P vs No-P2P at high concurrency (MiniMax M2.5 test as proxy):
- P2P enabled: 5000 tok/s
- P2P disabled: 10000 tok/s

For low concurrency, P2P generates faster per-token; for high concurrency, DRAM routing wins.

---

## GLM-5 Benchmarks

### GLM-5 Single-Batch Decode Speed

All on 8x RTX 6000 Pro, SGLang, NVFP4.

| Configuration | 0K Context | 15K Context | 100K Context | 200K Context |
|---------------|:----------:|:-----------:|:------------:|:------------:|
| NVFP4 no MTP (early, luke) | ~50 | -- | -- | -- |
| NVFP4 no MTP (Festr/JTazz) | 35-44 | 30 | -- | -- |
| NVFP4 + MTP (EAGLE) | 70-105 | -- | 60-80 | -- |
| NVFP4 + MTP (latest, Festr) | ~100 | -- | 60-80 | ~50 |
| NVFP4 + MTP (orangezed) | 97.2 | -- | -- | -- |

### GLM-5 MTP Stats

- Accept rate: 0.55-0.94 (varies by context)
- Accept length: 2.19-2.80 tokens
- Speed improvement: roughly **2x** over non-MTP baseline

### GLM-5 Concurrent Throughput

3 running requests with MTP: **133-135 tok/s** generation throughput.

### GLM-5 Memory Usage

Per-GPU breakdown (8x TP8, NVFP4 + MTP):

| Component | Size |
|-----------|-----:|
| Weights (NVFP4) | 57.06 GB |
| KV Cache (bf16) | 29.32 GB |
| Total allocated | ~86.38 GB |
| Available | 7.43-7.53 GB |

KV cache capacity with `--mem-fraction-static 0.92`: 314,304 tokens total, context_len 202,752.

### GLM-5 Startup Time

| Phase | Duration |
|-------|:--------:|
| Model load (multithread, 8 threads) | ~36 sec |
| CUDA graph capture | ~208 sec |
| Total | ~7-8 min |

---

## MiniMax-M2.5 Benchmarks

4x RTX 6000 Pro:

| Quant | Engine | Concurrency | Decode tok/s |
|:-----:|:------:|:-----------:|:------------:|
| FP8 | vLLM | 1 | 74-76 |
| NVFP4 (2 GPUs) | -- | 1 | Competes well with FP8 on 4 GPUs |

### Wattage Scaling (MiniMax-M2.5 NVFP4, 4 cards)

| Concurrency | 300W tok/s | 500W tok/s | Improvement |
|:-----------:|:----------:|:----------:|:-----------:|
| 64 | 1206 | 1558 | +29% |
| 32 | -- | -- | ~25% |
| 16 | -- | -- | ~16% |
| 4 or below | -- | -- | ~0% |

---

## Cross-Model Comparison

### Single-Batch Decode Speed Summary (best configs per model)

| Model | GPUs | Quant | Engine | MTP | Best tok/s |
|-------|:----:|:-----:|:------:|:---:|:----------:|
| Qwen3.5-397B | 4x | NVFP4 | vLLM | MTP=2 | 130 |
| Qwen3.5-397B | 8x | NVFP4 | SGLang | Yes | 350 |
| Kimi K2.5 | 8x | INT4 | SGLang | No (no MTP) | 101 |
| Kimi K2.5 | 8x | INT4 | vLLM | No | 90 |
| GLM-5 | 8x | NVFP4 | SGLang | MTP | ~100 |
| MiniMax-M2.5 | 4x | FP8 | vLLM | No | 76 |

### Model Sizing Guide

| GPUs | NVFP4 Models | FP8 Models |
|:----:|:-------------|:-----------|
| 1x 96GB | Qwen3.5-27B | -- |
| 2x 96GB | MiniMax-M2.5 NVFP4, Qwen3.5-122B NVFP4 | -- |
| 4x 96GB | Qwen3.5-397B NVFP4, GLM-4.7 NVFP4, MiniMax-M2.5 FP8 | MiniMax-M2.5 FP8 |
| 6x 96GB | GLM-5 NVFP4 (TP2 PP3) | -- |
| 8x 96GB | All current models | GLM-4.7 FP8, Qwen3.5-397B FP8, Kimi K2.5 INT4 |
| 16x 96GB | All models with massive KV cache | GLM-5 FP8 |

---

## Wattage-Performance Scaling

Based on wattage-performance benchmarks at https://shihanqu.github.io/Blackwell-Wattage-Performance/

- **500W vs 600W**: Nearly identical performance.
- **300W vs 500W**: 4% loss at single-user, up to 30% loss at 64 concurrent users.
- **400W to 300W**: Significant performance drop at high concurrency.
- **300W**: Almost no penalty at 4 concurrent users or below.

MaxQ (300W) vs Workstation (600W): ~20% faster prefill on WS, similar decode speed (VRAM/PCIe limited).

---

## NCCL AllReduce Benchmarks

### Bus Bandwidth at 32M-2G Message Sizes (8 GPUs)

| System | Config | Avg Bus BW (GB/s) |
|--------|--------|:------------------:|
| luke (8x MaxQ, 3 switches) | NCCL_MIN_NCHANNELS=8 | **41.1** |
| Grimulkan (8x, 4 switches) | NCCL_MIN_NCHANNELS=8 | ~39.4 |
| Festr (8x Server, dual Turin) | NCCL_MIN_NCHANNELS=8 | 37.6 |
| Festr (8x Server, dual Turin) | Default | 22.2 |

### NCCL Graph XML Impact (AMD Turin, small messages)

| Message Size | Without XML | With XML | Speedup |
|:------------:|:-----------:|:--------:|:-------:|
| 32 KB | 48.16 us | 26.20 us | **1.84x** |
| 64 KB | 48.69 us | 25.59 us | **1.90x** |
| 128 KB | 51.56 us | 32.09 us | 1.61x |
| 256 KB | 56.48 us | 37.26 us | 1.52x |

---

## P2P Interconnect Benchmarks

### P2P Bandwidth and Latency

| Metric | Value |
|--------|:-----:|
| P2P unidirectional write bandwidth | ~55-56 GB/s |
| P2P bidirectional write bandwidth | ~111 GB/s |
| P2P enabled latency (same switch/NUMA) | 0.36-0.45 us |
| P2P disabled latency | ~14 us |

### p2pmark Scores (8 GPUs)

| System | PCIe Link Score | Dense Interconnect Score | Effective Latency |
|--------|:---------------:|:------------------------:|:-----------------:|
| luke (switches) | 0.86 (54.3 GB/s) | 0.44 (191.8/434.7 GB/s) | 6.79 us |
| Festr Turin (dual CPU) | 0.84 (52.7 GB/s) | 0.41 (173.1/421.3 GB/s) | 6.03 us |
| Grimulkan (switches) | 0.86 (53.9 GB/s) | 0.38 (164.3/431.2 GB/s) | 7.04 us |

### Custom Allreduce vs NCCL (luke's switches, 8 GPUs)

| Size | Custom (us) | NCCL (us) | Winner |
|:----:|:-----------:|:---------:|:------:|
| 256 B | 7.5 | 24.6 | Custom 3.3x |
| 1 KB | 7.5 | 24.1 | Custom 3.2x |
| 8 KB | 9.2 | 24.2 | Custom 2.6x |
| 32 KB | 16.5 | 24.5 | Custom 1.5x |
| 64 KB | 25.9 | 24.1 | NCCL 1.1x |
| 256 KB | 73.6 | 28.0 | NCCL 2.6x |

Custom allreduce is optimized for PCIe switch topologies. On dual-CPU systems without switches, it is **slower** than default NCCL.

---

## Benchmark Tools

### vllm-benchmark-suite

- URL: https://github.com/shihanqu/vllm-benchmark-suite
- Setup:
  ```bash
  uv venv vllm-benchmark-suite --python 3.12
  source vllm-benchmark-suite/bin/activate
  git clone https://github.com/notaDestroyer/vllm-benchmark-suite.git
  cd vllm-benchmark-suite.git
  uv pip install -r requirements.txt
  uv pip install transformers torch
  # Edit vllm_benchmark_suitev2.py and change API_BASE_URL
  python vllm_benchmark_suitev2.py
  ```
- Model name must be full HuggingFace name
- `HF_HUB_OFFLINE=1` helps avoid tokenizer download issues

### SGLang bench_serving

- Guide: https://github.com/nvjullin/sglang/blob/update-benchmark-doc/docs/developer_guide/bench_serving.md
- Built-in benchmarking via `sglang.bench_one_batch_server`

### Pinchbench (OpenClaw Coding Benchmark)

- URL: https://pinchbench.com/
- Repo: https://github.com/pinchbench/skill
- Requires OpenClaw CLI for task execution and LLM-judge grading

### EleutherAI lm-evaluation-harness

- URL: https://github.com/EleutherAI/lm-evaluation-harness
- Standard eval suite (MMLU-Pro, GPQA, IFEval, etc.)

### NCCL Performance Tests

```bash
# Located at /usr/src/nccl-tests (in NVIDIA containers)
NCCL_P2P_LEVEL=SYS NCCL_NET_GDR_LEVEL=SYS ./all_reduce_perf -b 32M -g 8 -c 0
NCCL_NET_GDR_LEVEL=SYS NCCL_MIN_NCHANNELS=8 ./all_reduce_perf -b 8M -e 2G -f 2 -g 8 -n 50
```

### p2pmark (PCIe Interconnect Benchmarking)

- URL: https://github.com/lukealonso/p2pmark
- Commands:
  ```bash
  ./p2pmark              # bandwidth and topology
  ./p2pmark --latency    # P2P latency
  ./p2pmark --allreduce  # custom vs NCCL allreduce comparison
  ```

### AMD xGMI Fabric Monitor

- URL: https://github.com/voipmonitor/amd-epyc-gpu-fabric-monitor
- Real-time monitoring of AMD EPYC GPU fabric transfers

### Wattage-Performance Dashboard

- URL: https://shihanqu.github.io/Blackwell-Wattage-Performance/
- Tests MiniMax-M2.5 NVFP4 at various power limits and concurrency levels

### Quality Benchmarks

| Benchmark | Use Case | Notes |
|-----------|----------|-------|
| MMLU-Pro | Knowledge testing | Use temp=0.01 |
| GPQA | Long-context reasoning | Traces can reach 64K tokens |
| AIME 2025 | Math reasoning | Requires nemo-skill install |
| WikiText perplexity | Quant quality assessment | Test across context lengths |
