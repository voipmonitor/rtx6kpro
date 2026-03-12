# Inference Benchmark: Qwen3.5-397B-A17B Quantizations

Comprehensive prefill + decode benchmark comparing quantized Qwen3.5-397B-A17B checkpoints on 4x RTX PRO 6000 Blackwell (TP4) with MTP speculative decoding.

## Summary

**AWQ-INT4 wins on both quality AND throughput.** At C=64 it delivers 38% more throughput than NVFP4 while having 1.5x better quantization quality (KLD 0.024 vs 0.035). There is no reason to use NVFP4 for this model on Blackwell.

| Model | Mean KLD | C=1 | C=8 | C=32 | C=64 | Verdict |
|-------|----------|-----|-----|------|------|---------|
| **QuantTrio/AWQ-INT4** | **0.024** | **152** | **665** | **1516** | **1662** | Best quality + best speed |
| lukealonso/NVFP4 | 0.035 | 132 | 581 | 1191 | 1202 | Good quality, 15-38% slower |
| nvidia/NVFP4 | 0.109 | — | — | — | — | Worst quality, not recommended |

KLD source: [kld-evaluation.md](../kld-evaluation.md). nvidia/NVFP4 not re-tested after benchmark improvements (original data in [old results](#old-results-context0-only)).

---

## Prefill Speed

Measured at C=1 with `max_tokens=1` (pure prefill, no decode). Baseline TTFT subtracted to isolate prefill from HTTP/scheduling overhead.

### QuantTrio/AWQ-INT4 (baseline TTFT=0.126s)

| Context | TTFT (s) | Prefill (s) | Prefill tok/s |
|---------|----------|-------------|---------------|
| 8k      | 0.62     | 0.49        | 16,680        |
| 16k     | 1.09     | 0.96        | 17,042        |
| 32k     | 2.23     | 2.11        | 15,545        |
| 64k     | 4.68     | 4.55        | 14,404        |
| 128k    | 10.41    | 10.29       | 12,740        |

### lukealonso/NVFP4 (baseline TTFT=0.088s)

| Context | TTFT (s) | Prefill (s) | Prefill tok/s |
|---------|----------|-------------|---------------|
| 8k      | 0.59     | 0.50        | 16,301        |
| 16k     | 1.08     | 1.00        | 16,431        |
| 32k     | 2.23     | 2.14        | 15,323        |
| 64k     | 4.64     | 4.55        | 14,389        |
| 128k    | 10.23    | 10.14       | 12,927        |

### Prefill Takeaways

- **Both models have virtually identical prefill speed** — ~16-17k tok/s at short contexts, ~13k tok/s at 128k.
- Prefill is limited by chunked prefill pipeline (`--chunked-prefill-size 4096`), not quantization format.
- Prefill throughput decreases with context length due to memory bandwidth pressure from growing KV cache.
- Prefill does NOT scale with concurrency — aggregate throughput is constant (~12.5k tok/s regardless of C=1/2/4/8).

---

## Decode Throughput

All measurements use radix-cached prefill (pure decode speed). Server-side `gen_throughput` Prometheus metric, median of samples with 4s warmup skip, 30s per cell.

### QuantTrio/AWQ-INT4

#### Aggregate Throughput (tok/s)

| ctx\conc |     1 |     2 |     4 |     8 |    16 |    32 |    64 |
|----------|------:|------:|------:|------:|------:|------:|------:|
| 0        | 152.4 | 263.2 | 438.4 | 665.3 | 975.7 |1516.2 |1661.7 |
| 16k      |  92.1 | 173.6 | 301.7 | 493.7 | 788.1 |1058.3 |   N/A |
| 32k      |  59.4 | 113.6 | 225.1 | 399.2 | 651.8 |   N/A |   N/A |
| 64k      |  46.1 |  76.2 | 146.9 | 274.9 | 463.9 |   N/A |   N/A |
| 128k     |  28.4 |  42.5 |  88.6 | 163.9 |   N/A |   N/A |   N/A |

#### Per-Request Throughput (tok/s)

| ctx\conc |     1 |     2 |     4 |     8 |    16 |    32 |    64 |
|----------|------:|------:|------:|------:|------:|------:|------:|
| 0        | 152.4 | 131.6 | 109.6 |  83.2 |  61.0 |  47.4 |  26.0 |
| 16k      |  92.1 |  86.8 |  75.4 |  61.7 |  49.3 |  33.1 |   N/A |
| 32k      |  59.4 |  56.8 |  56.3 |  49.9 |  40.7 |   N/A |   N/A |
| 64k      |  46.1 |  38.1 |  36.7 |  34.4 |  29.0 |   N/A |   N/A |
| 128k     |  28.4 |  21.2 |  22.2 |  20.5 |   N/A |   N/A |   N/A |

#### TTFT (seconds)

| ctx\conc |     1 |     2 |     4 |     8 |    16 |    32 |    64 |
|----------|------:|------:|------:|------:|------:|------:|------:|
| 0        |  0.07 |  0.11 |  0.12 |  0.14 |  0.19 |  0.23 |  0.27 |
| 16k      |  1.17 |  0.23 |  0.25 |  0.34 |  0.57 |  0.97 |   N/A |
| 32k      |  2.30 |  0.29 |  0.39 |  0.57 |  1.18 |   N/A |   N/A |
| 64k      |  4.65 |  0.48 |  0.85 |  1.55 |  2.93 |   N/A |   N/A |
| 128k     | 10.44 |  0.90 |  1.53 |  3.08 |   N/A |   N/A |   N/A |

### lukealonso/NVFP4

#### Aggregate Throughput (tok/s)

| ctx\conc |     1 |     2 |     4 |     8 |    16 |    32 |    64 |
|----------|------:|------:|------:|------:|------:|------:|------:|
| 0        | 132.0 | 222.0 | 390.4 | 580.8 | 851.9 |1190.7 |1202.2 |
| 16k      |  80.8 | 148.8 | 268.6 | 456.5 | 715.0 | 886.9 |   N/A |
| 32k      |  62.6 | 110.5 | 212.6 | 359.9 | 590.3 |   N/A |   N/A |
| 64k      |  42.8 |  74.9 | 141.0 | 254.3 |   N/A |   N/A |   N/A |
| 128k     |  32.6 |  43.6 |  82.6 |   N/A |   N/A |   N/A |   N/A |

#### Per-Request Throughput (tok/s)

| ctx\conc |     1 |     2 |     4 |     8 |    16 |    32 |    64 |
|----------|------:|------:|------:|------:|------:|------:|------:|
| 0        | 132.0 | 111.0 |  97.6 |  72.6 |  53.2 |  37.2 |  18.8 |
| 16k      |  80.8 |  74.4 |  67.1 |  57.1 |  44.7 |  27.7 |   N/A |
| 32k      |  62.6 |  55.2 |  53.2 |  45.0 |  36.9 |   N/A |   N/A |
| 64k      |  42.8 |  37.5 |  35.3 |  31.8 |   N/A |   N/A |   N/A |
| 128k     |  32.6 |  21.8 |  20.7 |   N/A |   N/A |   N/A |   N/A |

#### TTFT (seconds)

| ctx\conc |     1 |     2 |     4 |     8 |    16 |    32 |    64 |
|----------|------:|------:|------:|------:|------:|------:|------:|
| 0        |  0.10 |  0.19 |  0.18 |  0.21 |  0.23 |  0.71 |  0.29 |
| 16k      |  1.10 |  0.24 |  0.28 |  0.36 |  0.58 |  0.90 |   N/A |
| 32k      |  2.19 |  0.31 |  0.38 |  0.64 |  1.22 |   N/A |   N/A |
| 64k      |  4.58 |  0.42 |  0.71 |  1.53 |   N/A |   N/A |   N/A |
| 128k     | 10.14 |  0.98 |  1.79 |   N/A |   N/A |   N/A |   N/A |

### AWQ vs NVFP4 — Head to Head

| Metric | AWQ (QuantTrio) | NVFP4 (lukealonso) | AWQ advantage |
|--------|-----------------|--------------------|--------------:|
| **Quality (Mean KLD)** | **0.024** | 0.035 | 1.5x better |
| **C=1 decode (ctx=0)** | **152** tok/s | 132 tok/s | +15% |
| **C=8 decode (ctx=0)** | **665** tok/s | 581 tok/s | +14% |
| **C=32 decode (ctx=0)** | **1,516** tok/s | 1,191 tok/s | +27% |
| **C=64 decode (ctx=0)** | **1,662** tok/s | 1,202 tok/s | +38% |
| **Prefill (16k)** | 17,042 tok/s | 16,431 tok/s | ~same |

### Key Takeaways

1. **AWQ is faster at ALL concurrency levels** — 15% faster at C=1, growing to 38% at C=64.
2. **AWQ scales better** — at C=64, AWQ still gains throughput (1662 tok/s) while NVFP4 plateaus at C=32 (1191→1202).
3. **Context length heavily impacts per-request speed** — at 128k, per-request decode drops to ~28-33 tok/s regardless of model.
4. **Prefill is identical** — both models push ~16-17k tok/s, limited by the chunked prefill pipeline.
5. **TTFT at C=1 reflects prefill time** — at C≥2 with cached context, TTFT drops to sub-second even at 128k.
6. **N/A cells exceed KV cache budget** — the server auto-reports its KV capacity and the benchmark skips combinations that would exceed it.

---

## Old Results (context=0 only)

Earlier benchmark run (2026-03-12, `--duration 15`, `--max-running-requests 32`):

```
Aggregate Throughput (tok/s), context=0
Model                                  C=1     C=4    C=16    C=32
----------------------------------------------------------------------
QuantTrio/Qwen3.5-397B-A17B-AWQ       124.8   411.9  1013.0  1470.0
lukealonso/Qwen3.5-397B-A17B-NVFP4    129.1   361.0   879.3  1062.5
nvidia/Qwen3.5-397B-A17B-NVFP4        110.5   343.2   827.5  1032.3
```

These were measured with shorter duration (15s) and lower max-running-requests (32). The updated results above use 30s duration, 64 max-running-requests, and include context length scaling.

---

## Hardware

- **Server:** 8x NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB each)
- **GPUs used:** 4x (TP4) for all quantized models
- **Container:** `voipmonitor/llm-pytorch-blackwell:nightly-cuda132`
- **SGLang version:** nightly (2026-03-12)

## How to Reproduce

### 1. Start container

```bash
docker run -it --rm \
  --gpus all --ipc=host --shm-size=8g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v vllm-nightly-jit:/cache/jit \
  voipmonitor/llm-pytorch-blackwell:nightly-cuda132 \
  bash
```

### 2. Install benchmark dependencies

```bash
pip install rich httpx
```

### 3. Launch model server

#### AWQ (QuantTrio)

```bash
NCCL_P2P_LEVEL=SYS SGLANG_ENABLE_SPEC_V2=True python3 -m sglang.launch_server \
  --model QuantTrio/Qwen3.5-397B-A17B-AWQ --served-model-name Qwen3.5 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --tensor-parallel-size 4 --kv-cache-dtype fp8_e4m3 --trust-remote-code \
  --cuda-graph-max-bs 64 --max-running-requests 64 \
  --chunked-prefill-size 4096 \
  --speculative-algo NEXTN --speculative-num-steps 5 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 \
  --mamba-scheduler-strategy extra_buffer \
  --mem-fraction-static 0.95 --host 0.0.0.0 --port 5000 \
  --disable-custom-all-reduce --attention-backend triton --enable-metrics
```

#### NVFP4 (lukealonso)

```bash
NCCL_P2P_LEVEL=SYS SGLANG_ENABLE_SPEC_V2=True python3 -m sglang.launch_server \
  --model lukealonso/Qwen3.5-397B-A17B-NVFP4 --served-model-name Qwen3.5 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --tensor-parallel-size 4 --kv-cache-dtype fp8_e4m3 --trust-remote-code \
  --quantization modelopt_fp4 \
  --moe-runner-backend flashinfer_cutlass --fp4-gemm-backend flashinfer_cudnn \
  --cuda-graph-max-bs 64 --max-running-requests 64 \
  --chunked-prefill-size 4096 \
  --speculative-algo NEXTN --speculative-num-steps 5 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 \
  --mamba-scheduler-strategy extra_buffer \
  --mem-fraction-static 0.97 --host 0.0.0.0 --port 5000 \
  --disable-custom-all-reduce --attention-backend triton --enable-metrics
```

### 4. Run benchmark

```bash
python3 benchmark_sglang.py --port 5000 --output results.json
```

The benchmark automatically:
- Reads KV cache capacity and max concurrent requests from the server
- Runs prefill speed tests (8k–128k context)
- Runs decode throughput matrix with cached prefill
- Skips cells that would exceed KV cache budget

---

## Measurement Methodology

- **Throughput source:** Server-side `sglang:gen_throughput` Prometheus metric (not client-side SSE counting). With MTP, SGLang batches ~3-4 tokens per SSE event — client-side counting under-reports by ~3x.
- **Aggregation:** Median of samples (robust to outliers from MTP accept rate variation).
- **Warmup:** First 4 seconds of each cell discarded (CUDA graph warmup, ramp-up).
- **Duration:** 30 seconds per cell (~26 samples after warmup).
- **Prefill measurement:** Baseline TTFT (ctx=0) subtracted to isolate pure prefill time from HTTP/scheduling overhead.
- **Cache isolation:** Each prefill test uses a unique random prefix per run — no cross-run or cross-context cache contamination.
- **KV budget:** Cells where `concurrency × (context + max_tokens) > max_total_num_tokens` are automatically skipped.

## Benchmark Script

The benchmark script (`benchmark_sglang.py`) features:
- Two-phase design: prefill speed → decode throughput (cached)
- Auto-detection of server limits (`/get_server_info`)
- Rich TUI dashboard with live server metrics
- Unique text per context length (defeats radix cache for prefill)
- JSON output with full per-cell results + prefill data
- Graceful interrupt handling with partial results saved
