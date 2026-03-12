# Inference Throughput Benchmark: Qwen3.5-397B-A17B Quantizations

Decode throughput comparison of three quantized Qwen3.5-397B-A17B checkpoints on 4x RTX PRO 6000 Blackwell (TP4) with MTP speculative decoding.

## Results

All models tested with identical SGLang configuration, MTP enabled (NEXTN, 5 steps, 6 draft tokens), context=0, max_tokens=8192. Throughput measured from server-side `sglang:gen_throughput` Prometheus metric (not client-side SSE counting).

### Aggregate Throughput (tok/s)

```
Model                                  C=1     C=4    C=16    C=32
----------------------------------------------------------------------
QuantTrio/Qwen3.5-397B-A17B-AWQ       124.8   411.9  1013.0  1470.0
lukealonso/Qwen3.5-397B-A17B-NVFP4    129.1   361.0   879.3  1062.5
nvidia/Qwen3.5-397B-A17B-NVFP4        110.5   343.2   827.5  1032.3
```

### Per-Request Throughput (tok/s)

```
Model                                  C=1     C=4    C=16    C=32
----------------------------------------------------------------------
QuantTrio/Qwen3.5-397B-A17B-AWQ       147.5   117.4   67.3    50.9
lukealonso/Qwen3.5-397B-A17B-NVFP4    124.6   101.2   59.5    34.2
nvidia/Qwen3.5-397B-A17B-NVFP4        116.6    94.1   56.5    33.6
```

### Time to First Token (seconds)

```
Model                                  C=1     C=4    C=16    C=32
----------------------------------------------------------------------
QuantTrio/Qwen3.5-397B-A17B-AWQ        0.09    0.12   0.29    0.22
lukealonso/Qwen3.5-397B-A17B-NVFP4     0.09    0.14   0.18    0.24
nvidia/Qwen3.5-397B-A17B-NVFP4         0.32    0.19   0.24    0.30
```

### Key Takeaways

1. **AWQ dominates at high concurrency** — 38% faster than NVFP4 at C=32 (1470 vs 1063/1032 tok/s).
2. **Similar single-request speed** — all three models produce ~110-130 tok/s at C=1.
3. **AWQ scales better with batching** — aggregate throughput grows nearly linearly up to C=32, while NVFP4 models plateau earlier.
4. **TTFT is similar across all models** — under 0.3s in most cases.

### Combined with KLD Quality

| Model | Mean KLD | C=1 tok/s | C=32 tok/s | Verdict |
|-------|----------|-----------|------------|---------|
| QuantTrio/AWQ-INT4 | **0.024** | 125 | **1470** | Best quality + best throughput |
| lukealonso/NVFP4 | 0.035 | 129 | 1063 | Good quality, moderate throughput |
| nvidia/NVFP4 | 0.109 | 111 | 1032 | Worst quality, worst throughput |

AWQ-INT4 wins on **both quality and throughput** for this model on Blackwell.

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
  --cuda-graph-max-bs 32 --max-running-requests 32 \
  --context-length 262144 --chunked-prefill-size 32768 \
  --speculative-algo NEXTN --speculative-num-steps 5 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 \
  --mamba-scheduler-strategy extra_buffer --page-size 64 \
  --mem-fraction-static 0.85 --host 0.0.0.0 --port 5000 \
  --disable-custom-all-reduce --attention-backend triton --enable-metrics
```

#### NVFP4 (lukealonso or nvidia)

Same as AWQ but add FP4-specific flags:

```bash
NCCL_P2P_LEVEL=SYS SGLANG_ENABLE_SPEC_V2=True python3 -m sglang.launch_server \
  --model lukealonso/Qwen3.5-397B-A17B-NVFP4 --served-model-name Qwen3.5 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --tensor-parallel-size 4 --kv-cache-dtype fp8_e4m3 --trust-remote-code \
  --quantization modelopt_fp4 \
  --moe-runner-backend flashinfer_cutlass --fp4-gemm-backend flashinfer_cudnn \
  --cuda-graph-max-bs 32 --max-running-requests 32 \
  --context-length 262144 --chunked-prefill-size 32768 \
  --speculative-algo NEXTN --speculative-num-steps 5 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 \
  --mamba-scheduler-strategy extra_buffer --page-size 64 \
  --mem-fraction-static 0.85 --host 0.0.0.0 --port 5000 \
  --disable-custom-all-reduce --attention-backend triton --enable-metrics
```

For nvidia checkpoint, replace `lukealonso/Qwen3.5-397B-A17B-NVFP4` with `nvidia/Qwen3.5-397B-A17B-NVFP4`.

### 4. Run benchmark

```bash
python3 benchmark_sglang.py \
  --concurrency 1,4,16,32 \
  --contexts 0 \
  --duration 15 \
  --model Qwen3.5 \
  --max-tokens 8192 \
  --output results.json
```

---

## Server Configuration Details

All flags explained:

| Flag | Value | Purpose |
|------|-------|---------|
| `--tensor-parallel-size 4` | 4 | Use 4 GPUs (TP4) |
| `--kv-cache-dtype fp8_e4m3` | fp8_e4m3 | FP8 KV cache to save memory |
| `--attention-backend triton` | triton | Required for Blackwell SM120 with Qwen3.5 hybrid GDN attention |
| `--speculative-algo NEXTN` | NEXTN | Enable MTP speculative decoding (Qwen3.5 has built-in MTP layers) |
| `--speculative-num-steps 5` | 5 | Number of speculative decoding steps |
| `--speculative-num-draft-tokens 6` | 6 | Draft tokens per step |
| `--speculative-eagle-topk 1` | 1 | Top-k for EAGLE draft selection |
| `--mamba-scheduler-strategy extra_buffer` | extra_buffer | Required for MTP with VLM-format models (ConditionalGeneration) |
| `--page-size 64` | 64 | Memory page size for extra_buffer strategy |
| `--cuda-graph-max-bs 32` | 32 | Max CUDA graph batch size |
| `--max-running-requests 32` | 32 | Max concurrent requests in-flight |
| `--context-length 262144` | 256K | Maximum context window |
| `--chunked-prefill-size 32768` | 32K | Chunk size for prefill processing |
| `--mem-fraction-static 0.85` | 0.85 | GPU memory fraction for KV cache |
| `--disable-custom-all-reduce` | - | Required on Blackwell for stability |
| `--enable-metrics` | - | Enable Prometheus metrics endpoint |
| `NCCL_P2P_LEVEL=SYS` | SYS | NCCL peer-to-peer at system level |
| `SGLANG_ENABLE_SPEC_V2=True` | True | Enable v2 speculative decoding engine |

### MTP (Multi-Token Prediction) Notes

- **Both** `SGLANG_ENABLE_SPEC_V2=True` (env) AND `--speculative-algo NEXTN` (CLI) are required. The env var alone does NOT enable MTP.
- All Qwen3.5-397B-A17B variants (FP8, AWQ, NVFP4) have MTP layers in their weights.
- VLM-format models (`Qwen3_5MoeForConditionalGeneration`) require `--mamba-scheduler-strategy extra_buffer` for MTP. Using `--disable-radix-cache` also works but is slower.
- MTP acceptance rate is typically 58-78% with accept length ~3.0-3.5 out of 6 draft tokens.

### NVFP4-specific flags

| Flag | Purpose |
|------|---------|
| `--quantization modelopt_fp4` | Enable FP4 weight loading (ModelOpt format) |
| `--moe-runner-backend flashinfer_cutlass` | Use FlashInfer+CUTLASS for MoE kernels |
| `--fp4-gemm-backend flashinfer_cudnn` | Use FlashInfer+cuDNN for FP4 GEMM |

AWQ models do NOT need `--quantization` — SGLang auto-detects AWQ from the checkpoint config.

## Benchmark Script

The benchmark script (`benchmark_sglang.py`) features:
- Rich TUI dashboard with live server metrics
- Throughput measured from server-side `sglang:gen_throughput` Prometheus metric (accurate with MTP)
- Configurable concurrency levels, context lengths, and duration
- `--cached-prefill` mode for pure decode speed measurement (populates radix cache)
- JSON output with full per-cell results
- Graceful interrupt handling with partial results saved

### Why server-side metrics?

With MTP speculative decoding, SGLang batches multiple accepted tokens into each SSE streaming event (~3-4 tokens per event). Client-side SSE delta counting under-reports throughput by ~3x. The server's `gen_throughput` metric counts actual generated tokens accurately.
