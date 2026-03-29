# KLD Evaluation for Quantized Models

Measure how much quality is lost in quantized models (NVFP4, AWQ, etc.) compared to a higher-precision reference (FP8) using KL divergence over full vocabulary logit distributions.

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [How It Works](#how-it-works)
- [Step-by-Step Guide](#step-by-step-guide)
- [Automation Script](#automation-script)
- [Interpreting Results](#interpreting-results)
- [Known Issues](#known-issues)

---

## Overview

Standard benchmarks (MMLU, HumanEval, etc.) are noisy and coarse. KL divergence measures the **exact difference in output probability distributions** between two models, giving a much more sensitive quality metric.

**Reference model:** `Qwen/Qwen3.5-397B-A17B-FP8` (TP8, 8x RTX PRO 6000 Blackwell)
**Test models:** See results below
**Dataset:** WikiText-2, 100 sliding windows (2048 tokens, stride 512), 204,800 total positions

## Results

All measurements taken on 8x RTX PRO 6000 Blackwell Server Edition, SGLang with `--attention-backend triton`, same container and reference for all models. Log-probabilities stored in float32 (computed on CPU), MTP speculative-head logits excluded via call-stack filtering.

```
KLD Evaluation Results (ref: Qwen3.5-397B-A17B-FP8, dataset: wikitext-2, 204,800 positions)
============================================================================================

Model                                      Mean KLD   Median KLD    P95 KLD    P99 KLD    Max KLD
------------------------------------------------------------------------------------------------
QuantTrio/Qwen3.5-397B-A17B-AWQ (INT4)    0.024057     0.004778   0.097600   0.349900     4.3300
lukealonso/Qwen3.5-397B-A17B-NVFP4        0.035637     0.006939   0.147900   0.534100     4.4300
nvidia/Qwen3.5-397B-A17B-NVFP4            0.108526     0.027302   0.467703   1.411015    19.6018
```

### MoE Backend Comparison (lukealonso/NVFP4)

Different MoE/FP4 backends produce equivalent KLD — the MoE kernel choice does not affect quality:

```
Model                                      Mean KLD   Median KLD    P95 KLD    P99 KLD    Max KLD
------------------------------------------------------------------------------------------------
flashinfer_cutlass (fp4 + moe)             0.035637     0.006939   0.147900   0.534100     4.4300
cutedsl + cudnn (moe cutedsl, fp4 cudnn)   0.036000     0.006900   0.148700   0.538100     4.4300
cutlass MoE                                0.036000     0.006900   0.148800   0.538100     4.4300
```

### Ranking

1. **QuantTrio/AWQ-INT4** — best quality across all metrics. Mean KLD 0.024 (near-lossless).
2. **lukealonso/NVFP4** — 1.5x worse than AWQ but still good. Mean KLD 0.036.
3. **nvidia/NVFP4** — 4.5x worse than AWQ, 3x worse than lukealonso. Mean KLD 0.109, with a heavy tail (Max KLD 19.6).

### Why AWQ beats NVFP4 in quality

- **INT4 (AWQ)** has 16 quantization levels with per-channel scaling and salient weight protection — smarter allocation of precision to important weights.
- **FP4 (NVFP4, E2M1)** has only 8 unique values — less effective precision, but has dedicated Blackwell FP4 Tensor Core hardware for faster matmul.
- NVFP4 trades quality for throughput; AWQ trades throughput for quality — **however, our throughput benchmarks show AWQ is also faster** (see below).

### Throughput Benchmark (MTP Speculative Decoding)

All models tested with identical SGLang configuration, MTP enabled (NEXTN, 5 steps, 6 draft tokens), `--mamba-scheduler-strategy extra_buffer`, 4x RTX PRO 6000 Blackwell (TP4). Throughput measured from server-side `sglang:gen_throughput` Prometheus metric (median, 30s per cell, 4s warmup skip).

```
Aggregate decode throughput (tok/s), context=0
=========================================================================

Model                                 C=1    C=8    C=16    C=32    C=64
------------------------------------------------------------------------
QuantTrio/Qwen3.5-397B-A17B-AWQ      152    665     976    1516    1662
lukealonso/Qwen3.5-397B-A17B-NVFP4   132    581     852    1191    1202
```

**AWQ wins on both quality AND throughput** at every concurrency level. 15% faster at C=1, growing to 38% at C=64 where AWQ still gains throughput (1662 tok/s) while NVFP4 plateaus (1191→1202). Prefill speed is identical (~16-17k tok/s at 16k context).

For full decode + prefill tables across context lengths, reproduction details, and the benchmark script, see [inference-throughput/](inference-throughput/).

#### MTP setup (critical)

MTP (Multi-Token Prediction) speculative decoding requires **both**:
1. `SGLANG_ENABLE_SPEC_V2=True` (environment variable)
2. `--speculative-algo NEXTN` (CLI flag)

The env var alone does NOT enable MTP. Without the CLI flag, `speculative_algorithm=None` in the server config and MTP is completely disabled.

VLM-format models (`Qwen3_5MoeForConditionalGeneration`) require `--mamba-scheduler-strategy extra_buffer` for MTP to work.

### Interpretation scale

| Mean KLD | Quantization quality |
|----------|---------------------|
| < 0.01 | Near-lossless |
| 0.01 - 0.05 | Good, minimal quality loss |
| 0.05 - 0.1 | Noticeable quality loss |
| > 0.1 | Significant quality loss |

---

## How It Works

### Problem

SGLang only exposes top-k logprobs via its API, not full vocabulary logits. KLD needs full distributions over all 152,064 tokens.

### Solution

1. **Patch SGLang** at runtime to capture full `[N, vocab_size]` log-probability tensors during prefill
2. **Run reference model** (FP8) on sliding windows over WikiText-2, saving logits to disk as safetensors
3. **Run test model(s)** on the same windows, saving logits to disk
4. **Compute KLD** between reference and test logit distributions

### Architecture

```
Phase 1: FP8 Reference (TP8)          Phase 2: Test Model (TP4)
+-----------------------+               +-----------------------+
| SGLang Server         |               | SGLang Server         |
| + logit capture       |               | + logit capture       |
|   patch               |               |   patch               |
+-----------+-----------+               +-----------+-----------+
            | saves logits                          | saves logits
            v                                       v
      /tmp/kld_ref/                           /tmp/kld_test/
      +-- 0.safetensors                       +-- 0.safetensors
      +-- 1.safetensors         --KLD-->      +-- 1.safetensors
      +-- ...99.safetensors                   +-- ...99.safetensors
```

### Storage requirements

- Per window: 2048 x 152,064 x 4 bytes = **1,188 MB** (text-only models, float32)
- Per window: 2048 x 248,320 x 4 bytes = **1,940 MB** (VLM models like AWQ, float32)
- 100 windows = **~116-190 GB** per model
- Runtime: ~130-250s per phase (100 windows), KLD compute takes seconds

### What the patch does

The patch (`patches/sglang-kld-logit-capture.py`) modifies `LogitsProcessor.forward()` in SGLang to insert a `_kld_maybe_save()` hook:

```python
# BEFORE (in LogitsProcessor.forward, non-chunked path):
input_logits = logits[input_logprob_indices]
del logits
logprobs_result = self.process_input_logprobs(input_logits, logits_metadata)

# AFTER:
input_logits = logits[input_logprob_indices]
del logits
_kld_maybe_save(input_logits, logits_metadata)  # saves full [N, vocab_size] log-softmax
logprobs_result = self.process_input_logprobs(input_logits, logits_metadata)
```

The hook:
- Is a no-op unless `SGLANG_KLD_SAVE_DIR` env var is set
- **Skips MTP/NextN speculative-head calls** by inspecting the call stack for MTP model files (`*mtp*.py`, `*nextn*.py`) — without this, MTP models save 2x files per window, contaminating KLD by ~18%
- **Skips `DRAFT_EXTEND` forward mode** (post-decode MTP speculative passes)
- Only saves from TP rank 0 (avoids duplicate writes across tensor-parallel workers)
- Trims TP padding columns to actual `vocab_size` (controlled by `SGLANG_KLD_VOCAB_SIZE`, default 152064)
- Computes `log_softmax` in float32 on CPU (avoids GPU OOM), saves as float32 safetensors

---

## Step-by-Step Guide

### Prerequisites

- Docker image: `voipmonitor/llm-pytorch-blackwell:nightly` or `voipmonitor/llm-pytorch-blackwell:nightly-cuda132`
- 8x GPUs for FP8 reference (TP8), 4x GPUs for quantized test models (TP4)
- ~120 GB free disk space per model pair
- Files from this repo:
  - `patches/sglang-kld-logit-capture.py`
  - `scripts/sglang_kld_eval.py`

### Step 1: Start container

```bash
docker run --rm -it \
  --gpus all --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 5000:5000 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v vllm-nightly-jit:/cache/jit \
  -v /tmp/kld:/tmp/kld \
  voipmonitor/llm-pytorch-blackwell:nightly \
  bash
```

### Step 2: Apply the logit capture patch

Inside the container:

```bash
pip install datasets  # needed for wikitext loading
python /workspace/sglang-kld-logit-capture.py
```

> If the image already has the patch baked in (nightly images do), this step is a no-op.

### Step 3: Run FP8 reference server

```bash
mkdir -p /tmp/kld/ref

SGLANG_KLD_SAVE_DIR=/tmp/kld/ref \
SGLANG_KLD_VOCAB_SIZE=152064 \
SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=0 \
NCCL_P2P_DISABLE=1 \
python -m sglang.launch_server \
  --model Qwen/Qwen3.5-397B-A17B-FP8 \
  --tp 8 --trust-remote-code \
  --kv-cache-dtype bfloat16 \
  --mem-fraction-static 0.85 \
  --disable-custom-all-reduce \
  --attention-backend triton \
  --host 0.0.0.0 --port 5000
```

> **Blackwell note:** `--attention-backend triton` is required for Qwen3.5-397B on Blackwell (SM120) due to hybrid GDN attention architecture. The server will fail without it.

### Step 4: Generate reference logits

From a **second terminal**:

```bash
docker exec -it <container_id> \
  python /workspace/sglang_kld_eval.py --phase ref \
    --server-url http://localhost:5000 \
    --tokenizer Qwen/Qwen3.5-397B-A17B-FP8 \
    --logits-dir /tmp/kld/ref
```

Expected output:
```
Done. 100 windows in 69.1s
Files saved: 100
First file shape: torch.Size([2048, 152064])
```

### Step 5: Stop reference server, start test model

Ctrl+C the server, then start a test model. Examples for each quantization type:

#### NVFP4 (nvidia or lukealonso)

```bash
mkdir -p /tmp/kld/test_nvfp4
rm -f /tmp/kld/test_nvfp4/*

SGLANG_KLD_SAVE_DIR=/tmp/kld/test_nvfp4 \
SGLANG_KLD_VOCAB_SIZE=152064 \
SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=0 \
NCCL_P2P_LEVEL=SYS \
python -m sglang.launch_server \
  --model nvidia/Qwen3.5-397B-A17B-NVFP4 \
  --tp 4 --trust-remote-code \
  --quantization modelopt_fp4 \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend triton \
  --moe-runner-backend flashinfer_cutlass \
  --fp4-gemm-backend flashinfer_cudnn \
  --mem-fraction-static 0.85 \
  --disable-custom-all-reduce \
  --host 0.0.0.0 --port 5000
```

#### AWQ (QuantTrio)

```bash
mkdir -p /tmp/kld/test_awq
rm -f /tmp/kld/test_awq/*

SGLANG_KLD_SAVE_DIR=/tmp/kld/test_awq \
SGLANG_KLD_VOCAB_SIZE=248320 \
SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=0 \
NCCL_P2P_LEVEL=SYS \
python -m sglang.launch_server \
  --model QuantTrio/Qwen3.5-397B-A17B-AWQ \
  --tp 4 --trust-remote-code \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend triton \
  --mem-fraction-static 0.85 \
  --disable-custom-all-reduce \
  --host 0.0.0.0 --port 5000
```

> **AWQ note:** This model uses VLM format (`Qwen3_5MoeForConditionalGeneration`) with `vocab_size=248320`. Set `SGLANG_KLD_VOCAB_SIZE=248320`. The compute phase automatically handles vocab mismatch by truncating to the common 152,064 text tokens and re-normalizing. See [VLM models](#vlm-models-different-vocab-size) in Known Issues.

> **Do NOT** add `--speculative-*` or `--quantization` flags for AWQ (SGLang auto-detects AWQ from the config).

### Step 6: Generate test logits

```bash
docker exec -it <container_id> \
  python /workspace/sglang_kld_eval.py --phase test \
    --server-url http://localhost:5000 \
    --tokenizer Qwen/Qwen3.5-397B-A17B-FP8 \
    --logits-dir /tmp/kld/test_nvfp4   # or test_awq
```

> **Important:** Always use the **same tokenizer** (`Qwen/Qwen3.5-397B-A17B-FP8`) for both ref and test phases to ensure identical sliding windows.

### Step 7: Compute KLD

Stop the server first (KLD compute needs GPU memory), then:

```bash
python /workspace/sglang_kld_eval.py --phase compute \
  --ref-dir /tmp/kld/ref \
  --test-dirs /tmp/kld/test_awq /tmp/kld/test_nvfp4 \
  --test-names "QuantTrio/AWQ" "nvidia/NVFP4"
```

> If the server is still running and using all GPUs, use `CUDA_VISIBLE_DEVICES=4` (or any free GPU) to run compute on a different GPU.

---

## Automation Script

The full pipeline can be run with `scripts/kld_eval_pipeline.sh`:

```bash
# Run everything: FP8 reference + all test models + compute KLD
./scripts/kld_eval_pipeline.sh
```

The script:
1. Starts an FP8 reference server, generates reference logits, stops it
2. For each test model: starts server, generates logits, stops it
3. Computes KLD for all test models against the reference
4. Prints results table

### Configuration

Edit the variables at the top of the script, or override via environment:

| Variable | Default | Purpose |
|----------|---------|---------|
| `KLD_BASE_DIR` | `/tmp/kld` | Base directory for all logit files |
| `KLD_REF_MODEL` | `Qwen/Qwen3.5-397B-A17B-FP8` | Reference model |
| `KLD_REF_TP` | `8` | Reference TP size |
| `KLD_TOKENIZER` | `Qwen/Qwen3.5-397B-A17B-FP8` | Tokenizer (same for all phases) |
| `KLD_PORT` | `5000` | Server port |
| `KLD_STARTUP_TIMEOUT` | `600` | Max seconds to wait for server startup |

### Adding test models

Test models are defined in the `TEST_MODELS` array inside the script. Each entry specifies the model path, display name, vocab size, and any extra server flags:

```bash
TEST_MODELS=(
  "nvidia/Qwen3.5-397B-A17B-NVFP4|nvidia/NVFP4|152064|--tp 4 --quantization modelopt_fp4 ..."
  "lukealonso/Qwen3.5-397B-A17B-NVFP4|lukealonso/NVFP4|152064|--tp 4 --quantization modelopt_fp4 ..."
  "QuantTrio/Qwen3.5-397B-A17B-AWQ|QuantTrio/AWQ|248320|--tp 4 ..."
)
```

---

## Interpreting Results

### KLD scale

| Mean KLD | Quantization quality |
|----------|---------------------|
| < 0.01 | Near-lossless |
| 0.01 - 0.05 | Good, minimal quality loss |
| 0.05 - 0.1 | Noticeable quality loss |
| > 0.1 | Significant quality loss |

### What the metrics mean

- **Mean KLD** -- average divergence across all token positions. The primary quality metric.
- **Median KLD** -- if much lower than mean, the distribution has a heavy right tail (a few positions are very wrong, most are fine).
- **P95 / P99** -- tail behavior. High P95 means 5% of positions have substantially different predictions than the reference.
- **Max KLD** -- worst single position. Values > 10 indicate completely broken predictions at some positions.

### KLD formula

For each token position, KLD is computed as:

```
KL(P_ref || Q_test) = sum_x  P_ref(x) * (log P_ref(x) - log Q_test(x))
```

Where the sum is over all vocabulary tokens. This measures how many nats of information are lost when using the test model's distribution instead of the reference.

### Determinism

KLD evaluation is fully deterministic -- running the same model twice on the same inputs produces bit-identical results. This makes it reliable for A/B comparisons.

---

## Known Issues

### VLM models (different vocab size)

Some checkpoints use VLM format (`Qwen3_5MoeForConditionalGeneration`) with `vocab_size=248320` instead of the text-only `vocab_size=152064`. Examples: `QuantTrio/Qwen3.5-397B-A17B-AWQ`.

**Impact on KLD capture:**
- Set `SGLANG_KLD_VOCAB_SIZE=248320` when running the server

**Impact on KLD compute:**
- The compute script automatically detects vocab size mismatch and truncates both distributions to the common 152,064 text tokens, then re-normalizes via `logsumexp`. This is mathematically equivalent to computing log-softmax over text tokens only.
- Visual tokens (indices 152064-248319) are irrelevant for text-only benchmarks like WikiText.

### AWQ + FusedMoE modules_to_not_convert

AWQ checkpoints with `modules_to_not_convert` (like `QuantTrio/Qwen3.5-397B-A17B-AWQ` which keeps layer 0 in BF16) require SGLang PR [#20439](https://github.com/sgl-project/sglang/pull/20439) or later. Without this fix, loading fails with `KeyError: 'model.layers.0.mlp.experts.w13_weight'` because the FusedMoE layer doesn't check the skip list.

### Blackwell attention backend

On Blackwell GPUs (SM120), Qwen3.5-397B requires `--attention-backend triton` due to its hybrid GDN attention architecture. The server will crash with an `AssertionError` without this flag. This applies to both FP8 reference and all test models.

### Sehyo/Qwen3.5-397B-A17B-NVFP4 produces NaN on SGLang

Sehyo's checkpoint uses `compressed-tensors` quantization format. SGLang's `compressed-tensors` weight loader does not support `linear_attn` layers used by Qwen3.5-397B's mixed attention architecture (3 linear attention layers + 1 full attention, repeating). All `linear_attn` weights fail to load, leaving 45 out of 60 attention layers uninitialized, producing 100% NaN logits.

**Workaround:** None on current SGLang. vLLM may have better `compressed-tensors` support for this architecture.

### TP padding in logits

With tensor parallelism, SGLang pads the vocabulary dimension to a multiple of TP size. The patch trims these padding columns via `SGLANG_KLD_VOCAB_SIZE` before computing log-softmax. Without trimming, the padding columns (containing garbage values) corrupt the probability distribution.

### Chunked logits processing

The patch only hooks the non-chunked logits path. Set `SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=0` to ensure this path is used. With 2048-token windows this is fine -- chunking is only needed for very large prefills.

### Speculative decoding (MTP)

MTP speculative decoding (`--speculative-algo NEXTN`) is now safe to use during KLD evaluation. The patch automatically detects and skips MTP head forward passes by inspecting the call stack for MTP model files (`*mtp*.py`, `*nextn*.py`) and checking for `DRAFT_EXTEND` forward mode.

**Previous bug (fixed 2026-03-29):** Before this fix, MTP models saved 2 files per window (one from the main head, one from the MTP speculative head). The MTP head has higher entropy and a different distribution, which inflated mean KLD by ~18%. If you have old logit captures with 200 files for 100 windows, only the even-numbered files (0, 2, 4, ...) contain main-head logits.

### NaN logits in quantized models

Some quantized checkpoints produce NaN logits at certain positions (observed with MiniMax-M2.5-NVFP4: 8% of positions had NaN). The compute script automatically detects and excludes NaN positions, reporting the count in the results.
