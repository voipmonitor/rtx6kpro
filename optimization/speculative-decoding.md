# Speculative Decoding & MTP

Speculative decoding is the single largest throughput optimization for single-user inference on RTX PRO 6000 Blackwell systems, providing 50-100% speedup at low concurrency.

## Table of Contents

- [Multi-Token Prediction (MTP) Explained](#multi-token-prediction-mtp-explained)
- [MTP Configuration](#mtp-configuration)
- [MTP=2 as the Sweet Spot](#mtp2-as-the-sweet-spot)
- [EAGLE Speculative Decoding](#eagle-speculative-decoding)
- [Acceptance Rate Data](#acceptance-rate-data)
- [Throughput Improvements](#throughput-improvements)
- [Known Issues and Limitations](#known-issues-and-limitations)
- [Model-Specific Notes](#model-specific-notes)

---

## Multi-Token Prediction (MTP) Explained

MTP is a speculative decoding technique where the model predicts multiple tokens ahead in parallel, then verifies them. Instead of generating one token per forward pass, the model drafts N candidate tokens and verifies them in a single pass, accepting those that match what the model would have generated autoregressively.

**How it works:**
1. The main model generates token T
2. MTP heads (lightweight layers trained alongside the model) draft tokens T+1, T+2, ..., T+N
3. The main model verifies all drafted tokens in a single forward pass
4. Accepted tokens are emitted; rejected tokens are discarded and generation continues from the last accepted token

**Key advantage:** MTP heads are part of the model itself (not a separate draft model), so there is no additional VRAM overhead for a draft model -- only the MTP layer weights (~19 GB for GLM-5's layer 78 in BF16).

**Models with native MTP support:**
- Qwen3.5 (all sizes)
- GLM-5
- GLM-4.7

**Models WITHOUT MTP:**
- Kimi K2.5 (uses EAGLE3 instead)
- MiniMax-M2.5

---

## MTP Configuration

### vLLM MTP Flags

```bash
# Standard MTP (recommended for most models):
--speculative-config '{"method":"mtp","num_speculative_tokens":2}'

# Qwen3.5-specific (newer vLLM versions):
--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'

# Higher token count (requires PR #35615 for >1, unstable at >3):
--speculative-config '{"method":"mtp","num_speculative_tokens":5}'
```

### SGLang MTP Flags

```bash
# Environment variable (MANDATORY):
export SGLANG_ENABLE_SPEC_V2=True

# Launch flags:
--speculative-algorithm NEXTN
--speculative-num-steps 3
--speculative-eagle-topk 1
--speculative-num-draft-tokens 4
```

**CRITICAL:** `SGLANG_ENABLE_SPEC_V2=True` is **mandatory**. Without it, SGLang silently converts NEXTN to EAGLE and tries to load the full model a second time as a draft model, causing instant OOM (e.g., 57 GB x 2 = 114 GB per GPU on a 96 GB card for GLM-5).

### SGLang EAGLE Flags (for models with EAGLE support)

```bash
export SGLANG_ENABLE_SPEC_V2=1

--speculative-algorithm EAGLE
--speculative-num-steps 3
--speculative-eagle-topk 1
--speculative-num-draft-tokens 4
```

---

## MTP=2 as the Sweet Spot

Community testing consistently shows **MTP=2 as the optimal setting** for the NVIDIA NVFP4 checkpoint on vLLM:

### Why MTP=2

| MTP Tokens | Stability | Performance | Recommendation |
|------------|-----------|-------------|----------------|
| 0 (disabled) | Stable | Baseline | Safe fallback |
| 1 | Stable | ~30% improvement | Conservative |
| **2** | **Stable** | **~50-55% improvement** | **Recommended** |
| 3 | Mostly stable | ~55-60% improvement | Risky |
| 5 | Unstable at long context | ~70%+ improvement | Not recommended for production |

### Per-Position Acceptance Rates (MTP=5)

| Position | Acceptance Rate |
|----------|----------------|
| T+1 | ~85% |
| T+2 | ~67% |
| T+3 | ~51% |
| T+4 | ~36% |
| T+5 | ~31% |

The diminishing returns past position 2-3, combined with increased instability, make MTP=2 the practical sweet spot.

---

## EAGLE Speculative Decoding

EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency) uses a separate lightweight draft model rather than built-in MTP heads.

### EAGLE3 for Kimi K2.5

Kimi K2.5 does not have native MTP, so EAGLE3 is the speculative decoding path:

- **Draft model:** `AQ-MedAI/Kimi-K2-Instruct-eagle3`
- **SGLang PR:** [sgl-project/sglang#19689](https://github.com/sgl-project/sglang/pull/19689)
- **vLLM PR:** [vllm-project/vllm#35966](https://github.com/vllm-project/vllm/pull/35966)
- **Current status:** Grimulkan reported speculative decoding was "strictly worse than no speculation" for Kimi K2.5 in all vLLM configurations tried. Potential for 30% improvement once EAGLE3 + FA2 is working.

### EAGLE in SGLang for Qwen3.5

boo's optimized EAGLE config for Qwen3.5-397B NVFP4:

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

Result: ~130 TPS with MTP=3 on Qwen3.5 NVFP4.

---

## Acceptance Rate Data

### Qwen3.5-397B NVFP4, MTP=2 (vLLM)

| Metric | Value |
|--------|-------|
| Draft acceptance rate | 89.2% |
| Tokens drafted | 165,550 |
| Tokens accepted | 147,689 |
| Mean acceptance length | 2.73-3.69 tokens |

### GLM-5 NVFP4 + MTP (SGLang)

| Metric | Value |
|--------|-------|
| Accept rate | 0.55-0.94 (varies by context) |
| Accept length | 2.19-2.80 tokens |
| Speed improvement | ~2x over non-MTP baseline |

### Qwen3.5 MTP=5 Per-Position Rates

| Position | Acceptance Rate |
|----------|----------------|
| 1 | ~0.85 |
| 2 | ~0.67 |
| 3 | ~0.51 |
| 4 | ~0.36 |
| 5 | ~0.31 |

---

## Throughput Improvements

### Qwen3.5-397B NVFP4, 4x RTX PRO 6000 (vLLM, MTP=2)

| Concurrent Users | No MTP (tok/s) | MTP=2 (tok/s) | Improvement |
|-----------------|----------------|---------------|-------------|
| 1 | 85.8 | 130.0 | **+51.5%** |
| 2 | 137.1 | 212.7 | **+55.1%** |
| 5 | 234.2 | 358.6 | **+53.1%** |
| 10 | 334.3 | 573.5 | **+71.6%** |
| 20 | 491.5 | 744.1 | **+51.4%** |
| 32 | 605.9 | 922.6 | **+52.3%** |

Peak throughput with MTP=2: **1,127 tok/s at 50 concurrent users** (1K context each).

### Qwen3.5-397B NVFP4, Single-Stream Comparison

| Engine | MTP Config | tok/s | Notes |
|--------|-----------|-------|-------|
| vLLM | None | 70-86 | Multiple users, stable |
| vLLM | MTP=2 | 130 | malaiwah |
| vLLM | MTP=5 | 150-250 | Unstable at long context |
| SGLang | None | 42-51 | kcramp, Ixtrix |
| SGLang | MTP=3 (EAGLE) | 85 | Unstable |
| SGLang | MTP=3 (NEXTN) | ~130 | boo's config |

### GLM-5 NVFP4, 8x RTX PRO 6000 (SGLang)

| Configuration | 0 Context | 100K Context | 200K Context |
|---------------|-----------|--------------|--------------|
| NVFP4, no MTP | 35-44 tok/s | -- | -- |
| NVFP4 + MTP (NEXTN) | 97-105 tok/s | 60-80 tok/s | ~50 tok/s |

### GLM-5 Concurrent Throughput

- 3 running requests with MTP: **133-135 tok/s** total generation throughput

---

## Known Issues and Limitations

### 1. MTP >3 Crashes (vLLM)

MTP with `num_speculative_tokens` > 3 causes illegal memory access errors, especially at long context.

**Error:**
```
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

**Workaround:** Keep `num_speculative_tokens` at 2 (stable) or 3 (mostly stable). PR #35615 partially fixes this.

### 2. Tool Call Failures with MTP Enabled

When MTP is enabled with `tool_choice='required'`, the model often outputs XML tool calls instead of JSON:

```xml
<tool_call>
  <function>ask_wiki</function>
  <parameters>{"question": "..."}</parameters>
</tool_call>
```

**Impact:** 50-70% of tool calls fail.

**Fix:** PR #35936. Using `tool_choice='auto'` can handle both XML and JSON formats.

**Key insight:** "if thinking is false, even with mtp there is no problem" -- the combination of thinking mode + MTP causes the most tool call issues.

### 3. SGLANG_ENABLE_SPEC_V2 is Mandatory

Without `SGLANG_ENABLE_SPEC_V2=True`, SGLang converts NEXTN to EAGLE and attempts to load the model twice, causing OOM.

### 4. CARVE Models Should Not Use MTP

MTP heads were trained on censored content. Using MTP with abliterated (CARVE) models produces inconsistent behavior.

### 5. Radix Cache + Speculative Decoding Crash (SGLang)

EAGLE V2 speculative decoding crashes when a request hits the radix cache prefix. NaN in logits from flashinfer CUTLASS race condition propagates to the verify step.

**Error:**
```
eagle_worker_v2.py:510 _zero_fill_draft_kv_for_cached_prefix
torch.AcceleratorError: CUDA error: device-side assert triggered
```

**Fix:** SGLang PR [#19897](https://github.com/sgl-project/sglang/pull/19897). Root cause is the flashinfer CUTLASS race condition ([flashinfer#2708](https://github.com/flashinfer-ai/flashinfer/issues/2708)).

### 6. MTP Fused Kernel Performance

vLLM PR #35581 fixes the MTP fused kernel, providing ~6% throughput boost. This is a sed one-liner fix.

---

## Model-Specific Notes

### Qwen3.5

- MTP=2 is the production-safe setting on vLLM
- MTP=5 works with PR #35615 but crashes at long context
- SGLang uses NEXTN algorithm with `speculative-num-steps 3`
- `--speculative-draft-model-quantization unquant` may be needed for some checkpoints

### GLM-5

- MTP layer is layer 78, kept in BF16 precision (~19 GB)
- Use `festr2/GLM-5-NVFP4-MTP` (includes MTP weights)
- `lukealonso/GLM-5-NVFP4` does NOT include MTP weights
- `--moe-runner-backend cutlass` is fastest for MTP on SM120
- FP8 MTP is possible but decreases accept rate; only useful for high-throughput batch scenarios

### Kimi K2.5

- No native MTP support
- EAGLE3 is the speculative decoding path (in development)
- Draft model: `AQ-MedAI/Kimi-K2-Instruct-eagle3`
- Currently, speculative decoding is "strictly worse" in vLLM for this model

### MiniMax-M2.5

- No MTP or speculative decoding support
- Single-stream speed is 74-100 tok/s without speculation

### Configuration Summary Table

| Model | Engine | Spec Method | Flags | Expected Speedup |
|-------|--------|-------------|-------|-----------------|
| Qwen3.5 | vLLM | MTP | `'{"method":"mtp","num_speculative_tokens":2}'` | +50-55% |
| Qwen3.5 | SGLang | NEXTN | `--speculative-algorithm NEXTN --speculative-num-steps 3` | +50-100% |
| GLM-5 | SGLang | NEXTN | `--speculative-algorithm NEXTN --speculative-num-steps 3` | ~2x |
| Kimi K2.5 | SGLang | EAGLE3 | `--speculative-algorithm EAGLE` (+ draft model) | WIP |
| Kimi K2.5 | vLLM | EAGLE3 | PR #35966 | WIP |
