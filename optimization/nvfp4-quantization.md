# NVFP4 Quantization

NVFP4 is NVIDIA's native 4-bit floating-point format for Blackwell GPUs (SM120). It enables running 400B+ parameter models on just 4x RTX PRO 6000 Blackwell cards with minimal quality loss.

## Table of Contents

- [What is NVFP4](#what-is-nvfp4)
- [Available NVFP4 Models](#available-nvfp4-models)
- [Calibration and Quantization Process](#calibration-and-quantization-process)
- [Performance: NVFP4 vs FP8](#performance-nvfp4-vs-fp8)
- [KV Cache Considerations](#kv-cache-considerations)
- [CARVE: Unlocked/Abliterated Models](#carve-unlocked-abliterated-models)
- [SM120f Compilation and GEMM Backends](#sm120f-compilation-and-gemm-backends)
- [Known Issues](#known-issues)

---

## What is NVFP4

NVFP4 (NVIDIA FP4) is a 4-bit floating-point format native to Blackwell architecture (SM120). It uses the E2M1 (2-bit exponent, 1-bit mantissa) format with blockwise quantization and FP8 scaling factors.

**Key properties:**
- 4 bits per weight element (vs 8 bits for FP8, 16 bits for BF16)
- Blockwise quantization with calibrated FP8 scales
- Native hardware support on SM120 via `cvt.rn.satfinite.e2m1x2.f32` PTX instruction
- Requires SM120f family-conditional compilation for optimal performance
- Typically quantized using NVIDIA ModelOpt toolkit

**Why it matters:** A 397B-parameter MoE model that requires 8 GPUs at FP8 fits on just 4 GPUs at NVFP4, halving hardware cost while maintaining ~99% of FP8 quality.

---

## Available NVFP4 Models

### Qwen3.5-397B-A17B

| Checkpoint | Source | KV Cache Scales | Notes |
|------------|--------|----------------|-------|
| `nvidia/Qwen3.5-397B-A17B-NVFP4` | NVIDIA ModelOpt | Yes (FP8 calibrated) | Official, best quality |
| `vincentzed-hf/Qwen3.5-397B-A17B-NVFP4` | Community | No | Early quant |
| `Sehyo/Qwen3.5-397B-A17B-NVFP4` | llm-compressor | No (defaults to bf16 KV) | 2x KV cache memory |
| `vpyn/Qwen3.5-397B-A17B-CARVE-v1-NVFP4` | Abliterated | Yes | Uncensored, better >300K context |

### Qwen3.5 Other Sizes (Sehyo Series)

| Checkpoint | Notes |
|------------|-------|
| `Sehyo/Qwen3.5-27B-NVFP4` | Multimodal + MTP support |
| `Sehyo/Qwen3.5-35B-A3B-NVFP4` | Multimodal + MTP support |
| `Sehyo/Qwen3.5-122B-A10B-NVFP4` | Multimodal + MTP support |

### Other Models

| Checkpoint | Notes |
|------------|-------|
| `nvidia/Kimi-K2.5-NVFP4` | Slower than native INT4 Marlin on this model |
| `lukealonso/GLM-5-NVFP4` | No MTP weights |
| `festr2/GLM-5-NVFP4-MTP` | With MTP layer 78 in BF16 (~19 GB) |

### NVIDIA vs Sehyo/llm-compressor NVFP4

The key difference is KV cache calibration:

| Property | NVIDIA (ModelOpt) | Sehyo (llm-compressor) |
|----------|------------------|----------------------|
| KV cache scheme | `{num_bits: 8, type: float, dynamic: false}` | `null` |
| KV cache scales | Calibrated k_scale/v_scale tensors | None (defaults to scale=1.0) |
| Runtime KV dtype | FP8 with proper calibration | BF16 (2x memory) or uncalibrated FP8 |
| VRAM for KV cache | 1x | 2x (if bf16) |

**Recommendation:** Use NVIDIA ModelOpt checkpoints when available for the best VRAM efficiency and quality.

---

## Calibration and Quantization Process

### Using NVIDIA ModelOpt

NVIDIA's official NVFP4 checkpoints are produced using the ModelOpt toolkit, which performs:

1. **Weight quantization:** BF16 weights -> NVFP4 (E2M1) with blockwise FP8 scales
2. **KV cache calibration:** Runs calibration data through the model to compute per-layer k_scale and v_scale tensors for FP8 KV cache
3. **Activation calibration:** Computes scaling factors for activations

### CARVE Quantization Recipe

The CARVE (abliterated) model was created by:

1. Starting from BF16 weights
2. Applying abliteration (removing refusal behavior)
3. Quantizing back to NVFP4 using ModelOpt

This preserves KV cache calibration quality while removing censorship.

### Inference Engine Flags

**SGLang:**
```bash
--quantization modelopt_fp4
--kv-cache-dtype fp8_e4m3    # Only with NVIDIA checkpoint
```

**vLLM:**
```bash
--quantization modelopt       # or leave unset for auto-detection
--kv-cache-dtype fp8          # Only with NVIDIA checkpoint
```

---

## Performance: NVFP4 vs FP8

### Speed Comparison

| Model | Quant | GPUs | Decode tok/s | Notes |
|-------|-------|------|-------------|-------|
| Qwen3.5-397B | NVFP4 | 4x | 70-86 | vLLM, no MTP |
| Qwen3.5-397B | NVFP4 + MTP=2 | 4x | 130 | vLLM |
| Qwen3.5-397B | FP8 | 8x | 75-125 | SGLang |
| GLM-4.7 | FP8 | 4x | 90-120 | Fastest |
| GLM-4.7 | NVFP4 | 4x | 60-90 | 20-30 tok/s slower than FP8 |
| Kimi K2.5 | INT4 (native) | 8x | 90 | Faster than NVFP4 variant |
| Kimi K2.5 | NVFP4 | 8x | 53-55 | Slower due to PTQ overhead |

**General rule:** NVFP4 is consistently 20-30 tok/s slower than FP8 for the same model, due to slower fused MoE kernels and GEMM operations. The trade-off is halved GPU count.

### Quality Comparison

| Model | NVFP4 | FP8 | Delta |
|-------|-------|-----|-------|
| Qwen3.5-397B (MMLU-Pro) | 90.0% | ~90% | Within noise |
| GLM-5 (MMLU) | 0.873 | 0.877 (official BF16) | -0.004 |
| MiniMax-M2.5 (MMLU-Pro) | Higher than FP8 | Baseline | NVFP4 +0.4% |

> "nvfp4 has 1% degradation and sometimes its even better than fp8 so the differences are really within noise probability" -- Festr

### When NVFP4 Beats FP8

- MiniMax-M2.5: NVFP4 outperformed official FP8 by 0.4% on MMLU-Pro
- NVFP4 on 2 GPUs competes well with FP8 on 4 GPUs for throughput
- Half the hardware cost for similar quality

### When FP8 is Better

- Higher decode throughput when you have enough GPUs
- Kimi K2.5: native INT4 Marlin is faster than NVFP4 PTQ
- GLM-4.7: FP8 is noticeably faster in decode

---

## KV Cache Considerations

### FP8 KV Cache (NVIDIA Checkpoints)

NVIDIA ModelOpt checkpoints include calibrated k_scale and v_scale tensors, enabling FP8 KV cache with proper quality:

```bash
# SGLang
--kv-cache-dtype fp8_e4m3

# vLLM
--kv-cache-dtype fp8
```

This uses half the memory of BF16 KV cache, allowing roughly 2x the context length.

### BF16 KV Cache (llm-compressor Checkpoints)

Checkpoints without calibrated KV scales default to BF16 KV cache:
- 2x memory usage for KV cache
- Shorter maximum context length
- No quality risk from uncalibrated scales

### FP8 KV Cache Limitations on SM120

**GLM-5:** FP8 KV cache is **broken** on SM120 -- produces garbled output or emits 1 token and stops. Only BF16 KV cache works.

**Kimi K2.5 (SGLang):** FP8 KV on the original INT4 checkpoint drops to 16 tok/s. The NVFP4 checkpoint supports FP8 KV at 55 tok/s.

---

## CARVE: Unlocked/Abliterated Models

### What is CARVE

CARVE models have been "abliterated" -- their refusal behavior has been surgically removed without retraining. The model is first converted to BF16, abliteration is applied, then it is quantized back to NVFP4.

### Available CARVE Model

`vpyn/Qwen3.5-397B-A17B-CARVE-v1-NVFP4`

### CARVE vs NVIDIA Reference at Long Context

| Context Length | CARVE tok/s | REF tok/s | Winner |
|---------------|------------|-----------|--------|
| 10K | 76.9 | 92.3 | REF +20% |
| 50K | 75.5 | 91.3 | REF +21% |
| 100K | 74.8 | 73.6 | ~tied |
| 200K | 74.3 | 95.5 | REF +29% |
| 300K | 73.3 | 43.8 | **CARVE +67%** |
| 400K | 67.9 | 42.3 | **CARVE +61%** |
| 500K | 67.0 | 42.2 | **CARVE +59%** |

**Key finding:** CARVE maintains much better performance at >300K context than the NVIDIA reference NVFP4.

### CARVE with YaRN Extended Context

CARVE supports YaRN rope scaling up to ~900K context:

```bash
--hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}'
--max-model-len 921600
```

Requires `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`.

### CARVE Limitations

- Do NOT use MTP with CARVE -- "MTP was trained on censored content" and the model is abliterated
- Slightly slower than reference at short context (<200K)

---

## SM120f Compilation and GEMM Backends

### The SM120f Issue

NVFP4 on vLLM was consistently slower than INT4 AWQ due to the absence of the `cvt.rn.satfinite.e2m1x2.f32` PTX instruction for FP32->FP4 conversion. This instruction is available on SM120 family but only when compiled for `sm120f` (family-conditional instructions).

FlashInfer initially did not compile for sm120f, resulting in suboptimal NVFP4 performance. This was fixed in FlashInfer PR #2650 and #2716.

### GEMM Backend Options

**SGLang FP4 GEMM backends:**

| Backend | Notes |
|---------|-------|
| `flashinfer_cutlass` | Default. Has a race condition bug causing silent memory corruption at high concurrency. |
| `flashinfer_cudnn` | Faster and more stable. Recommended. |

```bash
# Recommended:
--fp4-gemm-backend flashinfer_cudnn

# Requires:
pip install nvidia-cudnn-cu13==9.19.1.2
```

**vLLM FP4 GEMM backends:**

```bash
VLLM_NVFP4_GEMM_BACKEND=cutlass
```

vLLM internal CUTLASS GEMMs compiled for sm120f achieve ~67 tok/s vs FlashInfer with CUDA 13.1 at ~65 tok/s.

### MoE Runner Backend for SM120

| Backend | Speed | Notes |
|---------|-------|-------|
| `cutlass` | Fastest for MTP | Only compatible SM120 MoE backend |
| `flashinfer_cutlass` | Default, slightly slower | |
| `deep_gemm` | Falls back to cutlass on SM120 | DeepGemm requires WGMMA/TCGEN05 |

---

## Known Issues

### 1. flashinfer_cutlass Race Condition

The default `flashinfer_cutlass` FP4 GEMM backend has a race condition that silently corrupts memory, leading to crashes or token degradation under high concurrency.

**Fix:** Use `--fp4-gemm-backend flashinfer_cudnn` instead.

Reference: [flashinfer#2708](https://github.com/flashinfer-ai/flashinfer/issues/2708)

### 2. NVFP4 Cache Corruption on Nightly Builds

Some vLLM nightly builds exhibit NVFP4 cache corruption.

**Fix:** Pin to known-good builds or use the `orthozany/vllm-qwen35-mtp` Docker image.

### 3. NVFP4 Quality Concerns in Coding Tasks

Subjective reports of NVFP4 producing lower-quality code compared to FP8, though this has not been formally benchmarked.

### 4. DeepGemm Scale Format Detection

On SM120, the DeepGemm scale format detection incorrectly assumes `ue8m0` scales. NVFP4 uses `float8_e4m3fn` scales, causing NaN output.

**Fix:**
```bash
sed -i "s/DEEPGEMM_SCALE_UE8M0 = DEEPGEMM_BLACKWELL/DEEPGEMM_SCALE_UE8M0 = False/" \
    /sgl-workspace/sglang/python/sglang/srt/layers/deep_gemm_wrapper/configurer.py
```

### 5. Kimi K2.5 NVFP4 vs Native INT4

For Kimi K2.5, the NVFP4 variant (PTQ from BF16) is slower than the native INT4 with Marlin kernels.

> "There's no point in doing nvfp4 kimi imo, the source weights were int4." -- luke

### 6. MTP FC Layer Shape Mismatch (vLLM)

NVIDIA's NVFP4 Qwen3.5 checkpoint requires adding `"mtp.fc"` to the `quantization_config.ignore` list in `config.json`:

```json
"ignore": [
    "...existing entries...",
    "mtp.fc"
]
```

Also add `"model.language_model.layers..mlp.gate"` to both `config.json` and `hf_quant_config.json`.

Related PRs: vLLM #35156, #35675.
