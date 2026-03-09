# FlashInfer on RTX 6000 Pro Blackwell (SM120)

## Table of Contents

- [What is FlashInfer](#what-is-flashinfer)
- [SM120 Backend Landscape](#sm120-backend-landscape)
- [CUTLASS Backend and SM120 Support](#cutlass-backend-and-sm120-support)
- [SM120f Family Conditional Instructions](#sm120f-family-conditional-instructions)
- [flashinfer_cudnn Workaround](#flashinfer_cudnn-workaround)
- [CUTLASS Race Condition Bug](#cutlass-race-condition-bug)
- [FA2 vs CUTLASS Performance Comparison](#fa2-vs-cutlass-performance-comparison)
- [JIT Cache Management](#jit-cache-management)
- [MLA Kernels for SM120](#mla-kernels-for-sm120)
- [Relevant PRs and Issues](#relevant-prs-and-issues)

---

## What is FlashInfer

FlashInfer is a library of GPU kernels for LLM inference, providing attention backends, MoE GEMM runners, and allreduce fusion primitives. On RTX 6000 Pro Blackwell (SM120), FlashInfer is the primary kernel library used by SGLang and partially by vLLM.

FlashInfer provides:
- FlashAttention 2 (FA2) kernels for SM120 (SM89 path)
- CUTLASS-based FP4 GEMM kernels
- cuDNN-based FP4 GEMM kernels
- MLA (Multi-head Latent Attention) kernels for DeepSeek-style models
- XQA FP8 MLA kernels
- Allreduce fusion for PCIe-connected GPUs

---

## SM120 Backend Landscape

SM120 (Blackwell workstation/consumer) lacks several hardware features present in SM90 (Hopper) and SM100 (Blackwell datacenter):

| Feature | SM90 | SM100 | SM120 |
|---------|:----:|:-----:|:-----:|
| TMEM (Tensor Memory) | Yes | Yes | **No** |
| TCGEN05 instructions | No | Yes | **No** |
| WGMMA instructions | Yes | No | **No** |
| FlashAttention 3+ | Yes | Yes | **No** |
| DeepGemm | Yes | Yes | **No** |
| FlashMLA Sparse native | Yes | Yes | **No** |

SM120 is limited to FlashAttention 2 via SM89 kernels. All FA3, DeepGemm, and FlashMLA-based backends must fall back to FlashInfer FA2 or CUTLASS alternatives.

---

## CUTLASS Backend and SM120 Support

On SM120, CUTLASS is the only compatible MoE backend when compiled for SM120. Performance comparison:

| Backend | Speed (Qwen3.5 NVFP4, single batch) |
|---------|--------------------------------------|
| vLLM internal CUTLASS GEMMs (compiled sm120f) | 67 tok/s |
| FlashInfer with CUDA 13.1 (120f JIT) | 65 tok/s |

vLLM's internal CUTLASS GEMMs are slightly faster than FlashInfer's JIT-compiled CUTLASS for SM120.

### MOE Runner Backend on SM120

In SGLang, the `--moe-runner-backend` flag controls which kernel runs MoE expert routing:

| Backend | Actual behavior on SM120 |
|---------|--------------------------|
| `cutlass` | Uses CUTLASS directly. **Fastest for MTP.** |
| `flashinfer_cutlass` | FlashInfer wrapper around CUTLASS. Default. Slightly slower. |
| `deep_gemm` | Falls back to CUTLASS on SM120 (DeepGemm unsupported). Same speed as `cutlass`. |
| `triton` | Uses Triton kernels. Viable alternative. |

---

## SM120f Family Conditional Instructions

NVFP4 on SM120 was initially much slower than INT4 AWQ. The root cause was the absence of the `cvt.rn.satfinite.e2m1x2.f32` PTX instruction for FP32-to-FP4 conversion. This instruction is available on the SM120 family but **only when compiled for `sm120f`** (family conditional target).

FlashInfer initially did not compile for sm120f. The `gen_fp4_quantization_sm120f_module` was added to enable this:

- PR [#2650](https://github.com/flashinfer-ai/flashinfer/pull/2650): sm120f compilation support
- PR [#2460](https://github.com/flashinfer-ai/flashinfer/pull/2460): Doubled throughput for NVFP4 on some tiles

---

## flashinfer_cudnn Workaround

The default FP4 GEMM backend (`flashinfer_cutlass`) has a race condition bug that silently corrupts memory. The recommended workaround is to use cuDNN:

```bash
# Upgrade nvidia-cudnn:
pip install nvidia-cudnn-cu13==9.19.1.2

# Use cuDNN backend in SGLang:
--fp4-gemm-backend flashinfer_cudnn
```

The cuDNN backend is not only safer (no race condition) but may also be slightly faster than the buggy CUTLASS backend.

---

## CUTLASS Race Condition Bug

### The Bug

FlashInfer's CUTLASS FP4 GEMM kernel has a race condition that silently corrupts memory, leading to:

1. NaN values in model outputs
2. `probability tensor contains either inf, nan or element < 0` assertion failures
3. Crash cascades where NaN propagates through speculative decoding verification
4. Silent token quality degradation at high concurrency

### Bug Report

- Issue: https://github.com/flashinfer-ai/flashinfer/issues/2708
- Fix PR: https://github.com/flashinfer-ai/flashinfer/pull/2716

### Workarounds (in order of preference)

1. **Use `flashinfer_cudnn` backend**: `--fp4-gemm-backend flashinfer_cudnn`
2. **Upgrade to CUTLASS 4.4.1** and rebuild FlashInfer JIT cache (`rm -rf /cache/jit/*`)
3. **Use `--enable-nan-detection`** in SGLang (prevents crashes but may produce garbage tokens)
4. **Apply luke's sampler patch** that validates/fixes probabilities before multinomial sampling

### Important

When upgrading Docker images that include the CUTLASS fix, you **must** wipe the JIT kernel cache for the fix to take effect:

```bash
rm -rf /cache/jit/*
```

Old cached kernels contain the buggy code and will continue to be used unless deleted.

---

## FA2 vs CUTLASS Performance Comparison

Tested on Kimi K2.5 with 8x RTX 6000 Pro (vLLM):

| Backend | KV Cache | DCP | Decode tok/s | KV Cache Capacity |
|---------|----------|:---:|:------------:|:-----------------:|
| FlashInfer FA2 | BF16 | 1 | **90** | 190K tokens |
| Triton MLA | BF16 | 1 | 78 | 190K tokens |
| Triton MLA | FP8 | 1 | 79 | 449K tokens |
| FlashInfer FA2 | BF16 | 8 | 72 | 1.5M tokens |
| Triton MLA | FP8 | 8 | 68 | 3.6M tokens |
| Triton MLA | BF16 | 8 | 67 | 1.5M tokens |

**Conclusions**:
- FA2 is faster (90 vs 78 tok/s in BF16 single batch) but **does not support FP8 KV cache**
- For maximum throughput at short context: FA2 + BF16 KV
- For maximum context capacity: Triton MLA + FP8 KV + DCP
- XQA kernel is a dead end currently (no DCP, no MTP, no LSE return needed for DCP integration)

---

## JIT Cache Management

FlashInfer uses Just-In-Time (JIT) compilation for SM120 kernels. The compiled kernels are cached on disk.

### Cache Location

The default location inside Docker is `/cache/jit/`. Mount a Docker volume to persist across container restarts:

```bash
-v vllm-nightly-jit:/cache/jit
```

### When to Wipe the Cache

Wipe the JIT cache when:
- Upgrading FlashInfer version
- Upgrading CUTLASS version
- Applying bug fixes (especially the CUTLASS race condition fix)
- Changing CUDA toolkit version

```bash
rm -rf /cache/jit/*
```

### First-Run Compilation Time

On first launch with a new cache, JIT compilation adds several minutes to startup. Subsequent launches are fast as they reuse cached kernels.

---

## MLA Kernels for SM120

FlashInfer provides two MLA kernel variants for SM120:

1. **FlashInfer FA-based BF16 MLA kernel** (SM120 specific)
   - Used for Kimi K2.5, GLM-5 in SGLang
   - Supports BF16 KV cache only
   - Highest single-batch throughput (90 tok/s on Kimi K2.5)

2. **XQA FP8 MLA kernel** (SM120 specific)
   - Supports FP8 KV cache
   - Does not return LSE (Log-Sum-Exp), so incompatible with DCP
   - No MTP support
   - Currently a dead end for production use

Neither kernel is available in vLLM as of 2026-03-08. vLLM uses Triton MLA for SM120.

### Relevant FlashInfer PRs for SM120 MLA

| PR | Description |
|----|-------------|
| [#2689](https://github.com/flashinfer-ai/flashinfer/pull/2689) | Fix 10 bugs in BF16 XQA MLA kernel for SM120/SM121 |
| [#2675](https://github.com/flashinfer-ai/flashinfer/pull/2675) | Support BF16 MLA on SM120 with shared-mem fallback |
| [#1566](https://github.com/flashinfer-ai/flashinfer/pull/1566) | Add LSE return to XQA (non-sm120 path, needed for DCP+XQA) |

---

## Relevant PRs and Issues

| PR/Issue | Description |
|----------|-------------|
| [flashinfer#2460](https://github.com/flashinfer-ai/flashinfer/pull/2460) | Doubled throughput for NVFP4 on some tiles |
| [flashinfer#2650](https://github.com/flashinfer-ai/flashinfer/pull/2650) | sm120f compilation support |
| [flashinfer#2675](https://github.com/flashinfer-ai/flashinfer/pull/2675) | BF16 MLA on SM120 with shared-mem fallback |
| [flashinfer#2689](https://github.com/flashinfer-ai/flashinfer/pull/2689) | Fix 10 bugs in BF16 XQA MLA kernel for SM120 |
| [flashinfer#2708](https://github.com/flashinfer-ai/flashinfer/issues/2708) | CUTLASS FP4 GEMM race condition bug |
| [flashinfer#2716](https://github.com/flashinfer-ai/flashinfer/pull/2716) | Fix for CUTLASS race condition |
