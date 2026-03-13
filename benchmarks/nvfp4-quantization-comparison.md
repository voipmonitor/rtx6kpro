# Qwen3.5-397B-A17B Quantization Comparison: AWQ vs NVFP4

## Three-way benchmark: QuantTrio/AWQ vs lukealonso/NVFP4 vs nvidia/NVFP4

### Test Environment

- **GPU:** 8x NVIDIA RTX PRO 6000 Blackwell Server Edition (98GB each)
- **SGLang version:** 0.5.9
- **Container:** voipmonitor/llm-pytorch-blackwell:nightly-cuda132
- **Date:** 2026-03-11 to 2026-03-13

### Models Tested

| Model | Quantization | Format | Precision |
|-------|-------------|--------|-----------|
| QuantTrio/Qwen3.5-397B-A17B-AWQ | AWQ | INT4 | 16 levels, per-channel scaling, salient weight protection |
| lukealonso/Qwen3.5-397B-A17B-NVFP4 | ModelOpt FP4 | E2M1 | 8 unique values, Blackwell FP4 Tensor Core hardware |
| nvidia/Qwen3.5-397B-A17B-NVFP4 | ModelOpt FP4 | E2M1 | 8 unique values, Blackwell FP4 Tensor Core hardware |

### Common Server Config (NVFP4)
```
--tensor-parallel-size 8
--quantization modelopt_fp4
--kv-cache-dtype fp8_e4m3
--attention-backend triton
--moe-runner-backend flashinfer_cutlass
--fp4-gemm-backend flashinfer_cudnn
--cuda-graph-max-bs 128
--max-running-requests 128
--context-length 262144
--chunked-prefill-size 32768
--mem-fraction-static 0.80
--disable-custom-all-reduce
--disable-shared-experts-fusion
--schedule-conservativeness 0.1
```

### AWQ Server Config
Same as above but without `--quantization modelopt_fp4` and `--fp4-gemm-backend` (SGLang auto-detects AWQ from checkpoint config).

### MTP-specific flags (speculative decoding)
```
SGLANG_ENABLE_SPEC_V2=True
--speculative-algo NEXTN
--speculative-num-steps 5
--speculative-eagle-topk 1
--speculative-num-draft-tokens 6
```

### Eval Command
All GPQA evaluations were run using the same command:
```bash
python3 -u -m sglang.test.run_eval \
  --eval-name gpqa \
  --model Qwen3.5 \
  --base-url http://localhost:5000 \
  --num-examples 198 \
  --repeat 1 \
  --thinking-mode qwen3 \
  --max-tokens 64000
```
For 8-repeat tests, each repeat was run sequentially (not parallel) to avoid server overload. Results were collected from 8 independent runs.

---

## Full Comparison: All Three Quantizations

### Summary Table

| Benchmark | AWQ (QuantTrio) | lukealonso NVFP4 | nvidia NVFP4 | Notes |
|-----------|----------------|------------------|--------------|-------|
| **GPQA** (thinking, 8-repeat mean) | **88.40%** ±1.39 | 88.28% ±1.06 | 87.46% ±1.57 | 198 examples, 8 runs, MTP ON |
| **GSM8K** (thinking) | **99.0%** | **99.0%** | 97.5% | 200 examples, max-tokens 16000 |
| **Hard Math** (no thinking) | **89.5%** (17/19) | **89.5%** (17/19) | 84.2% (16/19) | 19 custom questions |
| **KL Divergence** (vs FP8) | **0.024** | 0.035 | 0.109 | 204,800 positions, WikiText-2 |

**On GPQA, all three models are statistically indistinguishable** (overlapping 95% CIs, Welch t-test p>0.05 for all pairs). AWQ and lukealonso tie on GSM8K and Hard Math. nvidia is consistently the weakest. AWQ's advantage is clearest in KLD (deterministic, 0.024 vs 0.035 vs 0.109) and throughput (15-38% faster than NVFP4).

---

## Part 1: KL Divergence (Quantization Quality)

KL divergence measures the exact difference in output probability distributions between a quantized model and the FP8 reference. Lower = better.

**Reference model:** Qwen/Qwen3.5-397B-A17B-FP8 (TP8)
**Dataset:** WikiText-2, 100 sliding windows (2048 tokens, stride 512), 204,800 total positions

```
KLD Evaluation Results (ref: Qwen3.5-397B-A17B-FP8)
============================================================================================

Model                                      Mean KLD   Median KLD    P95 KLD    P99 KLD    Max KLD
------------------------------------------------------------------------------------------------
QuantTrio/Qwen3.5-397B-A17B-AWQ (INT4)    0.024042     0.004788   0.097537   0.346614     3.7282
lukealonso/Qwen3.5-397B-A17B-NVFP4        0.035269     0.006830   0.146239   0.529040     4.5687
nvidia/Qwen3.5-397B-A17B-NVFP4            0.108526     0.027302   0.467703   1.411015    19.6018
```

### KLD Ranking

1. **QuantTrio/AWQ** — best quality. Mean KLD 0.024 (near-lossless).
2. **lukealonso/NVFP4** — 1.5x worse than AWQ but still good. Mean KLD 0.035.
3. **nvidia/NVFP4** — 4.5x worse than AWQ, 3x worse than lukealonso. Mean KLD 0.109, with heavy tail (Max KLD 19.6).

### KLD Interpretation Scale

| Mean KLD | Quantization quality |
|----------|---------------------|
| < 0.01 | Near-lossless |
| 0.01 - 0.05 | Good, minimal quality loss |
| 0.05 - 0.1 | Noticeable quality loss |
| > 0.1 | Significant quality loss |

nvidia NVFP4 falls into "significant quality loss" territory (0.109), while both AWQ and lukealonso are in the "good" range.

For full KLD methodology, reproduction steps, and automation script, see [kld-evaluation.md](kld-evaluation.md).

---

## Part 2: GPQA (Graduate-Level Google-Proof Q&A)

### GPQA 8-Repeat Detail (all three models, thinking mode, MTP ON)

| # | Model | GPQA Mean | Scores (8 repeats) | Std | Runtime per repeat |
|---|-------|-----------|---------------------|-----|--------------------|
| 1 | **AWQ (QuantTrio)** | **88.40%** | 87.9, 89.9, 88.9, 89.9, 87.4, 89.4, 85.9, 87.9 | 1.389 | ~20 min |
| 2 | lukealonso NVFP4 | 88.28% | 88.9, 87.9, 86.9, 88.4, 90.4, 88.4, 87.9, 87.4 | 1.06 | ~24 min |
| 3 | nvidia NVFP4 | 87.46% | 85.9, 90.4, 85.9, 88.4, 87.9, 85.9, 87.4, 87.9 | 1.57 | ~24 min |

**Statistical significance (Welch t-test, two-tailed):**
- AWQ vs lukealonso: Δ=+0.12%, t=0.20, p>0.05 — **not significant**
- AWQ vs nvidia: Δ=+0.94%, t=1.27, p>0.05 — **not significant**
- lukealonso vs nvidia: Δ=+0.81%, t=1.21, p>0.05 — **not significant**

95% confidence intervals overlap for all three models: AWQ [87.4–89.4], lukealonso [87.5–89.0], nvidia [86.4–88.5]. With 8 repeats and std ~1.0–1.6, the GPQA benchmark cannot distinguish these quantizations at the 95% confidence level.

### GPQA 8-Repeat Detail (lukealonso & nvidia, MTP OFF)

| # | Model | GPQA Mean | Scores (8 repeats) | Std | Runtime per repeat |
|---|-------|-----------|---------------------|-----|--------------------|
| 1 | lukealonso NVFP4 | 87.53% | 86.4, 87.4, 89.4, 86.9, 88.9, 86.4, 87.4, 87.4 | 1.09 | ~30 min |
| 2 | nvidia NVFP4 | 86.58% | 86.4, 85.9, 86.9, 86.4, 84.8, 86.4, 86.9, 88.9 | 1.15 | ~34 min |

### MTP Impact on GPQA (lukealonso & nvidia)

| Model | MTP ON | MTP OFF | Delta |
|-------|--------|---------|-------|
| lukealonso | 88.28% | 87.53% | **+0.75%** (within noise) |
| nvidia | 87.46% | 86.58% | **+0.88%** (within noise) |

MTP does NOT degrade accuracy — both models score marginally higher with MTP (within statistical noise). MTP provides 18-24% inference speedup.

---

## Part 3: GSM8K (Grade School Math 8K)

### GSM8K with thinking mode (200 examples, max-tokens 16000)

| Model | Score | Std |
|-------|-------|-----|
| **AWQ (QuantTrio)** | **99.0%** | — |
| lukealonso NVFP4 | **99.0%** | 0.099 |
| nvidia NVFP4 | 97.5% | 0.156 |

AWQ and lukealonso tie at 99.0%. nvidia lags at 97.5% with higher variance.

### GSM8K without thinking (5-shot, lukealonso vs nvidia only)

| Model | Score |
|-------|-------|
| lukealonso NVFP4 | 44% |
| nvidia NVFP4 | 39% |

Without thinking mode, the gap between NVFP4 quantizations widens to 5%, consistent with vLLM Issue #36094 reporting nvidia NVFP4 accuracy problems.

---

## Part 4: Hard Math Test (19 custom questions, no thinking mode)

| Model | Score | Correct |
|-------|-------|---------|
| **AWQ (QuantTrio)** | **89.5%** | 17/19 |
| lukealonso NVFP4 | **89.5%** | 17/19 |
| nvidia NVFP4 | 84.2% | 16/19 |

### Per-Question Detail

| Q# | Question | AWQ | lukealonso | nvidia |
|----|----------|-----|-----------|--------|
| 1 | (37*43)-(29*51)+17 | FAIL | FAIL (139) | FAIL (10) |
| 2 | 123^2 - 113^2 | OK | OK | OK |
| 3 | 2^31 mod 7 | FAIL | FAIL (4) | FAIL (1) |
| 4 | log_2(x)=5.5, x=? | OK | OK | OK |
| 5 | P(2 aces in row) | OK | OK | OK |
| 6 | LCM(12,18,20) | OK | OK | OK |
| 7 | Primes < 50 | OK | OK | OK |
| 8 | Sum primes < 30 | OK | OK | OK |
| 9 | (root1+1)(root2+1) for x^2-7x+12=0 | OK | **OK (20)** | **FAIL (30)** |
| 10 | 2^a*3^b=72, a+b=? | OK | OK | OK |
| 11 | Altitude to hypotenuse | OK | OK | OK |
| 12 | 10th Fibonacci | OK | OK | OK |
| 13 | Geometric sequence 8th term | OK | OK | OK |
| 14 | MISSISSIPPI arrangements | OK | OK | OK |
| 15 | C(8,3) | OK | OK | OK |
| 16 | Infinite geometric series sum | OK | OK | OK |
| 17 | 2x2 determinant | OK | OK | OK |
| 18 | Sum 1 to 100 | OK | OK | OK |
| 19 | 13^3 | OK | OK | OK |

Q1 and Q3 are failed by all models (multi-digit arithmetic without thinking). Q9 differentiates nvidia (FAIL) from AWQ and lukealonso (OK).

---

## Part 5: Throughput Benchmark

All models tested with MTP enabled (NEXTN, 5 steps, 6 draft tokens), 4x RTX PRO 6000 Blackwell (TP4). Server-side `sglang:gen_throughput` Prometheus metric.

```
Aggregate decode throughput (tok/s), context=0
=========================================================================

Model                                 C=1    C=8    C=16    C=32    C=64
------------------------------------------------------------------------
QuantTrio/Qwen3.5-397B-A17B-AWQ      152    665     976    1516    1662
lukealonso/Qwen3.5-397B-A17B-NVFP4   132    581     852    1191    1202
```

**AWQ is faster at every concurrency level:** 15% faster at C=1, growing to 38% at C=64 where AWQ still scales (1662 tok/s) while NVFP4 plateaus (1202 tok/s).

For full decode + prefill tables across context lengths, see [inference-throughput/](inference-throughput/).

---

## Overall Conclusions

### 1. AWQ (QuantTrio) is the best quantization for Qwen3.5-397B-A17B

| Metric | AWQ | lukealonso NVFP4 | nvidia NVFP4 |
|--------|-----|------------------|--------------|
| GPQA (8-repeat mean) | **88.40%** | 88.28% | 87.46% |
| GSM8K | **99.0%** | **99.0%** | 97.5% |
| Hard Math | **89.5%** | **89.5%** | 84.2% |
| KL Divergence | **0.024** | 0.035 | 0.109 |
| Throughput (C=64) | **1662 tok/s** | 1202 tok/s | — |

On task benchmarks (GPQA, GSM8K, Hard Math), AWQ and lukealonso NVFP4 are **statistically equivalent** — the 0.12% GPQA difference is well within noise (Welch t-test p>0.05). AWQ's clear advantages are in **KLD** (0.024 vs 0.035 — 32% closer to FP8 reference) and **throughput** (15-38% faster at all concurrency levels).

### 2. If NVFP4 is required, use lukealonso over nvidia
lukealonso NVFP4 trends higher than nvidia NVFP4 across all benchmarks (+0.8% on GPQA, though not statistically significant). The advantage is clear without thinking mode (GSM8K: +5%, Hard Math: +5.3%). nvidia NVFP4 has significant KLD (0.109, 3x worse than lukealonso), consistent with community reports (vLLM Issue #36094).

### 3. Enable MTP for production serving
MTP provides 18-24% inference speedup with no measurable accuracy degradation.

### 4. Recommended production config
```bash
# Model (best overall)
--model QuantTrio/Qwen3.5-397B-A17B-AWQ

# MTP (speculative decoding)
SGLANG_ENABLE_SPEC_V2=True
--speculative-algo NEXTN
--speculative-num-steps 5
--speculative-eagle-topk 1
--speculative-num-draft-tokens 6

# Required flags
--disable-shared-experts-fusion
--disable-custom-all-reduce
--attention-backend triton
--mamba-scheduler-strategy extra_buffer
```

### Warning (NVFP4 only)
```
DeepGemm is enabled but the scale_fmt of checkpoint is not ue8m0.
This might cause accuracy degradation on Blackwell.
```
This warning appears for all NVFP4 runs. AWQ does not trigger this warning.
