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

| Benchmark | AWQ (QuantTrio) | lukealonso NVFP4 | nvidia NVFP4 | nvidia NVFP4 (vLLM) | Notes |
|-----------|----------------|------------------|--------------|---------------------|-------|
| **GPQA** (thinking, 8-repeat mean) | **88.40%** ±1.39 | 88.28% ±1.06 | 87.46% ±1.57 | **88.53%** ±1.92 | 198 examples, 8 runs, MTP ON |
| **GSM8K** (thinking) | **99.0%** | **99.0%** | 97.5% | 98.5% | 200 examples, max-tokens 16000 |
| **Hard Math** (no thinking) | **89.5%** (17/19) | **89.5%** (17/19) | 84.2% (16/19) | 84.2% (16/19) | 19 custom questions |
| **KL Divergence** (vs FP8) | **0.024** | 0.035 | 0.109 | — | 204,800 positions, WikiText-2 |

**On GPQA, all four configurations are statistically indistinguishable** (overlapping 95% CIs, Welch t-test p>0.05 for all pairs). lukealonso has the best run-to-run consistency (std 1.06), while nvidia/vLLM scores highest but with the worst consistency (std 1.92). AWQ and lukealonso tie on GSM8K and Hard Math. AWQ's advantages are in KLD (0.024 vs 0.035 vs 0.109) and throughput (15-38% faster than NVFP4 on SGLang). lukealonso's advantage is lower variance across repeated evaluations.

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

### GPQA 8-Repeat Detail (nvidia NVFP4 on vLLM, thinking mode, MTP ON vs OFF)

| # | Model | Engine | MTP | GPQA Mean | Scores (8 repeats) | Std |
|---|-------|--------|-----|-----------|---------------------|-----|
| 1 | nvidia NVFP4 | vLLM | ON | **88.53%** | 91.9, 87.9, 89.4, 85.9, 86.4, 88.4, 89.9, 88.4 | 1.92 |
| 2 | nvidia NVFP4 | vLLM | OFF | 86.90% | 87.4, 86.4, 86.9, 86.9, 85.9, 85.9, 89.4, 86.4 | 1.13 |
| 3 | nvidia NVFP4 | SGLang | ON | 87.46% | 85.9, 90.4, 85.9, 88.4, 87.9, 85.9, 87.4, 87.9 | 1.57 |
| 4 | nvidia NVFP4 | SGLang | OFF | 86.58% | 86.4, 85.9, 86.9, 86.4, 84.8, 86.4, 86.9, 88.9 | 1.15 |

vLLM server config:
```bash
VLLM_LOG_STATS_INTERVAL=1 NCCL_P2P_LEVEL=SYS SAFETENSORS_FAST_GPU=1 \
python3 -m vllm.entrypoints.openai.api_server \
  --model nvidia/Qwen3.5-397B-A17B-NVFP4 \
  --host 0.0.0.0 --port 5199 \
  --served-model-name Qwen3_5-397B-A17B-NVFP4 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 128 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-type shm \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}' \
  --enable-prefix-caching --enable-chunked-prefill
```

MTP ON vs OFF on vLLM: Δ=+1.62%, t=2.06, p>0.05 (borderline, not significant).
MTP ON vs OFF on SGLang: Δ=+0.88%, t=1.28, p>0.05 (not significant).
Without MTP, both engines converge: vLLM 86.90% vs SGLang 86.58% (Δ=+0.31%, t=0.61, ns).

**Statistical significance (Welch t-test, two-tailed, all MTP-ON pairs):**
- AWQ (SGLang) vs lukealonso (SGLang): Δ=+0.12%, t=0.20, p>0.05 — **not significant**
- AWQ (SGLang) vs nvidia (SGLang): Δ=+0.94%, t=1.27, p>0.05 — **not significant**
- AWQ (SGLang) vs nvidia (vLLM): Δ=−0.13%, t=0.15, p>0.05 — **not significant**
- lukealonso (SGLang) vs nvidia (SGLang): Δ=+0.81%, t=1.21, p>0.05 — **not significant**
- nvidia (vLLM) vs nvidia (SGLang): Δ=+1.07%, t=1.22, p>0.05 — **not significant**

95% confidence intervals overlap for all configurations. With 8 repeats and std ~1.0–1.9, the GPQA benchmark cannot distinguish these quantizations or inference engines at the 95% confidence level.

For full MTP quality analysis including per-run data, see [mtp-quality-evaluation.md](mtp-quality-evaluation.md).

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

| Model | Engine | Score | Std |
|-------|--------|-------|-----|
| **AWQ (QuantTrio)** | SGLang | **99.0%** | — |
| lukealonso NVFP4 | SGLang | **99.0%** | 0.099 |
| nvidia NVFP4 | vLLM | 98.5% | — |
| nvidia NVFP4 | SGLang | 97.5% | 0.156 |

AWQ and lukealonso tie at 99.0%. nvidia on vLLM (98.5%) outperforms nvidia on SGLang (97.5%).

### GSM8K without thinking (5-shot, lukealonso vs nvidia only)

| Model | Score |
|-------|-------|
| lukealonso NVFP4 | 44% |
| nvidia NVFP4 | 39% |

Without thinking mode, the gap between NVFP4 quantizations widens to 5%, consistent with vLLM Issue #36094 reporting nvidia NVFP4 accuracy problems.

---

## Part 4: Hard Math Test (19 custom questions, no thinking mode)

| Model | Engine | Score | Correct |
|-------|--------|-------|---------|
| **AWQ (QuantTrio)** | SGLang | **89.5%** | 17/19 |
| lukealonso NVFP4 | SGLang | **89.5%** | 17/19 |
| nvidia NVFP4 | SGLang | 84.2% | 16/19 |
| nvidia NVFP4 | vLLM | 84.2% | 16/19 |

nvidia scores identically on both engines (16/19), failing the same three questions (Q1, Q3, Q9). This confirms the errors are from quantization, not the inference engine.

### Per-Question Detail

| Q# | Question | AWQ | lukealonso | nvidia (SGLang) | nvidia (vLLM) |
|----|----------|-----|-----------|--------|
| 1 | (37*43)-(29*51)+17 | FAIL | FAIL (139) | FAIL (10) | FAIL (17) |
| 2 | 123^2 - 113^2 | OK | OK | OK | OK |
| 3 | 2^31 mod 7 | FAIL | FAIL (4) | FAIL (1) | FAIL (1) |
| 4 | log_2(x)=5.5, x=? | OK | OK | OK | OK |
| 5 | P(2 aces in row) | OK | OK | OK | OK |
| 6 | LCM(12,18,20) | OK | OK | OK | OK |
| 7 | Primes < 50 | OK | OK | OK | OK |
| 8 | Sum primes < 30 | OK | OK | OK | OK |
| 9 | (root1+1)(root2+1) for x^2-7x+12=0 | OK | **OK (20)** | **FAIL (30)** | **FAIL (30)** |
| 10 | 2^a*3^b=72, a+b=? | OK | OK | OK | OK |
| 11 | Altitude to hypotenuse | OK | OK | OK | OK |
| 12 | 10th Fibonacci | OK | OK | OK | OK |
| 13 | Geometric sequence 8th term | OK | OK | OK | OK |
| 14 | MISSISSIPPI arrangements | OK | OK | OK | OK |
| 15 | C(8,3) | OK | OK | OK | OK |
| 16 | Infinite geometric series sum | OK | OK | OK | OK |
| 17 | 2x2 determinant | OK | OK | OK | OK |
| 18 | Sum 1 to 100 | OK | OK | OK | OK |
| 19 | 13^3 | OK | OK | OK | OK |

Q1 and Q3 are failed by all models (multi-digit arithmetic without thinking). Q9 differentiates nvidia (FAIL on both engines) from AWQ and lukealonso (OK). nvidia produces the same wrong answer (30) on both SGLang and vLLM, confirming the error is in the quantized weights.

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

### 1. AWQ and lukealonso NVFP4 are both excellent — AWQ wins on quality and throughput, lukealonso on consistency

| Metric | AWQ (SGLang) | lukealonso NVFP4 (SGLang) | nvidia NVFP4 (SGLang) | nvidia NVFP4 (vLLM) |
|--------|-----|------------------|--------------|---------------------|
| GPQA (8-repeat mean) | 88.40% | 88.28% | 87.46% | **88.53%** |
| GPQA (std) | 1.39 | **1.06** | 1.57 | 1.92 |
| GSM8K | **99.0%** | **99.0%** | 97.5% | 98.5% |
| Hard Math | **89.5%** | **89.5%** | 84.2% | 84.2% |
| KL Divergence | **0.024** | 0.035 | 0.109 | — |
| Throughput (C=64) | **1662 tok/s** | 1202 tok/s | — | — |

On GPQA, all four configurations are **statistically equivalent** (Welch t-test p>0.05 for all pairs). nvidia NVFP4 on vLLM (88.53%) scores highest but also has the worst run-to-run consistency (std 1.92). lukealonso has the **best consistency** (std 1.06) — the most predictable results across repeated evaluations. AWQ and lukealonso tie on all accuracy benchmarks (GPQA, GSM8K, Hard Math).

AWQ's advantages are in **KLD** (0.024 vs 0.035 — 32% lower divergence from FP8 reference, meaning AWQ preserves more of the original model's output distribution) and **throughput** (15-38% faster than NVFP4 on SGLang). lukealonso's advantage is **lower variance** on GPQA, making it the more consistent choice for accuracy-sensitive deployments.

**Overall recommendation:** AWQ offers the best combination of quality (lowest KLD) and throughput, making it the best choice for production serving where both speed and fidelity matter. lukealonso NVFP4 is an equally valid choice if consistency is prioritized over throughput.

### 2. If NVFP4 is required, use lukealonso over nvidia
lukealonso NVFP4 trends higher than nvidia NVFP4 across all benchmarks (+0.8% on GPQA, though not statistically significant) with better consistency (std 1.06 vs 1.57). The advantage is clear without thinking mode (GSM8K: +5%, Hard Math: +5.3%). nvidia NVFP4 has significant KLD (0.109, 3x worse than lukealonso), consistent with community reports (vLLM Issue #36094).

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
