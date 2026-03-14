# Qwen3.5-397B-A17B Quantization Comparison: AWQ vs NVFP4

## Three-way benchmark: QuantTrio/AWQ vs lukealonso/NVFP4 vs nvidia/NVFP4

### Test Environment

- **GPUs:** 8x NVIDIA RTX PRO 6000 Blackwell Server Edition (98GB each)
- **Engines:** SGLang 0.5.9 (TP8), vLLM (TP4)
- **Container:** voipmonitor/llm-pytorch-blackwell:nightly-cuda132
- **Date:** 2026-03-11 to 2026-03-14

### Models Tested

| Model | Quantization | Format | Precision |
|-------|-------------|--------|-----------|
| QuantTrio/Qwen3.5-397B-A17B-AWQ | AWQ | INT4 | 16 levels, per-channel scaling, salient weight protection |
| lukealonso/Qwen3.5-397B-A17B-NVFP4 | ModelOpt FP4 | E2M1 | 8 unique values, Blackwell FP4 Tensor Core hardware |
| nvidia/Qwen3.5-397B-A17B-NVFP4 | ModelOpt FP4 | E2M1 | 8 unique values, Blackwell FP4 Tensor Core hardware |

### Server Configs

#### SGLang (TP8)

```
--tensor-parallel-size 8
--quantization modelopt_fp4      # omit for AWQ (auto-detected)
--kv-cache-dtype fp8_e4m3
--attention-backend triton
--moe-runner-backend flashinfer_cutlass
--fp4-gemm-backend flashinfer_cudnn  # omit for AWQ
--cuda-graph-max-bs 128
--max-running-requests 128
--context-length 262144
--chunked-prefill-size 32768
--mem-fraction-static 0.80
--disable-custom-all-reduce
--disable-shared-experts-fusion
--schedule-conservativeness 0.1
```

MTP flags (SGLang):
```
SGLANG_ENABLE_SPEC_V2=True
--speculative-algo NEXTN
--speculative-num-steps 5
--speculative-eagle-topk 1
--speculative-num-draft-tokens 6
```

#### vLLM (TP4)

```bash
VLLM_LOG_STATS_INTERVAL=1 NCCL_P2P_LEVEL=SYS SAFETENSORS_FAST_GPU=1 \
python3 -m vllm.entrypoints.openai.api_server \
  --model nvidia/Qwen3.5-397B-A17B-NVFP4 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 128 \
  --reasoning-parser qwen3 \
  --enable-prefix-caching --enable-chunked-prefill
```

MTP flags (vLLM):
```
--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
```

### Eval Command

```bash
python3 -u -m sglang.test.run_eval \
  --eval-name gpqa \
  --model <served-model-name> \
  --base-url http://localhost:<port> \
  --num-examples 198 \
  --repeat 1 \
  --thinking-mode qwen3 \
  --max-tokens 64000
```

8-repeat tests were run sequentially (not parallel) to avoid server overload.

---

## Summary Table

All GPQA results are 8-repeat means with thinking mode enabled, 198 examples.

| Benchmark | AWQ SGLang | luke NVFP4 SGLang | nvidia NVFP4 SGLang | nvidia NVFP4 vLLM |
|-----------|-----------|-------------------|--------------------|--------------------|
| **GPQA MTP ON** | **88.40%** ±1.39 | 88.28% ±1.06 | 87.46% ±1.57 | **88.53%** ±1.92 |
| **GPQA MTP OFF** | — | 87.53% ±1.09 | 86.58% ±1.15 | 86.90% ±1.13 |
| **GSM8K** (thinking) | **99.0%** | **99.0%** | 97.5% | 98.5% |
| **Hard Math** (no thinking) | **89.5%** | **89.5%** | 84.2% | 84.2% |
| **KL Divergence** (vs FP8) | **0.024** | 0.035 | 0.109 | — |
| **Throughput ctx=0** (C=128, MTP ON) | **3519** | 3220 | 3232 | — |
| **Throughput ctx=64k** (C=64, MTP ON) | 1747 | **1905** | **1912** | — |

**Key findings:**
- **GPQA with MTP ON:** all four configurations are statistically indistinguishable (Welch t-test p>0.05 for all pairs)
- **GPQA MTP OFF:** both engines converge to ~86.6-86.9% for nvidia, confirming MTP ON does not hurt accuracy
- **GSM8K/Hard Math:** AWQ and lukealonso tie; nvidia is weaker (97.5% / 84.2%), same results on both engines
- **KLD:** AWQ clearly best (0.024), lukealonso good (0.035), nvidia poor (0.109)
- **Throughput (vLLM):** AWQ fastest at short context (3519 tok/s ctx=0), but NVFP4 is 9% faster at long context (1912 vs 1747 at ctx=64k/C=64 MTP ON). AWQ collapses at 128k/C=128 MTP ON (646 tok/s vs NVFP4's 2157)

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

198 examples, thinking mode, 8 independent runs per configuration.

### All GPQA Results

| Model | Engine | MTP | Mean | Std | Scores (8 repeats) |
|-------|--------|-----|------|-----|---------------------|
| AWQ (QuantTrio) | SGLang | ON | **88.40%** | 1.39 | 87.9, 89.9, 88.9, 89.9, 87.4, 89.4, 85.9, 87.9 |
| lukealonso NVFP4 | SGLang | ON | 88.28% | **1.06** | 88.9, 87.9, 86.9, 88.4, 90.4, 88.4, 87.9, 87.4 |
| nvidia NVFP4 | vLLM | ON | **88.53%** | 1.92 | 91.9, 87.9, 89.4, 85.9, 86.4, 88.4, 89.9, 88.4 |
| nvidia NVFP4 | SGLang | ON | 87.46% | 1.57 | 85.9, 90.4, 85.9, 88.4, 87.9, 85.9, 87.4, 87.9 |
| lukealonso NVFP4 | SGLang | OFF | 87.53% | 1.09 | 86.4, 87.4, 89.4, 86.9, 88.9, 86.4, 87.4, 87.4 |
| nvidia NVFP4 | vLLM | OFF | 86.90% | 1.13 | 87.4, 86.4, 86.9, 86.9, 85.9, 85.9, 89.4, 86.4 |
| nvidia NVFP4 | SGLang | OFF | 86.58% | 1.15 | 86.4, 85.9, 86.9, 86.4, 84.8, 86.4, 86.9, 88.9 |

### Statistical Significance (Welch t-test, two-tailed)

**MTP ON pairs:**

| Comparison | Delta | t-stat | Significant? |
|:---|:---:|:---:|:---:|
| AWQ SGLang vs lukealonso SGLang | +0.12pp | 0.20 | No (p>0.05) |
| AWQ SGLang vs nvidia SGLang | +0.94pp | 1.27 | No (p>0.05) |
| AWQ SGLang vs nvidia vLLM | −0.13pp | 0.15 | No (p>0.05) |
| lukealonso SGLang vs nvidia SGLang | +0.81pp | 1.21 | No (p>0.05) |
| nvidia vLLM vs nvidia SGLang | +1.07pp | 1.22 | No (p>0.05) |

**MTP ON vs OFF (same model, same engine):**

| Comparison | Delta | t-stat | Significant? |
|:---|:---:|:---:|:---:|
| nvidia vLLM: MTP ON vs OFF | +1.62pp | 2.06 | No (p=0.06, borderline) |
| nvidia SGLang: MTP ON vs OFF | +0.88pp | 1.28 | No (p>0.05) |
| lukealonso SGLang: MTP ON vs OFF | +0.75pp | 1.41 | No (p>0.05) |

**Without MTP, engines converge:**

| Comparison | Delta | t-stat | Significant? |
|:---|:---:|:---:|:---:|
| nvidia vLLM OFF vs nvidia SGLang OFF | +0.31pp | 0.61 | No (p>0.05) |

No pair reaches statistical significance. With 8 repeats and std ~1.0–1.9, GPQA cannot distinguish these configurations at the 95% confidence level.

For full MTP quality analysis, see [mtp-quality-evaluation.md](mtp-quality-evaluation.md).

---

## Part 3: GSM8K (Grade School Math 8K)

### With thinking mode (200 examples, max-tokens 16000)

| Model | Engine | MTP | Score |
|-------|--------|-----|-------|
| **AWQ (QuantTrio)** | SGLang | ON | **99.0%** |
| lukealonso NVFP4 | SGLang | ON | **99.0%** |
| nvidia NVFP4 | vLLM | OFF | 98.5% |
| nvidia NVFP4 | SGLang | ON | 97.5% |

nvidia on vLLM (98.5%) outperforms nvidia on SGLang (97.5%), but both are below AWQ/lukealonso (99.0%).

### Without thinking (5-shot, SGLang only)

| Model | Engine | Score |
|-------|--------|-------|
| lukealonso NVFP4 | SGLang | **44%** |
| nvidia NVFP4 | SGLang | 39% |

Without chain-of-thought reasoning, the quantization quality gap is much more pronounced (+5pp).

---

## Part 4: Hard Math Test (19 custom questions, no thinking mode)

| Model | Engine | Score | Correct |
|-------|--------|-------|---------|
| **AWQ (QuantTrio)** | SGLang | **89.5%** | 17/19 |
| lukealonso NVFP4 | SGLang | **89.5%** | 17/19 |
| nvidia NVFP4 | SGLang | 84.2% | 16/19 |
| nvidia NVFP4 | vLLM | 84.2% | 16/19 |

nvidia scores identically on both engines (16/19), failing the same three questions (Q1, Q3, Q9) with the same wrong answers. This confirms the errors are from quantization, not the inference engine.

### Per-Question Detail

| Q# | Question | AWQ | lukealonso | nvidia SGLang | nvidia vLLM |
|----|----------|-----|-----------|---------------|-------------|
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

Q1 and Q3 are failed by all models (multi-digit arithmetic without thinking). Q9 differentiates nvidia (FAIL, answer=30, both engines) from AWQ and lukealonso (OK, answer=20).

---

## Part 5: Throughput Benchmark (vLLM, TP4)

All models tested on vLLM 0.17.0rc1, 4x RTX PRO 6000 Blackwell (TP4), with and without MTP.

### Decode throughput at context=0 (tok/s)

| Model | MTP | C=1 | C=8 | C=16 | C=32 | C=64 | C=128 |
|-------|-----|-----|-----|------|------|------|-------|
| **AWQ** | **ON** | **147** | **767** | **1163** | **1679** | **2622** | **3519** |
| lukealonso NVFP4 | ON | 127 | 615 | 934 | 1441 | 2283 | 3220 |
| nvidia NVFP4 | ON | 121 | 577 | 918 | 1418 | 2252 | 3232 |
| AWQ | OFF | 104 | 509 | 843 | 1272 | 1909 | 2796 |
| lukealonso NVFP4 | OFF | 81 | 414 | 668 | 987 | 1590 | 2291 |
| nvidia NVFP4 | OFF | 79 | 406 | 652 | 987 | 1590 | 2294 |

### Decode throughput at context=64k (tok/s)

| Model | MTP | C=1 | C=8 | C=16 | C=32 | C=64 | C=128 |
|-------|-----|-----|-----|------|------|------|-------|
| lukealonso NVFP4 | ON | 128 | 525 | 904 | 1295 | **1905** | **2183** |
| nvidia NVFP4 | ON | 125 | 581 | 877 | 1271 | **1912** | **2159** |
| AWQ | ON | 61 | 389 | 680 | 1074 | 1747 | 2303 |
| AWQ | OFF | 100 | 477 | 748 | 1080 | 1464 | 1909 |
| lukealonso NVFP4 | OFF | 78 | 398 | 636 | 922 | 1338 | 1907 |
| nvidia NVFP4 | OFF | 76 | 390 | 621 | 891 | 1339 | 1783 |

### MTP speedup

| Model | C=1 | C=8 | C=32 | C=64 | C=128 |
|-------|-----|-----|------|------|-------|
| AWQ | +41% | +51% | +32% | +37% | +26% |
| lukealonso NVFP4 | +57% | +49% | +46% | +44% | +41% |
| nvidia NVFP4 | +53% | +42% | +44% | +42% | +41% |

**AWQ is fastest at short context (ctx=0)** but NVFP4 overtakes at long context (64k+) with MTP ON. AWQ has a severe anomaly at 128k/C=128 MTP ON (646 tok/s, queue=81%) due to its larger vocab_size (248320 vs 152064) exhausting KV cache. Without MTP, all models converge at 128k to ~1527 tok/s.

For full results across all context lengths, see [inference-throughput/](inference-throughput/).

---

## Overall Conclusions

### 1. All quantizations produce equivalent GPQA accuracy

| Metric | AWQ SGLang | luke NVFP4 SGLang | nvidia NVFP4 SGLang | nvidia NVFP4 vLLM |
|--------|-----------|-------------------|--------------------|--------------------|
| GPQA MTP ON | 88.40% ±1.39 | 88.28% ±1.06 | 87.46% ±1.57 | 88.53% ±1.92 |
| GPQA MTP OFF | — | 87.53% ±1.09 | 86.58% ±1.15 | 86.90% ±1.13 |
| GSM8K | **99.0%** | **99.0%** | 97.5% | 98.5% |
| Hard Math | **89.5%** | **89.5%** | 84.2% | 84.2% |
| KL Divergence | **0.024** | 0.035 | 0.109 | — |
| Throughput ctx=0 (C=128, MTP) | **3519 tok/s** | 3220 tok/s | 3232 tok/s | — |
| Throughput ctx=64k (C=64, MTP) | 1747 tok/s | **1905 tok/s** | **1912 tok/s** | — |

On GPQA, no pair of configurations is statistically distinguishable (p>0.05 for all). nvidia on vLLM (88.53%) and AWQ on SGLang (88.40%) score highest, but with std 1.0–1.9, 8 repeats cannot resolve differences below ~2pp.

### 2. AWQ wins on KLD and short-context throughput, NVFP4 wins long-context

AWQ has the lowest KL divergence from FP8 (0.024 vs 0.035 vs 0.109) and is fastest at short context (3519 vs 3220 tok/s at ctx=0/C=128). But at long context (64k+) with MTP, NVFP4 is 9% faster (1912 vs 1747 at ctx=64k/C=64). AWQ collapses at 128k/C=128 MTP (646 tok/s) due to larger vocab_size. Choose AWQ for short-context batch workloads, NVFP4 for long-context deployments.

### 3. If NVFP4 is required, use lukealonso over nvidia

lukealonso NVFP4 ties AWQ on GSM8K (99.0%) and Hard Math (89.5%). nvidia is consistently weaker: GSM8K 97.5-98.5%, Hard Math 84.2% (same failures on both engines). nvidia has 3x worse KLD (0.109 vs 0.035), consistent with community reports (vLLM Issue #36094).

### 4. Inference engine does not significantly affect accuracy

nvidia NVFP4 scores similarly on vLLM and SGLang across all benchmarks. Hard Math produces identical results (same questions wrong, same wrong answers). Without MTP, GPQA converges to ~86.6-86.9% on both engines.

### 5. Enable MTP for production serving

MTP provides 18-24% inference speedup with no measurable accuracy degradation on either engine (p>0.05 for all MTP ON vs OFF comparisons).

### 6. Recommended production config

```bash
# Model (best overall)
--model QuantTrio/Qwen3.5-397B-A17B-AWQ

# MTP — SGLang
SGLANG_ENABLE_SPEC_V2=True
--speculative-algo NEXTN
--speculative-num-steps 5
--speculative-eagle-topk 1
--speculative-num-draft-tokens 6

# MTP — vLLM
--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'

# Required flags (SGLang)
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
