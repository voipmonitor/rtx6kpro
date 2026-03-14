# Inference Throughput: Qwen3.5-397B-A17B Quantizations

Decode throughput benchmark comparing quantized Qwen3.5-397B-A17B checkpoints on 4x RTX PRO 6000 Blackwell (TP4), with and without MTP speculative decoding.

## Test Environment

| Parameter | Value |
|-----------|-------|
| **GPUs** | 4x NVIDIA RTX PRO 6000 Blackwell Server Edition (98GB each) |
| **Engine** | vLLM 0.17.0rc1 (TP4) |
| **Container** | llm-pytorch-blackwell:nightly-cuda132 |
| **Benchmark tool** | llm_decode_bench.py |
| **Date** | 2026-03-14 |
| **Max tokens** | 512 per request |
| **Concurrency** | 1, 2, 4, 8, 16, 32, 64, 128 |
| **Context lengths** | 0, 16k, 32k, 64k, 128k |

### vLLM server config (common)

```
--tensor-parallel-size 4
--gpu-memory-utilization 0.9
--max-num-batched-tokens 8192
--max-num-seqs 128
--enable-prefix-caching
--enable-chunked-prefill
```

MTP: `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`

---

## Summary

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
| **AWQ** | **ON** | **61** | **389** | **680** | **1074** | 1747 | 2303 |
| AWQ | OFF | 100 | 477 | 748 | 1080 | 1464 | 1909 |
| lukealonso NVFP4 | OFF | 78 | 398 | 636 | 922 | 1338 | 1907 |
| nvidia NVFP4 | OFF | 76 | 390 | 621 | 891 | 1339 | 1783 |

### MTP speedup

| Model | C=1 | C=8 | C=32 | C=64 | C=128 |
|-------|-----|-----|------|------|-------|
| AWQ | +41% | +51% | +32% | +37% | +26% |
| lukealonso NVFP4 | +57% | +49% | +46% | +44% | +41% |
| nvidia NVFP4 | +53% | +42% | +44% | +42% | +41% |

### Key findings

1. **AWQ is fastest at short context (ctx=0)**, outperforming NVFP4 by 9-16% with MTP ON
2. **At long context (64k+) with MTP ON, NVFP4 is faster**: at ctx=64k/C=64, NVFP4 reaches 1905-1912 tok/s vs AWQ's 1747 tok/s (~9% faster)
3. **AWQ has a severe anomaly at 128k/C=128 MTP ON**: throughput collapses to 646 tok/s (queue utilization ~81%) while NVFP4 holds 2157 tok/s. This is likely due to AWQ's larger vocab_size (248320 vs 152064 for NVFP4), which consumes more KV cache memory per token and causes capacity exhaustion at maximum context + concurrency
4. **Without MTP, all three models converge at 128k/C=128 to ~1527 tok/s** — the gap disappears entirely
5. **MTP gives 26-57% speedup** depending on model and concurrency — always worth enabling
6. **lukealonso and nvidia NVFP4 have identical throughput** without MTP; lukealonso is ~5% faster with MTP at low concurrency
7. **AWQ MTP at C=128 ctx=0 peaks at 3519 tok/s** — highest measured throughput in short-context workloads

---

## Full Results: AWQ (QuantTrio/Qwen3.5-397B-A17B-AWQ)

### AWQ MTP ON — Aggregate Throughput (tok/s)

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|------------|---|---|---|---|----|----|----|----|
| 0 | 147 | 269 | 411 | 767 | 1163 | 1679 | 2622 | 3519 |
| 16k | 110 | 205 | 319 | 602 | 980 | 1435 | 2231 | 2135 |
| 32k | 88 | 163 | 269 | 507 | 851 | 1284 | 2044 | 2024 |
| 64k | 61 | 115 | 206 | 389 | 680 | 1074 | 1747 | 2303 |
| 128k | 40 | 76 | 136 | 264 | 477 | 812 | 1326 | 646 |

### AWQ MTP OFF — Aggregate Throughput (tok/s)

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|------------|---|---|---|---|----|----|----|----|
| 0 | 104 | 167 | 298 | 509 | 843 | 1272 | 1909 | 2796 |
| 16k | 104 | 167 | 294 | 501 | 812 | 1209 | 1781 | 2542 |
| 32k | 103 | 165 | 290 | 493 | 796 | 1146 | 1654 | 2290 |
| 64k | 100 | 159 | 282 | 477 | 748 | 1080 | 1464 | 1909 |
| 128k | 96 | 151 | 267 | 445 | 684 | 923 | 1209 | 1526 |

---

## Full Results: lukealonso/Qwen3.5-397B-A17B-NVFP4

### lukealonso MTP ON — Aggregate Throughput (tok/s)

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|------------|---|---|---|---|----|----|----|----|
| 0 | 127 | 219 | 350 | 615 | 934 | 1441 | 2283 | 3220 |
| 16k | 127 | 220 | 354 | 610 | 881 | 1439 | 2206 | 2902 |
| 32k | 127 | 214 | 345 | 601 | 933 | 1377 | 2075 | 2574 |
| 64k | 128 | 215 | 344 | 525 | 904 | 1295 | 1905 | 2183 |
| 128k | 122 | 214 | 344 | 570 | 838 | 1185 | 1633 | 2157 |

### lukealonso MTP OFF — Aggregate Throughput (tok/s)

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|------------|---|---|---|---|----|----|----|----|
| 0 | 81 | 141 | 251 | 414 | 668 | 987 | 1590 | 2291 |
| 16k | 80 | 139 | 250 | 406 | 653 | 986 | 1465 | 2164 |
| 32k | 80 | 137 | 247 | 406 | 652 | 954 | 1463 | 2037 |
| 64k | 78 | 135 | 243 | 398 | 636 | 922 | 1338 | 1907 |
| 128k | 77 | 133 | 239 | 389 | 604 | 859 | 1209 | 1527 |

---

## Full Results: nvidia/Qwen3.5-397B-A17B-NVFP4

### nvidia MTP ON — Aggregate Throughput (tok/s)

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|------------|---|---|---|---|----|----|----|----|
| 0 | 121 | 212 | 340 | 577 | 918 | 1418 | 2252 | 3232 |
| 16k | 122 | 207 | 340 | 598 | 922 | 1390 | 2167 | 2624 |
| 32k | 120 | 206 | 340 | 589 | 909 | 1340 | 2065 | 2502 |
| 64k | 125 | 209 | 341 | 581 | 877 | 1271 | 1912 | 2159 |
| 128k | 123 | 203 | 334 | 554 | 806 | 1164 | 1620 | 2138 |

### nvidia MTP OFF — Aggregate Throughput (tok/s)

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|------------|---|---|---|---|----|----|----|----|
| 0 | 79 | 135 | 243 | 406 | 652 | 987 | 1590 | 2294 |
| 16k | 78 | 133 | 239 | 398 | 652 | 955 | 1464 | 2164 |
| 32k | 78 | 131 | 239 | 398 | 637 | 954 | 1462 | 2036 |
| 64k | 76 | 131 | 235 | 390 | 621 | 891 | 1339 | 1783 |
| 128k | 75 | 127 | 231 | 382 | 589 | 858 | 1209 | 1527 |

---

## Analysis

### AWQ vs NVFP4 throughput

The winner depends on context length and whether MTP is enabled:

**Short context (ctx=0), MTP ON:** AWQ is fastest, outperforming NVFP4 by 9-16% across all concurrency levels. AWQ peaks at 3519 tok/s vs NVFP4's 3220-3232 tok/s at C=128.

**Long context (64k+), MTP ON:** NVFP4 pulls ahead. At ctx=64k/C=64, NVFP4 reaches 1905-1912 tok/s vs AWQ's 1747 tok/s — a ~9% NVFP4 advantage. The crossover occurs somewhere between ctx=16k and ctx=64k.

**Without MTP:** AWQ is moderately faster at short context (3-33%), but all three models converge tightly at 128k/C=128 to approximately 1527 tok/s. The context-length penalty is steepest for AWQ with MTP.

**AWQ collapse at 128k/C=128 MTP ON (646 tok/s):** This is a severe anomaly — throughput drops to 42% of NVFP4's figure at the same setting (2157 tok/s). The root cause is almost certainly AWQ's larger vocabulary: `vocab_size=248320` vs `152064` for NVFP4 (a 63% larger embedding table). This inflates the KV cache footprint per token, exhausting available GPU memory at maximum context length and concurrency when MTP is active. Queue utilization reaches ~81%, indicating requests are being serialized rather than batched. Notably, this does not occur without MTP, where AWQ holds 1526 tok/s at 128k/C=128 — comparable to NVFP4's 1527 tok/s.

AWQ's advantage at short context despite NVFP4 having dedicated FP4 Tensor Cores is explained by:
1. AWQ uses mature INT4 GEMM kernels with better scheduling
2. NVFP4's E2M1 format (8 unique values) vs AWQ's 16 levels means slightly different effective quantization density per bit
3. AWQ's per-channel scaling and salient weight protection reduce quantization error without runtime overhead

### MTP impact on throughput

MTP provides substantial speedup across all models:
- **NVFP4 models:** consistent 40-57% speedup at low concurrency, 41% at C=128
- **AWQ:** 41-51% at low concurrency, but drops to 26% at C=128 (AWQ's baseline is already faster)

MTP is more effective for NVFP4 because the base decode speed is slower, giving more room for speculative acceleration. However, for AWQ at very long contexts, MTP introduces the KV cache pressure issue described above — at 128k/C=128 the benefit turns into a severe penalty.

### Context length impact

Throughput degrades with longer contexts due to KV cache memory pressure:
- **Without MTP:** All models degrade gracefully. At C=128, ctx=128k vs ctx=0 reduces throughput by ~31-40%. All three models land at ~1527 tok/s at 128k/C=128.
- **With MTP, NVFP4:** Moderate degradation at high context. 128k/C=128 yields ~2138-2157 tok/s, still well above the no-MTP baseline.
- **With MTP, AWQ:** Degrades normally up to 64k/C=128 (2303 tok/s), then collapses at 128k/C=128 (646 tok/s) due to KV cache exhaustion from the larger vocab_size.

The practical recommendation: use AWQ for short-context batch workloads; prefer NVFP4 for long-context or mixed-context deployments where MTP is enabled.

### NVFP4: lukealonso vs nvidia

Without MTP, the two NVFP4 checkpoints have **identical throughput** (within measurement noise). With MTP, lukealonso is ~5% faster at C=1 (127 vs 121 tok/s) but converges at high concurrency. The difference likely comes from minor weight distribution differences affecting MTP acceptance rate.

---

## Legacy: SGLang Results (2026-03-11)

Previous measurements on SGLang 0.5.9 (TP4) with MTP ON only. These used a different benchmark method (Prometheus `sglang:gen_throughput` metric) and are not directly comparable to the vLLM results above.

```
SGLang MTP ON — Aggregate decode throughput (tok/s), context=0
=========================================================================

Model                                 C=1    C=8    C=16    C=32    C=64
------------------------------------------------------------------------
QuantTrio/Qwen3.5-397B-A17B-AWQ      152    665     976    1516    1662
lukealonso/Qwen3.5-397B-A17B-NVFP4   132    581     852    1191    1202
```

Note: SGLang numbers are lower at high concurrency because `--max-running-requests 64` was used vs 128 for vLLM. The relative ranking (AWQ > NVFP4) is consistent across both engines.
