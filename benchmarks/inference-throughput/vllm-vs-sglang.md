# vLLM vs SGLang Throughput: lukealonso/Qwen3.5-397B-A17B-NVFP4

## Setup

| | SGLang | vLLM |
|--|--------|------|
| **Version** | 0.5.9 | 0.17.0rc1 |
| **TP** | 4 | 4 |
| **GPUs** | 4x RTX PRO 6000 Blackwell (98GB) | 4x RTX PRO 6000 Blackwell (98GB) |
| **Max concurrent** | 64 | 128 |
| **GPU mem** | 0.97 (MTP) / 0.90 (no-MTP) | 0.9 |
| **Chunked prefill** | 4096 | default |
| **KV cache** | fp8_e4m3 | auto |
| **Attention** | triton | default |
| **MoE runner** | flashinfer_cutlass | default |
| **FP4 GEMM** | flashinfer_cudnn | default |
| **MTP spec** | NEXTN steps=4 topk=1 draft=3 | mtp num_speculative_tokens=2 |
| **Date** | 2026-03-14 | 2026-03-14 |

---

## MTP ON

### SGLang MTP ON (tok/s)

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|------------|------|------|------|------|------|------|------|------|
| 0 | 130 | 220 | 390 | 590 | 862 | 1342 | 1248 | — |
| 16k | 79 | 147 | 260 | 447 | 701 | 973 | — | — |
| 32k | 58 | 113 | 205 | 360 | 575 | — | — | — |
| 64k | 37 | 71 | 137 | 252 | — | — | — | — |
| 128k | 29 | 41 | 78 | 168 | — | — | — | — |

### vLLM MTP ON (tok/s)

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|------------|------|------|------|------|------|------|------|------|
| 0 | 127 | 219 | 350 | 615 | 934 | 1441 | 2283 | 3220 |
| 16k | 127 | 220 | 354 | 610 | 880 | 1439 | 2206 | 2901 |
| 32k | 127 | 214 | 345 | 601 | 933 | 1377 | 2075 | 2574 |
| 64k | 128 | 215 | 344 | 525 | 904 | 1295 | 1905 | 2183 |
| 128k | 122 | 214 | 344 | 570 | 838 | 1185 | 1633 | 2157 |

### SGLang / vLLM ratio — MTP ON (>1.00 = SGLang faster)

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|------------|------|------|------|------|------|------|------|------|
| 0 | 1.02x | 1.00x | 1.11x | 0.96x | 0.92x | 0.93x | 0.55x | — |
| 16k | 0.62x | 0.67x | 0.73x | 0.73x | 0.80x | 0.68x | — | — |
| 32k | 0.45x | 0.53x | 0.59x | 0.60x | 0.62x | — | — | — |
| 64k | 0.29x | 0.33x | 0.40x | 0.48x | — | — | — | — |
| 128k | 0.24x | 0.19x | 0.23x | 0.30x | — | — | — | — |

---

## MTP OFF (no-MTP)

### SGLang MTP OFF (no-MTP) (tok/s)

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|------------|------|------|------|------|------|------|------|------|
| 0 | 74 | 123 | 223 | 379 | 615 | 996 | 1160 | — |
| 16k | 65 | 111 | 203 | 353 | 574 | 929 | 1151 | — |
| 32k | 58 | 100 | 186 | 327 | 541 | 884 | — | — |
| 64k | 48 | 85 | 159 | 285 | 482 | — | — | — |
| 128k | 35 | 64 | 122 | 227 | — | — | — | — |

### vLLM MTP OFF (no-MTP) (tok/s)

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|------------|------|------|------|------|------|------|------|------|
| 0 | 81 | 141 | 251 | 414 | 668 | 987 | 1590 | 2291 |
| 16k | 80 | 139 | 250 | 406 | 653 | 986 | 1465 | 2164 |
| 32k | 80 | 137 | 247 | 406 | 652 | 954 | 1463 | 2037 |
| 64k | 78 | 135 | 243 | 398 | 636 | 922 | 1338 | 1906 |
| 128k | 77 | 133 | 239 | 389 | 604 | 859 | 1209 | 1527 |

### SGLang / vLLM ratio — MTP OFF (no-MTP) (>1.00 = SGLang faster)

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|------------|------|------|------|------|------|------|------|------|
| 0 | 0.92x | 0.87x | 0.89x | 0.92x | 0.92x | 1.01x | 0.73x | — |
| 16k | 0.82x | 0.80x | 0.81x | 0.87x | 0.88x | 0.94x | 0.79x | — |
| 32k | 0.73x | 0.73x | 0.76x | 0.81x | 0.83x | 0.93x | — | — |
| 64k | 0.61x | 0.63x | 0.66x | 0.72x | 0.76x | — | — | — |
| 128k | 0.46x | 0.48x | 0.51x | 0.58x | — | — | — | — |

---

## Per-request decode at C=1 (tok/s)

| ctx | SGLang MTP | vLLM MTP | SGLang/vLLM | SGLang noMTP | vLLM noMTP | SGLang/vLLM |
|-----|-----------|---------|-------------|-------------|-----------|-------------|
| 0 | 130 | 127 | 1.02x | 74 | 81 | 0.92x |
| 16k | 79 | 127 | 0.62x | 65 | 80 | 0.82x |
| 32k | 58 | 127 | 0.45x | 58 | 80 | 0.73x |
| 64k | 37 | 128 | 0.29x | 48 | 78 | 0.61x |
| 128k | 29 | 122 | 0.24x | 35 | 77 | 0.46x |

## Notes

- SGLang max-running-requests=64, vLLM max-num-seqs=128. SGLang C=128 not tested.
- SGLang skips cells where KV cache budget is exceeded (shown as "—" in tables).
- SGLang no-MTP required --mem-fraction-static 0.90 (0.97 caused OOM in fused_moe workspace). MTP used 0.97.
- SGLang no-MTP also required --disable-shared-experts-fusion (crash without it, bug #5702).
- SGLang MTP config differs from vLLM: NEXTN steps=4 topk=1 draft=3 vs mtp num_speculative_tokens=2.
- Both engines use the same container image (llm-pytorch-blackwell:nightly-cuda132) and same GPUs.

