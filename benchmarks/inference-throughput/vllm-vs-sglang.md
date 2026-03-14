# vLLM vs SGLang: Decode Throughput

## Setup

| | vLLM | SGLang |
|--|------|--------|
| **Version** | 0.17.0rc1 | 0.5.9 |
| **TP** | 4 | 4 |
| **GPUs** | 4x RTX PRO 6000 Blackwell (98GB) | 4x RTX PRO 6000 Blackwell (98GB) |
| **MTP** | `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'` | `SGLANG_ENABLE_SPEC_V2=True --speculative-algo NEXTN --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6` |
| **Max seqs** | 128 | 64 |
| **Date** | 2026-03-14 | 2026-03-11 |

## Decode throughput — MTP ON, context=0 (tok/s)

### AWQ (QuantTrio/Qwen3.5-397B-A17B-AWQ)

| C | vLLM | SGLang | vLLM/SGLang |
|---|------|--------|-------------|
| 1 | 147 | 152 | 0.97x |
| 8 | 767 | 665 | 1.15x |
| 16 | 1163 | 976 | 1.19x |
| 32 | 1679 | 1516 | 1.11x |
| 64 | 2622 | 1662 | 1.58x |
| 128 | 3519 | — | — |

### lukealonso NVFP4

| C | vLLM | SGLang | vLLM/SGLang |
|---|------|--------|-------------|
| 1 | 127 | 132 | 0.96x |
| 8 | 615 | 581 | 1.06x |
| 16 | 934 | 852 | 1.10x |
| 32 | 1441 | 1191 | 1.21x |
| 64 | 2283 | 1202 | 1.90x |
| 128 | 3220 | — | — |

## Notes

- SGLang used `--max-running-requests 64`, vLLM used `--max-num-seqs 128` — SGLang C=64 is at its limit, vLLM scales to C=128
- SGLang throughput was measured via Prometheus `sglang:gen_throughput` metric; vLLM via `llm_decode_bench.py`
- SGLang data is context=0 MTP ON only; no long-context or MTP OFF SGLang throughput data available
- nvidia NVFP4 was not tested on SGLang throughput

## vLLM-only: Full context sweep, MTP ON, C=1 (tok/s)

| ctx | AWQ | luke NVFP4 | nvidia NVFP4 |
|-----|-----|-----------|--------------|
| 0 | 147 | 127 | 121 |
| 16k | 110 | 127 | 122 |
| 32k | 88 | 127 | 120 |
| 64k | 61 | 128 | 125 |
| 128k | 40 | 122 | 123 |

## vLLM-only: Full context sweep, MTP ON, C=128 (tok/s)

| ctx | AWQ | luke NVFP4 | nvidia NVFP4 |
|-----|-----|-----------|--------------|
| 0 | 3519 | 3220 | 3232 |
| 16k | 2135 | 2902 | 2624 |
| 32k | 2024 | 2574 | 2502 |
| 64k | 2303 | 2183 | 2159 |
| 128k | 646 | 2157 | 2138 |

## vLLM-only: MTP OFF, C=1 (tok/s)

| ctx | AWQ | luke NVFP4 | nvidia NVFP4 |
|-----|-----|-----------|--------------|
| 0 | 104 | 81 | 79 |
| 16k | 104 | 80 | 78 |
| 32k | 103 | 80 | 78 |
| 64k | 100 | 78 | 76 |
| 128k | 96 | 77 | 75 |

## vLLM-only: MTP OFF, C=128 (tok/s)

| ctx | AWQ | luke NVFP4 | nvidia NVFP4 |
|-----|-----|-----------|--------------|
| 0 | 2796 | 2291 | 2294 |
| 16k | 2542 | 2164 | 2164 |
| 32k | 2290 | 2037 | 2036 |
| 64k | 1909 | 1907 | 1783 |
| 128k | 1526 | 1527 | 1527 |
