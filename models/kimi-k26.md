# Kimi K2.6 on RTX PRO 6000 Blackwell

## Overview

This page tracks the current public community recipe for the Kimi MLA serving stack on RTX PRO 6000 Blackwell systems.

The reproducible public demo path currently uses:
- target model: `moonshotai/Kimi-K2.5`
- speculative draft: `lightseekorg/kimi-k2.5-eagle3-mla`
- serving engine: `vLLM`
- attention backend: `TRITON_MLA`
- KV cache: `fp8`
- decode context parallelism: `DCP=4` or `DCP=8`

That setup is used here because it exercises the same MLA serving path that matters for Kimi K2.x, is public, and is significantly more representative for Kimi than the non-MLA Llama-style EAGLE draft.

## Community Image

Docker image:

```bash
docker pull voipmonitor/vllm:kimi-k25-eagle3mla-nccl2297-community-20260422
```

What is inside:
- vLLM `0.19.2rc1.dev48+g47fcb8ca6.d20260420`
- patched NCCL `2.29.7`
- Kimi MLA runtime path: `TRITON_MLA + fp8 KV`
- DCP=4 XML-scoped workaround included in the image
- community-tested `Kimi-K2.5 + eagle3-mla` launch path

## Why This Uses `eagle3-mla` Instead of the Llama Draft

The MLA draft is the right public demo for Kimi because:
- it is MLA-aware, so it uses the same `TRITON_MLA + fp8 KV + DCP` stack as the target model
- it is a better proxy for real Kimi serving than the non-MLA Llama-style draft
- in practical Kimi testing it behaved better than the Llama draft for this serving path
- it keeps the benchmark focused on the Kimi MLA runtime instead of mixing in a different draft backend/runtime path

## Recommended Launch Commands

### Fastest known: DCP=4

```bash
docker run --rm --gpus all --network host --ipc host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /mnt/nccl_graph_opt.xml:/mnt/nccl_graph_opt.xml:ro \
  voipmonitor/vllm:kimi-k25-eagle3mla-nccl2297-community-20260422 \
  bash -lc '
VLLM_SPECULATIVE_DISABLE_ABOVE_SEQ_LEN=7000 \
VLLM_ENABLE_PCIE_ALLREDUCE=1 \
NCCL_P2P_LEVEL=SYS \
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
NCCL_GRAPH_FILE=/mnt/nccl_graph_opt.xml \
VLLM_LOG_STATS_INTERVAL=1 \
VLLM_TEST_FORCE_FP8_MARLIN=1 \
VLLM_MARLIN_USE_ATOMIC_ADD=1 \
VLLM_MARLIN_INPUT_DTYPE=fp8 \
/opt/venv/bin/vllm serve moonshotai/Kimi-K2.5 \
  --served-model-name Kimi-K2.5 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 5000 \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 1 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --load-format fastsafetensors \
  --async-scheduling \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 128 \
  --mm-processor-cache-gb 0 \
  --mm-encoder-tp-mode weights \
  --attention-backend TRITON_MLA \
  --kv-cache-dtype fp8 \
  --decode-context-parallel-size 4 \
  --tool-call-parser kimi_k2 \
  --enable-auto-tool-choice \
  --reasoning-parser kimi_k2 \
  --speculative-config '\''{"model":"lightseekorg/kimi-k2.5-eagle3-mla","method":"eagle3","num_speculative_tokens":3,"draft_attention_backend":"TRITON_MLA","draft_kv_cache_dtype":"fp8","rejection_sample_method":"probabilistic"}'\''
'
```

### Reference alternative: DCP=8

```bash
docker run --rm --gpus all --network host --ipc host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /mnt/nccl_graph_opt.xml:/mnt/nccl_graph_opt.xml:ro \
  voipmonitor/vllm:kimi-k25-eagle3mla-nccl2297-community-20260422 \
  bash -lc '
VLLM_SPECULATIVE_DISABLE_ABOVE_SEQ_LEN=7000 \
VLLM_ENABLE_PCIE_ALLREDUCE=1 \
NCCL_P2P_LEVEL=SYS \
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
NCCL_GRAPH_FILE=/mnt/nccl_graph_opt.xml \
VLLM_LOG_STATS_INTERVAL=1 \
VLLM_TEST_FORCE_FP8_MARLIN=1 \
VLLM_MARLIN_USE_ATOMIC_ADD=1 \
VLLM_MARLIN_INPUT_DTYPE=fp8 \
/opt/venv/bin/vllm serve moonshotai/Kimi-K2.5 \
  --served-model-name Kimi-K2.5 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 5000 \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 1 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --load-format fastsafetensors \
  --async-scheduling \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 128 \
  --mm-processor-cache-gb 0 \
  --mm-encoder-tp-mode weights \
  --attention-backend TRITON_MLA \
  --kv-cache-dtype fp8 \
  --decode-context-parallel-size 8 \
  --tool-call-parser kimi_k2 \
  --enable-auto-tool-choice \
  --reasoning-parser kimi_k2 \
  --speculative-config '\''{"model":"lightseekorg/kimi-k2.5-eagle3-mla","method":"eagle3","num_speculative_tokens":3,"draft_attention_backend":"TRITON_MLA","draft_kv_cache_dtype":"fp8","rejection_sample_method":"probabilistic"}'\''
'
```

## Expected `llm_decode_bench.py` Results

Command:

```bash
python3 /mnt/llm_decode_bench.py --port 5000
```

### DCP=4 decode expectations (tok/s)

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 85.5 | 116.2 | 174.0 | 281.3 | 427.5 | 639.3 | 929.6 | 1224.5 |
| 16k | 52.7 | 97.3 | 174.8 | 262.2 | 365.5 | 572.8 | 826.7 | 1144.5 |
| 32k | 52.7 | 97.2 | 159.0 | 246.3 | 317.9 | 477.4 | 698.8 | 889.8 |
| 64k | 52.7 | 89.5 | 151.1 | 206.7 | 270.1 | 381.6 | 508.4 | 509.1 |
| 128k | 51.6 | 83.4 | 123.2 | 166.9 | 204.9 | 254.3 | 317.7 | 381.1 |

Prefill expectations:
- 8k: `7885 tok/s`, TTFT `0.713s`
- 16k: `7998 tok/s`, TTFT `1.350s`
- 32k: `7693 tok/s`, TTFT `2.746s`
- 64k: `6999 tok/s`, TTFT `5.969s`
- 128k: `6108 tok/s`, TTFT `13.602s`

### DCP=8 decode expectations (tok/s)

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 78.6 | 111.4 | 172.0 | 268.1 | 435.8 | 608.1 | 924.6 | 1024.8 |
| 16k | 47.7 | 87.4 | 151.1 | 238.1 | 318.1 | 508.4 | 763.5 | 1018.1 |
| 32k | 49.7 | 89.4 | 151.2 | 230.7 | 302.2 | 445.2 | 635.7 | 763.5 |
| 64k | 49.7 | 85.4 | 143.0 | 198.8 | 254.5 | 349.7 | 445.2 | 508.6 |
| 128k | 47.7 | 79.5 | 119.3 | 159.1 | 190.9 | 254.3 | 318.0 | 381.1 |

Prefill expectations:
- 8k: `7907 tok/s`, TTFT `0.712s`
- 16k: `8015 tok/s`, TTFT `1.347s`
- 32k: `7695 tok/s`, TTFT `2.746s`
- 64k: `6991 tok/s`, TTFT `5.977s`
- 128k: `6096 tok/s`, TTFT `13.630s`

At the moment `DCP=4` is the recommended community setting and `DCP=8` is the reference comparison point.

## KV Cache Capacity from the Benchmark Logs

The values below are taken directly from startup logs of the benchmark launch used for validation:
- target model: `moonshotai/Kimi-K2.5`
- draft model: `lightseekorg/kimi-k2.5-eagle3-mla`
- `TRITON_MLA`
- `fp8` KV cache
- `--max-model-len 65536`
- `--gpu-memory-utilization 0.90`
- speculative decode enabled

They are useful as a sanity check for the benchmark profile, but they should not be treated as universal values for every Kimi launch. The reported KV cache size changes with launch options such as `--max-model-len`, `--gpu-memory-utilization`, multimodal flags, and other memory consumers.

GPU KV cache size reported in those benchmark logs:

| Setting | Expected GPU KV cache size |
|---|---:|
| `DCP=4` | `512,192 tokens` |
| `DCP=8` | `1,022,592 tokens` |

Source log line:

```text
GPU KV cache size: ... tokens
```

If the reported cache size is far lower than these numbers, the launch configuration is not matching the benchmark profile above.

## NCCL XML vs no-XML Validation

`NCCL_GRAPH_FILE=/mnt/nccl_graph_opt.xml` is still the shipped public recipe for the fastest known configuration.

Patched NCCL without XML was functional, but prefill was still slower than the XML-based path. The cleanest validation was a prefill-only test on the same Kimi MLA stack with `DCP=8`, `TRITON_MLA`, `fp8 KV`, `max_tokens=1`, and no prefix cache.

### Prefill-only A/B

| Mode | Concurrency | Avg TTFT | Aggregate Prompt tok/s |
|---|---:|---:|---:|
| XML | 1 | 0.310s | 20.7k tok/s |
| no-XML | 1 | 1.329s | 6.75k tok/s |
| XML | 16 | 0.903s | 115.3k tok/s |
| no-XML | 16 | 0.999s | 108.1k tok/s |

So:
- no-XML was about `3.07x` slower on the single-request prefill test
- no-XML was still about `6.7%` slower at concurrency 16

### What NCCL Debug Showed

The transport was the same in both cases (`P2P/CUMEM`). The difference was the graph/search result:

With XML:
- `Pattern 4 ... bw 38/32, type PHB/PIX`
- `Pattern 1 ... bw 38/32, type PHB/PIX`
- 4 total channels

Without XML:
- `Pattern 4 ... bw 15/15, type SYS/PIX`
- `Pattern 1 ... bw 0.1/0.1, type SYS/SYS`
- 2 total channels

That is why the XML path is still the best known public recipe for Kimi MLA.

### Upstream no-XML NCCL status

There is now an upstream NCCL draft PR that specifically targets the Turin no-XML pathological ring selection:

- [`NVIDIA/nccl#2127`](https://github.com/NVIDIA/nccl/pull/2127) — Improve AMD Turin no-XML ring graph selection

That fix was validated on the real Kimi MLA community serving path using a single cold `8k` prefill request:

| Mode | TTFT | Prompt tokens | Prompt tok/s |
|---|---:|---:|---:|
| broken no-XML (before fix) | `9.138s` | `8005` | `875.98` |
| no-XML with PR `#2127` | `1.074s` | `8005` | `7455.28` |
| XML baseline | `1.071s` | `8005` | `7477.64` |

So the no-XML path is now effectively at parity with XML on that targeted reproducer.

Practical interpretation:
- today: keep using `NCCL_GRAPH_FILE` in the public recipe
- medium term: if NCCL PR `#2127` lands and ships, the XML file should no longer be required on this Turin setup

## Minimal Discord Summary

> Community image: `voipmonitor/vllm:kimi-k25-eagle3mla-nccl2297-community-20260422`
>
> The public demo path uses `moonshotai/Kimi-K2.5` + `lightseekorg/kimi-k2.5-eagle3-mla` because that is the most representative MLA setup for Kimi on vLLM: same `TRITON_MLA + fp8 KV + DCP` serving path, and better behavior than the non-MLA Llama draft.
>
> Expected `llm_decode_bench.py --port 5000` results:
> - DCP=4: `85.5 tok/s` at `ctx=0, C=1`, `52.7 tok/s` at `ctx=16k, C=1`, `1224.5 tok/s` at `ctx=0, C=128`
> - DCP=8: `78.6 tok/s` at `ctx=0, C=1`, `47.7 tok/s` at `ctx=16k, C=1`, `1024.8 tok/s` at `ctx=0, C=128`
>
> Current recommendation: `DCP=4` with `NCCL_GRAPH_FILE=/mnt/nccl_graph_opt.xml`. `DCP=8` remains the reference alternative. Patched NCCL without XML is functional, but still slower on prefill.

## MTP long-context investigation (WIP, 2026-04-23)

Exploratory debugging session on why MTP collapses at 30k+ context:
- full write-up, benchmarks, patches, and docker image tags: **[kimi-k26-mtp-long-ctx-wip/](kimi-k26-mtp-long-ctx-wip/README.md)**
- TL;DR: root cause is target-forward GPU time at `num_tokens=4` (≈85 ms at 30k ctx with `TRITON_MLA`); the draft/metadata side was ruled out. `FLASHINFER_MLA` would cut target forward to ~19 ms and flatten interarrival across ctx but on SM120 FP8 it silently gives 0 % draft acceptance (XQA can't causally-mask within the 4-query verification span, trtllm-gen not compiled for SM120).
