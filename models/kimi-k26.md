# Kimi-K2.6 on 8x RTX PRO 6000 Blackwell

This is the current practical recipe for running `moonshotai/Kimi-K2.6` with
vLLM on 8x RTX PRO 6000 Blackwell / sm120. It uses the Kimi MLA path:

- target model: `moonshotai/Kimi-K2.6`
- draft model for MTP: `lightseekorg/kimi-k2.5-eagle3-mla`
- attention backend: `TRITON_MLA` for target and draft
- KV cache: `fp8`
- tensor parallel: `TP=8`
- decode context parallelism: `DCP=1`, `DCP=4`, or `DCP=8`
- runtime topology: PCIe P2P custom allreduce + NCCL XML graph file

## Contents

- [Quick Start](#quick-start)
- [Choose A Profile](#choose-a-profile)
- [Launch: MTP Enabled](#launch-mtp-enabled)
- [Launch: Maximum KV Cache, No MTP](#launch-maximum-kv-cache-no-mtp)
- [Speed Vs Context Length](#speed-vs-context-length)
- [Expected Decode Throughput](#expected-decode-throughput)
- [Prefill Sanity Checks](#prefill-sanity-checks)
- [NCCL XML Status](#nccl-xml-status)
- [Why The MLA Draft](#why-the-mla-draft)
- [FP8 Tensor Draft Check](#fp8-tensor-draft-check)
- [Current Upstream Patch Stack](#current-upstream-patch-stack)
- [Legacy Kimi-K2.5 Community Image](#legacy-kimi-k25-community-image)

## Quick Start

Use this Docker image:

```bash
docker pull voipmonitor/vllm:kimi-k26-mtp-upstream-stack-pcie-env-test-20260424
```

The image is built from upstream vLLM plus the Kimi/K2.6 MTP patch stack tracked
in:

- vLLM issue: <https://github.com/vllm-project/vllm/issues/40608>
- vLLM draft PR: <https://github.com/vllm-project/vllm/pull/40750>
- detailed work log: [kimi-k26-mtp-long-ctx-wip/](kimi-k26-mtp-long-ctx-wip/README.md)

The current validated startup profile is `DCP=8 + MTP + XML` on port `5002`:

```text
GPU KV cache size:           1,753,088 tokens
Maximum concurrency @262144: 6.69x
PCIe custom allreduce:       enabled on all 8 workers
TRITON_MLA:                  target and draft
```

## Choose A Profile

| Goal | DCP | MTP | max model len | Why |
|---|---:|---:|---:|---|
| Fastest single-stream / short-ctx speed | 1 | on | 150k or 262k | highest C=1 decode speed; smallest KV pool |
| Balanced long-ctx + concurrency | 4 | on | 262k | good speed with much larger KV pool |
| Maximum KV cache with MTP | 8 | on | 262k | best MTP capacity; slightly lower single-stream speed |
| Maximum KV cache / many short requests | 8 | off | 262k | no draft model VRAM; largest KV pool |

For most community validation start with `DCP=8 + MTP + --max-model-len 262144`.
For latency or single-stream comparisons also test `DCP=1 + MTP`. For maximum
batch capacity remove the speculative config and use `DCP=8`.

## Launch: MTP Enabled

Set the profile variables first:

```bash
export IMAGE=voipmonitor/vllm:kimi-k26-mtp-upstream-stack-pcie-env-test-20260424
export PORT=5002
export DCP=8
export MAX_MODEL_LEN=262144
export MAX_NUM_BATCHED_TOKENS=8192
export MAX_NUM_SEQS=128
```

Then start vLLM:

```bash
docker run --rm --gpus all --network host --ipc host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /mnt/nccl_graph_opt.xml:/mnt/nccl_graph_opt.xml:ro \
  --entrypoint /bin/bash \
  "$IMAGE" \
  -lc "VLLM_ENABLE_PCIE_ALLREDUCE=1 \
NCCL_P2P_LEVEL=SYS \
NCCL_GRAPH_FILE=/mnt/nccl_graph_opt.xml \
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
VLLM_TEST_FORCE_FP8_MARLIN=1 \
VLLM_MARLIN_USE_ATOMIC_ADD=1 \
VLLM_MARLIN_INPUT_DTYPE=fp8 \
VLLM_LOG_STATS_INTERVAL=1 \
/opt/venv/bin/vllm serve moonshotai/Kimi-K2.6 \
  --served-model-name Kimi-K2.6 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port ${PORT} \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 1 \
  --decode-context-parallel-size ${DCP} \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --load-format fastsafetensors \
  --async-scheduling \
  --gpu-memory-utilization 0.94 \
  --max-model-len ${MAX_MODEL_LEN} \
  --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \
  --max-num-seqs ${MAX_NUM_SEQS} \
  --mm-processor-cache-gb 0 \
  --mm-encoder-tp-mode weights \
  --attention-backend TRITON_MLA \
  --kv-cache-dtype fp8 \
  --tool-call-parser kimi_k2 \
  --enable-auto-tool-choice \
  --reasoning-parser kimi_k2 \
  --speculative-config '{\"model\":\"lightseekorg/kimi-k2.5-eagle3-mla\",\"method\":\"eagle3\",\"num_speculative_tokens\":3,\"draft_attention_backend\":\"TRITON_MLA\",\"draft_kv_cache_dtype\":\"fp8\",\"rejection_sample_method\":\"probabilistic\"}'"
```

Sanity check:

```bash
curl http://127.0.0.1:${PORT}/v1/models
```

Run the benchmark:

```bash
python3 /mnt/llm_decode_bench.py --port ${PORT}
```

## Launch: Maximum KV Cache, No MTP

Use this when you want the largest KV cache and many concurrent requests more
than single-stream MTP latency. It is the same command as above, but remove the
entire `--speculative-config ...` argument.

Recommended variables:

```bash
export DCP=8
export MAX_MODEL_LEN=262144
export MAX_NUM_BATCHED_TOKENS=8192
export MAX_NUM_SEQS=128
```

Expected KV cache at `--max-model-len 262144` from the final benchmark matrix:

| Config | KV tokens | Max concurrency |
|---|---:|---:|
| DCP=8, no MTP | 3,424,256 | 13.03x |
| DCP=8, MTP=3 | 1,769,600 | 6.75x |
| DCP=4, MTP=3 | 1,126,528 | 4.30x |
| DCP=1, MTP=3 | 337,296 | 1.29x |

The current upstream-stack test image reported `1,753,088` KV tokens for
`DCP=8 + MTP=3`, which is effectively the same deployment class as the final
matrix above.

## Speed Vs Context Length

`--max-model-len` is a real performance knob, not just a limit. Larger values
increase the captured block-table stride and cost decode throughput even for
short prompts.

Use this for short-context speed:

```bash
export MAX_MODEL_LEN=150000
export MAX_NUM_BATCHED_TOKENS=4096
export MAX_NUM_SEQS=256
```

Use this for full long-context flexibility:

```bash
export MAX_MODEL_LEN=262144
export MAX_NUM_BATCHED_TOKENS=8192
export MAX_NUM_SEQS=128
```

Measured isolation result from the long-context work: changing only
`--max-model-len` from `262144` to `150000` restored short-context
high-concurrency throughput from about `1397 tok/s` to `1527 tok/s` in the
DCP=8 no-MTP diagnostic run. The trade-off is obvious: lower max request length,
better short-context throughput.

## Expected Decode Throughput

Final image matrix, `llm_decode_bench.py --skip-prefill`, 10 seconds per cell,
aggregate tok/s.

### DCP=1 + MTP=3

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 114.1 | 178.9 | 243.7 | 417.4 | 659.0 | 1055.2 | 1522.0 | 1992.3 |
| 16k | 108.1 | 166.4 | 226.4 | 353.3 | 502.9 | 680.6 | 879.5 | 1080.1 |
| 32k | skip | 151.6 | 189.1 | 293.3 | 396.6 | 495.4 | 580.2 | 605.7 |
| 64k | 85.6 | 136.3 | 157.5 | 222.8 | 286.5 | 354.1 | 420.4 | 384.8 |
| 128k | 66.6 | 105.4 | 121.2 | 152.6 | 186.5 | 211.1 | 189.1 | 374.1 |

### DCP=1, No MTP

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 88.5 | 148.4 | 233.8 | 340.7 | 475.2 | 823.6 | 1330.2 | 2026.3 |
| 16k | 82.6 | 139.1 | 214.6 | 310.2 | 413.8 | 667.9 | 954.0 | 1272.2 |
| 32k | 77.5 | 129.4 | 197.8 | 286.2 | 366.1 | 572.4 | 758.7 | 890.3 |
| 64k | 68.6 | 117.2 | 170.9 | 246.5 | 302.1 | 411.8 | 508.9 | 635.5 |
| 128k | 55.4 | 97.3 | 131.3 | 190.7 | 222.6 | 286.0 | 317.9 | 380.5 |

### DCP=4 + MTP=3

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 98.8 | 160.0 | 211.4 | 363.3 | 576.5 | 853.2 | 1239.1 | 1506.2 |
| 16k | 96.4 | 145.1 | 190.0 | 311.0 | 453.4 | 603.1 | 741.0 | 891.9 |
| 32k | 88.5 | 130.2 | 180.8 | 266.2 | 367.5 | 451.9 | 575.0 | 597.2 |
| 64k | 76.5 | 114.3 | 151.9 | 205.4 | 267.5 | 303.4 | 404.9 | 432.4 |
| 128k | 67.6 | 87.4 | 112.4 | 152.0 | 175.8 | 213.5 | 247.7 | 383.9 |

### DCP=4, No MTP

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 74.5 | 122.5 | 205.2 | 308.2 | 426.6 | 726.2 | 1136.1 | 1640.3 |
| 16k | 73.5 | 117.3 | 193.5 | 278.5 | 365.8 | 600.0 | 827.1 | 1145.0 |
| 32k | 71.5 | 113.5 | 182.8 | 254.7 | 334.0 | 509.1 | 694.1 | 882.4 |
| 64k | 66.7 | 105.3 | 163.0 | 222.7 | 270.3 | 379.0 | 508.8 | 629.3 |
| 128k | 60.3 | 92.8 | 135.1 | 174.7 | 206.7 | 254.6 | 318.0 | 379.7 |

### DCP=8 + MTP=3

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 90.4 | 142.7 | 176.1 | 311.3 | 498.3 | 641.2 | 908.0 | 1137.0 |
| 16k | 85.4 | 131.2 | 186.9 | 297.1 | 466.0 | 590.7 | 744.4 | 861.0 |
| 32k | 87.4 | 121.1 | 163.9 | 275.8 | 400.2 | 520.8 | 699.6 | 828.2 |
| 64k | 79.3 | 114.2 | 156.1 | 237.3 | 340.5 | 432.9 | 560.7 | 581.7 |
| 128k | 75.4 | 99.4 | 135.2 | 197.9 | 252.2 | 312.5 | 417.6 | 416.2 |

### DCP=8, No MTP

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 74.1 | 113.4 | 185.8 | 276.6 | 363.5 | 600.9 | 948.4 | 1260.6 |
| 16k | 73.6 | 111.2 | 178.7 | 262.2 | 333.9 | 571.3 | 826.7 | 1137.4 |
| 32k | 73.0 | 109.3 | 174.8 | 245.0 | 317.7 | 509.3 | 757.7 | 1008.6 |
| 64k | 69.7 | 104.7 | 162.9 | 229.1 | 301.8 | 445.7 | 632.2 | 757.0 |
| 128k | 65.6 | 95.5 | 146.8 | 197.5 | 252.9 | 350.0 | 476.8 | 635.3 |

Interpretation:

- MTP wins single-stream at every tested context length.
- At high concurrency and short context, no-MTP can be faster because the GPU is
  already saturated and speculative verification adds work.
- `VLLM_SPECULATIVE_DISABLE_ABOVE_SEQ_LEN=7000` is obsolete for this image. Do
  not use it for the Kimi-K2.6 MTP path.

## Prefill Sanity Checks

DCP=1 + MTP=3 prefill, C=1:

| ctx | prompt tokens | TTFT | prefill tok/s |
|---|---:|---:|---:|
| 8k | 5,312 | 0.71s | 7,892 |
| 16k | 10,480 | 1.44s | 7,478 |
| 32k | 20,822 | 2.94s | 7,177 |
| 64k | 41,505 | 6.48s | 6,449 |
| 128k | 82,845 | 14.82s | 5,604 |

If your numbers are much lower, first check:

- `NCCL_GRAPH_FILE=/mnt/nccl_graph_opt.xml` is mounted and set.
- `VLLM_ENABLE_PCIE_ALLREDUCE=1` is present.
- logs say `PCIe custom allreduce enabled via VLLM_ENABLE_PCIE_ALLREDUCE=1`.
- logs say `Using AttentionBackendEnum.TRITON_MLA backend`.
- `--max-model-len` matches the profile you are comparing against.

## NCCL XML Status

The public recipe still uses:

```bash
NCCL_P2P_LEVEL=SYS
NCCL_GRAPH_FILE=/mnt/nccl_graph_opt.xml
```

Without the XML, NCCL historically picked a much worse ring graph on this Turin
system. The bad no-XML path showed single cold 8k prefill around `876 tok/s`,
while XML was around `7478 tok/s`.

There is now an upstream NCCL draft PR for the no-XML topology issue:

- <https://github.com/NVIDIA/nccl/pull/2127>

With that NCCL fix, the targeted single cold 8k prefill reproducer reached
`7455 tok/s`, effectively matching XML. Until that fix is released in a normal
NCCL package, keep the XML file in the public recipe.

## Why The MLA Draft

Use `lightseekorg/kimi-k2.5-eagle3-mla` rather than a Llama-style Eagle draft
because it exercises the same MLA runtime path as Kimi:

- `TRITON_MLA` target and draft
- fp8 KV cache target and draft
- DCP correctness path
- Kimi tool-call and reasoning parsers

The non-MLA Llama draft is useful for separate debugging, but it is not the
representative public Kimi serving path.

## FP8 Tensor Draft Check

An experimental locally quantized draft was tested against the public
`lightseekorg/kimi-k2.5-eagle3-mla` draft to decide whether the FP8 tensor draft
should become the default recipe.

Common server settings:

```text
target model:              moonshotai/Kimi-K2.6
TP / DCP:                  8 / 1
attention backend:         TRITON_MLA for target and draft
target KV cache dtype:     fp8
draft KV cache dtype:      fp8
speculative tokens:        3
rejection sample method:   probabilistic
max_num_batched_tokens:    8192
gpu_memory_utilization:    0.97
NCCL graph file:           /mnt/nccl_graph_opt.xml
```

Only the draft model changed:

| Variant | Draft model |
|---|---|
| Public draft | `lightseekorg/kimi-k2.5-eagle3-mla` |
| Local FP8 tensor draft | `/mnt/kimi-k2.5-eagle3-mla-fp8-tensor` |

Benchmark command:

```bash
python3 /mnt/llm_decode_bench.py \
  --port 5002 \
  --model Kimi-K2.6 \
  --concurrency 1,16,64 \
  --contexts 0,16k,32k,64k,128k \
  --duration 20 \
  --max-tokens 512 \
  --skip-prefill
```

Manual KV budgets were set from the vLLM startup log:

| Variant | GPU KV cache size |
|---|---:|
| Public draft | 424,576 tokens |
| Local FP8 tensor draft | 431,328 tokens |

Decode result, aggregate tok/s:

| ctx / conc | Local FP8 tensor draft | Public draft | Delta |
|---|---:|---:|---:|
| 0 / 1 | 129.3 | 114.4 | +13.0% |
| 0 / 16 | 747.5 | 758.5 | -1.4% |
| 0 / 64 | 1603.3 | 1614.5 | -0.7% |
| 16k / 1 | 119.1 | 122.2 | -2.6% |
| 16k / 16 | 524.8 | 523.2 | +0.3% |
| 32k / 1 | 102.4 | 105.4 | -2.8% |
| 64k / 1 | 92.4 | 89.5 | +3.2% |
| 128k / 1 | 70.5 | 72.0 | -2.0% |

Conclusion: the local FP8 tensor draft is not a better default. It only clearly
wins the `0ctx / C=1` cell, is otherwise within noise or slightly slower, and
adds only about 1.6% more KV cache. The default community recipe should keep the
public `lightseekorg/kimi-k2.5-eagle3-mla` draft with `draft_kv_cache_dtype=fp8`.
The local FP8 tensor draft can remain an experimental option for local testing.

## Current Upstream Patch Stack

The current reconstruction branch includes:

- `vllm-project/vllm#39633`: explicit PCIe custom-allreduce opt-in
- `vllm-project/vllm#40609`: MLA + DCP + fp8 KV support
- `vllm-project/vllm#40610`: async proposer synchronization fix
- `vllm-project/vllm#40611`: draft-specific attention backend and KV dtype
- `vllm-project/vllm#40750`: TRITON_MLA full CUDA graphs, DCP correctness,
  batch-aware KV split selection, and sm120/fp8 tuning table

The draft PR is intentionally a runnable consolidation branch. It can be split
for upstream review after the end-to-end recipe is fully validated.

## Legacy Kimi-K2.5 Community Image

The older public Kimi-K2.5 image remains useful for historical comparison:

```bash
voipmonitor/vllm:kimi-k25-eagle3mla-nccl2297-community-20260422
```

Do not use that image as the current Kimi-K2.6 recipe. It predates the final
Kimi-K2.6 MTP full-CG path, still documented the old speculation kill-switch,
and used older benchmark assumptions.
