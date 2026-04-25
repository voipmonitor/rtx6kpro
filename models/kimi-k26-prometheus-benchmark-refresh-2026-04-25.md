# Kimi-K2.6 Prometheus Benchmark Refresh - 2026-04-25

This is a side report for the benchmark methodology refresh before replacing
the published tables in `models/kimi-k26.md`.

## What Changed

Benchmark tooling changed, not the vLLM/SGLang server code in this pass:

- `llm_decode_bench.py` is now `0.4.2-prom-decode-warmup`.
- Decode throughput for vLLM now uses the exact delta of
  `vllm:generation_tokens`/`vllm:generation_tokens_total` over the measured
  window instead of the median of 1-second samples.
- Decode requests now send `ignore_eos=true` by default. Use `--respect-eos`
  for application-level latency tests. Benchmarking decode with EOS enabled was
  contaminated by repeated short completions.
- Decode matrices can now use `--decode-warmup-seconds 20`. This runs a hidden
  decode cell before the recorded matrix. It fixes the first-cell cold-start
  artifact seen on Kimi MTP/EAGLE where `0/C1` was under-reported.
- Prompt sizes are now targeted through `/tokenize` when available, so `16k`
  means the actual API prompt is 16,384 tokens, not an approximate character
  count.
- The old TTFT-subtraction prefill table is no longer trusted. Exact prefill is
  measured separately through Prometheus counters:
  `vllm:prompt_tokens` and `vllm:request_prefill_time_seconds`.
- The terminal colors changed to a bright yellow/blue/magenta palette so new
  screenshots are visually distinguishable from old screenshots.
- SGLang metric names are now recognized in the helper scripts. SGLang prefill
  should use `sglang:prompt_tokens_total` and
  `sglang:per_stage_req_latency_seconds{stage="prefill_forward"}`. SGLang
  decode needs extra care because `sglang:generation_tokens_total` is emitted
  on finished requests rather than as a clean live in-window counter.

## Hardware/Driver State

Initial NVIDIA module state before the forced-P2P test:

```text
EnableResizableBar: 0
DmaRemapPeerMmio: 1
RegistryDwords: ""
RegistryDwordsPerDevice: ""
```

Forced P2P was then enabled without reboot by installing:

```bash
cp /root/nvidia-p2p-override.conf /etc/modprobe.d/nvidia-p2p-override.conf
modprobe -r nvidia_uvm nvidia_modeset nvidia
modprobe nvidia
modprobe nvidia_uvm
```

Active state after the reload:

```text
EnableResizableBar: 1
DmaRemapPeerMmio: 1
GrdmaPciTopoCheckOverride: 1
RegistryDwords: "ForceP2P=0x11;RMForceP2PType=1;RMPcieP2PType=2;GrdmaPciTopoCheckOverride=1;EnableResizableBar=1"
RegistryDwordsPerDevice: ""
```

## Kimi Runtime

Common runtime for the new Kimi measurements:

```text
image:                  voipmonitor/vllm:kimi-k26-mtp-upstream-stack-pcie-env-test-20260424
target model:           moonshotai/Kimi-K2.6
draft model:            lightseekorg/kimi-k2.5-eagle3-mla
TP / DCP:               8 / 1, 4, or 8
speculative method:     eagle3
speculative tokens:     3
attention backend:      TRITON_MLA
target KV dtype:        fp8
draft KV dtype:         fp8
max_model_len:          262144
max_num_batched_tokens: 8192
max_num_seqs:           128
gpu_memory_utilization: 0.94
Marlin FP8 force envs:  disabled
```

Best measured DCP=1 + MTP=3 profile without forced-P2P modprobe:

```bash
unset VLLM_ENABLE_PCIE_ALLREDUCE
export NCCL_P2P_DISABLE=1
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=32
export NCCL_BUFFSIZE=67108864
export CUDA_DEVICE_MAX_CONNECTIONS=64
export NCCL_P2P_LEVEL=SYS
export NCCL_GRAPH_FILE=/mnt/nccl_graph_opt.xml
export VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
```

Best measured DCP=1 + MTP=3 profile with forced-P2P modprobe active:

```bash
unset VLLM_ENABLE_PCIE_ALLREDUCE
export NCCL_P2P_DISABLE=0
unset NCCL_MIN_NCHANNELS
unset NCCL_MAX_NCHANNELS
unset NCCL_BUFFSIZE
export CUDA_DEVICE_MAX_CONNECTIONS=32
export NCCL_P2P_LEVEL=SYS
export NCCL_GRAPH_FILE=/mnt/nccl_graph_opt.xml
export VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
```

For DCP=1, forced-P2P modprobe makes custom AR safe, but not necessary. Keep
`VLLM_ENABLE_PCIE_ALLREDUCE` unset unless a DCP=4/DCP=8 run proves otherwise.

## Exact Prefill, DCP=1 + MTP=3

Measured with `/mnt/vllm_prefill_prom_bench.py`, 2 valid samples per context.

| Context | Prompt tokens | Median prefill tok/s |
|---:|---:|---:|
| 8k | 8,192 | 7,283 |
| 16k | 16,384 | 7,049 |
| 32k | 32,768 | 6,604 |
| 64k | 65,536 | 5,852 |
| 128k | 131,072 | 4,766 |

Raw result:

```text
/mnt/kimi_prom_bench_20260425/kimi_docimage_dcp1_mtp3_opt_prefill.json
```

## Exact Decode Matrix, DCP=1 + MTP=3

Measured with `/mnt/llm_decode_bench.py --skip-prefill`, 10 seconds per cell,
`ignore_eos=true`.

Important correction: this matrix was captured before the hidden decode warmup
option existed. The first cell, `0/C1=76.2`, is a cold-start artifact. Reruns of
that single cell after warmup measured `110.0 tok/s` over 30 seconds and
`123.2 tok/s` over 10 seconds. New matrices should be generated with
`--decode-warmup-seconds 20`.

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 76.2* | 198.3 | 355.9 | 527.5 | 781.0 | 1100.8 | 1749.6 | 2235.4 |
| 16k | 101.6 | 171.1 | 270.1 | 373.1 | 468.5 | 597.8 | 786.3 | 980.4 |
| 32k | 95.3 | 158.8 | 220.9 | 286.8 | 345.9 | 430.5 | 538.5 | 624.1 |
| 64k | 78.8 | 122.9 | 160.7 | 191.9 | 228.4 | 282.5 | 345.4 | 349.0 |
| 128k | 52.5 | 81.1 | 103.6 | 126.0 | 146.1 | 190.5 | 176.8 | 62.2 |

The `128k / C=128` cell is capacity/pressure limited in practice:
`avg_running_reqs=118.5` rather than 128. Treat it as unstable and do not use it
as a headline number.

Raw result:

```text
/mnt/kimi_prom_bench_20260425/kimi_docimage_dcp1_mtp3_opt_decode_matrix.json
```

## Exact DCP=4 and DCP=8, MTP=3

Measured after forced-P2P modprobe was active, with:

```bash
unset VLLM_ENABLE_PCIE_ALLREDUCE
export NCCL_P2P_DISABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=32
export NCCL_P2P_LEVEL=SYS
export NCCL_GRAPH_FILE=/mnt/nccl_graph_opt.xml
```

The same `voipmonitor/vllm:kimi-k26-mtp-upstream-stack-pcie-env-test-20260424`
image was used. For DCP=4/DCP=8, the runtime DCP/XML guard from
`/root/vllm/runtime_dcp_xml_patch` was bind-mounted over the three distributed
communicator files. Without that guard, DCP=4 startup failed during DCP group
creation with `NCCL error: unhandled system error`, because the full 8-GPU XML
graph was applied to the smaller DCP subgroup.

Startup KV cache from vLLM logs:

| DCP | GPU KV cache size | Max concurrency at 262,144 tokens |
|---:|---:|---:|
| 4 | 1,115,136 tokens | 4.25x |
| 8 | 1,753,088 tokens | 6.69x |

### DCP=4 Prefill

Measured with `/mnt/vllm_prefill_prom_bench.py`, 2 warm samples per context.

| Context | Prompt tokens | Median prefill tok/s |
|---:|---:|---:|
| 8k | 8,192 | 7,646 |
| 16k | 16,384 | 7,343 |
| 32k | 32,768 | 6,803 |
| 64k | 65,536 | 5,913 |
| 128k | 131,072 | 4,685 |

Raw result:

```text
/mnt/kimi_prom_bench_20260425/kimi_modp2p_dcp4_mtp3_aroff_ncclp2pon_prefill_warm.json
```

### DCP=4 Decode Matrix

Measured with `/mnt/llm_decode_bench.py --skip-prefill`, 10 seconds per cell,
`ignore_eos=true`, `--decode-warmup-seconds 20`.

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 105.3 | 174.2 | 277.4 | 423.1 | 632.1 | 881.4 | 1337.1 | 1825.3 |
| 16k | 88.5 | 142.5 | 225.5 | 315.0 | 428.6 | 544.1 | 762.3 | 1069.5 |
| 32k | 83.7 | 133.2 | 186.4 | 244.2 | 340.8 | 420.8 | 583.3 | 740.6 |
| 64k | 80.4 | 115.3 | 143.9 | 181.7 | 221.3 | 299.4 | 386.5 | 444.2 |
| 128k | 62.1 | 73.6 | 91.4 | 121.9 | 147.6 | 190.8 | 218.8 | 35.5* |

`128k / C=128` is capacity-limited: the benchmark requested 128 concurrent
requests, but vLLM averaged 116 running requests. Treat that cell as invalid for
headline throughput.

Raw result:

```text
/mnt/kimi_prom_bench_20260425/kimi_modp2p_dcp4_mtp3_aroff_ncclp2pon_decode_matrix.json
```

### DCP=8 Prefill

Measured with `/mnt/vllm_prefill_prom_bench.py`, 2 warm samples per context.

| Context | Prompt tokens | Median prefill tok/s |
|---:|---:|---:|
| 8k | 8,192 | 7,639 |
| 16k | 16,384 | 7,333 |
| 32k | 32,768 | 6,779 |
| 64k | 65,536 | 5,883 |
| 128k | 131,072 | 4,643 |

Raw result:

```text
/mnt/kimi_prom_bench_20260425/kimi_modp2p_dcp8_mtp3_aroff_ncclp2pon_prefill_warm.json
```

### DCP=8 Decode Matrix

Measured with `/mnt/llm_decode_bench.py --skip-prefill`, 10 seconds per cell,
`ignore_eos=true`, `--decode-warmup-seconds 20`.

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 98.6 | 148.3 | 233.1 | 358.9 | 553.9 | 645.3 | 959.9 | 1317.2 |
| 16k | 91.9 | 145.6 | 217.4 | 316.5 | 473.9 | 554.2 | 832.3 | 1115.9 |
| 32k | 86.7 | 137.1 | 196.7 | 278.7 | 399.5 | 499.8 | 742.0 | 911.8 |
| 64k | 79.7 | 124.8 | 169.7 | 236.9 | 314.0 | 401.3 | 515.5 | 664.2 |
| 128k | 68.1 | 100.0 | 130.0 | 173.2 | 224.1 | 303.9 | 369.9 | 26.2* |

`128k / C=128` is capacity-limited: the benchmark requested 128 concurrent
requests, but vLLM averaged 107.1 running requests and had queue pressure during
90.9% of the measured samples. Treat that cell as invalid for headline
throughput.

Raw result:

```text
/mnt/kimi_prom_bench_20260425/kimi_modp2p_dcp8_mtp3_aroff_ncclp2pon_decode_matrix.json
```

## Communication Variant Matrix

Subset decode cells are aggregate tok/s. Prefill cells are exact Prometheus
prefill tok/s.

The first block was measured before forced P2P was enabled in the NVIDIA module.

| Variant | Prefill 8k | Prefill 32k | Prefill 128k | Decode 0/C1 | Decode 0/C16 | Decode 16k/C1 | Decode 16k/C16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| custom AR on, default NCCL | 6,138 | 5,686 | 4,273 | 28.9 | 140.7 | 26.4 | 140.7 |
| custom AR off, default NCCL | 6,194 | 5,696 | 4,275 | 73.3 | 716.8 | 104.4 | 451.0 |
| custom AR off, P2P off, channels 16/32 | 7,277 | 6,604 | 4,767 | 70.2 | 805.2 | 104.5 | 471.3 |
| custom AR off, P2P off, channels 32/64 | 7,263 | 6,590 | 4,760 | 71.0 | 791.3 | 101.5 | 468.6 |
| custom AR on, P2P off, channels 16/32 | 7,168 | 6,580 | 4,763 | 27.8 | 154.7 | 24.3 | 149.2 |

The second block was measured after the module reload with:

```text
ForceP2P=0x11
RMForceP2PType=1
RMPcieP2PType=2
GrdmaPciTopoCheckOverride=1
EnableResizableBar=1
```

Decode subset command includes `--decode-warmup-seconds 20`, so `0/C1` is not a
first-cell cold-start artifact.

| Variant, forced P2P modprobe ON | Prefill 8k | Prefill 32k | Prefill 128k | Decode 0/C1 | Decode 0/C16 | Decode 16k/C1 | Decode 16k/C16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| custom AR off, NCCL P2P on/default | 7,633 | 6,896 | 4,918 | 125.3 | 779.1 | 109.5 | 468.5 |
| custom AR on, NCCL P2P on/default | 7,645 | 6,902 | 4,921 | 121.6 | 738.6 | 106.7 | 453.4 |
| custom AR off, NCCL P2P off, channels 16/32 | 7,284 | 6,604 | 4,767 | 124.2 | 802.6 | 107.7 | 478.0 |
| custom AR on, NCCL P2P off, channels 16/32 | 7,283 | 6,605 | 4,768 | 128.2 | 755.0 | 114.4 | 442.7 |

Interpretation:

- Without forced P2P in the NVIDIA module, `VLLM_ENABLE_PCIE_ALLREDUCE=1` is
  bad for this DCP=1 + MTP=3 profile. It collapses decode throughput even when
  prefill is good.
- With forced P2P in the NVIDIA module, custom AR no longer collapses decode.
  It is not clearly better for DCP=1, though: `custom AR off, NCCL P2P
  on/default` is the best balanced measured variant because it keeps the higher
  prefill numbers and only slightly trails the best single decode cell.
- `NCCL_P2P_DISABLE=1` plus higher NCCL channels improves prefill materially
  only in the no-forced-P2P state. With forced P2P active, leaving NCCL P2P
  enabled is faster for prefill.
- 16/32 channels and 32/64 channels are effectively tied. 16/32 is the more
  conservative recommendation when `NCCL_P2P_DISABLE=1` is used.
- The prefill improvement is not a measurement artifact: it comes from exact
  `prompt_tokens / request_prefill_time_seconds`, not TTFT subtraction.

Raw variant files:

```text
/mnt/kimi_prom_bench_20260425/kimi_docimage_dcp1_mtp3_ar_on_clean_decode.json
/mnt/kimi_prom_bench_20260425/kimi_docimage_dcp1_mtp3_ar_on_clean_prefill_subset.json
/mnt/kimi_prom_bench_20260425/kimi_docimage_dcp1_mtp3_ar_off_sanity_decode.json
/mnt/kimi_prom_bench_20260425/kimi_docimage_dcp1_mtp3_ar_off_prefill.json
/mnt/kimi_prom_bench_20260425/kimi_docimage_dcp1_mtp3_ar_off_p2pdisable_ch16_32_decode.json
/mnt/kimi_prom_bench_20260425/kimi_docimage_dcp1_mtp3_ar_off_p2pdisable_ch16_32_prefill.json
/mnt/kimi_prom_bench_20260425/kimi_docimage_dcp1_mtp3_ar_off_p2pdisable_ch32_64_decode.json
/mnt/kimi_prom_bench_20260425/kimi_docimage_dcp1_mtp3_ar_off_p2pdisable_ch32_64_prefill.json
/mnt/kimi_prom_bench_20260425/kimi_docimage_dcp1_mtp3_ar_on_p2pdisable_ch16_32_decode.json
/mnt/kimi_prom_bench_20260425/kimi_docimage_dcp1_mtp3_ar_on_p2pdisable_ch16_32_prefill.json
/mnt/kimi_prom_bench_20260425/kimi_modp2p_aroff_decode_subset.json
/mnt/kimi_prom_bench_20260425/kimi_modp2p_aroff_prefill_subset.json
/mnt/kimi_prom_bench_20260425/kimi_modp2p_aron_decode_subset.json
/mnt/kimi_prom_bench_20260425/kimi_modp2p_aron_prefill_subset.json
/mnt/kimi_prom_bench_20260425/kimi_modp2p_aroff_ncclp2poff_ch16_32_decode_subset.json
/mnt/kimi_prom_bench_20260425/kimi_modp2p_aroff_ncclp2poff_ch16_32_prefill_subset.json
/mnt/kimi_prom_bench_20260425/kimi_modp2p_aron_ncclp2poff_ch16_32_decode_subset.json
/mnt/kimi_prom_bench_20260425/kimi_modp2p_aron_ncclp2poff_ch16_32_prefill_subset.json
```

## AIPerf Check

Repository checked: <https://github.com/ai-dynamo/aiperf>

Installed locally into:

```text
/tmp/aiperf-venv
```

Command used:

```bash
/tmp/aiperf-venv/bin/aiperf profile \
  --model Kimi-K2.6 \
  --url http://127.0.0.1:8000 \
  --endpoint-type chat \
  --streaming \
  --tokenizer moonshotai/Kimi-K2.6 \
  --tokenizer-trust-remote-code \
  --use-server-token-count \
  --use-legacy-max-tokens \
  --isl 16384 \
  --isl-stddev 0 \
  --osl 512 \
  --osl-stddev 0 \
  --concurrency 1 \
  --request-count 2 \
  --extra-inputs '{"ignore_eos": true, "temperature": 0}' \
  --output-artifact-dir /mnt/aiperf_kimi_test_nowarm \
  --export-level records
```

AIPerf result summary:

| Metric | Value |
|---|---:|
| TTFT avg | 2,485 ms |
| Output token throughput per user | 112.7 tok/s/user |
| E2E output token throughput | 72.9 tok/s/user |
| AIPerf prefill throughput per user | 6,595 tok/s/user |
| Server metrics exact prefill | 6,779 tok/s |

The AIPerf built-in prefill metric is `input_tokens / TTFT`, so it is not the
same as exact server prefill. Its exported server metrics can be post-processed:

```text
vllm:prompt_tokens total = 32,784
vllm:request_prefill_time_seconds sum = 4.836299185997632
exact server prefill = 6,779 tok/s
```

Conclusion: AIPerf is useful as an application-level OpenAI-compatible latency
bench and it can collect Prometheus server metrics. It should not replace the
custom Prometheus prefill/decode scripts for the Kimi tables unless the AIPerf
server metrics export is post-processed per isolated cell.

Artifacts:

```text
/mnt/aiperf_kimi_test_nowarm/profile_export_aiperf.json
/mnt/aiperf_kimi_test_nowarm/server_metrics_export.json
/mnt/aiperf_kimi_test/profile_export_aiperf.json
/mnt/aiperf_kimi_test/server_metrics_export.json
```

## Pending Before Updating Public Tables

- Re-run DCP=4 and DCP=8 with the new exact decode/pre-fill methodology before
  replacing their published matrices.
- Re-run no-MTP profiles if the public page should keep both MTP and no-MTP
  recommendations.
- Test the forced P2P modprobe configuration after reboot/driver reload; it is
  not active in the current boot.
- Run the GLM-5.1 SGLang/vLLM comparison with the SGLang Prometheus path before
  updating `models/glm5.1/compare-dense-mla-vs-nsa-benchmark-2026-04-20`.
