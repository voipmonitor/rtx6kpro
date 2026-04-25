# Kimi / GLM AR benchmark report - 2026-04-25

This is a standalone benchmark report for Kimi K2.6 and GLM 5.1 runs with
all-reduce variants. It intentionally does not update the main model recipe
pages yet.

## Scope

- Kimi ran on `10.229.14.14`, using the first 8 GPUs on the 16-GPU PCIe-switch server.
- GLM ran locally on the 8-GPU server used for previous GLM tests.
- Both hosts had the NVIDIA P2P override active:
  - `EnableResizableBar: 1`
  - `DmaRemapPeerMmio: 1`
  - `GrdmaPciTopoCheckOverride: 1`
  - `RegistryDwords: "ForceP2P=0x11;RMForceP2PType=1;RMPcieP2PType=2;GrdmaPciTopoCheckOverride=1;EnableResizableBar=1"`
- Decode was measured with `/mnt/llm_decode_bench.py` v0.4.3, Prometheus effective-concurrency checks, `--skip-prefill`, 10 seconds per cell, 512 max generated tokens.
- `N/A(x/y)` means the cell did not actually reach the requested effective concurrency or timed out during warmup. The raw value is not suitable as a clean benchmark number.

## Kimi remote: 10.229.14.14

Runtime:

- Image: `voipmonitor/vllm:kimi-k26-mtp-upstream-stack-pcie-env-test-20260424`
- Model: `moonshotai/Kimi-K2.6`
- Draft: `lightseekorg/kimi-k2.5-eagle3-mla`
- GPUs: first 8 cards only
- Common options: TP=8, MTP/Eagle3, `TRITON_MLA`, target KV `fp8`, draft KV `fp8`, `num_speculative_tokens=3`
- NCCL XML was mounted in all variants: `NCCL_GRAPH_FILE=/mnt/nccl_graph_opt.xml`
- AR-on means `VLLM_ENABLE_PCIE_ALLREDUCE=1`; AR-off means unset.

### Kimi DCP=1, AR-off

| ctx \ conc | 1 | 16 | 64 | 128 |
|---|---:|---:|---:|---:|
| 0 | 157.3 | 906.7 | 1972.9 | 2729.9 |
| 16k | 126.0 | 521.8 | 805.0 | 991.0 |
| 64k | 80.5 | 233.1 | 364.6 | 395.6 |
| 128k | 56.2 | 157.0 | 155.4 | N/A(69/128), raw 151.7 |

### Kimi DCP=1, AR-on

| ctx \ conc | 1 | 16 | 64 | 128 |
|---|---:|---:|---:|---:|
| 0 | 138.7 | 825.4 | 1485.0 | 1746.1 |
| 16k | 108.5 | 490.8 | 731.1 | 860.6 |
| 64k | 78.2 | 228.8 | 327.2 | 370.9 |
| 128k | 58.8 | 160.4 | 157.9 | N/A(84/128), raw 186.1 |

### Kimi DCP=8, AR-off

| ctx \ conc | 1 | 16 | 64 | 128 |
|---|---:|---:|---:|---:|
| 0 | 110.6 | 621.5 | 1104.2 | 1210.5 |
| 16k | 102.0 | 496.4 | 990.2 | 1202.3 |
| 64k | 84.3 | 316.9 | 409.6 | 634.6 |
| 128k | 66.3 | 242.5 | 194.1 | N/A(76/128), raw 272.6 |

### Kimi DCP=8, AR-on

| ctx \ conc | 1 | 16 | 64 | 128 |
|---|---:|---:|---:|---:|
| 0 | 99.5 | 562.8 | 903.1 | 1121.4 |
| 16k | 93.9 | 490.1 | 864.7 | 1022.4 |
| 64k | 68.6 | 309.9 | 538.6 | 500.4 |
| 128k | 66.9 | 241.0 | 358.8 | N/A(81/128), raw 118.8 |

### Kimi decision

- On this PCIe-switch server, DCP=1 should use AR-off. It is clearly faster at short and medium contexts and only marginally worse at low-concurrency 128k.
- For DCP=8, AR-off is better for 0/16k and most high-throughput cases. AR-on only clearly wins at `64k/C64` and `128k/C64`.
- Practical default for this server: use AR-off unless specifically optimizing DCP=8 long-context C64-style traffic, where AR-on should be re-tested against the actual workload.
- DCP=8 is still useful for long-context single-request decode: at 128k/C1 it reaches about 66 tok/s versus DCP=1 at about 56-59 tok/s.

## GLM local SGLang NSA

Runtime:

- Image: `voipmonitor/sglang:luke-main-f4b7830-b12x096-20260419`
- Model: `lukealonso/GLM-5.1-NVFP4`
- `mem_fraction_static=0.76`
- `cuda_graph_max_bs=16`
- `--attention-backend nsa`
- `--nsa-prefill-backend b12x`
- `--nsa-decode-backend b12x`
- `--moe-runner-backend b12x`
- `--fp4-gemm-backend b12x`
- `--speculative-algorithm EAGLE`
- AR-on means `--enable-pcie-oneshot-allreduce`; AR-off means omitted.

### GLM prefill, Prometheus exact counters

SGLang `/tokenize` rejected chat-style tokenize requests, so prompt sizing used approximate chars/token fallback. Prefill throughput is still taken from Prometheus prompt-token deltas and is based on actual prompt tokens, not labels.

| variant | 8k actual tokens | 8k tok/s | 32k actual tokens | 32k tok/s | 128k actual tokens | 128k tok/s |
|---|---:|---:|---:|---:|---:|---:|
| AR-off | 4,800 | 2,655 | 18,947 | 2,109 | 75,554 | 1,541 |
| AR-on | 4,801 | 2,649 | 18,948 | 2,121 | 75,555 | 1,542 |

The first 8k sample in both variants was a cold/compile outlier and was excluded.

### GLM decode

| variant | ctx \ conc | 1 | 4 | 16 | 30 |
|---|---|---:|---:|---:|---:|
| AR-off | 0 | 87.1 | 222.8 | 539.3 | 316.1 |
| AR-off | 16k | 59.2 | 195.4 | - | - |
| AR-off | 64k | 58.4 | - | - | - |
| AR-off | 128k | N/A(1/1), raw 46.8 | - | - | - |
| AR-on | 0 | 80.8 | N/A(4/4), raw 254.3 | 535.4 | 177.6 |
| AR-on | 16k | 68.7 | 205.1 | - | - |
| AR-on | 64k | 61.9 | - | - | - |
| AR-on | 128k | N/A(1/1), raw 54.5 | - | - | - |

### GLM decision

- Prefill is essentially unchanged by SGLang oneshot AR.
- Decode is mixed:
  - AR-off is better for short-context throughput, especially 0/C30.
  - AR-on is slightly better for 16k/64k single-request and 16k/C4.
  - 128k/C1 did not pass effective-concurrency/warmup validation in either variant, so do not use that cell as a clean benchmark.
- For GLM NSA on this local box, AR-off is the safer default for throughput testing unless the target workload is long-context low-concurrency decode.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` should not be used with this SGLang b12x setup; earlier it broke pcie_oneshot graph/cuda-ipc capture. This successful run omitted it.

## GLM local SGLang NSA: Luke c314a13 + b12x 0.10.0 retest

Runtime:

- Image: `voipmonitor/sglang:luke-main-c314a13-b12x0100-glm-nsa-20260425`
- SGLang: `0.5.10.post2.dev584+gc314a13f7`
- `b12x`: `0.10.0`
- Model: `lukealonso/GLM-5.1-NVFP4`
- `cuda_graph_max_bs=16`
- Complete results below use `mem_fraction_static=0.70`.

The first attempt with `mem_fraction_static=0.76` failed with CUDA OOM during draft/speculative MoE prefill. The failing allocation was a 768 MiB scratch buffer while each GPU had only about 0.4 GiB free, so the retest was repeated at `mem_fraction_static=0.70`.

### Updated GLM prefill, Prometheus counters

SGLang `/tokenize` still rejected the benchmark's tokenize request and prompt sizing used the approximate chars/token fallback. Throughput is from Prometheus prompt-token deltas and actual prompt token counts.

The first 8k sample in both variants was a cold/compile outlier and was excluded. 32k and 128k still showed large first-sample variance, so both the median summary and the second warm sample are shown.

| variant | 8k actual tokens | 8k tok/s | 32k actual tokens | 32k median tok/s | 32k warm tok/s | 128k actual tokens | 128k median tok/s | 128k warm tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AR-off | 4,800 | 2,740 | 18,947 | 1,808 | 2,619 | 75,554 | 1,943 | 2,204 |
| AR-on | 4,798 | 2,745 | 18,945 | 1,803 | 2,614 | 75,552 | 1,940 | 2,202 |

### Updated GLM decode

| variant | ctx \ conc | 1 | 4 | 16 | 30 |
|---|---|---:|---:|---:|---:|
| AR-off | 0 | 92.7 | 253.3 | 536.3 | 242.6 |
| AR-off | 16k | 75.5 | 202.5 | - | - |
| AR-off | 64k | 71.1 | - | - | - |
| AR-off | 128k | skip | - | - | - |
| AR-on | 0 | 89.2 | 251.5 | 525.7 | 260.5 |
| AR-on | 16k | 87.1 | 201.7 | - | - |
| AR-on | 64k | 68.9 | - | - | - |
| AR-on | 128k | skip | - | - | - |

### Updated GLM decision

- The new Luke `c314a13` + `b12x 0.10.0` stack works, but it needs lower `mem_fraction_static` than the old image for this test shape.
- Prefill does not materially change between AR-off and AR-on.
- Decode is also mostly unchanged by oneshot AR:
  - AR-off is slightly better at `0/C1`, `0/C4`, `0/C16`, and `64k/C1`.
  - AR-on is better at `0/C30` and `16k/C1`.
  - `16k/C4` is effectively tied.
- Compared with the previous `f4b7830` + `b12x 0.9.6` run, short-context decode is roughly similar, but the new stack did not produce a clear improvement in this benchmark.
- Do not publish the `0.76` memory setting for this new image without further tuning; it OOMed in the measured configuration.

## Tooling notes

- `/mnt/llm_decode_bench.py` v0.4.3 distinguishes raw throughput from effective-concurrency-valid throughput.
- High-context/high-concurrency cells can now correctly become `N/A` instead of publishing misleading per-request-looking numbers.
- `/mnt/vllm_prefill_prom_bench.py` was adjusted so SGLang can still be measured when its prefill metric appears only after the first request, and when `/tokenize` rejects chat-message payloads.

## Result paths

- Remote Kimi: `10.229.14.14:/mnt/kimi_remote8_bench_20260425_201159`
- Local GLM: `/mnt/glm_local_sglang_bench_20260425_204500`
- Local GLM Luke c314a13 + b12x 0.10.0 retest: `/mnt/glm_luke_c314_b12x0100_bench_mem070_20260425_223046`
