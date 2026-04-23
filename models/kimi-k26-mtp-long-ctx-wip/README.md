# MTP long-context debugging: findings and recommendations

## Status (read first)

- **No throughput win from this session.** On the user's exact single-request 30k ctx workload, MTP-on tok/s stays at ~25 with or without any of my changes.
- **Root cause identified:** target forward with `num_tokens=4` (1 real + 3 spec) at 30k ctx takes **85 ms of GPU time**, vs 13 ms at `num_tokens=1`. That 85 ms is the iter budget. Draft compute is ~2.5 ms; metadata sync was ~80 ms but overlapped with GPU already, so removing it did not help a single request.
- **Clear next step identified:** FLASHINFER_MLA would give target forward at ~19 ms and flat tok/s across ctx — but on this FlashInfer build for sm120 FP8, spec-verification drops draft acceptance to 0 % (the XQA backend is the only shipped fp8 MLA path, and it does not causally-mask within the 4-query verification span). See Next Steps §1 for what would need to change upstream.
- **Committed images:**
  - `voipmonitor/vllm:cu130-mtp-baseline-20260423` — the container as I found it, pre-changes.
  - `voipmonitor/vllm:cu130-mtp-fix-v2-20260423` — current container with safe, behaviour-neutral changes: defensive CPU-path in MLA builder (falls back to original sync when CPU shadows unavailable — same as baseline in async spec mode), timing module for future profiling, instrumentation hooks in eagle.py (no-op unless `MTP_TIMING=1`). **Expected tok/s: identical to baseline.**
  - The earlier `voipmonitor/vllm:cu130-mtp-fix-v1-20260423` image had an overeager `gpu_model_runner.py` change that introduced a 2k-ctx regression; it was reverted.

## TL;DR

The user's exact config (TRITON_MLA + eagle3 + fp8 KV + 8× Blackwell sm120) produces **25 tok/s at 30k ctx with MTP on** vs **75 tok/s at 30k without MTP**. sglang is reported at 70 tok/s with MTP at similar ctx.

Root cause is not the TRITON_MLA decode kernel (benchmarked at 56 μs/call at 30k B=1) nor draft compute (draft is 1-layer). It is:

1. **TRITON_MLA's metadata builder enters the expensive prefill branch any time `max_query_len > 1`** (i.e., target is verifying spec tokens, or draft first-pass processes accepted+new tokens). Inside that branch `compute_num_computed_tokens().cpu()` forces a GPU→CPU sync that stalls on all in-flight target kernels — ~40-80 ms per decode step at 30k context. (**Fixed: +0 tok/s on single-req benchmark but frees CPU for batched workloads.**)

2. **Target forward itself takes ~85 ms of GPU time at `num_tokens=4` at 30k context** vs ~13 ms at `num_tokens=1`. Attention at B=4 is only ~8 ms total across 61 layers — the remaining 77 ms is MoE/MLP. This is the dominant cost and is not fixable without kernel work or switching attention backend.

3. **FLASHINFER_MLA would deliver FULL cudagraph** (the only MLA backend on sm120 that supports `UNIFORM_BATCH`, which covers spec verification). In our test it reduces target forward time to ~19 ms and flattens interarrival at 19 ms regardless of context (confirmed at 2k/10k/30k). **BUT it silently produces 0 % draft-token acceptance with eagle3 on sm120**, so the apparent "2× speedup" is because target just runs at B=4 without benefit from spec. Needs a correctness fix before it can ship.

## Reproduction table

All benches: single request, `test.py`-style padding, 200-500 output tokens, after my sync fix is applied.

| backend        | ctx    | MTP | tok/s | interarrival p50 | accept | notes |
|----------------|-------:|:---:|------:|-----------------:|:------:|-------|
| TRITON_MLA     |  2 000 | off |  74.6 |             13.0 |   n/a  | baseline |
| TRITON_MLA     | 10 000 | off |  73.6 |             13.6 |   n/a  | baseline |
| TRITON_MLA     | 30 000 | off |  75.5 |             13.3 |   n/a  | baseline |
| TRITON_MLA     |  2 000 | on  |  75.0 |             31.4 | 50-70% | MTP helps |
| TRITON_MLA     | 10 000 | on  |  32.2 |             71.2 | ~45%  | MTP hurts|
| TRITON_MLA     | 30 000 | on  |  25.3 |             92.4 | ~45%  | MTP hurts badly |
| TRITON_MLA (num_spec=1) | 30 000 | on | 18.8 |      85.9 | ~50% | even worse |
| FLASHINFER_MLA |  2 000 | on  |  52.2 |             18.5 | **0%** | target runs fast but spec broken |
| FLASHINFER_MLA | 10 000 | on  |  52.7 |             18.9 | **0%** | flat interarrival (FULL CG works) |
| FLASHINFER_MLA | 30 000 | on  |  53.6 |             19.1 | **0%** | 2× target speedup but useless |

## Findings

### 1. Microbenchmark: TRITON_MLA kernel is fine

Synthetic FP8 MLA decode kernel sweep (per-rank shape, q_heads=8, block_size=64):

| seq_len | batch | splits | median ms |
|--------:|------:|-------:|----------:|
|   1 000 |     1 |      1 |     0.083 |
|  10 000 |     1 |     32 |     0.038 |
|  30 000 |     1 |     64 |     0.056 |
| 100 000 |     1 |    128 |     0.096 |
|  30 000 |     4 |     64 |     0.134 |
|  30 000 |     8 |     64 |     0.280 |

So 61 target layers × 0.134 ms ≈ 8 ms of MLA attention at B=4, 30k. Draft is 1 layer → 56 μs × 3 steps = 170 μs.

### 2. Instrumented trace: 80 ms/iter was in a GPU→CPU sync

Added `_mtp_timing.py` + enter/exit wrappers in propose() and MLACommonMetadataBuilder.build(). Per 50 propose calls at ctx≈18k (BEFORE any fix):

```
00_propose_total              GPU  300 ms      CPU 4415 ms (88 ms/iter)
03_build_attn_metadata_first  GPU   10 ms      CPU 4199 ms (84 ms/iter)
```

Narrowed with finer wrappers:
```
P00_prefill_branch_total                 CPU 42 ms/iter   ← prefill-like branch taken because max_query_len=4>threshold=1
P00a_fallback_cpu_sync                   CPU 41 ms/iter   ← compute_num_computed_tokens().cpu() sync
```

In async-spec-decode mode, `gpu_model_runner` **sets `_seq_lens_cpu` and `_num_computed_tokens_cpu` to None** ("GPU tensors are authoritative") — so the MLA builder always hits the `.cpu()` path because the CPU shadows aren't there to derive from.

### 3. Fix 1 (applied): avoid the `.cpu()` sync in the prefill branch

Files (in container `73f963a03703` at `/opt/vllm/…`):

- **`vllm/model_executor/layers/attention/mla_attention.py` ~line 1745**: when `_num_computed_tokens_cpu` is available, use it; else when `_seq_lens_cpu` is available, derive `num_computed_tokens_cpu = _seq_lens_cpu − query_seq_lens_cpu` on CPU; else fall back to the old sync path.
- **`vllm/v1/worker/gpu_model_runner.py` ~line 2186**: in `use_async_spec_decode` mode, **keep `_seq_lens_cpu = self.optimistic_seq_lens_cpu[:num_reqs_padded]`** instead of None; only `_num_computed_tokens_cpu` is nulled. The CPU-derivation path in MLA now works.

After Fix 1:
```
03_build_attn_metadata_first  GPU  14 ms   CPU  13 ms  (0.26 ms/iter)   ~320× faster CPU
P00_prefill_branch_total      GPU  24 ms   CPU  19 ms  (0.19 ms/iter)
P00a_seq_lens_cpu_path        CPU 0.75 ms  (0.008 ms/iter)   ← fast path taken
```

End-to-end tok/s is unchanged — for a single request the GPU was already on the critical path, CPU work was overlapped. Fix 1 still matters for **concurrent requests** (CPU freed to queue more GPU work) and is a prerequisite for the next improvements; it is a correctness-neutral change.

### 4. The remaining 80 ms is target forward at B=4

Added `TGT_model_forward` timing in `gpu_model_runner.execute_model`:
```
TGT_model_forward  50 calls  GPU 4260 ms (85 ms/iter)  CPU 423 ms (8 ms/iter)
```

i.e. 85 ms of GPU work per iter, 8 ms of wall-time (the rest overlaps). Target forward for (batch=1, num_tokens=4) at 30k dominates the iter. Attention is only ~8 ms of that; the other ~77 ms is MoE/MLP (Marlin WNA16, fp8 activations). I did NOT track this further — it would need kernel-level profiling (nsys) and likely a MoE kernel fix or a different backend.

### 5. FLASHINFER_MLA is 2× faster but has a correctness bug

Under FLASHINFER_MLA (supports `UNIFORM_BATCH` CG on sm120 with DeepSeek/Kimi dims):
- Target forward B=4 at 30k drops to ~19 ms GPU.
- **Interarrival becomes flat across context (18-19 ms at 2k/10k/30k).** Proves the O(ctx) symptom was target-forward cost at B=4 with PIECEWISE CG / non-captured attention.
- But: `Mean acceptance length: 1.00`, `Per-position acceptance rate: 0.000`. Every draft token is rejected. Output text is still produced (target generates one token per step), so without the correctness metric you'd miss the bug; raw tok/s is 52.4 which looks "twice as fast".

Likely cause: FLASHINFER_MLA's spec-verification attention on sm120 fp8 does not apply causal masking inside the 4-query batch (or doesn't write KV in the exact layout the rejection sampler expects). Needs investigation before shipping — see `vllm/v1/attention/backends/mla/flashinfer_mla.py` `forward_mqa` around the `is_fp8_kv_cache` branch.

## What I recommend next

Priority order:

1. **Debug FLASHINFER_MLA 0% acceptance** on sm120 fp8 eagle3. The prize is huge (2× target forward → roughly 2× tok/s with correct MTP, + flat scaling in ctx). What I learned while trying to fix it:

   - The flatten path in `_build_decode` (line ~237-289) converts a 1-request × 4-query spec-verify batch into 4 independent 1-query decodes. This lets the XQA kernel run (XQA requires `tokens_per_req == 1`), but XQA does not apply causal masking across the synthesised 4 requests → garbage logits → 0% acceptance.
   - Tried removing `self._is_fp8_kv_cache` from the flatten condition so tpr>1 would fall through to the `trtllm-gen` backend instead → runtime error `TllmGenFmhaRunner: Unsupported architecture` — **trtllm-gen FP8 MLA is not compiled for sm120** in this FlashInfer build.
   - Tried `--block-size 64` to bypass a related `Supported block_size are 32 and 64, got 16` assertion in the same trtllm-gen path; that got past the validator but still hits the `Unsupported architecture` error.

   So on this box the only shipped FP8 MLA kernel is XQA, and XQA does not handle causal-within-query-group. Fixing this requires either (a) a kernel upstream fix in FlashInfer to make XQA handle spec verification, or (b) compiling trtllm-gen for sm120, or (c) falling back to TRITON_MLA for spec-verification only while using FLASHINFER_MLA/XQA for pure decodes. Option (c) is feasible in vLLM but would mean swapping attention backends per layer/step, which is not a one-line change.

2. **Also try `VLLM_MARLIN_USE_ATOMIC_ADD=0`** and/or `VLLM_MARLIN_INPUT_DTYPE=bf16` to see whether Marlin-MoE at B=4 is faster without these. They may be optimised for large-batch MoE and hurt small-batch.

2. **Enable FULL cudagraph for TRITON_MLA in spec-verification shapes.** The backend currently declares `AttentionCGSupport.NEVER`. Even PARTIAL support for fixed spec shapes (num_queries_per_req ∈ {2,3,4,5}) would capture stage-1+stage-2 + block-table lookups as one graph and cut the 77-ms MoE surround overhead. Related: `_cudagraph_support` in `triton_mla.py`, and the warning at `compilation.py:1346`.

3. **Spec-as-decode for TRITON_MLA.** Set `query_len_support = UNIFORM` and `reorder_batch_threshold = 1+num_spec`. Then max_query_len=4 goes through the decode path (no prefill branch at all). Requires kernel to handle multi-query per request — block_table has to be expanded per query (cur_batch_req_idx currently equals cur_batch; with B queries/request × num_reqs total queries this is wrong). A minimal path: `block_table = block_table.repeat_interleave(num_queries_per_req, dim=0)` right before kernel call, plus per-query `seq_lens` of shape `(num_decode_tokens,)`.

4. **MoE kernel at B=4.** Profile with nsys to see whether Marlin WNA16 at B=4 is really 8× B=1 (it shouldn't be for bandwidth-bound MoE). Compare against sglang's MoE path for the same Kimi-K2 FP8 quantization. If sglang uses a different MoE kernel at B>1 (e.g. FlashInfer CUTLASS MoE), we should try that. Env switch candidates: `VLLM_MARLIN_USE_ATOMIC_ADD=0`, `VLLM_MARLIN_INPUT_DTYPE=bf16`, unset `VLLM_TEST_FORCE_FP8_MARLIN`.

5. **Cross-check with sglang source** under `/mnt/sglang*` for how they do spec-verification metadata build + attention: whether they avoid the equivalent of vllm's prefill branch, and which MoE kernel they call at B=1+num_spec.

## Files modified in the container (`73f963a03703`)

- `/opt/vllm/vllm/model_executor/layers/attention/mla_attention.py` — sync fix + instrumentation (see backup `.baseline_20260423`).
- `/opt/vllm/vllm/v1/worker/gpu_model_runner.py` — keep optimistic seq_lens_cpu in async spec mode + TGT_model_forward timing (backup `.baseline_20260423`).
- `/opt/vllm/vllm/v1/spec_decode/eagle.py` — propose()-timing wrappers (backup `.baseline_20260423`).
- `/opt/vllm/vllm/v1/spec_decode/_mtp_timing.py` — new file, shared timer. Enable with `MTP_TIMING=1`; default 0, no-op.

The committed baseline image `voipmonitor/vllm:cu130-mtp-baseline-20260423` is a snapshot of the container just before my changes.

## Benchmark scripts

- `/root/vllm/mtp-long-ctx-fix/bench/bench_triton_mla.py` — microbench the decode kernel directly.
- `/root/vllm/mtp-long-ctx-fix/bench/e2e_bench.py` — streaming tok/s + per-token interarrival against the server. Use `--port 5002 -c <ctx> --max-tokens <N>`.
