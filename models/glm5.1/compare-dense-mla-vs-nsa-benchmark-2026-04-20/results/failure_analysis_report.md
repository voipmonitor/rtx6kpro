# GLM-5.1 Failure Analysis: Dense MLA vs NSA

Date: 2026-04-20 14:14:32 UTC

This report summarizes the benchmark failure modes observed while comparing two SGLang launch configurations for `lukealonso/GLM-5.1-NVFP4`:

- `dense_mla`: dense MLA path (`flashinfer`, BF16 KV cache)
- `nsa`: NSA path (`b12x`, FP8 KV cache)

The benchmark executed 30 end-to-end runs per variant against the same prompt file (`/mnt/testLuke5.txt`) using `/mnt/test.py --max-tokens 40000` and judged correctness from the **final answer**, not from intermediate reasoning mentions.

## Launch Configurations

### dense_mla
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_ENABLE_SPEC_V2=True
export SGLANG_ENABLE_JIT_DEEPGEMM=0
export SGLANG_ENABLE_DEEP_GEMM=0
export NCCL_GRAPH_FILE=/mnt/nccl_graph_opt.xml
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=SYS
export NCCL_ALLOC_P2P_NET_LL_BUFFERS=1
export NCCL_MIN_NCHANNELS=8
export OMP_NUM_THREADS=8
export SAFETENSORS_FAST_GPU=1
python3 -m sglang.launch_server   --model-path lukealonso/GLM-5.1-NVFP4   --served-model-name GLM-5   --reasoning-parser glm45   --tool-call-parser glm47   --tensor-parallel-size 8   --quantization modelopt_fp4   --kv-cache-dtype bfloat16   --trust-remote-code   --disable-shared-experts-fusion   --attention-backend flashinfer   --moe-runner-backend b12x   --fp4-gemm-backend b12x   --cuda-graph-max-bs 30   --speculative-algorithm EAGLE   --speculative-num-steps 4   --speculative-num-draft-tokens 6   --speculative-eagle-topk 1   --chunked-prefill-size 8192   --max-running-requests 30   --mem-fraction-static 0.80   --host 0.0.0.0   --port 8001   --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 16}'   --enable-metrics   --enable-piecewise-cuda-graph
```

### nsa
```bash
export CUTE_DSL_ARCH=sm_120a
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_ENABLE_SPEC_V2=True
export SGLANG_ENABLE_JIT_DEEPGEMM=0
export SGLANG_ENABLE_DEEP_GEMM=0
export NCCL_GRAPH_FILE=/mnt/nccl_graph_opt.xml
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=SYS
export NCCL_ALLOC_P2P_NET_LL_BUFFERS=1
export NCCL_MIN_NCHANNELS=8
export OMP_NUM_THREADS=8
export SAFETENSORS_FAST_GPU=1
python3 -m sglang.launch_server   --model-path lukealonso/GLM-5.1-NVFP4   --served-model-name GLM-5   --reasoning-parser glm45   --tool-call-parser glm47   --tensor-parallel-size 8   --quantization modelopt_fp4   --kv-cache-dtype fp8_e4m3   --trust-remote-code   --disable-shared-experts-fusion   --nsa-prefill-backend b12x   --nsa-decode-backend b12x   --page-size 64   --attention-backend nsa   --moe-runner-backend b12x   --fp4-gemm-backend b12x   --cuda-graph-max-bs 4   --enable-pcie-oneshot-allreduce   --speculative-algorithm EAGLE   --speculative-num-steps 3   --speculative-num-draft-tokens 4   --speculative-eagle-topk 1   --chunked-prefill-size 8192   --max-running-requests 4   --mem-fraction-static 0.76   --host 0.0.0.0   --port 8001   --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 16}'   --json-model-override-args '{"index_topk_pattern": "FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSSFSFFFSFSSSFSFFSFFSSS"}'   --preferred-sampling-params '{"temperature": 1.0, "top_p": 0.95}'   --enable-metrics
```

## Recommended Headline Statistics

The main comparison below excludes the pathological breakdown runs:

- `dense_mla`: exclude run `8`
- `nsa`: exclude runs `21` and `29`

### dense_mla (run 8 excluded)
- completed: 29
- correct: 22
- wrong: 7
- correct rate: 75.86%
- completion tokens min / median / mean / max: 3219 / 8631 / 8560.000 / 16875
- elapsed seconds min / median / mean / max: 36.813 / 100.648 / 103.488 / 189.575
- generation tok/s min / median / mean / max: 80.198 / 86.621 / 85.731 / 90.039
- mean TTFT: 3.577 s
- mean end-to-end tok/s: 83.173

### nsa (runs 21 and 29 excluded)
- completed: 28
- correct: 25
- wrong: 3
- correct rate: 89.29%
- completion tokens min / median / mean / max: 1784 / 4965.5 / 5618.464 / 27297
- elapsed seconds min / median / mean / max: 25.270 / 76.562 / 97.013 / 615.137
- generation tok/s min / median / mean / max: 44.408 / 68.029 / 67.085 / 71.871
- mean TTFT: 6.365 s
- mean end-to-end tok/s: 63.533

## Raw All-Run Statistics

These numbers are preserved for completeness, but they include the pathological breakdown runs and should be treated as stability/tail-risk evidence rather than the main quality comparison.

### dense_mla
- completed: 30
- correct: 22
- wrong: 8
- correct rate: 73.33%
- completion tokens min / median / mean / max: 3219 / 8681.0 / 8852.767 / 17343
- elapsed seconds min / median / mean / max: 36.813 / 101.255 / 106.872 / 204.995
- generation tok/s min / median / mean / max: 80.198 / 86.558 / 85.702 / 90.039
- mean TTFT: 3.477 s
- mean end-to-end tok/s: 83.221

### nsa
- completed: 30
- correct: 25
- wrong: 5
- correct rate: 83.33%
- completion tokens min / median / mean / max: 1784 / 5051.5 / 7910.567 / 40000
- elapsed seconds min / median / mean / max: 25.270 / 77.088 / 164.002 / 1438.703
- generation tok/s min / median / mean / max: 27.812 / 67.854 / 65.284 / 71.871
- mean TTFT: 5.971 s
- mean end-to-end tok/s: 61.967

## Failure-Type Classification

The wrong runs do **not** all belong to the same failure class.

### dense_mla
- coherent but wrong answers: runs 15, 19, 21, 22, 24, 27, 28
- breakdown / truncated / degenerate generation: run 8
- interpretation: most dense failures were logically coherent but ended in the wrong country (`Latvia`), with one clear generation breakdown that never produced a clean final answer.

### nsa
- coherent but wrong answers: runs 6, 26, 27
- breakdown / repetition failures: runs 21, 29
- interpretation: NSA had better nominal accuracy, but it also produced two severe long-form degeneration failures that hit the token limit.

## Wrong Runs: Detailed Classification

| Variant | Run | Classification | Completion tokens | Elapsed s | Finish reason | Final answer summary |
|---|---:|---|---:|---:|---|---|
| dense_mla | 8 | breakdown_or_degeneration | 17343 | 204.995 | None | *   The "reagent vendor ledger says stock code AR-12 is booked under procurement |
| dense_mla | 15 | coherent_wrong_answer | 11398 | 132.369 | stop | Therefore, the manufacturer of the material used by the Glass Current salinity bench is **Mirel Industrial**, headquartered in **Latvia**. |
| dense_mla | 19 | coherent_wrong_answer | 8844 | 100.648 | stop | Latvia |
| dense_mla | 21 | coherent_wrong_answer | 3615 | 44.171 | stop | Based on the instrument maintenance appendix and the adjacent vendor master update in the packet, the **Glass Current** salinity bench curre |
| dense_mla | 22 | coherent_wrong_answer | 11503 | 131.810 | stop | **Latvia** |
| dense_mla | 24 | coherent_wrong_answer | 10117 | 117.005 | stop | Therefore, the manufacturer of the material (reagent AR-12) used by the Glass Current salinity bench is headquartered in **Latvia**. |
| dense_mla | 27 | coherent_wrong_answer | 16875 | 189.575 | stop | The manufacturer of the material (stock code AR-12) used by the Glass Current salinity bench is Mirel Industrial, which is headquartered in  |
| dense_mla | 28 | coherent_wrong_answer | 7500 | 89.990 | stop | Therefore, the manufacturer of the material used by the Glass Current salinity bench is Mirel Industrial, which is headquartered in **Latvia |
| nsa | 6 | coherent_wrong_answer | 5323 | 77.306 | stop | Therefore, the manufacturer of the material used by the Glass Current salinity bench is headquartered in **Latvia**. |
| nsa | 21 | breakdown_or_degeneration | 40000 | 764.986 | length | * within information about small driver seat:**   */           *   Within information about small driver seat**:   */           *            |
| nsa | 26 | coherent_wrong_answer | 5814 | 86.137 | stop | **Latvia** |
| nsa | 27 | coherent_wrong_answer | 4698 | 67.371 | stop | Therefore, the manufacturer of the material (adapter collar N-4 / stock code AR-12) used by the Glass Current salinity bench is headquartere |
| nsa | 29 | breakdown_or_degeneration | 40000 | 1438.703 | length | establish   *   * the $< text.   *   *Issue for     for`    Multiple but the *   *   *   * The logic       something not descriptive,     *  |


## Outlier Review

The worst observed outlier was `nsa` run `21`:

- completion tokens: 40000
- elapsed: 764.986 s
- finish_reason: `length`
- result: incorrect
- failure mode: clear repetition / degeneration, not just a long but valid reasoning trace

A second NSA degeneration run was also observed in run `29`:

- completion tokens: 40000
- elapsed: 1438.703 s
- finish_reason: `length`
- result: incorrect
- failure mode: repetition / gibberish / unstable continuation

For comparison, dense MLA also had one non-coherent failure in run `8`:

- completion tokens: 17343
- elapsed: 204.995 s
- finish_reason: `None`
- result: incorrect
- failure mode: truncated / looping reasoning that never reached a stable final answer

## Filtered Statistics

### NSA with only the single worst outlier removed (run 21 removed)
- completed: 29
- correct: 25
- wrong: 4
- correct rate: 86.21%
- completion tokens min / median / mean / max: 1784 / 5011 / 6804.034 / 40000
- elapsed seconds min / median / mean / max: 25.270 / 76.870 / 143.278 / 1438.703
- generation tok/s min / median / mean / max: 27.812 / 67.855 / 65.731 / 71.871
- mean TTFT: 6.161 s
- mean end-to-end tok/s: 62.301

### dense_mla with breakdown runs removed
- completed: 29
- correct: 22
- wrong: 7
- correct rate: 75.86%
- completion tokens min / median / mean / max: 3219 / 8631 / 8560.000 / 16875
- elapsed seconds min / median / mean / max: 36.813 / 100.648 / 103.488 / 189.575
- generation tok/s min / median / mean / max: 80.198 / 86.621 / 85.731 / 90.039
- mean TTFT: 3.577 s
- mean end-to-end tok/s: 83.173

### nsa with breakdown runs removed (runs 21 and 29 removed)
- completed: 28
- correct: 25
- wrong: 3
- correct rate: 89.29%
- completion tokens min / median / mean / max: 1784 / 4965.5 / 5618.464 / 27297
- elapsed seconds min / median / mean / max: 25.270 / 76.562 / 97.013 / 615.137
- generation tok/s min / median / mean / max: 44.408 / 68.029 / 67.085 / 71.871
- mean TTFT: 6.365 s
- mean end-to-end tok/s: 63.533

## Interpretation

1. `nsa` achieved a higher raw correct rate than `dense_mla` (`83.33%` vs `73.33%`).
2. `dense_mla` was faster on generated tokens per second, but its answers were more often coherently wrong.
3. `nsa` had better nominal accuracy, but its tail-risk was materially worse because of two catastrophic repetition failures.
4. Once the pathological NSA breakdown runs are excluded, the cleaned NSA dataset still has a better correct rate than dense MLA, but the existence of two max-token degeneration failures is important and should not be ignored.
5. The failure modes are therefore mixed:
   - `dense_mla`: mostly wrong-but-coherent reasoning paths to `Latvia`
   - `nsa`: a mix of wrong-but-coherent `Latvia` answers and true repetition collapses

## Files Included in This Bundle

- `report.md`: this report
- `wrong_run_classification.json`: per-run failure classification
- `datasets/nsa_without_run21.jsonl`: NSA dataset with the single worst outlier removed
- `datasets/dense_mla_without_breakdowns.jsonl`: dense MLA dataset excluding breakdown run 8
- `datasets/nsa_without_breakdowns.jsonl`: NSA dataset excluding breakdown runs 21 and 29
- `full_outputs/nsa_run_21_full_output.txt`: full raw output for the main NSA outlier
- `full_outputs/nsa_run_29_full_output.txt`: full raw output for the second NSA repetition failure
- `full_outputs/dense_mla_run_8_full_output.txt`: full raw output for the dense MLA breakdown run
- original benchmark source data: `/root/glm/benchmarks/glm_dense_vs_nsa_20260420T113047Z`
