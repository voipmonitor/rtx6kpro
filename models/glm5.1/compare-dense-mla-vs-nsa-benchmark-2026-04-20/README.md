# Dense MLA vs NSA Benchmark for GLM-5.1-NVFP4

This directory is a self-contained reproduction bundle for comparing **dense MLA** and **NSA** inference paths on `lukealonso/GLM-5.1-NVFP4` in SGLang.

## Why this test exists

Luke's working hypothesis is that **dense MLA is not a harmless superset** for a model that was co-trained with an NSA indexer. The argument is that, if the model was always trained with sparse retrieval in the loop, the sparse routing mask is part of the learned architecture rather than an optional optimization. Under that view:

- the indexer shapes which tokens ever compete in attention,
- training gradients are conditioned on that restricted candidate set,
- attention score calibration is learned in the sparse world,
- removing the indexer at inference time changes the function being computed.

That leads to a concrete expectation:

- short contexts may still look mostly fine,
- long contexts may become more distractible under dense inference,
- dense attention may spread probability mass across many distractors,
- some heads may no longer do the job they were trained for,
- quality can fall even when dense attention has access to more tokens.

This benchmark was created to test whether that theoretical claim shows up in a real long-context task.

## Main result

The **headline numbers below exclude pathological breakdown runs**. Those runs are preserved in `results/`, but they are not used for the top-line comparison because they are better understood as implementation/pathology failures than as normal model-quality failures.

```text
+-----------+-----------------------+-----------+---------+-------+-------------------------------+--------------------------------------+
| Variant   | Excluded runs         | Completed | Correct | Wrong | Completion tokens min/med/avg/max    | Elapsed s min/med/avg/max            |
+-----------+-----------------------+-----------+---------+-------+-------------------------------+--------------------------------------+
| dense_mla | 8                     | 29        | 22      | 7     | 3219 / 8631.0 / 8560.000 / 16875     | 36.813 / 100.648 / 103.488 / 189.575 |
| nsa       | 21, 29                | 28        | 25      | 3     | 1784 / 4965.5 / 5618.464 / 27297     | 25.270 / 76.562 / 97.013 / 615.137   |
+-----------+-----------------------+-----------+---------+-------+-------------------------------+--------------------------------------+
```

Interpretation:

- `dense_mla` was faster on generation throughput, but it more often ended in a coherent **wrong answer** (`Latvia`).
- `nsa` reached the correct final answer more often, which supports Luke's concern that dense MLA changes the computation in a harmful way for a co-trained NSA model.
- The excluded NSA runs (`21`, `29`) are documented separately as repetition/degeneration bugs and should be read as tail-pathology evidence, not as normal quality data points.

## High-level findings

1. The results do **not** support the idea that dense MLA is a harmless substitute for NSA on this model.
2. The dense path failed mostly by staying fluent but reasoning to the wrong country.
3. The NSA path was more often right, but when it failed badly, it failed much more dramatically.
4. So the comparison is not just about mean accuracy; it is also about **failure shape**:
   - dense MLA: more coherent-but-wrong
   - NSA: fewer wrong answers overall, but worse pathological outliers

## Exact runtime state used for the benchmark

Base SGLang source inside the container:

- repo: `lukealonso/sglang`
- commit: `f4b7830ed8d3d570d9662e273009334c824c8227`
- `b12x`: `0.9.6`
- container image used as runtime base: `voipmonitor/sglang:cu130test`

Local modifications present during the benchmark:

- `patches/tokenizer_manager_glm_generate_template.diff`
  - auto-applies the GLM chat template for raw `/generate` requests so that the benchmark does not exercise the wrong prompt contract.
- `patches/nsa_backend_extend_capacity_144k.diff`
  - raises the b12x NSA extend gather workspace cap from `128 * 1024` to `144 * 1024` to avoid a long-context crash.
- `patches/cuda_piecewise_backend_skip_capture.diff`
  - skips piecewise CUDA graph capture for runtime-recompiled shapes when no PCG capture stream is active.
- extra Blackwell tuning JSONs copied under `configs/`
  - these are the local performance tuning configs that were present in the test runtime.

## What is included here

```text
compare-dense-mla-vs-nsa-benchmark-2026-04-20/
├── README.md
├── scripts/
│   ├── benchmark_glm_variants.py
│   ├── launch_dense_mla.sh
│   ├── launch_nsa.sh
│   ├── run_benchmark.sh
│   └── test.py
├── prompts/
│   └── testLuke5.txt
├── patches/
│   ├── tokenizer_manager_glm_generate_template.diff
│   ├── nsa_backend_extend_capacity_144k.diff
│   └── cuda_piecewise_backend_skip_capture.diff
├── configs/
│   ├── quantization/
│   └── moe_triton/
└── results/
    ├── failure_analysis_report.md
    ├── summary.json
    ├── final_summary.csv
    ├── final_summary.json
    ├── wrong_run_classification.json
    ├── dense_mla_runs.jsonl
    ├── nsa_runs.jsonl
    ├── datasets/
    │   ├── nsa_without_run21.jsonl
    │   ├── nsa_without_breakdowns.jsonl
    │   └── dense_mla_without_breakdowns.jsonl
    └── full_outputs/
        ├── dense_mla_run_8_full_output.txt
        ├── nsa_run_21_full_output.txt
        └── nsa_run_29_full_output.txt
```

## How to reproduce the test

### 1. Prepare the runtime

Use the same SGLang source state and apply the local diffs in `patches/`. The benchmark was not run on a fully clean upstream tree.

You should also have the Blackwell tuning JSONs from `configs/` installed into the corresponding SGLang paths:

- quantization configs -> `python/sglang/srt/layers/quantization/configs/`
- MoE triton configs -> `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/`

### 2. Start one of the two GLM variants

Dense MLA:

```bash
bash scripts/launch_dense_mla.sh
```

NSA:

```bash
bash scripts/launch_nsa.sh
```

### 3. Run a single evaluation manually

```bash
python3 scripts/test.py --port 8001 -f prompts/testLuke5.txt --max-tokens 40000
```

### 4. Run the full comparison benchmark

```bash
bash scripts/run_benchmark.sh
```

The benchmark script:

- launches `dense_mla`, runs 30 evaluations, records all outputs,
- stops the server,
- launches `nsa`, runs 30 evaluations, records all outputs,
- scores correctness from the **final answer**, not from intermediate reasoning tokens.

## How correctness was judged

This is important.

The benchmark does **not** stop as soon as it first sees the token `Estonia` anywhere in the stream. That would be misleading because the model often mentions both candidate countries during reasoning.

Instead, correctness is scored from the **final answer line** emitted at the end of generation.

This makes the benchmark much closer to the real user-facing behavior.

## Failure analysis summary

### dense_mla wrong runs

- coherent but wrong answers: runs `15, 19, 21, 22, 24, 27, 28`
- breakdown / truncated generation: run `8`

Dense MLA therefore mostly failed by producing a fluent but wrong answer rather than by collapsing into repetition.

### nsa wrong runs

- coherent but wrong answers: runs `6, 26, 27`
- repetition / degeneration failures: runs `21, 29`

NSA therefore had a better raw success rate, but a worse tail failure mode.

## Raw all-run statistics

These numbers are retained for completeness, but they are **not** the recommended comparison because they include the pathological breakdown runs.

```text
+-----------+-----------+---------+-------+-------------------------------+--------------------------------------+-------------------------------------------+
| Variant   | Completed | Correct | Wrong | Correct rate                  | Completion tokens min/med/avg/max    | Elapsed s min/med/avg/max                 |
+-----------+-----------+---------+-------+-------------------------------+--------------------------------------+-------------------------------------------+
| dense_mla | 30        | 22      | 8     | 73.33%                        | 3219 / 8681.0 / 8852.767 / 17343     | 36.813 / 101.255 / 106.872 / 204.995      |
| nsa       | 30        | 25      | 5     | 83.33%                        | 1784 / 5051.5 / 7910.567 / 40000     | 25.270 / 77.088 / 164.002 / 1438.703      |
+-----------+-----------+---------+-------+-------------------------------+--------------------------------------+-------------------------------------------+
```

## Filtered views

### NSA with only the single worst outlier removed (`run 21`)

- completed: 29
- correct: 25
- wrong: 4
- correct rate: 86.21%
- completion tokens min / median / mean / max: 1784 / 5011 / 6804.034 / 40000
- elapsed seconds min / median / mean / max: 25.270 / 76.870 / 143.278 / 1438.703

### NSA with both degeneration runs removed (`runs 21 and 29`)

- completed: 28
- correct: 25
- wrong: 3
- correct rate: 89.29%
- completion tokens min / median / mean / max: 1784 / 4965.5 / 5618.464 / 27297
- elapsed seconds min / median / mean / max: 25.270 / 76.562 / 97.013 / 615.137

## Recommended reading order for Luke

1. Start with this `README.md`.
2. Read `results/failure_analysis_report.md` for the detailed breakdown.
3. Read `results/outlier_report_for_luke.md` for the dedicated explanation of the excluded outlier runs.
4. Open `results/wrong_run_classification.json` for per-run labels.
5. Inspect `results/full_outputs/nsa_run_21_full_output.txt` and `results/full_outputs/nsa_run_29_full_output.txt` to see the two pathological NSA failures.
6. Compare that against `results/full_outputs/dense_mla_run_8_full_output.txt` to see the single dense breakdown.

## Bottom line

This benchmark supports the claim that, for this GLM-5.1 NSA-trained model, **dense MLA is not just a harmless superset of NSA**. On this task, dense MLA was more likely to stay coherent while arriving at the wrong answer. NSA was more often correct, which fits the theory that the indexer is part of the learned computation. At the same time, the current NSA implementation showed worse pathological outliers, so the correct conclusion is not simply “NSA wins,” but rather:

- **accuracy / faithfulness to trained behavior:** NSA looks better
- **runtime stability / tail failures:** dense MLA currently looks safer
