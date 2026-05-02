# vLLM b12x NSA/MTP port for GLM-5.1

Date: 2026-05-02

Audience: maintainers working on GLM-5.1, b12x, SGLang, and vLLM on RTX PRO 6000 Blackwell PCIe systems.

This is the state report for the custom vLLM port of `lukealonso/GLM-5.1-NVFP4-MTP` with b12x sparse NSA attention, ModelOpt FP4/NVFP4 weights, b12x MoE, MTP speculative decode, FP8 KV cache, and PCIe oneshot allreduce.

The important result is that vLLM was made to run long prefill smoothly after removing runtime shape/JIT churn in the b12x integration. The final Docker is not a vanilla vLLM or vanilla b12x wheel: it contains a custom vLLM tree plus a small audited b12x runtime overlay. The correctness-critical b12x overlay is the PCIe oneshot completion barrier needed with CUDA graph/no-copy speculative decode on our 8-GPU PCIe topology.

## Final reference image

```text
voipmonitor/vllm:glm51-mtp-pciebarrier-b12x0111-kv432k-20260502
digest/image id: sha256:4fc8f9dd9b33beda530e3865cdf388e93aecc4cc460b967582f463fd22aff11e
```

Runtime components verified in the image:

| Component | Version / state |
|---|---|
| Model | `lukealonso/GLM-5.1-NVFP4-MTP` |
| vLLM | `0.0.0+local` custom tree |
| PyTorch | `2.11.0+cu130` |
| b12x | `0.11.1` plus local runtime overlay listed below |
| flashinfer-python | `0.6.8` |
| transformers | `5.3.0` |
| fastsafetensors | `0.2.2` |
| NCCL graph XML | baked/mounted at `/opt/vllm/nccl_graph_opt.xml` |
| b12x `attention/nsa_indexer/api.py` hash | `14d1fdb4585fbdc71ba828c97cd7fc013e2c15b57142e6f8ca7d3352a7587411` |
| b12x `attention/nsa_indexer/tiled_topk.py` hash | `d6e920cf2431f8bdc5311f6e06d86cd142de0682663a5688b6f26143cb1596e7` |
| b12x `pcie_oneshot.cu` hash | `2b8ccaf2294fc10795c923540eff2c8578d65d87a0fef92348d0fe6736673ee7` |
| vLLM `b12x_mla_sparse.py` hash in final image | `dff7bd04773bca00c645f05030e1348d93c0f0e20d5eca4b7c6fb5d3167ae4b2` |

Audit commands used before publishing this page:

```bash
docker run --rm --entrypoint /bin/bash \
  voipmonitor/vllm:glm51-mtp-pciebarrier-b12x0111-kv432k-20260502 \
  -lc '/opt/venv/bin/python - <<PY
import importlib.metadata as md
for p in ["vllm", "torch", "b12x", "flashinfer-python", "transformers", "fastsafetensors"]:
    print(p, md.version(p))
PY
sha256sum /opt/venv/lib/python3.12/site-packages/b12x/attention/nsa_indexer/api.py
sha256sum /opt/venv/lib/python3.12/site-packages/b12x/attention/nsa_indexer/tiled_topk.py
sha256sum /opt/venv/lib/python3.12/site-packages/b12x/distributed/pcie_oneshot.cu
sha256sum /opt/vllm/vllm/v1/attention/backends/mla/b12x_mla_sparse.py
grep -n "multi_gpu_barrier<ngpus, false>" /opt/venv/lib/python3.12/site-packages/b12x/distributed/pcie_oneshot.cu
ls -l /opt/vllm/nccl_graph_opt.xml'

docker logs vllm-glm51-local-pciebarrier-mtp-memtest-20260502 2>&1 \
  | rg "non-default args|GPU KV cache size|Available KV cache memory|Maximum concurrency"

python3 -m pip download b12x==0.11.1 --no-deps -d /tmp/b12x_audit/wheel
# The wheel was extracted under /tmp/b12x_audit/vanilla and the final image
# package under /tmp/b12x_audit/final before comparing source hashes/diffs.
```

The same image was also tagged locally during development as:

```text
voipmonitor/vllm:exp-glm51-graphtrue-nocopy-specserial0-pciebarrier-20260502
```

## Launch command

This is the recommended DCP=1 / TP=8 reference command. It binds persistent cache directories from the host so CUTE DSL, torchinductor, Triton, vLLM, and CUDA extension caches survive container restarts.

```bash
mkdir -p \
  "$HOME/.cache/huggingface" \
  "$HOME/.cache/vllm-glm51/jit" \
  "$HOME/.cache/vllm-glm51/cutlass_dsl" \
  "$HOME/.cache/vllm-glm51/torchinductor" \
  "$HOME/.cache/vllm-glm51/triton" \
  "$HOME/.cache/vllm-glm51/vllm"

docker run -d --name vllm-glm51-mtp-b12x \
  --gpus all \
  --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  --entrypoint /bin/bash \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/vllm-glm51/jit:/cache/jit" \
  -v "$HOME/.cache/vllm-glm51/cutlass_dsl:/root/.cache/cutlass_dsl" \
  -v "$HOME/.cache/vllm-glm51/torchinductor:/root/.cache/torchinductor" \
  -v "$HOME/.cache/vllm-glm51/triton:/root/.cache/triton" \
  -v "$HOME/.cache/vllm-glm51/vllm:/root/.cache/vllm" \
  -v /mnt/nccl_graph_opt.xml:/opt/vllm/nccl_graph_opt.xml:ro \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e PORT=5269 \
  -e OMP_NUM_THREADS=16 \
  -e CUTE_DSL_ARCH=sm_120a \
  -e VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel \
  -e VLLM_USE_B12X_SPARSE_INDEXER=1 \
  -e VLLM_DISABLE_SHARED_EXPERTS_STREAM=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
  -e VLLM_B12X_MLA_SPEC_SERIAL_DECODE=0 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
  -e VLLM_PCIE_ONESHOT_ALLOW_CROSS_NUMA=1 \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_GRAPH_FILE=/opt/vllm/nccl_graph_opt.xml \
  -e CUDA_CACHE_PATH=/cache/jit \
  -e TORCH_EXTENSIONS_DIR=/cache/jit/torch_extensions \
  -e VLLM_CACHE_DIR=/cache/jit/vllm \
  -e TVM_FFI_CACHE_DIR=/cache/jit/tvm-ffi \
  -e CUTE_DSL_CACHE_DIR=/root/.cache/cutlass_dsl \
  -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor \
  -e TRITON_CACHE_DIR=/root/.cache/triton \
  -e VLLM_CACHE_ROOT=/root/.cache/vllm \
  -e XDG_CACHE_HOME=/cache/jit \
  -e FLASHINFER_WORKSPACE_BASE=/cache/jit/flashinfer \
  -e VLLM_LOG_STATS_INTERVAL=1 \
  -e SPEC_CONFIG='{"model":"lukealonso/GLM-5.1-NVFP4-MTP","method":"mtp","num_speculative_tokens":3,"rejection_sample_method":"probabilistic","moe_backend":"b12x","use_local_argmax_reduction":true}' \
  -e HF_OVERRIDES='{"index_topk_pattern":"FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSSFSFFFSFSSSFSFFSFFSSSF"}' \
  voipmonitor/vllm:glm51-mtp-pciebarrier-b12x0111-kv432k-20260502 \
  -lc 'exec /opt/venv/bin/vllm serve lukealonso/GLM-5.1-NVFP4-MTP \
    --served-model-name GLM-5 \
    --trust-remote-code \
    --host 0.0.0.0 --port "${PORT}" \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 1 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --load-format fastsafetensors \
    --async-scheduling \
    --gpu-memory-utilization 0.89 \
    --max-model-len 202752 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 64 \
    --mm-processor-cache-gb 0 \
    --mm-encoder-tp-mode weights \
    --attention-backend B12X_MLA_SPARSE \
    --moe-backend b12x \
    --kv-cache-dtype fp8 \
    --tool-call-parser glm47 \
    --enable-auto-tool-choice \
    --reasoning-parser glm45 \
    --speculative-config "${SPEC_CONFIG}" \
    --hf-overrides "${HF_OVERRIDES}" \
    --max-cudagraph-capture-size 8'
```

The image is a DCP=1 reference image. DCP=8 support was investigated, but it is not the final published state described here.

## Memory and KV cache result

The safe published configuration is:

```text
--gpu-memory-utilization 0.89
--max-model-len 202752
--kv-cache-dtype fp8
```

Observed vLLM startup memory:

| Setting | Result |
|---|---|
| GPU KV cache size | `432,832 tokens` |
| Available KV cache memory | `25.09 GiB` |
| Max concurrency for `202,752` tokens | `2.13x` |
| Source | Current container startup log for `vllm-glm51-local-pciebarrier-mtp-memtest-20260502` |

Stress test:

```bash
python3 /mnt/test.py --port 5269 --context-tokens 310000 --max-tokens 128
```

This long-prefill test was performed during the session and was used to choose the published `0.89` setting. The exact per-GPU free-memory trace for the higher `0.895`/`0.90` experiments was not archived in this repository, so this page intentionally does not cite those higher settings as audited data.

Practical conclusion from the archived, reproducible state: use `0.89` with `max_model_len=202752`. Higher settings need to be remeasured before being published.

## Voipmonitor LLM completion-stats result

The current final vLLM container was tested with the internal long-document GLM task after the Docker/source audit above.

Raw result artifact:

```text
models/glm5.1/results/vllm-pciebarrier-mtp-voipmonitor-llmtest-20260502.json
```

Exact test command:

```bash
python3 /mnt/llm_decode_bench.py \
  --host localhost \
  --port 5269 \
  --model GLM-5 \
  --completion-stats \
  --completion-stats-min-results 30 \
  --completion-stats-concurrency-levels 1,2,4,8,16,30 \
  --display-mode plain \
  --no-hw-monitor \
  --output /mnt/glm51_voipmonitor_llmtest_20260502/vllm_pciebarrier_mtp_5269_completion_stats.json
```

Test metadata:

| Field | Value |
|---|---|
| Timestamp | `2026-05-02T19:22:06.310498` |
| Server | `http://localhost:5269` |
| Model name | `GLM-5` |
| Prompt | `models/glm5.1/compare-dense-mla-vs-nsa-benchmark-2026-04-20/prompts/testLuke5.txt` |
| Prompt size | `707,372 chars`, `133,179 prompt tokens` in measured runs |
| Max tokens | `40,000` |
| Correctness regex | `\bestonia\b` against the final non-empty answer line |
| Prefix cache scout | Enabled, one `max_tokens=1` scout before measured runs |

Result summary:

| Metric | Value |
|---|---:|
| Selected concurrency | `1` |
| Completed | `30/30` |
| Correct | `28/30` (`93.3%`) |
| Hit `max_tokens` | `0` |
| Aggregate generation throughput | `81.3 tok/s` |
| Aggregate E2E throughput | `80.6 tok/s` |
| Mean per-request generation throughput | `81.6 tok/s` |
| Completion tokens avg | `6,423` |
| Completion tokens p50 | `5,864` |
| Completion tokens p90 | `10,709` |
| Completion tokens p99 | `16,055` |
| TTFT avg | `0.61s` |

Adaptive probe details:

| Concurrency | Runs | Correct | Aggregate gen tok/s | Selected |
|---:|---:|---:|---:|---|
| `1` | `30/30` | `28/30` | `81.3` | yes |
| `2` | `2/2` | `2/2` | `67.2` | no |

The two wrong final answers were both `Latvia` instead of `Estonia`:

| Run | Completion tokens | Final answer |
|---:|---:|---|
| `1` | `9,222` | `**Answer:** Latvia` |
| `22` | `16,099` | `Answer: Latvia` |

## What had to be added to vLLM

The upstream SGLang patches could not be copied into vLLM as a mechanical backport. vLLM needed additional integration work around its scheduler metadata, CUDA graph capture, speculative decode API, block/page tables, prefix cache, and custom allreduce stack.

Main vLLM-side areas touched during the port. The final image does not include a `.git` checkout under `/opt/vllm`, so this table is a functional source audit of the shipped image rather than a clean upstream diffstat.

| Area | Why it was needed |
|---|---|
| `vllm/v1/attention/backends/mla/b12x_mla_sparse.py` | New GLM-5.1 b12x sparse MLA backend, fixed-capacity b12x workspaces, CUDA-graph-compatible workspace usage, NSA extend/decode plumbing, and stable runtime tensor contracts. |
| `vllm/v1/attention/backends/mla/indexer.py` | b12x sparse indexer integration, page/block metadata conversion, fixed workspace capacity, supertile-K capacity, and removal of host shape hints that caused CUTE/JIT specialization. |
| `vllm/model_executor/layers/sparse_attn_indexer.py` | Front-end sparse indexer module, `index_topk_pattern` support, b12x NSA API compatibility, and contract phantom threading where available. |
| `vllm/model_executor/layers/fused_moe/b12x_moe.py` | b12x MoE path, import/env cleanup, and prewarm hooks for common prefill/decode M shapes. |
| `vllm/model_executor/kernels/linear/nvfp4/b12x.py` | b12x FP4/NVFP4 GEMM integration and avoiding fallback kernels that are wrong or slow for this stack. |
| `vllm/model_executor/models/deepseek_mtp.py` | GLM-5.1 MTP/NVFP4 model loading, Hugging Face cache resolution for remote speculative model ids, and compatibility with the ModelOpt FP4 draft layer. |
| `vllm/v1/spec_decode/llm_base_proposer.py` | vLLM speculative proposer integration for MTP, local argmax reduction, and draft/target sharing behavior. |
| `vllm/v1/worker/gpu_model_runner.py` | CUDA graph/no-copy capture setup, preinstalling arenas before capture, prewarm passes, and making the b12x integration stable under vLLM's runner lifecycle. |
| `vllm/config/speculative.py` | MTP configuration extensions used by GLM-5.1 and b12x MoE draft execution. |
| `vllm/distributed/device_communicators/custom_all_reduce.py` | Runtime wiring for b12x PCIe oneshot allreduce inside vLLM, including the env-controlled path and topology handling. |

Notable vLLM-only fixes:

- `os.getenv()` calls were removed from per-layer/per-decode hot paths and moved to startup/module constants. This was a measurable CPU-side decode improvement and was a vLLM port artifact, not something seen in Luke's SGLang path.
- The b12x sparse indexer was changed from live-shape scratch allocation to fixed-capacity arena/workspace usage.
- Decode/indexer workspace and MLA arena paths were made CUDA-graph compatible where required because SGLang runs this way and the vLLM path should not require a slower eager-only backend. One NSA extend workspace in the final source still uses eager workspace creation, so this should not be read as "every b12x workspace uses `use_cuda_graph=True`".
- Runtime CUTE/JIT specialization was reduced by making the scratch capacity stable and passing live lengths as runtime data instead of changing launcher-visible structural shapes.
- vLLM had to prewarm b12x MoE and attention/indexer paths during engine startup so the first real long prompt does not repeatedly trigger compiles.
- `index_topk_pattern` was wired through `--hf-overrides` so the GLM-5.1 NSA skip pattern can be supplied at launch.
- MTP uses `use_local_argmax_reduction=true`; otherwise full-vocab allreduce creates unnecessary decode overhead.

## How the fast prefill was reached

The original symptom was visible in `nvitop`: during a long prefill, GPUs would run briefly, go idle, run again, then idle again. That pattern looked like CPU synchronization or runtime compilation between chunks.

The useful finding was that the stalls were not just "normal prefill". They were caused by runtime b12x/CUTE/torch compilation being triggered from shape-varying vLLM metadata and scratch tensors during chunked prefill.

The fixes that mattered:

1. The b12x NSA/indexer workspace was made fixed-capacity instead of sized from the current prompt/chunk shape.
2. The K-side scratch capacity was made stable using full supertile capacity instead of live active width.
3. Host hints such as `active_width_hint` were removed from the launcher-visible structural key. Live lengths are passed as runtime tensors instead.
4. Workspace/arena install was moved earlier so CUDA graph capture sees stable buffers.
5. CUDA graph/no-copy mode was enabled for the b12x sparse path rather than falling back to eager-only behavior.
6. b12x MoE kernels were prewarmed for common M values: `1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192`.
7. Persistent cache directories were mounted for CUTE DSL, torchinductor, Triton, vLLM, and CUDA extensions so any remaining compile products survive restarts.

After these changes, long prefill became much smoother. This is the main difference versus the earlier vLLM state where long prefill repeatedly paused mid-request.

Luke's SGLang comment was that SGLang "never compiles at runtime" because the kernels are written to be dynamic with respect to sequence length. That matches the diagnosis: vLLM was accidentally exposing sequence-length-derived host shapes to the JIT/launcher cache. The fix was to make the vLLM/b12x contract closer to SGLang's dynamic runtime contract.

## Why the PCIe allreduce barrier is present

The final image includes a local patch in b12x `pcie_oneshot.cu`. The patch adds a completion barrier after the reduce loop:

```cpp
multi_gpu_barrier<ngpus, false>(sg, self_sg, rank);
```

Reason:

- vLLM's CUDA graph/no-copy path reuses allreduce buffers aggressively.
- On our 8-GPU PCIe topology, especially across NUMA domains, the unbarriered oneshot allreduce could let one rank reuse a slot while another rank was still peer-reading it.
- The corruption was easiest to reproduce with MTP/speculative decode plus `VLLM_ENABLE_PCIE_ALLREDUCE=1`.
- Eager execution could appear fine while CUDA graph mode failed, which made this look like an MTP/model problem until the allreduce path was isolated.

Correctness testing during the investigation:

| Variant | Result |
|---|---|
| Unpatched b12x PCIe oneshot | Development tests and model-level smoke tests showed CUDA graph/no-copy corruption in the MTP/allreduce path. |
| Patched b12x PCIe oneshot with completion barrier | The GLM-5.1 MTP path returned coherent output again under the tested vLLM launch. |
| Final image audit | The final image has the barrier line at `pcie_oneshot.cu:187` and the patched file hash listed above. |

Cost:

- A/B testing against a no-barrier image suggests about a 1 tok/s no-MTP decode cost on the local system.
- Example observed by user: about `50.4 tok/s` without the barrier path versus about `49.1 tok/s` with the barrier path in a no-MTP local test.

This barrier is the most important correctness delta versus vanilla b12x 0.11.1. It should be reviewed upstream. A lower-cost upstream solution might use a safer slot-lifetime/epoch protocol instead of a full completion barrier, but the current barrier is the known working option for this vLLM CUDA graph/no-copy configuration.

## Difference from vanilla b12x

The Docker reports `b12x 0.11.1`, but the installed b12x package is not byte-for-byte identical to the vanilla `b12x==0.11.1` wheel. The shipped package was compared against a freshly downloaded vanilla wheel. Ignoring generated `__pycache__` files, exactly three b12x source files differ:

| b12x file | Vanilla hash | Final image hash | Purpose |
|---|---|---|---|
| `attention/nsa_indexer/api.py` | `4944b39c856c7f9aab014f2839bd7fedefc07cb2a29326a048220c4b8cb4be1a` | `14d1fdb4585fbdc71ba828c97cd7fc013e2c15b57142e6f8ca7d3352a7587411` | Avoids a CUDA-device `.amax().item()` metadata sync in vLLM's DCP/MTP decode path and gates page-id validation behind `B12X_NSA_VALIDATE_PAGE_IDS`. |
| `attention/nsa_indexer/tiled_topk.py` | `dcf2ef22f197a9d37d6d4b427cd6d30ae9dac021fe434c5905d661c3c87a5195` | `d6e920cf2431f8bdc5311f6e06d86cd142de0682663a5688b6f26143cb1596e7` | Makes selected 1D row/index tensors dynamic in the CUTE launcher cache key and bumps the key to `tiled_topk_v18_dynamic_rows`, reducing sequence-length-driven recompilation. |
| `distributed/pcie_oneshot.cu` | `3944baa025e33d98cd7a9e79434d9dad5e42437e1db04a3e234ef3b6a166dd42` | `2b8ccaf2294fc10795c923540eff2c8578d65d87a0fef92348d0fe6736673ee7` | Adds the PCIe oneshot completion barrier described above. |

Docker history confirms these runtime overlays were baked after `pip install --upgrade b12x==0.11.1 --no-deps`:

```text
COPY patches/b12x_runtime/tiled_topk_dynamic_rows.patch ...
COPY patches/b12x_runtime/nsa_indexer_api.py ...
COPY vllm /opt/venv/lib/python3.12/site-packages/vllm ...
```

So the accurate description is: final image = vanilla `b12x==0.11.1` plus three audited source overlays, plus the custom vLLM tree. It is not a broad b12x fork, but it is also not just the upstream wheel.

Open upstream question for Luke/b12x:

- Is the missing completion barrier a real b12x race for all CUDA graph/no-copy users, or only exposed by vLLM's buffer reuse policy?
- If the barrier is too expensive, can b12x expose a slot lifetime or stream-ordering contract that gives the same correctness without globally stalling the reduce group?
- Should the b12x sparse MLA/indexer API make the "dynamic sequence length, stable structural cache key" contract explicit so vLLM cannot accidentally pass shape-varying host hints again?

## Difference from Luke's SGLang patches

Luke's SGLang patches already contain the core b12x backend ideas: NSA attention, b12x MoE, ModelOpt FP4/NVFP4, GLM-5.1 skip pattern, and speculative decode support.

The vLLM port had to solve additional vLLM-specific problems:

| Topic | SGLang state | vLLM-specific work |
|---|---|---|
| Runtime compile behavior | SGLang kernels are dynamic with respect to sequence length and do not compile during normal prefill. | vLLM initially caused runtime compiles by passing live shape data into structural launcher keys. Fixed via fixed-capacity arenas, runtime tensors for live lengths, and prewarm. |
| CUDA graph/no-copy | SGLang uses its own graph/runtime model. | vLLM needed arena preinstallation before capture and stable no-copy source tensors. |
| Metadata | SGLang owns its server-side metadata path. | vLLM needed conversion from scheduler/block-table/page metadata to b12x sparse NSA/indexer metadata. |
| MTP | SGLang speculative path is integrated with its runner. | vLLM needed speculative config extensions, draft model loading fixes, local argmax reduction, and target/draft sharing compatibility. |
| Allreduce | SGLang has its own launch/server allreduce integration. | vLLM needed a custom allreduce communicator path and exposed the PCIe oneshot buffer lifetime race under CUDA graph/no-copy. |
| Env/config hot path | SGLang did not have the expensive `os.getenv()` checks in the vLLM per-layer decode path. | vLLM hot-path env reads had to be moved to startup/module constants. |
| DCP | SGLang DCP support is not the same code path. | vLLM DCP8 work required extra metadata and allreduce correctness work; it was not finalized in this published DCP1 reference image. |

In short: the vLLM patchset is not just "Luke's SGLang patch copied over." The backend math is shared, but vLLM needed substantial integration work around graph capture, scheduling metadata, cache-key stability, speculative decode, and allreduce correctness.

## Decode speed notes

There were two separate performance tracks:

1. Prefill stalls were fixed by removing dynamic JIT/shape specialization and prewarming stable b12x paths.
2. Decode speed still has a tradeoff from the PCIe allreduce completion barrier.

Reference images preserved during the investigation:

| Image | Purpose | Image id |
|---|---|---|
| `voipmonitor/vllm:ref-glm51-fast-nomtp-graphtrue-nocopy-b12x0111-20260502` | Fast no-MTP reference without the PCIe completion barrier. Useful for measuring raw decode speed. | `sha256:8915f484f98445d28133509e4b9d8fe089a1719cc864b47bbfd5bc9d5b389b42` |
| `voipmonitor/vllm:glm51-mtp-pciebarrier-b12x0111-kv432k-20260502` | Correct MTP reference with PCIe completion barrier and safe KV setting. | `sha256:4fc8f9dd9b33beda530e3865cdf388e93aecc4cc460b967582f463fd22aff11e` |

Comparing all `1686` Python files under `/opt/vllm/vllm` between the fast no-MTP reference and the final barrier image showed only one changed Python file, and the diff was one default:

```text
VLLM_B12X_MLA_SPEC_SERIAL_DECODE default: "1" -> "0"
```

That default is inert for no-MTP. The b12x `attention/nsa_indexer/api.py` and `attention/nsa_indexer/tiled_topk.py` hashes are identical between these two images; the meaningful no-MTP decode difference was the b12x `pcie_oneshot.cu` barrier. The no-barrier reference has `pcie_oneshot.cu` hash `3944baa025e33d98cd7a9e79434d9dad5e42437e1db04a3e234ef3b6a166dd42`; the final image has `2b8ccaf2294fc10795c923540eff2c8578d65d87a0fef92348d0fe6736673ee7`.

## DCP8 status

DCP8 was investigated but is not the final state of this report.

What was learned:

- DCP8 requires additional vLLM metadata work and correctness validation beyond the DCP1 path.
- PCIe allreduce behavior matters more for DCP8 because communication pressure is higher.
- Some DCP8 runs were slow or incoherent depending on allreduce/barrier/cuda-graph combinations.
- The final published image should be treated as the DCP1/TP8 reference, not a DCP8 production image.

## What to watch next

1. Upstream-review the b12x PCIe oneshot completion barrier or replace it with a lower-cost slot-lifetime fix.
2. Keep the fixed-capacity/dynamic-length contract for b12x NSA/indexer in vLLM; do not reintroduce host shape hints into launcher-visible keys.
3. Keep `os.getenv()` and other Python/env branching out of per-layer decode paths.
4. Measure whether the allreduce barrier cost can be hidden or narrowed to the unsafe CUDA graph/no-copy cases.
5. Finish DCP8 separately from the DCP1 reference image; do not mix DCP8 experiments into the known-good DCP1 Docker tag.
