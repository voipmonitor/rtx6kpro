# GLM-5.1 cc32 SGLang vs vLLM runtime state

Date: 2026-05-03

This page records the current local DCP=1 / TP=8 state for `lukealonso/GLM-5.1-NVFP4-MTP` on RTX PRO 6000 Blackwell PCIe. The goal of this checkpoint is to preserve the exact launch state used for 32 concurrent decode testing before continuing with MTP acceptance and vLLM-vs-SGLang optimization work.

Remote host `10.229.14.14` was not used for this checkpoint. All numbers below were measured on the local 8-GPU run.

## Images

| Engine | DockerHub image | Repo digest observed and DockerHub manifest verified |
|---|---|---|
| vLLM | `voipmonitor/vllm:glm51-mtp-pciebarrier-b12x0111-kv432k-cg128-envhoist2-c32workspacefix-moechunk64-greedystrict-20260503` | `sha256:972c914d5e9385734733a1774752d0b73edc87a3a9f6a5ce9bd748e0cf29c9fc` |
| SGLang | `voipmonitor/sglang:glm51-nsa-luke-a2573ab-b12x0110-20260430` | `sha256:ebd34cf67b8dbde53abc0181eacdd903198a5b874a235b835c021ed73a26eca8` |

Important vLLM reproducibility note: the measured vLLM launcher bind-mounted a local `rejection_sampler.py` over the file inside the image.

| File | SHA256 |
|---|---|
| Image file `/opt/venv/lib/python3.12/site-packages/vllm/v1/sample/rejection_sampler.py` | `d54ae03b5a5335ea133a5644bc9ea6331db68544b77bfd4f9970bbaaf7309313` |
| Runtime bind mount `/root/vllm-nsa-vllm-backport-20260427/vllm/v1/sample/rejection_sampler.py` | `e51acacd6b763b273a7050c8b227be2719378700f08ab81903f5848f9e33bd8b` |

The vLLM image itself contained these package versions when checked:

| Package | Version |
|---|---|
| `vllm` | `0.0.0+local` |
| `b12x` | `0.11.1` |
| `torch` | `2.11.0+cu130` |
| `flashinfer-python` | `0.6.8` |
| `transformers` | `5.3.0` |

## vLLM launch used for cc32

The vLLM launcher defaults to an aggressive non-distribution-preserving acceptance threshold. For this measurement the threshold was explicitly overridden to the safe/default target-only setting:

```bash
cd /root/vllm-nsa-vllm-backport-20260427

NAME=vllm-glm51-mtp-default-cc32-20260503 \
PORT=5275 \
GPUS=0,1,2,3,4,5,6,7 \
CACHE_ROOT=/root/.cache/vllm-glm51-mtp-default-cc32 \
SPEC_ACCEPT_THRESHOLD_ACC=1.0 \
SPEC_ACCEPT_THRESHOLD_SINGLE=1.0 \
MAX_NUM_SEQS=32 \
MAX_CUDAGRAPH_CAPTURE_SIZE=128 \
GPU_MEMORY_UTILIZATION=0.82 \
MAX_MODEL_LEN=65536 \
MAX_NUM_BATCHED_TOKENS=8192 \
IMAGE=voipmonitor/vllm:glm51-mtp-pciebarrier-b12x0111-kv432k-cg128-envhoist2-c32workspacefix-moechunk64-greedystrict-20260503 \
./tools/launch_scripts/run_vllm_glm51_mtp_cg256_benchmark.sh
```

Key effective settings from the launcher:

| Setting | Value |
|---|---|
| Model | `lukealonso/GLM-5.1-NVFP4-MTP` |
| Served model name | `GLM-5` |
| Tensor parallel | `8` |
| Attention backend | `B12X_MLA_SPARSE` |
| MoE backend | `b12x` |
| KV cache dtype | `fp8` |
| Speculative config | MTP, `num_speculative_tokens=3`, `rejection_sample_method=probabilistic`, `moe_backend=b12x`, `use_local_argmax_reduction=true` |
| Acceptance thresholds | `SPEC_ACCEPT_THRESHOLD_ACC=1.0`, `SPEC_ACCEPT_THRESHOLD_SINGLE=1.0` |
| Max running sequences | `32` |
| CUDA graph capture cap | `128` |
| NCCL | `NCCL_P2P_LEVEL=SYS`, `NCCL_GRAPH_FILE=/opt/vllm/nccl_graph_opt.xml` when `/mnt/nccl_graph_opt.xml` exists |
| PCIe allreduce | `VLLM_ENABLE_PCIE_ALLREDUCE=1`, `VLLM_PCIE_ONESHOT_ALLOW_CROSS_NUMA=0` |

Observed vLLM startup memory for this measurement:

| Metric | Value |
|---|---:|
| Available KV cache memory | `19.12 GiB` |
| GPU KV cache size | `329,792 tokens` |
| Maximum concurrency for `65,536` tokens/request | `5.03x` |

## SGLang launch used for cc32

The SGLang run used the default MTP acceptance behavior. No `--speculative-accept-threshold-acc` override was supplied.

```bash
CONTAINER_NAME=sglang-glm51-mtp-default-cc32-local \
PORT=8000 \
MAX_RUNNING_REQUESTS=32 \
CUDA_GRAPH_MAX_BS=32 \
MEM_FRACTION_STATIC=0.82 \
/root/vllm/run_sglang_glm51_mtp_default_local.sh
```

Key effective settings from the launcher:

| Setting | Value |
|---|---|
| Image | `voipmonitor/sglang:glm51-nsa-luke-a2573ab-b12x0110-20260430` |
| Model | `lukealonso/GLM-5.1-NVFP4-MTP` |
| Served model name | `GLM-5` |
| Tensor parallel | `8` |
| Quantization | `modelopt_fp4` |
| Load format | `instanttensor` |
| Attention backend | `nsa` |
| NSA prefill/decode backend | `b12x` / `b12x` |
| MoE runner backend | `b12x` |
| FP4 GEMM backend | `b12x` |
| KV cache dtype | `fp8_e4m3` |
| Page size | `64` |
| Speculative algorithm | `EAGLE`, `--speculative-num-steps 3`, `--speculative-eagle-topk 1`, `--speculative-num-draft-tokens 4` |
| Max running requests | `32` |
| CUDA graph max batch size | `32` |
| Static memory fraction | `0.82` |
| NCCL | `NCCL_IB_DISABLE=1`, `NCCL_P2P_LEVEL=SYS`, `NCCL_GRAPH_FILE=/mnt/nccl_graph_opt.xml`, `NCCL_MIN_NCHANNELS=8` |

Observed SGLang startup memory for this measurement:

| Metric | Value |
|---|---:|
| KV cache tokens | `404,800` |
| `max_running_requests` | `32` |
| `context_len` | `202,752` |
| Available GPU memory after graph capture | about `6.6-6.8 GiB` depending on rank |

## cc32 benchmark workload

Both engines were tested with the same OpenAI-compatible request shape:

```json
{
  "model": "GLM-5",
  "messages": [
    {
      "role": "user",
      "content": "Write an endless comma-separated list of integers starting at 1. Do not explain. Do not stop. Continue until the token limit."
    }
  ],
  "stream": false,
  "max_tokens": 1024,
  "temperature": 1.0,
  "top_p": 0.95,
  "ignore_eos": true
}
```

The client launched `32` concurrent requests and counted completion tokens from `usage.completion_tokens`. All measured requests finished by length with `1024` completion tokens.

Important measurement caveat: non-streaming client wall throughput includes request tail effects and API response timing. For engine throughput, the primary number below is steady server-side decode throughput from engine logs while `32` requests were running. The client wall number is still recorded as a sanity check.

## Results

| Engine | Client wall throughput | Steady server-side decode | MTP acceptance during steady decode | Notes |
|---|---:|---:|---|---|
| vLLM | `482.1 tok/s` on the final run, `464.1 tok/s` on the prior run | median `663.8 tok/s`, mean `661.7 tok/s` after dropping first four ramp samples | median draft-token acceptance `56.95%`, median mean accept length `2.71` | `32 x 1024` tokens, no request errors |
| SGLang | `377.4 tok/s` | median `841.0 tok/s`, mean about `849 tok/s` from steady `#running-req: 32` log lines | accept len about `3.4`, accept rate about `0.85-0.91` | `32 x 1024` tokens, no request errors |

vLLM steady `#running=32` generation throughput samples after ramp:

```text
519.8, 615.1, 582.9, 655.8, 566.6, 648.1, 592.2, 668.7,
605.2, 703.6, 724.2, 614.9, 684.9, 597.3, 693.7, 646.2,
709.1, 627.0, 736.8, 619.2, 693.0, 715.7, 610.1, 692.7,
663.8, 715.9, 633.0, 741.5, 672.8, 764.2, 670.6, 790.8,
661.8
```

SGLang steady `#running-req: 32` generation throughput samples after the initial low ramp sample:

```text
834.93, 851.50, 841.02, 839.53, 877.47
```

## Current conclusion

At `cc32` with safe/default MTP acceptance, vLLM is not yet at SGLang speed. Based on steady server-side decode, vLLM is about `79%` of SGLang throughput on this workload, or roughly `21%` slower.

The most visible difference is MTP acceptance: SGLang logs around `3.4` accepted length and `0.85-0.91` accept rate, while vLLM logs a median mean accept length around `2.71` and draft-token acceptance around `57%`. That makes MTP acceptance behavior the current lead for the remaining performance gap.

## Open follow-up

The next useful work is to explain why vLLM's safe/default MTP acceptance is lower than SGLang's on the same GLM-5.1 NVFP4-MTP model and similar NSA/b12x stack. Areas to check:

- Whether vLLM and SGLang apply identical target sampling masks for `temperature=1.0`, `top_p=0.95`.
- Whether draft hidden-state handoff and normalized hidden behavior are still exactly aligned.
- Whether vLLM speculative verification is paying extra overhead or rejecting positions that SGLang accepts.
- Whether the local `rejection_sampler.py` overlay should be baked into a new DockerHub image to make the vLLM cc32 reference self-contained.
