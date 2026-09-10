# GLM-5.3-Flash NVFP4 Jovian runtime

This stack overlays the GLM-5.3 DCP, CKV, prefix-cache, and speculative-decode
runtime changes on the CUDA 13.3 / Torch 2.13 / B12X image. The build is split
so Python-only vLLM changes rebuild a small runtime overlay without rebuilding
the compiled extensions.

## Source pins

- vLLM extensions: `4dbd82b9ced13114f90e93b8b6fae0966c942a3b`
- vLLM runtime: `e7a2a9a71187550105ba182030ac4dd937227126`
- B12X extensions: `903667d36aee19320776019a31dd06d1e9255b6a`
- B12X runtime: `903667d36aee19320776019a31dd06d1e9255b6a`
- Parent image: `voipmonitor/vllm@sha256:ef565229832e1f344fbe042dd97e950ec2cae10bc2fe7c6d158a47db840574f4`

Later branch heads `d43a0aa8` (vLLM) and `34505bcf` (B12X), checked on
2026-08-27, only add Qwen runtime/kernels and do not change GLM-5.3.

## Build and serve

```bash
./scripts/build-glm53-jovian-head-image.sh
STACK=jovian ./scripts/run-glm53-flash-nvfp4-compose.sh up
```

The qualified target is TP4/DCP4, MTP5, FP8 KV, prefix caching, a 524,288-token
maximum model length, B12X sparse attention/linear/all-reduce, CKV gather, and
Humming MoE. The defaults use graph mode, 16 maximum sequences, a 4,096-token
batch ceiling, and 0.95 GPU-memory utilization. Use a cache directory specific
to the source and configuration while qualifying a new image:

```bash
CACHE_DIR=/data/vllm-cache/glm53-jovian-e7a2a9a-dcp4-mtp5 \
  STACK=jovian \
  ./scripts/run-glm53-flash-nvfp4-compose.sh up
```

Humming is the correctness-preserving MoE default. GLM-5.3 applies a SwiGLU
clamp that native B12X W4A4 does not currently reproduce. The optional
`B12X_MOE_FORCE_A16=1 MOE_BACKEND=b12x` path preserves that clamp, but measured
slower than Humming and is not the production default.

## Qualified results

On a stock-clock 4x RTX PRO 6000 Blackwell workstation, the 0.95 configuration
completed the C16/8k regression load without an OOM or container restart. vLLM
reported 14,034,786 usable KV tokens after hybrid-cache reservations (26.77
concurrent 524,288-token requests). The benchmark's raw DCP-scaled block count
is larger and must not be reported as usable capacity.

A focused run measured 7,306/8,208/8,154 tok/s prefill at 8k/64k/128k.
Sustained zero-context aggregate decode at C1/C8/C16 was
128.1/575.7/792.0 tok/s. These are initial qualification numbers, not the
>10k prefill or >170 C1 decode performance target. MTP acceptance is
workload-dependent: observed effective output per verification step ranged
from 2.50 on two coding/operations prompts to 2.84-2.96 on synthetic load;
an earlier Rust sample on the experimental lineage reached 3.37.

Two stock-clock tuning probes establish the current safe defaults:

- CKV prefetch depth 1 reserved 1,190.3 MiB/GPU for two MTP execution lanes,
  reduced usable KV to 13,456,725 tokens, and measured
  7,376/8,228/8,174 tok/s prefill. The noise-level gain does not justify the
  memory and startup-pressure cost, so the default remains depth 0.
- Raising `--max-num-batched-tokens` to 8,192 increased estimated graph memory
  to 16.95 GiB/GPU, reduced usable KV to 9,464,070 tokens, and failed during
  Humming graph capture with a TMA illegal-memory-access error. The qualified
  ceiling remains 4,096 for this image.

The exact published image is
`ghcr.io/jackzampolin/glm53-flash-nvfp4-jovian:dcp4-prefix-mtp5-e7a2a9a-b12x903667d`
with digest
`sha256:8e621a7e381f46a56ece6e6b794a38fc5f0ddf18c7d1d507d54267367c02699d`.
The corresponding vLLM review is
[`local-inference-lab/vllm#488`](https://github.com/local-inference-lab/vllm/pull/488).

## Validate

```bash
curl -fsS http://127.0.0.1:8001/health
curl -sS http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.3-Flash-NVFP4","messages":[{"role":"user","content":"Say OK."}],"max_tokens":128,"temperature":0}' | jq .
```

Confirm that `vllm:spec_decode_num_accepted_tokens_total` increases, and retain
the startup lines for model memory, KV tokens, maximum 524k concurrency, DCP4
groups, B12X PCIe all-reduce, and CKV query selection. Per-request speculative
metrics are disabled by default to avoid perturbing the production path.

For the standard endpoint characterization from a workstation:

```bash
B=~/llm-inference-bench; $B/.venv/bin/python $B/llm_decode_bench.py --port 8001 --model GLM-5.3-Flash-NVFP4 --standalone-prefill --prefill-contexts 8k,64k,128k,256k,512k --contexts 0,8k,16k,32k --concurrency 1,2,4,8,16,32 --duration 30 --max-tokens 4096 --run-burst --output ~/glm53-$(hostname)-$(date -u +%Y%m%dT%H%M%SZ).json
```
