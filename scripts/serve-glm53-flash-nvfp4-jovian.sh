#!/usr/bin/env bash
set -euo pipefail

model=${MODEL_CONTAINER:-${MODEL:-local-inference-lab/GLM-5.3-Flash-NVFP4}}
served_model_name=${SERVED_MODEL_NAME:-GLM-5.3-Flash-NVFP4}
port=${PORT:-8001}
tp=${TP:-4}
dcp=${DCP:-4}
cp_kv_cache_interleave_size=${CP_KV_CACHE_INTERLEAVE_SIZE:-4}
dcp_comm_backend=${DCP_COMM_BACKEND:-ag_rs}
prefix_caching=${PREFIX_CACHING:-1}
enforce_eager=${ENFORCE_EAGER:-0}
disable_custom_all_reduce=${DISABLE_CUSTOM_ALL_REDUCE:-0}
per_request_spec_decode_metrics=${PER_REQUEST_SPEC_DECODE_METRICS:-0}
spec_attention_backend=${SPEC_ATTENTION_BACKEND:-B12X}
attention_backend=${ATTENTION_BACKEND:-B12X}
moe_backend=${MOE_BACKEND:-humming}
max_num_seqs=${MAX_NUM_SEQS:-16}
max_model_len=${MAX_MODEL_LEN:-524288}
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-4096}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.95}
num_speculative_tokens=${NUM_SPECULATIVE_TOKENS:-5}

[[ "${num_speculative_tokens}" =~ ^[0-9]+$ ]] || {
  printf 'NUM_SPECULATIVE_TOKENS must be a non-negative integer; got %s\n' \
    "${num_speculative_tokens}" >&2
  exit 2
}
[[ "${prefix_caching}" =~ ^[01]$ ]] || {
  printf 'PREFIX_CACHING must be 0 or 1; got %s\n' "${prefix_caching}" >&2
  exit 2
}
[[ "${enforce_eager}" =~ ^[01]$ ]] || {
  printf 'ENFORCE_EAGER must be 0 or 1; got %s\n' "${enforce_eager}" >&2
  exit 2
}
[[ "${disable_custom_all_reduce}" =~ ^[01]$ ]] || {
  printf 'DISABLE_CUSTOM_ALL_REDUCE must be 0 or 1; got %s\n' \
    "${disable_custom_all_reduce}" >&2
  exit 2
}
[[ "${per_request_spec_decode_metrics}" =~ ^[01]$ ]] || {
  printf 'PER_REQUEST_SPEC_DECODE_METRICS must be 0 or 1; got %s\n' \
    "${per_request_spec_decode_metrics}" >&2
  exit 2
}

cmd=(
  /opt/venv/bin/vllm serve "${model}"
  --served-model-name "${served_model_name}"
  --host 0.0.0.0
  --port "${port}"
  --tensor-parallel-size "${tp}"
  --pipeline-parallel-size 1
  --decode-context-parallel-size "${dcp}"
  --cp-kv-cache-interleave-size "${cp_kv_cache_interleave_size}"
  --dcp-comm-backend "${dcp_comm_backend}"
  --mamba-cache-mode align
  --enable-chunked-prefill
  --dtype bfloat16
  --kv-cache-dtype fp8
  --quantization modelopt_mixed
  --attention-backend "${attention_backend}"
  --block-size 256
  --moe-backend "${moe_backend}"
  --linear-backend b12x
  --no-enable-flashinfer-autotune
  --load-format instanttensor
  --gpu-memory-utilization "${gpu_memory_utilization}"
  --max-model-len "${max_model_len}"
  --max-num-seqs "${max_num_seqs}"
  --max-num-batched-tokens "${max_num_batched_tokens}"
  --reasoning-parser glm45
  --tool-call-parser glm47
  --enable-auto-tool-choice
)

if ((disable_custom_all_reduce)); then
  cmd+=(--disable-custom-all-reduce)
fi

if ((prefix_caching)); then
  cmd+=(--enable-prefix-caching)
else
  cmd+=(--no-enable-prefix-caching)
fi

if ((enforce_eager)); then
  cmd+=(--enforce-eager)
fi

if ((num_speculative_tokens > 0)); then
  printf -v speculative_config \
    '{"method":"mtp","num_speculative_tokens":%s,"moe_backend":"humming","attention_backend":"%s"}' \
    "${num_speculative_tokens}" "${spec_attention_backend}"
  cmd+=(--speculative-config "${speculative_config}")
  if ((per_request_spec_decode_metrics)); then
    cmd+=(--per-request-spec-decode-metrics summary)
  fi
fi

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'GLM-5.3-Flash NVFP4 Jovian launch:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

exec "${cmd[@]}"
