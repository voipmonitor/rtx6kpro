#!/usr/bin/env bash
set -euo pipefail

readonly base_launcher=/usr/local/bin/serve-glm53-flash-nvfp4-dflash2.sh

# LMCache is opt-in so the default entrypoint preserves the qualified GLM-5.3
# serving command, allocator, scheduler, and backend configuration exactly.
if [[ ${LMCACHE_ENABLED:-0} == 0 ]]; then
  exec "${base_launcher}" "$@"
fi
if [[ ${LMCACHE_ENABLED} != 1 ]]; then
  printf 'LMCACHE_ENABLED must be 0 or 1; got %s\n' "${LMCACHE_ENABLED}" >&2
  exit 2
fi

require_positive_integer() {
  local name=$1
  local value=$2
  if [[ ! ${value} =~ ^[1-9][0-9]*$ ]]; then
    printf '%s must be a positive integer; got %s\n' "${name}" "${value}" >&2
    exit 2
  fi
}

readonly mp_host=${LMCACHE_MP_HOST:-127.0.0.1}
readonly mp_port=${LMCACHE_MP_PORT:-5555}
readonly http_port=${LMCACHE_HTTP_PORT:-8085}
readonly prometheus_port=${LMCACHE_PROMETHEUS_PORT:-9095}
readonly startup_timeout=${LMCACHE_STARTUP_TIMEOUT_SECONDS:-120}
readonly transfer_mode=${LMCACHE_TRANSFER_MODE:-lmcache_driven}
# 'retain' promotes L2-loaded chunks into L1; 'default' leaves them temporary.
readonly prefetch_policy=${LMCACHE_L2_PREFETCH_POLICY:-retain}
readonly extra_args=${LMCACHE_SERVER_EXTRA_ARGS:-}
readonly broker_dir=${LMCACHE_CUMEM_BROKER_DIR:-/cache/lmcache-cumem}
readonly chunk_size=${LMCACHE_CHUNK_SIZE:-4096}
readonly target_token_budget=${LMCACHE_TARGET_TOKEN_BUDGET:-${MAX_NUM_BATCHED_TOKENS:-4096}}
readonly max_num_seqs=${MAX_NUM_SEQS:-16}
readonly min_shm_gib=${LMCACHE_MIN_SHM_GIB:-96}
readonly lmcache_kv_cache_dtype=${LMCACHE_KV_CACHE_DTYPE:-nvfp4_ds_mla}

require_positive_integer LMCACHE_MP_PORT "${mp_port}"
require_positive_integer LMCACHE_HTTP_PORT "${http_port}"
require_positive_integer LMCACHE_PROMETHEUS_PORT "${prometheus_port}"
require_positive_integer LMCACHE_STARTUP_TIMEOUT_SECONDS "${startup_timeout}"
require_positive_integer LMCACHE_CHUNK_SIZE "${chunk_size}"
require_positive_integer LMCACHE_TARGET_TOKEN_BUDGET "${target_token_budget}"
require_positive_integer MAX_NUM_SEQS "${max_num_seqs}"
require_positive_integer LMCACHE_MIN_SHM_GIB "${min_shm_gib}"
if [[ ! ${mp_host} =~ ^[A-Za-z0-9_.:-]+$ ]]; then
  printf 'LMCACHE_MP_HOST contains unsupported characters: %s\n' \
    "${mp_host}" >&2
  exit 2
fi
case "${transfer_mode}" in
  lmcache_driven | auto | engine_driven) ;;
  *)
    printf 'LMCACHE_TRANSFER_MODE must be lmcache_driven, auto, or engine_driven; got %s\n' \
      "${transfer_mode}" >&2
    exit 2
    ;;
esac

case "${prefetch_policy}" in
  default | retain) ;;
  *)
    printf 'LMCACHE_L2_PREFETCH_POLICY must be default or retain; got %s\n' \
      "${prefetch_policy}" >&2
    exit 2
    ;;
esac
case "${lmcache_kv_cache_dtype}" in
  fp8 | fp8_e4m3 | fp8_ds_mla | nvfp4_ds_mla) ;;
  *)
    printf 'LMCACHE_KV_CACHE_DTYPE must be fp8, fp8_e4m3, fp8_ds_mla, or nvfp4_ds_mla; got %s\n' \
      "${lmcache_kv_cache_dtype}" >&2
    exit 2
    ;;
esac

shm_bytes=$(df -B1 --output=size /dev/shm | awk 'NR == 2 {print $1}')
readonly shm_bytes
readonly min_shm_bytes=$((min_shm_gib * 1024 * 1024 * 1024))
if [[ ! ${shm_bytes} =~ ^[0-9]+$ ]] || ((shm_bytes < min_shm_bytes)); then
  printf 'LMCache native transfer requires a private /dev/shm of at least %s GiB; got %s bytes\n' \
    "${min_shm_gib}" "${shm_bytes:-unknown}" >&2
  exit 2
fi

lmcache_server=(
  /opt/venv/bin/lmcache server
  --instance-id "${LMCACHE_INSTANCE_ID:-glm53-jovian-judgement-lmcache}"
  --host "${mp_host}"
  --port "${mp_port}"
  --chunk-size "${chunk_size}"
  --max-workers "${LMCACHE_MAX_WORKERS:-8}"
  --max-gpu-workers "${LMCACHE_MAX_GPU_WORKERS:-8}"
  --max-cpu-workers "${LMCACHE_MAX_CPU_WORKERS:-16}"
  --hash-algorithm blake3
  --supported-transfer-mode "${transfer_mode}"
  --separate-object-groups
  --l1-size-gb "${LMCACHE_L1_SIZE_GB:-64}"
  --l1-use-lazy
  --l1-init-size-gb "${LMCACHE_L1_INIT_SIZE_GB:-2}"
  --eviction-policy LRU
  --l2-prefetch-policy "${prefetch_policy}"
  --http-host 127.0.0.1
  --http-port "${http_port}"
  --prometheus-port "${prometheus_port}"
)

case "${LMCACHE_L2_ENABLED:-1}" in
  0) ;;
  1)
    l2_path=${LMCACHE_L2_PATH:-/lmcache-l2}
    if [[ ${l2_path} != /* || ${l2_path} == *['"\']* ]]; then
      printf 'LMCACHE_L2_PATH must be an absolute path without quotes or backslashes\n' >&2
      exit 2
    fi
    mkdir -p "${l2_path}"
    l2_config=${LMCACHE_L2_CONFIG:-}
    if [[ -z ${l2_config} ]]; then
      printf -v l2_config \
        '{"type":"fs_native","base_path":"%s","num_workers":%s,"use_odirect":false,"max_capacity_gb":%s,"eviction":{"eviction_policy":"LRU","trigger_watermark":0.8,"eviction_ratio":0.2}}' \
        "${l2_path}" "${LMCACHE_L2_WORKERS:-8}" \
        "${LMCACHE_L2_MAX_CAPACITY_GB:-512}"
    fi
    /opt/venv/bin/python -c \
      'import json, sys; value=json.loads(sys.argv[1]); assert isinstance(value, dict)' \
      "${l2_config}"
    lmcache_server+=(--l2-adapter "${l2_config}")
    ;;
  *)
    printf 'LMCACHE_L2_ENABLED must be 0 or 1; got %s\n' \
      "${LMCACHE_L2_ENABLED}" >&2
    exit 2
    ;;
esac

lmcache_pid=
vllm_pid=

signal_children() {
  local signal=$1
  local pid
  for pid in "${vllm_pid}" "${lmcache_pid}"; do
    if [[ -n ${pid} ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -"${signal}" "${pid}" 2>/dev/null || true
    fi
  done
}

wait_children() {
  local pid
  set +e
  for pid in "${vllm_pid}" "${lmcache_pid}"; do
    if [[ -n ${pid} ]]; then
      wait "${pid}" 2>/dev/null
    fi
  done
  set -e
}

handle_signal() {
  local status=$1
  trap - TERM INT
  signal_children TERM
  wait_children
  exit "${status}"
}

trap 'handle_signal 143' TERM
trap 'handle_signal 130' INT

vllm_extra_args=()

# max_num_scheduled_tokens limits target-model work, while
# max_num_batched_tokens includes any additional drafter query rows. Keeping
# the limits separate lets DFlash end target prefills on LMCache chunk
# boundaries without reducing the qualified 4096-token target budget.
draft_slots_per_request=0
case "${SPECULATOR:-mtp}" in
  dflash | dflash2)
    num_speculative_tokens=${NUM_SPECULATIVE_TOKENS:-7}
    if [[ ! ${num_speculative_tokens} =~ ^[0-9]+$ ]]; then
      printf 'NUM_SPECULATIVE_TOKENS must be a non-negative integer; got %s\n' \
        "${num_speculative_tokens}" >&2
      exit 2
    fi
    draft_slots_per_request=${num_speculative_tokens}
    ;;
  mtp) ;;
  *)
    printf 'SPECULATOR must be mtp, dflash, or dflash2; got %s\n' \
      "${SPECULATOR}" >&2
    exit 2
    ;;
esac
readonly input_token_budget=$((
  target_token_budget + draft_slots_per_request * max_num_seqs
))
export MAX_NUM_BATCHED_TOKENS=${input_token_budget}
export KV_CACHE_DTYPE=${lmcache_kv_cache_dtype}
export VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE="${VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE:-auto}"
export VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE="${VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE:-auto}"
vllm_extra_args+=(--max-num-scheduled-tokens "${target_token_budget}")

if [[ ${transfer_mode} != engine_driven ]]; then
  if [[ -L ${broker_dir} ]]; then
    printf 'LMCACHE_CUMEM_BROKER_DIR must not be a symlink: %s\n' \
      "${broker_dir}" >&2
    exit 2
  fi
  mkdir -p "${broker_dir}"
  chmod 700 "${broker_dir}"
  if [[ ! -O ${broker_dir} || ! -w ${broker_dir} || ! -x ${broker_dir} ]]; then
    printf 'LMCACHE_CUMEM_BROKER_DIR must be owned and writable by the serving user: %s\n' \
      "${broker_dir}" >&2
    exit 2
  fi
  # Both the LMCache sidecar and vLLM workers must inherit the same path. The
  # standard container recipe mounts /cache, so the default also remains valid
  # if the two processes are later separated into distinct mount namespaces.
  export LMCACHE_CUMEM_BROKER_DIR="${broker_dir}"
  export LD_PRELOAD="/opt/lmcache/lib/liblmcache_cumem_shareable.so${LD_PRELOAD:+:${LD_PRELOAD}}"
  vllm_extra_args+=(--enable-cumem-allocator)
fi

if [[ -n ${extra_args} ]]; then
  # shellcheck disable=SC2086
  lmcache_server+=(${extra_args})
fi

"${lmcache_server[@]}" &
lmcache_pid=$!

readonly health_url="http://127.0.0.1:${http_port}/healthcheck"
readonly startup_start=${SECONDS}
until curl --fail --silent --show-error --max-time 2 "${health_url}" >/dev/null; do
  if ! kill -0 "${lmcache_pid}" 2>/dev/null; then
    set +e
    wait "${lmcache_pid}"
    status=$?
    set -e
    printf 'LMCache server exited before becoming ready with status %s\n' \
      "${status}" >&2
    if ((status == 0)); then
      status=1
    fi
    exit "${status}"
  fi
  if ((SECONDS - startup_start >= startup_timeout)); then
    printf 'LMCache server did not become ready within %s seconds\n' \
      "${startup_timeout}" >&2
    signal_children TERM
    wait_children
    exit 1
  fi
  sleep 0.25
done

connector_config=$(printf \
  '{"kv_connector":"LMCacheMPConnector","kv_connector_module_path":"lmcache.integration.vllm.lmcache_mp_connector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"%s","lmcache.mp.port":%s,"lmcache.mp.mp_transfer_mode":"%s"}}' \
  "${mp_host}" "${mp_port}" "${transfer_mode}")

# A hybrid recurrent cache object is usable only when vLLM retains the exact
# recurrent state at the same token boundary as the LMCache object. The vLLM
# argument also validates that the object size is compatible with the resolved
# scheduler block size.
vllm_extra_args+=(
  --prefix-cache-retention-interval "${chunk_size}"
  --kv-transfer-config "${connector_config}"
)

"${base_launcher}" "$@" "${vllm_extra_args[@]}" &
vllm_pid=$!

set +e
wait -n -p completed_pid "${lmcache_pid}" "${vllm_pid}"
status=$?
set -e
if [[ ${completed_pid:-} == "${lmcache_pid}" && ${status} == 0 ]]; then
  printf 'LMCache server exited while vLLM was still running\n' >&2
  status=1
fi
signal_children TERM
wait_children
exit "${status}"
