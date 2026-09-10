#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STACK="${STACK:-chris}"

case "${STACK}" in
  chris)
    DEFAULT_COMPOSE_FILE="${ROOT_DIR}/compose/glm53-flash-nvfp4.yml"
    DEFAULT_IMAGE="cstechdev/vllm@sha256:0bd709e80b8ff13ae5de8f7d7f708a499fade3a26970d56afb1be2ff3860fde5"
    DEFAULT_NAME="glm53-flash-nvfp4-chris"
    DEFAULT_DCP=1
    DEFAULT_CP_KV_CACHE_INTERLEAVE_SIZE=1
    # The pinned cstech image cannot map the current checkpoint's MTP scales.
    DEFAULT_MTP=0
    DEFAULT_MAX_NUM_SEQS=16
    DEFAULT_MAX_MODEL_LEN=524288
    DEFAULT_MAX_BATCHED_TOKENS=8192
    DEFAULT_GPU_MEMORY_UTILIZATION=0.95
    DEFAULT_CACHE_DIR=/data/vllm-cache/glm53-flash-nvfp4-chris
    ;;
  festr)
    DEFAULT_COMPOSE_FILE="${ROOT_DIR}/compose/glm53-flash-nvfp4-festr.yml"
    DEFAULT_IMAGE="voipmonitor/vllm@sha256:ef565229832e1f344fbe042dd97e950ec2cae10bc2fe7c6d158a47db840574f4"
    DEFAULT_NAME="glm53-flash-nvfp4-festr"
    DEFAULT_DCP=1
    DEFAULT_CP_KV_CACHE_INTERLEAVE_SIZE=1
    DEFAULT_MTP=0
    DEFAULT_MAX_NUM_SEQS=16
    DEFAULT_MAX_MODEL_LEN=524288
    DEFAULT_MAX_BATCHED_TOKENS=8192
    DEFAULT_GPU_MEMORY_UTILIZATION=0.95
    DEFAULT_CACHE_DIR=/data/vllm-cache/glm53-flash-nvfp4-festr
    ;;
  jovian)
    DEFAULT_COMPOSE_FILE="${ROOT_DIR}/compose/glm53-flash-nvfp4-jovian.yml"
    DEFAULT_IMAGE="ghcr.io/jackzampolin/glm53-flash-nvfp4-jovian:dcp4-prefix-mtp5-e7a2a9a-b12x903667d"
    DEFAULT_NAME="glm53-flash-nvfp4-jovian"
    DEFAULT_DCP=4
    DEFAULT_CP_KV_CACHE_INTERLEAVE_SIZE=4
    DEFAULT_MTP=5
    DEFAULT_MAX_NUM_SEQS=16
    DEFAULT_MAX_MODEL_LEN=524288
    DEFAULT_MAX_BATCHED_TOKENS=4096
    DEFAULT_GPU_MEMORY_UTILIZATION=0.95
    DEFAULT_CACHE_DIR=/data/vllm-cache/glm53-jovian-e7a2a9a-dcp4-mtp5-s16-b4096-humming
    ;;
  *)
    printf 'ERROR: STACK must be chris, festr, or jovian; got %s\n' "${STACK}" >&2
    exit 2
    ;;
esac

COMPOSE_FILE="${COMPOSE_FILE:-${DEFAULT_COMPOSE_FILE}}"

MODEL_REPO_ROOT="${MODEL_REPO_ROOT:-/data/models/models--local-inference-lab--GLM-5.3-Flash-NVFP4}"
MODEL_REVISION="${MODEL_REVISION:-520de24eabf507659eaef7c70f14fd584527facc}"
MODEL="${MODEL:-${MODEL_REPO_ROOT}/snapshots/${MODEL_REVISION}}"
MODEL_CONTAINER="${MODEL_CONTAINER:-/model-cache/snapshots/${MODEL_REVISION}}"
IMAGE="${IMAGE:-${DEFAULT_IMAGE}}"
NAME="${NAME:-${DEFAULT_NAME}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-GLM-5.3-Flash-NVFP4}"
PORT="${PORT:-8001}"
TP="${TP:-4}"
DCP="${DCP:-${DEFAULT_DCP}}"
CP_KV_CACHE_INTERLEAVE_SIZE="${CP_KV_CACHE_INTERLEAVE_SIZE:-${DEFAULT_CP_KV_CACHE_INTERLEAVE_SIZE}}"
DCP_COMM_BACKEND="${DCP_COMM_BACKEND:-ag_rs}"
PREFIX_CACHING="${PREFIX_CACHING:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
MTP="${MTP:-${DEFAULT_MTP}}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-${DEFAULT_MAX_NUM_SEQS}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-${DEFAULT_MAX_MODEL_LEN}}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-${DEFAULT_MAX_BATCHED_TOKENS}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-${DEFAULT_GPU_MEMORY_UTILIZATION}}"
CACHE_DIR="${CACHE_DIR:-${DEFAULT_CACHE_DIR}}"
ACTION="${1:-up}"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ "${TP}" == "4" ]] || die "TP must be 4 for the symmetric workstation deployment"
[[ "${DCP}" =~ ^(1|2|4)$ ]] || die "DCP must be 1, 2, or 4"
[[ "${CP_KV_CACHE_INTERLEAVE_SIZE}" =~ ^[0-9]+$ ]] || \
  die "CP_KV_CACHE_INTERLEAVE_SIZE must be an integer"
((CP_KV_CACHE_INTERLEAVE_SIZE > 0)) || \
  die "CP_KV_CACHE_INTERLEAVE_SIZE must be positive"
[[ "${DCP_COMM_BACKEND}" =~ ^(ag_rs|a2a)$ ]] || \
  die "DCP_COMM_BACKEND must be ag_rs or a2a"
[[ "${PREFIX_CACHING}" =~ ^[01]$ ]] || die "PREFIX_CACHING must be 0 or 1"
[[ "${ENFORCE_EAGER}" =~ ^[01]$ ]] || die "ENFORCE_EAGER must be 0 or 1"
[[ "${MTP}" =~ ^[0-9]+$ ]] || die "MTP must be an integer"
[[ "${MAX_NUM_SEQS}" =~ ^[0-9]+$ ]] || die "MAX_NUM_SEQS must be an integer"
[[ "${MAX_MODEL_LEN}" =~ ^[0-9]+$ ]] || die "MAX_MODEL_LEN must be an integer"
[[ "${MAX_BATCHED_TOKENS}" =~ ^[0-9]+$ ]] || die "MAX_BATCHED_TOKENS must be an integer"

if [[ "${ACTION}" =~ ^(up|start|restart|config)$ ]] && [[ ! -f "${MODEL}/config.json" ]]; then
  die "MODEL does not look like a local HF checkpoint: ${MODEL}"
fi

mkdir -p "${CACHE_DIR}"

export MODEL MODEL_REPO_ROOT MODEL_REVISION MODEL_CONTAINER
export IMAGE NAME SERVED_MODEL_NAME PORT TP DCP MTP MAX_NUM_SEQS STACK
export CP_KV_CACHE_INTERLEAVE_SIZE DCP_COMM_BACKEND PREFIX_CACHING
export ENFORCE_EAGER
export MAX_MODEL_LEN MAX_BATCHED_TOKENS GPU_MEMORY_UTILIZATION CACHE_DIR

compose=(docker compose -p "${NAME}" -f "${COMPOSE_FILE}")
case "${ACTION}" in
  up|start)
    "${compose[@]}" up -d --remove-orphans
    ;;
  restart)
    "${compose[@]}" up -d --force-recreate --remove-orphans
    ;;
  down|stop)
    "${compose[@]}" down
    ;;
  logs)
    "${compose[@]}" logs -f --tail=200
    ;;
  ps|config)
    "${compose[@]}" "${ACTION}"
    ;;
  *)
    die "usage: $0 [up|restart|down|logs|ps|config]"
    ;;
esac

printf '%s\n' \
  "stack=${STACK}" \
  "name=${NAME}" \
  "image=${IMAGE}" \
  "model=${MODEL}" \
  "model_repo_root=${MODEL_REPO_ROOT}" \
  "model_container=${MODEL_CONTAINER}" \
  "port=${PORT}" \
  "tp=${TP}" \
  "dcp=${DCP}" \
  "cp_kv_cache_interleave_size=${CP_KV_CACHE_INTERLEAVE_SIZE}" \
  "dcp_comm_backend=${DCP_COMM_BACKEND}" \
  "prefix_caching=${PREFIX_CACHING}" \
  "enforce_eager=${ENFORCE_EAGER}" \
  "mtp=${MTP}" \
  "max_num_seqs=${MAX_NUM_SEQS}" \
  "max_model_len=${MAX_MODEL_LEN}" \
  "max_batched_tokens=${MAX_BATCHED_TOKENS}" \
  "gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}" \
  "cache_dir=${CACHE_DIR}"
