#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_COMMIT="${VLLM_COMMIT:-4dbd82b9ced13114f90e93b8b6fae0966c942a3b}"
B12X_COMMIT="${B12X_COMMIT:-903667d36aee19320776019a31dd06d1e9255b6a}"
VLLM_RUNTIME_COMMIT="${VLLM_RUNTIME_COMMIT:-e7a2a9a71187550105ba182030ac4dd937227126}"
VLLM_RUNTIME_REPO="${VLLM_RUNTIME_REPO:-https://github.com/jackzampolin/vllm.git}"
B12X_RUNTIME_COMMIT="${B12X_RUNTIME_COMMIT:-903667d36aee19320776019a31dd06d1e9255b6a}"
CACHE_FINGERPRINT="${CACHE_FINGERPRINT:-cu133-torch213-glm53-jovian-${VLLM_RUNTIME_COMMIT:0:7}-b12x${B12X_RUNTIME_COMMIT:0:7}}"
BASE_IMAGE="${BASE_IMAGE:-local/glm53-flash-nvfp4-jovian:vllm${VLLM_COMMIT:0:7}-b12x${B12X_COMMIT:0:7}}"
IMAGE="${IMAGE:-local/glm53-flash-nvfp4-jovian:${VLLM_RUNTIME_COMMIT:0:7}-b12x${B12X_RUNTIME_COMMIT:0:7}}"

if ! docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1; then
  DOCKER_BUILDKIT=1 docker build \
    --progress=plain \
    --build-arg VLLM_COMMIT="${VLLM_COMMIT}" \
    --build-arg B12X_COMMIT="${B12X_COMMIT}" \
    --tag "${BASE_IMAGE}" \
    --file "${ROOT_DIR}/models/glm53-flash/Dockerfile.jovian-head" \
    "${ROOT_DIR}"
fi

DOCKER_BUILDKIT=1 docker build \
  --progress=plain \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  --build-arg VLLM_RUNTIME_REPO="${VLLM_RUNTIME_REPO}" \
  --build-arg VLLM_RUNTIME_COMMIT="${VLLM_RUNTIME_COMMIT}" \
  --build-arg B12X_RUNTIME_COMMIT="${B12X_RUNTIME_COMMIT}" \
  --build-arg CACHE_FINGERPRINT="${CACHE_FINGERPRINT}" \
  --tag "${IMAGE}" \
  --file "${ROOT_DIR}/models/glm53-flash/Dockerfile.jovian-runtime" \
  "${ROOT_DIR}"

docker image inspect "${IMAGE}" >/dev/null
printf 'image=%s\n' "${IMAGE}"
