#!/usr/bin/env bash
# =============================================================================
# KLD Evaluation Pipeline for Quantized Models
#
# Automates the full KLD evaluation: FP8 reference + multiple test models.
# Run inside a container with the KLD logit capture patch already applied.
#
# Usage:
#   bash kld_eval_pipeline.sh                    # run everything
#   bash kld_eval_pipeline.sh --skip-ref         # skip reference (reuse existing)
#   bash kld_eval_pipeline.sh --models nvfp4     # run only nvfp4 test
#   bash kld_eval_pipeline.sh --compute-only     # just compute KLD from existing logits
#
# Prerequisites:
#   - KLD logit capture patch applied (python patches/sglang-kld-logit-capture.py)
#   - pip install datasets (for wikitext loading)
#   - Enough disk space (~60 GB per model)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------

KLD_BASE_DIR="${KLD_BASE_DIR:-/tmp/kld}"
KLD_REF_MODEL="${KLD_REF_MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}"
KLD_REF_TP="${KLD_REF_TP:-8}"
KLD_TOKENIZER="${KLD_TOKENIZER:-Qwen/Qwen3.5-397B-A17B-FP8}"
KLD_PORT="${KLD_PORT:-5000}"
KLD_STARTUP_TIMEOUT="${KLD_STARTUP_TIMEOUT:-600}"
KLD_EVAL_SCRIPT="${KLD_EVAL_SCRIPT:-/workspace/sglang_kld_eval.py}"

# Common server flags for all models on Blackwell
COMMON_FLAGS="--trust-remote-code --disable-custom-all-reduce --attention-backend triton --host 0.0.0.0 --port ${KLD_PORT}"

# ---------------------------------------------------------------------------
# Test model definitions: model|name|vocab_size|extra_flags
# ---------------------------------------------------------------------------

declare -a ALL_MODELS=(
  "nvidia/Qwen3.5-397B-A17B-NVFP4|nvidia/NVFP4|152064|--tp 4 --quantization modelopt_fp4 --kv-cache-dtype fp8_e4m3 --moe-runner-backend flashinfer_cutlass --fp4-gemm-backend flashinfer_cudnn --mem-fraction-static 0.85"
  "lukealonso/Qwen3.5-397B-A17B-NVFP4|lukealonso/NVFP4|152064|--tp 4 --quantization modelopt_fp4 --kv-cache-dtype fp8_e4m3 --moe-runner-backend flashinfer_cutlass --fp4-gemm-backend flashinfer_cudnn --mem-fraction-static 0.85"
  "QuantTrio/Qwen3.5-397B-A17B-AWQ|QuantTrio/AWQ-INT4|248320|--tp 4 --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85"
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { log "ERROR: $*"; exit 1; }

wait_for_server() {
    local timeout=${1:-$KLD_STARTUP_TIMEOUT}
    local elapsed=0
    log "Waiting for server on port ${KLD_PORT}..."
    while [ $elapsed -lt $timeout ]; do
        if curl -s "http://localhost:${KLD_PORT}/health" 2>/dev/null | grep -q .; then
            log "Server ready (${elapsed}s)"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    die "Server did not start within ${timeout}s"
}

stop_server() {
    log "Stopping server..."
    pkill -f "sglang.launch_server" 2>/dev/null || true
    sleep 3
    # Verify GPU memory is freed
    local mem_used
    mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "$mem_used" -gt 100 ]; then
        log "WARNING: GPU 0 still has ${mem_used} MiB in use, waiting..."
        sleep 10
    fi
}

run_server() {
    local model="$1"
    local vocab_size="$2"
    local extra_flags="$3"
    local save_dir="$4"

    mkdir -p "$save_dir"
    rm -f "${save_dir}"/*.safetensors

    log "Starting server: ${model}"
    SGLANG_KLD_SAVE_DIR="$save_dir" \
    SGLANG_KLD_VOCAB_SIZE="$vocab_size" \
    SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=0 \
    NCCL_P2P_LEVEL=SYS \
    NCCL_P2P_DISABLE=1 \
    python -m sglang.launch_server \
        --model "$model" \
        $COMMON_FLAGS \
        $extra_flags \
        &

    wait_for_server
}

generate_logits() {
    local phase="$1"
    local save_dir="$2"

    log "Generating ${phase} logits -> ${save_dir}"
    python "$KLD_EVAL_SCRIPT" \
        --phase "$phase" \
        --server-url "http://localhost:${KLD_PORT}" \
        --tokenizer "$KLD_TOKENIZER" \
        --logits-dir "$save_dir"

    local file_count
    file_count=$(ls "${save_dir}"/*.safetensors 2>/dev/null | wc -l)
    log "Generated ${file_count} logit files"
}

# Handle VLM models that save 2 files per request
align_vlm_files() {
    local src_dir="$1"
    local dst_dir="$2"
    local expected_count="$3"

    local actual_count
    actual_count=$(ls "${src_dir}"/*.safetensors 2>/dev/null | wc -l)

    if [ "$actual_count" -eq "$((expected_count * 2))" ]; then
        log "VLM model detected (${actual_count} files for ${expected_count} windows) -- creating aligned symlinks"
        mkdir -p "$dst_dir"
        for i in $(seq 0 $((expected_count - 1))); do
            ln -sf "${src_dir}/$((i * 2)).safetensors" "${dst_dir}/${i}.safetensors"
        done
        echo "$dst_dir"
    else
        echo "$src_dir"
    fi
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

SKIP_REF=false
COMPUTE_ONLY=false
MODEL_FILTER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-ref)     SKIP_REF=true; shift ;;
        --compute-only) COMPUTE_ONLY=true; shift ;;
        --models)       MODEL_FILTER="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--skip-ref] [--compute-only] [--models <filter>]"
            echo "  --skip-ref      Skip FP8 reference generation (reuse existing)"
            echo "  --compute-only  Only compute KLD from existing logit files"
            echo "  --models <str>  Only run models whose name contains <str>"
            exit 0
            ;;
        *) die "Unknown argument: $1" ;;
    esac
done

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

REF_DIR="${KLD_BASE_DIR}/ref"

if [ "$COMPUTE_ONLY" = true ]; then
    log "=== Compute-only mode ==="
else
    # --- Phase 1: FP8 Reference ---
    if [ "$SKIP_REF" = true ]; then
        log "=== Skipping reference (--skip-ref) ==="
        [ -d "$REF_DIR" ] || die "Reference dir ${REF_DIR} does not exist"
    else
        log "=== Phase 1: FP8 Reference ==="
        stop_server
        run_server "$KLD_REF_MODEL" "152064" "--tp ${KLD_REF_TP} --kv-cache-dtype bfloat16 --mem-fraction-static 0.85" "$REF_DIR"
        generate_logits "ref" "$REF_DIR"
        stop_server
    fi

    # --- Phase 2: Test models ---
    for model_def in "${ALL_MODELS[@]}"; do
        IFS='|' read -r model_path model_name vocab_size extra_flags <<< "$model_def"

        # Apply filter if specified
        if [ -n "$MODEL_FILTER" ] && [[ "$model_name" != *"$MODEL_FILTER"* ]]; then
            log "Skipping ${model_name} (filter: ${MODEL_FILTER})"
            continue
        fi

        log "=== Phase 2: ${model_name} ==="
        local_dir="${KLD_BASE_DIR}/test_$(echo "$model_name" | tr '/' '_' | tr ' ' '_')"

        stop_server
        run_server "$model_path" "$vocab_size" "$extra_flags" "$local_dir"
        generate_logits "test" "$local_dir"
        stop_server
    done
fi

# --- Phase 3: Compute KLD ---
log "=== Phase 3: Compute KLD ==="

# Collect test dirs and names
TEST_DIRS=()
TEST_NAMES=()
REF_FILE_COUNT=$(ls "${REF_DIR}"/*.safetensors 2>/dev/null | wc -l)

for model_def in "${ALL_MODELS[@]}"; do
    IFS='|' read -r model_path model_name vocab_size extra_flags <<< "$model_def"

    if [ -n "$MODEL_FILTER" ] && [[ "$model_name" != *"$MODEL_FILTER"* ]]; then
        continue
    fi

    local_dir="${KLD_BASE_DIR}/test_$(echo "$model_name" | tr '/' '_' | tr ' ' '_')"

    if [ ! -d "$local_dir" ]; then
        log "WARNING: ${local_dir} not found, skipping ${model_name}"
        continue
    fi

    # Handle VLM 2x file issue
    aligned_dir=$(align_vlm_files "$local_dir" "${local_dir}_aligned" "$REF_FILE_COUNT")

    TEST_DIRS+=("$aligned_dir")
    TEST_NAMES+=("$model_name")
done

if [ ${#TEST_DIRS[@]} -eq 0 ]; then
    die "No test model logits found"
fi

python "$KLD_EVAL_SCRIPT" --phase compute \
    --ref-dir "$REF_DIR" \
    --test-dirs "${TEST_DIRS[@]}" \
    --test-names "${TEST_NAMES[@]}"

log "=== Done ==="
