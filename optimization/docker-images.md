# Docker Images & Container Setup

Docker is the primary deployment method for inference engines on RTX PRO 6000 Blackwell systems. CUDA 13.x and SM120 support require specific nightly or custom images.

## Table of Contents

- [Available Docker Images](#available-docker-images)
- [Docker Run Commands](#docker-run-commands)
- [Docker Compose Examples](#docker-compose-examples)
- [Custom Image Builds](#custom-image-builds)
- [Docker vs Venv Performance](#docker-vs-venv-performance)
- [Common Docker Issues](#common-docker-issues)
- [Tips and Best Practices](#tips-and-best-practices)

---

## Available Docker Images

### SGLang Images

| Image | Purpose | Notes |
|-------|---------|-------|
| `lmsysorg/sglang:dev-cu13` | SGLang nightly for CUDA 13 / Blackwell | Recommended starting point |
| `lmsysorg/sglang:glm5-blackwell` | GLM-5 specific build | Needs SM120 patches (built for SM90/SM100) |
| `lmsysorg/sglang:v0.5.9-cu130-amd64-runtime` | Stable release | May lag on Blackwell optimizations |

### vLLM Images

| Image | Purpose | Notes |
|-------|---------|-------|
| `vllm/vllm-openai:cu130-nightly` | vLLM nightly for CUDA 13 | Best general-purpose vLLM image |
| `vllm/vllm-openai:nightly` | vLLM generic nightly | May not have CUDA 13 support |
| `orthozany/vllm-qwen35-mtp` | Qwen3.5 with MTP patches | Cherry-picks PRs #35219, #35421, #35675 |

### Community Custom Images

| Image | Author | Purpose | Notes |
|-------|--------|---------|-------|
| `voipmonitor/vllm-openai:cu130-nightly-patched` | Festr | vLLM with Kimi K2.5 patches | Includes FP8 KV + DCP fixes |
| `voipmonitor/llm-pytorch-blackwell:nightly` | Festr | Full SGLang + vLLM for Blackwell | Recommended for GLM-5 |
| `voipmonitor/llm-pytorch-blackwell:customallreduce` | Festr | SGLang with luke's custom allreduce | For PCIe switch topologies |
| `voipmonitor/llm-pytorch-blackwell:nightly-fp4-prezero` | Festr | Experimental flashinfer pre-zero fix | For NVFP4 debugging |

### Pulling Images

```bash
# SGLang nightly
docker pull lmsysorg/sglang:dev-cu13

# vLLM nightly
docker pull vllm/vllm-openai:cu130-nightly

# Festr's comprehensive Blackwell image
docker pull voipmonitor/llm-pytorch-blackwell:nightly

# Qwen3.5 MTP patched vLLM
docker pull orthozany/vllm-qwen35-mtp:latest
```

---

## Docker Run Commands

### SGLang Docker Run

```bash
sudo docker run -it --rm \
  -v /home/gpusvr/:/home/gpusvr/ \
  --ipc=host \
  --shm-size=8g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --gpus all \
  --network host \
  lmsysorg/sglang:dev-cu13 bash
```

### vLLM Docker Run (All GPUs)

```bash
docker run -it --rm \
  --gpus all \
  --ipc=host \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --network host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:cu130-nightly bash
```

### Festr's Patched vLLM Docker (Kimi K2.5)

```bash
docker run -it --rm \
  --entrypoint /bin/bash \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v /mnt:/mnt/ \
  --ipc=host --shm-size=8g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus all --network host \
  --mount type=tmpfs,destination=/usr/local/cuda-13.0/compat \
  voipmonitor/vllm-openai:cu130-nightly-patched
```

Then inside the container: `pip install fastsafetensors`

### Festr's Blackwell Image (GLM-5)

```bash
docker run -it --rm \
    --entrypoint /bin/bash \
    --gpus all \
    --ipc=host \
    --shm-size=8g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --network host \
    --cpuset-cpus "0-63" \
    -v /root/.cache/huggingface:/root/.cache/huggingface \
    -v /mnt:/mnt \
    -v vllm-nightly-jit:/cache/jit \
    voipmonitor/llm-pytorch-blackwell:nightly
```

### Production vLLM Docker Run (Qwen3.5 MTP=2)

```bash
docker run -d --gpus all --ipc=host --shm-size=16g \
  -p 5000:8000 \
  -e NCCL_P2P_LEVEL=4 -e NCCL_IB_DISABLE=1 \
  -e SAFETENSORS_FAST_GPU=1 -e OMP_NUM_THREADS=8 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  orthozany/vllm-qwen35-mtp:latest \
  --model nvidia/Qwen3.5-397B-A17B-NVFP4 \
  --served-model-name qwen3.5 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.80 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 128 \
  --trust-remote-code \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --kv-cache-dtype fp8 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

Expected: ~130 tok/s single stream, ~1100+ tok/s at 50 concurrent users.

### Docker Run Flag Reference

| Flag | Purpose |
|------|---------|
| `--gpus all` | Pass all GPUs to the container |
| `--ipc=host` | Share IPC namespace (required for NCCL shared memory) |
| `--shm-size=8g` | Shared memory size (8-16 GB recommended) |
| `--ulimit memlock=-1` | Unlimited locked memory (required for GPU pinned memory) |
| `--ulimit stack=67108864` | 64 MB stack size |
| `--network host` | Use host networking (simplest for API servers) |
| `--cpuset-cpus "0-63"` | Pin to specific CPU cores |
| `--mount type=tmpfs,destination=/usr/local/cuda-13.0/compat` | Fix CUDA compat between host 13.1 and container 13.0 |

---

## Docker Compose Examples

### vLLM Qwen3.5 with MTP (Production)

```yaml
services:
  vllm-qwen35-mtp:
    image: orthozany/vllm-qwen35-mtp:latest
    container_name: vllm-qwen35-mtp-test
    ipc: host
    shm_size: "16g"
    ports:
      - "5001:8000"
    environment:
      - NVIDIA_VISIBLE_DEVICES=4,5,6,7
      - NCCL_P2P_LEVEL=4
      - SAFETENSORS_FAST_GPU=1
      - VLLM_LOG_STATS_INTERVAL=1
    volumes:
      - /mnt/raid0/models/nvidia/Qwen3.5-397B-A17B-NVFP4:/model
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["4", "5", "6", "7"]
              capabilities: [gpu]
    entrypoint: >
      python3 -m vllm.entrypoints.openai.api_server
      --model /model
      --served-model-name Qwen3.5-397B-A17B-NVFP4
      --host 0.0.0.0 --port 8000
      --trust-remote-code
      --tensor-parallel-size 4
      --gpu-memory-utilization 0.80
      --max-num-batched-tokens 4096
      --max-num-seqs 128
      --enable-auto-tool-choice
      --tool-call-parser qwen3_coder
      --reasoning-parser qwen3
      --mm-encoder-tp-mode data
      --mm-processor-cache-type shm
      --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":5}'
    restart: unless-stopped
```

### SGLang GLM-5 with MTP (Production)

```yaml
services:
  sglang-glm5:
    build: .
    image: sglang-glm5:latest
    container_name: sglang-glm5-nightly
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
      - CUDA_DEVICE_ORDER=PCI_BUS_ID
      - NCCL_IB_DISABLE=1
      - NCCL_P2P_LEVEL=SYS
      - NCCL_ALLOC_P2P_NET_LL_BUFFERS=1
      - NCCL_MIN_NCHANNELS=8
      - OMP_NUM_THREADS=8
      - SAFETENSORS_FAST_GPU=1
      - NCCL_CUMEM_HOST_ENABLE=0
      - FLASHINFER_DISABLE_VERSION_CHECK=1
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    volumes:
      - /mnt/raid0/models:/models:ro
      - huggingface-cache:/root/.cache/huggingface
    ports:
      - "8003:5000"
    command:
      - python3
      - -m
      - sglang.launch_server
      - --model-path=/models/festr2/GLM-5-NVFP4-MTP
      - --served-model-name=glm-5
      - --reasoning-parser=glm45
      - --tool-call-parser=glm47
      - --trust-remote-code
      - --tp=8
      - --mem-fraction-static=0.9
      - --max-running-requests=64
      - --kv-cache-dtype=bf16
      - --quantization=modelopt_fp4
      - --attention-backend=flashinfer
      - --moe-runner-backend=deep_gemm
      - --disable-custom-all-reduce
      - --cuda-graph-max-bs=32
      - --host=0.0.0.0
      - --port=5000
      - '--model-loader-extra-config={"enable_multithread_load": true, "num_threads": 8}'
      - --speculative-algorithm=EAGLE
      - --speculative-num-steps=3
      - --speculative-eagle-topk=1
      - --speculative-num-draft-tokens=4
    cpuset: "0-63"
    ipc: host
    shm_size: "8g"
    ulimits:
      memlock: -1
      stack: 67108864
```

--- \
--sleep-on-idle

## Custom Image Builds

### Patching the Official SGLang Image for GLM-5 on SM120

```dockerfile
# sglang dev-cu13 nightly pulled 2026-03-04
FROM lmsysorg/sglang@sha256:426d1fa4b10722688678b99d817c2caa92a89eed4a8ee2927ab44a848bbe77df

RUN pip install --no-cache-dir transformers==5.2.0

# Fix DeepGemm scale format detection for NVFP4 models on Blackwell (SM120)
# NVFP4 uses float8_e4m3fn scales, not ue8m0 -- hardcoded True causes NaN
RUN sed -i "s/DEEPGEMM_SCALE_UE8M0 = DEEPGEMM_BLACKWELL/DEEPGEMM_SCALE_UE8M0 = False/" \
    /sgl-workspace/sglang/python/sglang/srt/layers/deep_gemm_wrapper/configurer.py
```

### Patching vLLM for Qwen3.5 MTP

```dockerfile
FROM vllm/vllm-openai:cu130-nightly@sha256:cd7d78a3db7251ef785485bfcec2a6375f8f798691fb59e71af877d5e72d51f

COPY patches/vllm/ /usr/local/lib/python3.12/dist-packages/vllm/
```

Cherry-picks included:
- PR #35219: FlashInfer Blackwell accuracy fix, zeros freed SSM cache blocks
- PR #35421: Tool call streaming fix for speculative decoding
- PR #35675: Fix Qwen3.5-nvfp4 MTP fc layer shape mismatch

### What Festr's Comprehensive Image Contains

`voipmonitor/llm-pytorch-blackwell:nightly` includes:

| Component | Version |
|-----------|---------|
| SGLang | Compiled from source with SM120 patches |
| vLLM | Included |
| PyTorch | 2.12 |
| FlashInfer | Latest |
| CUTLASS | 4.4.1 |
| cuDNN | 91901 |
| Compilation target | SM_120f enabled |
| Triton MoE configs | Pre-generated for RTX PRO 6000 Blackwell Server Edition |

### Building vLLM from Source (Venv)

```bash
wget https://www.voipmonitor.org/build_vllm_venv.sh.txt -O build_vllm_venv.sh
chmod +x build_vllm_venv.sh
./build_vllm_venv.sh /root/venvtest
source /root/venvtest/activate.sh
```

Build time: 30-60 minutes. Limit CPU cores to avoid OOM: `./build_vllm_venv.sh /root/vllmvenv 64`.

---

## Docker vs Venv Performance

| Method | Throughput | Notes |
|--------|-----------|-------|
| Docker | ~76 tok/s | Small overhead from container layer |
| Venv (bare metal) | ~79 tok/s | Direct hardware access |
| **Delta** | **~3-4%** | **Washes away with context length** |

The performance difference is minimal and disappears entirely at longer context lengths where compute dominates over any container overhead.

---

## Common Docker Issues

### 1. CUDA Compatibility Between Host and Container

**Symptom:** Docker crashes initializing CUDA when host has CUDA 13.1 and container has CUDA 13.0.

**Fix:**
```bash
--mount type=tmpfs,destination=/usr/local/cuda-13.0/compat
```

### 2. Build Crashes / System Hangs

**Symptom:** System becomes unresponsive during Docker image builds.

**Fix:** Limit max CPU cores during build to avoid running out of memory. Use `MAX_JOBS=32` or similar.

### 3. Nightly Builds Breaking SM120 Support

**Symptom:** New nightly image doesn't work on Blackwell GPUs.

**Fix:** Use `dev` branch images or pin to known-good SHA digests.

### 4. fastsafetensors Missing

**Symptom:** vLLM crashes when `--load-format fastsafetensors` is used.

**Fix:** `pip install fastsafetensors` inside the container, or remove the `--load-format fastsafetensors` flag.

### 5. Old JIT Kernel Cache

**Symptom:** After upgrading Docker image, still getting NaN errors or old kernel behavior.

**Fix:** Clear the JIT cache:
```bash
rm -rf /cache/jit/*
```

When using a Docker volume for JIT cache (`-v vllm-nightly-jit:/cache/jit`), the old cache persists across image upgrades.

---

## Tips and Best Practices

### Model Caching

Mount your HuggingFace cache to avoid re-downloading models:

```bash
-v ~/.cache/huggingface:/root/.cache/huggingface
```

Use `HF_HUB_OFFLINE=1` to prevent models from phoning home to HuggingFace after initial download.

### GPU Selection

Select specific GPUs using environment variable:

```bash
-e NVIDIA_VISIBLE_DEVICES=4,5,6,7    # Use GPUs 4-7
-e CUDA_DEVICE_ORDER=PCI_BUS_ID      # Order by PCIe bus ID
```

Or with Docker `--gpus` flag:

```bash
--gpus '"device=0,1,2,3"'    # Specific GPUs
--gpus 4                      # First 4 GPUs
--gpus all                    # All GPUs
```

### Version Pinning

For production, pin images by SHA digest rather than tag:

```bash
docker pull lmsysorg/sglang@sha256:426d1fa4b10722688678b99d817c2caa92a89eed4a8ee2927ab44a848bbe77df
```

### Software Versions Known to Work

| Component | Version |
|-----------|---------|
| CUDA Toolkit | 13.1.1 (nvcc 13.1.115) |
| NVIDIA Driver | 590.48.01 |
| PyTorch | 2.10.0+cu130 |
| FlashInfer | 0.6.4 |
| OS | Ubuntu 24.04.3 LTS |
| Kernel | 6.17.0-14-generic |
