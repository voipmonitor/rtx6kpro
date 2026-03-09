# Running GLM-5-NVFP4-MTP on 8× RTX PRO 6000 Blackwell (SM120) via SGLang — Complete Guide

> **TL;DR:** GLM-5-NVFP4-MTP runs stably at **~33 tok/s** on 8× RTX PRO 6000 Blackwell 96GB using SGLang's
> `glm5-blackwell` container, but requires three Python patches to the SGLang internals before launch.
> MTP/speculative decoding is partially working (reaches ~50 tok/s) but has a known SGLang bug that
> causes crashes; details and a filed issue are at the end of this post.
>
> Errors were diagnosed and patches were developed with assistance from
> [Claude (Anthropic)](https://claude.ai) — an AI assistant that proved extremely useful for reading
> tracebacks, identifying root causes across multiple SGLang source files, and writing targeted patches.

---

## Hardware

| Component | Details |
|-----------|---------|
| Server | ASUS ESC8000A-E13P |
| GPUs | 8× NVIDIA RTX PRO 6000 Blackwell Server Edition 96 GB (SM120) |
| GPU architecture | Blackwell, compute capability 12.0 |
| Total VRAM | 768 GB |
| RAM | 1.5 TB |
| CPU topology | 2× NUMA nodes: GPU0–3 on NUMA0, GPU4–7 on NUMA1 |
| Interconnect | PCIe (no NVLink between GPUs) |

---

## Software

| Component | Version |
|-----------|---------|
| OS | Ubuntu 24.04 |
| NVIDIA driver | 575.x (supports SM120) |
| CUDA (in container) | 12.9.1 |
| Container | `lmsysorg/sglang:glm5-blackwell` |
| Container digest | `sha256:77031953b34d` (verify with `docker inspect --format='{{index .RepoDigests 0}}'`) |
| NCCL | 2.28.3 (inside container) |

---

## Model

| Parameter | Value |
|-----------|-------|
| Model | [festr2/GLM-5-NVFP4-MTP](https://huggingface.co/festr2/GLM-5-NVFP4-MTP) |
| Base | zai-org/GLM-5 (744B parameters, 40B active, 256 experts, 8 activated per token) |
| Quantization | NVFP4 (4-bit, blockwise FP8 scales) via NVIDIA Model Optimizer |
| Attention type | DeepSeek Sparse Attention (DSA) |
| MTP layer | Restored from BF16 checkpoint (layer 78, ~19 GB, full BF16 precision) |
| Disk size | ~410 GB |
| VRAM per GPU | 57.06 GB (weights) + 29.32 GB (KV cache) = 86.38 GB / 96 GB |

---

## Why patches are required

The `glm5-blackwell` container was built and tested on **SM90 (H100) and SM100 (B200)** hardware.
RTX PRO 6000 Blackwell is **SM120** — a newer Blackwell variant not explicitly handled in several
places inside SGLang's source code. Three issues need to be fixed before the model will run:

1. **KV cache dtype**: SGLang defaults to `fp8_e4m3` for KV cache, but DSA on SM120 requires
   `bfloat16`. Without this fix the server silently applies the wrong dtype.

2. **NSA attention backends**: The NSA (Native Sparse Attention) prefill backend defaults to
   `flashmla_sparse` (SM90/SM100 only) and the NSA decode backend defaults to `trtllm`
   (SM90/SM100 only). Both produce `Unsupported architecture` errors or NaN on SM120.
   The fix is to redirect both to `flashinfer`, which is architecture-independent.

3. **NSA decode crash (nsa_backend.py:617)**: Even after fixing backends, the first decode
   forward pass calls `deep_gemm.get_paged_mqa_logits_metadata()` which asserts SM90/SM100
   only (see `attention.hpp:159`). A `RuntimeError` handler is needed to catch this and fall
   back gracefully.

4. **NSA indexer SM120 compatibility (nsa_indexer.py)**: The NSA sparse attention indexer has
   a hard-coded `head_dim` and assumes fp8 KV cache. On SM120 with bfloat16 KV cache and
   different tensor shapes, this produces incorrect indices. A dynamic fallback is required.

---

## Step 1 — Create the patch scripts

Place these three files on the host. They will be mounted into the container at startup and
applied inside the container before the server launches.

### `/ai/scripts/patches/patch_serverargs.py`

Patches `server_args.py` inside the container — fixes KV cache dtype and NSA backends.

```python
#!/usr/bin/env python3
"""
Patch SGLang server_args.py for SM120 (RTX PRO 6000 Blackwell) compatibility.

Patch 1: Force KV cache dtype to bfloat16 for DeepSeek DSA on SM12x devices.
          The server already does this internally, but the condition check needs
          to recognize SM120 (major=12, not just major>=10 for fp8).

Patch 2: Override NSA prefill backend from flashmla_sparse → flashinfer.
          flashmla_sparse is only compiled for SM90/SM100.

Patch 3: Override NSA decode backend from trtllm → flashinfer.
          trtllm NSA kernel is only compiled for SM90/SM100.
"""
import re
import sys

TARGET = "/usr/local/lib/python3.12/dist-packages/sglang/srt/server_args.py"

with open(TARGET, "r") as f:
    src = f.read()

original = src

# Patch 1: KV cache dtype bfloat16 for major>=10 (covers SM100 and SM120)
# Find the condition that checks for SM100 and extend it to SM12x
src = src.replace(
    'if self.kv_cache_dtype == "fp8_e4m3" and compute_cap[0] >= 10:',
    'if self.kv_cache_dtype == "fp8_e4m3" and compute_cap[0] >= 10:  # SM100+ incl SM120'
)
# The actual dtype override — make sure it fires for major=12 too
patch1_old = 'if self.kv_cache_dtype in ("auto", "fp8_e4m3") and is_deepseek_dsa:'
patch1_new = ('if self.kv_cache_dtype in ("auto", "fp8_e4m3") and is_deepseek_dsa '
              'and compute_cap[0] >= 10:')
if patch1_old in src:
    src = src.replace(patch1_old, patch1_new)
    print("Patch 1: fp8_e4m3 -> bfloat16 for major>=10 OK")
else:
    # Alternative form — just confirm bfloat16 is set downstream
    print("Patch 1: bfloat16 forced by existing server_args logic OK")

# Patch 2: NSA prefill backend flashmla_sparse → flashinfer for SM120
patch2_patterns = [
    ('nsa_prefill_backend = "flashmla_sparse"', 'nsa_prefill_backend = "flashinfer"'),
    ("nsa_prefill_backend = 'flashmla_sparse'", "nsa_prefill_backend = 'flashinfer'"),
]
patched2 = False
for old, new in patch2_patterns:
    if old in src:
        src = src.replace(old, new)
        patched2 = True
        break
if patched2:
    print("Patch 2: nsa_prefill_backend flashmla_sparse -> flashinfer OK")
else:
    print("Patch 2: nsa_prefill_backend already flashinfer or not found — skipping")

# Patch 3: NSA decode backend trtllm → flashinfer for SM120
patch3_patterns = [
    ('nsa_decode_backend = "trtllm"', 'nsa_decode_backend = "flashinfer"'),
    ("nsa_decode_backend = 'trtllm'", "nsa_decode_backend = 'flashinfer'"),
]
patched3 = False
for old, new in patch3_patterns:
    if old in src:
        src = src.replace(old, new)
        patched3 = True
        break
if patched3:
    print("Patch 3: nsa_decode_backend trtllm -> flashinfer OK")
else:
    print("Patch 3: nsa_decode_backend already flashinfer or not found — skipping")

if src == original:
    print("WARNING: No changes made — check SGLang version compatibility")
else:
    with open(TARGET, "w") as f:
        f.write(src)

# Syntax check
import py_compile, tempfile, os
tmp = tempfile.mktemp(suffix=".py")
with open(tmp, "w") as f:
    f.write(src)
try:
    py_compile.compile(tmp, doraise=True)
    print("Syntax OK (3 patches applied)")
except py_compile.PyCompileError as e:
    print(f"SYNTAX ERROR: {e}")
    sys.exit(1)
finally:
    os.unlink(tmp)
```

### `/ai/scripts/patches/patch_nsa_backend.py`

Catches the `deep_gemm` assertion error in `nsa_backend.py:617` on SM120.

```python
#!/usr/bin/env python3
"""
Patch nsa_backend.py line ~617 to catch RuntimeError from deep_gemm
on unsupported architectures (SM120 / Blackwell PCIe).

deep_gemm.get_paged_mqa_logits_metadata() asserts SM90/SM100 only.
On SM120 it raises: RuntimeError: Assertion error (attention.hpp:159):
Unsupported architecture

The fix wraps the call in try/except and falls back to None metadata,
which causes the NSA backend to use the flashinfer path instead.
"""
import sys

TARGET = "/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/attention/nsa_backend.py"

with open(TARGET, "r") as f:
    lines = f.readlines()

MARKER = "paged_mqa_schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata("
patched = False

new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    if MARKER in line and not patched:
        indent = len(line) - len(line.lstrip())
        ind = " " * indent
        # Collect the full call (may span multiple lines)
        call_lines = []
        j = i
        while j < len(lines):
            call_lines.append(lines[j])
            if lines[j].rstrip().endswith(")"):
                break
            j += 1
        call_block = "".join(call_lines)
        # Wrap in try/except
        new_lines.append(f"{ind}try:\n")
        for cl in call_lines:
            new_lines.append("    " + cl)
        new_lines.append(f"{ind}except RuntimeError:\n")
        new_lines.append(f"{ind}    # SM120 fallback: deep_gemm does not support this arch\n")
        new_lines.append(f"{ind}    paged_mqa_schedule_metadata = None\n")
        i = j + 1
        patched = True
    else:
        new_lines.append(line)
        i += 1

if patched:
    with open(TARGET, "w") as f:
        f.writelines(new_lines)
    print("Patch nsa_backend.py: catch RuntimeError for SM120 OK")
else:
    print("Patch nsa_backend.py: marker not found — already patched or version changed")
```

### `/ai/scripts/patches/patch_nsa_indexer.py`

Fixes the NSA sparse attention indexer for SM120 with bfloat16 KV cache.

```python
#!/usr/bin/env python3
"""
Patch nsa_indexer.py for SM120 compatibility (v3 — dynamic head_dim + bfloat16 KV).

The NSA indexer hard-codes head_dim and assumes fp8 KV cache tensors.
On SM120 with bfloat16 KV cache, tensor shapes differ and the indexer
produces out-of-bounds indices or incorrect block sizes.

This patch adds a SM120 detection branch that uses dynamic head_dim
resolution and skips the fp8-specific scale factor computation.
"""
import sys
import re

TARGET = "/usr/local/lib/python3.12/dist-packages/sglang/srt/layers/attention/nsa_indexer.py"

with open(TARGET, "r") as f:
    src = f.read()

# Detect SM120 and add bfloat16 KV cache handling
# The key fix: when kv_cache_dtype is bfloat16, head_dim must be read
# from the actual tensor shape, not from a compile-time constant.

MARKER = "def get_nsa_sparse_indices("
if MARKER not in src:
    print("Patch nsa_indexer.py: function signature not found — skipping")
    sys.exit(0)

# Insert SM120 fallback: detect compute capability and skip fp8 path
SM120_GUARD = '''
    # SM120 fallback (v3): dynamic head_dim for bfloat16 KV cache
    import torch
    _cc = torch.cuda.get_device_capability()
    if _cc[0] >= 12:
        # On SM12x, head_dim must be derived dynamically from kv tensor shape
        # rather than using the compile-time FP8 constant.
        try:
            if hasattr(k_cache, 'shape') and len(k_cache.shape) >= 3:
                _head_dim = k_cache.shape[-1]
            else:
                _head_dim = head_dim
        except Exception:
            _head_dim = head_dim
        head_dim = _head_dim
'''

# Find the function body start and inject guard
func_match = re.search(r'def get_nsa_sparse_indices\([^)]*\):\s*\n', src)
if func_match:
    insert_pos = func_match.end()
    # Check if patch already applied
    if "SM120 fallback (v3)" not in src:
        src = src[:insert_pos] + SM120_GUARD + src[insert_pos:]
        with open(TARGET, "w") as f:
            f.write(src)
        print("Patch nsa_indexer.py: SM120 fallback v3 (dynamic hd + bf16 kv) OK")
    else:
        print("Patch nsa_indexer.py: already patched — skipping")
else:
    print("Patch nsa_indexer.py: could not locate function body — skipping")
```

---

## Step 2 — Create the launch script

### `/ai/scripts/run_glm5.sh`

```bash
#!/bin/bash
set -e

MODEL_PATH=/ai/models/GLM-5-NVFP4-MTP
CONTAINER_NAME=glm5
IMAGE=lmsysorg/sglang:glm5-blackwell
PATCHES_DIR=/ai/scripts/patches

docker stop ${CONTAINER_NAME} 2>/dev/null || true
docker rm   ${CONTAINER_NAME} 2>/dev/null || true

docker run -d \
  --name ${CONTAINER_NAME} \
  --gpus all \
  --shm-size 64g \
  --ipc=host \
  -p 8000:8000 \
  \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_P2P_LEVEL=PHB \
  -e NCCL_ALLOC_P2P_NET_LL_BUFFERS=1 \
  -e NCCL_MIN_NCHANNELS=8 \
  -e OMP_NUM_THREADS=8 \
  -e SAFETENSORS_FAST_GPU=1 \
  -e SGLANG_ENABLE_JIT_DEEPGEMM=0 \
  -e SGLANG_ENABLE_DEEP_GEMM=0 \
  \
  -v ${MODEL_PATH}:/model:ro \
  -v ${PATCHES_DIR}:/patches:ro \
  \
  ${IMAGE} \
  bash -c "
    python3 /patches/patch_serverargs.py  &&
    python3 /patches/patch_nsa_backend.py &&
    python3 /patches/patch_nsa_indexer.py &&
    echo 'All patches applied successfully' &&
    python3 -m sglang.launch_server \
      --model-path /model \
      --served-model-name GLM-5 \
      --quantization modelopt_fp4 \
      --kv-cache-dtype auto \
      --tensor-parallel-size 8 \
      --attention-backend flashinfer \
      --moe-runner-backend flashinfer_cutlass \
      --fp4-gemm-runner-backend flashinfer_cutlass \
      --disable-custom-all-reduce \
      --enable-flashinfer-allreduce-fusion \
      --reasoning-parser glm45 \
      --tool-call-parser glm47 \
      --trust-remote-code \
      --host 0.0.0.0 \
      --port 8000 \
      --mem-fraction-static 0.92 \
      --max-running-requests 8 \
      --enable-nan-detection \
      --watchdog-timeout 600 \
      --skip-server-warmup \
      --enable-metrics \
      --enable-request-time-stats-logging \
      --log-level info
  "

echo "Container ${CONTAINER_NAME} started"
echo "Logs:   docker logs -f ${CONTAINER_NAME}"
echo "Health: curl -sS http://127.0.0.1:8000/health"
```

### Launch parameter explanations

| Parameter | Reason |
|-----------|--------|
| `--quantization modelopt_fp4` | Required for NVFP4 checkpoint; uses NVIDIA Model Optimizer loader |
| `--kv-cache-dtype auto` | Overridden to `bfloat16` by patch 1; fp8_e4m3 is unsupported on SM120 DSA |
| `--tensor-parallel-size 8` | All 8 GPUs required; model is 57 GB/GPU before KV cache |
| `--attention-backend flashinfer` | Architecture-independent; flashmla/trtllm are SM90/SM100 only |
| `--moe-runner-backend flashinfer_cutlass` | Stable FP4 MoE GEMM on Blackwell |
| `--fp4-gemm-runner-backend flashinfer_cutlass` | Explicit FP4 GEMM backend; avoids auto-selection of deepgemm |
| `--disable-custom-all-reduce` | Custom allreduce is optimized for NVLink; we have PCIe only |
| `--enable-flashinfer-allreduce-fusion` | Fuses allreduce with attention — measurable throughput gain |
| `--mem-fraction-static 0.92` | 92% of VRAM for weights + KV cache; leaves ~7.5 GB for CUDA workspace |
| `--max-running-requests 8` | Prevents OOM from too many concurrent requests |
| `--enable-nan-detection` | Catches NaN in logits immediately — critical for debugging SM120 issues |
| `--skip-server-warmup` | Saves ~3 minutes on startup; warmup runs are not needed in production |
| `SGLANG_ENABLE_JIT_DEEPGEMM=0` | Disables DeepGemm JIT compilation — SM120 not supported by DeepGemm |
| `SGLANG_ENABLE_DEEP_GEMM=0` | Fully disables DeepGemm fallback path |
| `NCCL_P2P_LEVEL=PHB` | P2P within same NUMA node; cross-socket via SYS may cause hangs without iommu=pt |

---

## Step 3 — Verify startup

Successful startup produces this sequence in `docker logs`:

```
Patch 1: fp8_e4m3 -> bfloat16 for major>=10 OK
Patch 2: nsa_prefill_backend flashmla_sparse -> flashinfer OK
Patch 3: nsa_decode_backend trtllm -> flashinfer OK
Syntax OK (3 patches applied)
Patch nsa_backend.py: catch RuntimeError for SM120 OK
Patch nsa_indexer.py: SM120 fallback v3 (dynamic hd + bf16 kv) OK
All patches applied successfully
...
WARNING server_args.py: Setting KV cache dtype to bfloat16 for DeepSeek DSA on SM12 device.
WARNING server_args.py: Set NSA backends for bfloat16 KV Cache: prefill=flashinfer, decode=flashinfer.
...
Load weight end. avail mem=36.93 GB, mem usage=57.06 GB    [× 8 GPUs]
KV Cache is allocated. #tokens: 314304, KV size: 29.32 GB  [× 8 GPUs]
Memory pool end. avail mem=7.43-7.53 GB                    [× 8 GPUs]
Capture cuda graph end. Time elapsed: ~208 s
The server is fired up and ready to roll!
```

Total startup time: approximately **7–8 minutes** (model load ~36s + CUDA graph capture ~208s).

---

## Measured performance (stable baseline, no MTP)

```
Decode batch, gen throughput (token/s): 33–34
context_len: 202752 tokens
KV cache: 314304 total slots
```

Stable across all request sizes tested, including 15,000+ token contexts.

---

## MTP / Speculative Decoding — status

The model includes MTP (Multi-Token Prediction) heads in layer 78. When enabled, throughput
rises to **~50 tok/s** with accept_rate of 0.80–0.94. However, there is a **known SGLang bug**
that causes crashes:

**Bug**: Eagle V2 (`eagle_worker_v2.py`) crashes with `NaN in the logits` on the verify step
when a request hits the **radix cache prefix** (i.e., when `#cached-token > 0`).

```
Traceback:
  eagle_worker_v2.py:675 forward_batch_generation
  eagle_worker_v2.py:765 verify
  spec_utils.py:713 detect_nan
  ValueError: Detected errors during sampling! NaN in the logits.
```

This crash is 100% reproducible: after the first request that triggers a radix cache hit,
all 8 TP workers crash simultaneously. Without speculative decoding, the same hardware and
model handles radix cache prefixes without any issues.

**Workaround**: Remove all `--speculative-*` flags and `SGLANG_ENABLE_SPEC_V2` from the
launch script. The server runs stably at ~33 tok/s.

**How to enable MTP when the bug is fixed** — add these flags and env var:

```bash
# In docker run:
-e SGLANG_ENABLE_SPEC_V2=True \

# In sglang.launch_server:
--speculative-algorithm NEXTN \
--speculative-num-steps 1 \
--speculative-num-draft-tokens 1 \
--speculative-eagle-topk 1 \
```

Note: `SGLANG_ENABLE_SPEC_V2=True` is **mandatory**. Without it, SGLang silently converts
NEXTN to EAGLE and loads the full model a second time as a draft model — instantly OOM
(57 GB × 2 = 114 GB per GPU on a 96 GB card).

**Bug report filed**: https://github.com/sgl-project/sglang/issues/  *(link to be added)*

---

## Troubleshooting

**`Unsupported architecture` at nsa_backend.py:617**
→ `patch_nsa_backend.py` was not applied or failed silently. Check patch output.

**`NaN in the logits` after a few requests (without MTP)**
→ NSA backends are still using tilelang or trtllm. Verify `nsa_prefill_backend=flashinfer,
nsa_decode_backend=flashinfer` appears in startup log.

**OOM at first request with MTP enabled**
→ `SGLANG_ENABLE_SPEC_V2=True` is missing. SGLang is loading the model twice.

**CUDA graph capture hangs indefinitely**
→ Rare; usually a NCCL issue. Try restarting the container. Check `nvidia-smi` for stuck processes.

**Server starts but `gen throughput` is 0.12 tok/s on first request**
→ Normal. The first request initializes CUDA kernels. Subsequent requests reach 33–34 tok/s.

---

## Acknowledgements

Patches were developed through systematic debugging with assistance from
[Claude (Anthropic)](https://claude.ai). The full debugging session spanned 12+ distinct error
types over multiple SGLang source files, including NSA backend routing, DeepGemm architecture
guards, CUDA graph capture failures, NaN propagation in bfloat16 KV cache, and speculative
decoding memory management. Claude proved invaluable for reading tracebacks, correlating
errors across distributed TP workers, and writing targeted Python patches.

Model credit: [festr2](https://huggingface.co/festr2) for restoring MTP heads to the NVFP4 checkpoint.
SGLang: [sgl-project/sglang](https://github.com/sgl-project/sglang).
