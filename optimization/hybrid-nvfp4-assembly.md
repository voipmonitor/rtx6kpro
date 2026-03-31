# Hybrid NVFP4 Assembly: BF16 Sensitive Layers

Replace quality-sensitive layers in an NVFP4 checkpoint with full-precision BF16 weights from the original model. **No SGLang patches required** — layer exclusion is handled entirely through `config.json` ignore patterns.

## Layers Worth Keeping in BF16

| Layer | Why | VRAM cost |
|-------|-----|-----------|
| **Shared expert** (1/layer, 60 layers) | Runs on **every token** (unlike routed experts where 10/512 activate). Precision has outsized quality impact. | +1 GB (+0.4%) |
| **Layer 0 routed experts** (512 experts) | First layer — sets the representation for all subsequent layers. Quality-sensitive. | +3 GB (+1.3%) |

Both combined: ~237 GB vs 233 GB (+4 GB, +1.7%).

## How It Works: config.json Ignore Patterns

SGLang's `modelopt_fp4` quantization reads the `ignore` list from `config.json` → `quantization_config` section. Any layer matching an ignore pattern is loaded as unquantized BF16 instead of NVFP4.

The key patterns:

```json
{
  "quantization_config": {
    "ignore": [
      "*.mlp.shared_expert.*",
      "*.layers.0.mlp.experts*",
      "..."
    ]
  }
}
```

**How pattern matching works internally:**

1. SGLang reads `quantization_config.ignore` from `config.json` (takes priority over `hf_quant_config.json`)
2. `ModelOptFp4Config.from_config()` stores them as `exclude_modules`
3. `_get_quant_method()` calls `is_layer_excluded(prefix)` for each layer
4. Patterns are converted from glob to regex: `*.mlp.shared_expert.*` → `.*\.mlp\.shared_expert\..*`
5. For `LinearBase` layers (shared expert): returns `UnquantizedLinearMethod()` — loads as BF16
6. For `FusedMoE` layers (layer 0 experts): returns `None` → falls back to `UnquantizedFusedMoEMethod` — allocates full BF16 buffers

**Important:** The prefix SGLang sees is `model.layers.X.mlp...` (without `language_model`), so patterns must use wildcards like `*.layers.0.mlp.experts*` rather than full HuggingFace paths. The `model.language_model.layers.X...` patterns from the original checkpoint's ignore list only work for `LinearBase` layers because `is_layer_excluded` also does part-by-part matching.

## What Changes vs Pure NVFP4

| Component | NVFP4 | Hybrid | Changed? |
|-----------|-------|--------|----------|
| Routed experts layers 1-59 (512/layer) | NVFP4 (uint8 packed) | NVFP4 (uint8 packed) | No |
| **Routed experts layer 0** (512 experts) | **NVFP4 (uint8 packed)** | **BF16 (full precision)** | **Yes** |
| **Shared expert (1/layer, 60 layers)** | **NVFP4 (uint8 packed)** | **BF16 (full precision)** | **Yes** |
| Router / gate | BF16 | BF16 | No |
| Self-attention (15 layers) | BF16 | BF16 | No |
| Linear attention / GatedDeltaNet (45 layers) | BF16 | BF16 | No |
| KV cache scales (k_scale, v_scale) | FP8 | FP8 | No |
| Layer norms | BF16 | BF16 | No |
| Embeddings + lm_head | BF16 | BF16 | No |

## Prerequisites

- HuggingFace access to both models:
  - `lukealonso/Qwen3.5-397B-A17B-NVFP4` (~233 GB) — or `nvidia/Qwen3.5-397B-A17B-NVFP4`
  - `Qwen/Qwen3.5-397B-A17B` (~752 GB BF16)
- Both models downloaded to HF cache (`~/.cache/huggingface/hub/`)
- Python packages: `torch`, `safetensors`, `huggingface_hub`
- ~250 GB free disk space for output
- SGLang with CUDA 13 (e.g., `lmsysorg/sglang:dev-cu13`)

## Step 1: Assemble the Hybrid Checkpoint

The assembly script takes:
- Routed expert NVFP4 weights (layers 1-59) from the NVFP4 checkpoint
- Routed expert BF16 weights (layer 0) from the original model
- Shared expert BF16 weights (all layers) from the original model
- KV cache FP8 scales from the NVFP4 checkpoint
- Everything else (attention, norms, router, embeddings) from the original BF16 model

Save as `assemble_hybrid.py` and run:

```python
#!/usr/bin/env python3
"""Assemble hybrid NVFP4 model with BF16 sensitive layers.

Layer 0 routed experts + all shared experts kept in BF16.
Everything else from the NVFP4 source checkpoint.
"""

import json
import os
import re
import shutil
import logging
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

NVFP4_MODEL = os.environ.get("NVFP4_MODEL", "lukealonso/Qwen3.5-397B-A17B-NVFP4")
BF16_MODEL = os.environ.get("BF16_MODEL", "Qwen/Qwen3.5-397B-A17B")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./hybrid-nvfp4"))

# Layers whose routed experts stay in BF16
BF16_EXPERT_LAYERS = {0}


def get_layer_num(key: str) -> int | None:
    m = re.search(r"layers\.(\d+)\.", key)
    return int(m.group(1)) if m else None


def classify_key(key: str) -> str:
    """Classify tensor key to determine source model.

    Returns: 'nvfp4_expert', 'bf16_expert', 'bf16_shared', 'nvfp4_kv', 'bf16'
    """
    layer = get_layer_num(key)
    if key.endswith(".k_scale") or key.endswith(".v_scale"):
        return "nvfp4_kv"
    if ".mlp.experts." in key and ".shared_expert" not in key:
        if layer is not None and layer in BF16_EXPERT_LAYERS:
            return "bf16_expert"
        return "nvfp4_expert"
    if ".mlp.shared_expert." in key and ".shared_expert_gate" not in key:
        return "bf16_shared"
    return "bf16"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load weight map indices
    logger.info("Loading weight map indices...")
    nvfp4_idx = hf_hub_download(NVFP4_MODEL, "model.safetensors.index.json", token=HF_TOKEN)
    bf16_idx = hf_hub_download(BF16_MODEL, "model.safetensors.index.json", token=HF_TOKEN)
    with open(nvfp4_idx) as f:
        nvfp4_wm = json.load(f)["weight_map"]
    with open(bf16_idx) as f:
        bf16_wm = json.load(f)["weight_map"]
    logger.info("NVFP4 keys: %d, BF16 keys: %d", len(nvfp4_wm), len(bf16_wm))

    # Plan: which tensors from which source
    plan = {}
    for key in nvfp4_wm:
        cat = classify_key(key)
        if cat in ("nvfp4_expert", "nvfp4_kv"):
            plan[key] = {"source": "nvfp4", "file": nvfp4_wm[key], "category": cat}
        elif cat == "bf16_expert":
            # Skip ALL NVFP4 tensors for BF16 expert layers (weights + scales)
            # BF16 originals will be added from bf16_wm below
            pass
        elif cat == "bf16_shared":
            if key.endswith(".weight"):
                if key in bf16_wm:
                    plan[key] = {"source": "bf16", "file": bf16_wm[key], "category": cat}
            # else: skip .weight_scale, .weight_scale_2, .input_scale
        else:
            if key in bf16_wm:
                plan[key] = {"source": "bf16", "file": bf16_wm[key], "category": cat}
            elif not any(key.endswith(s) for s in (".input_scale", ".weight_scale", ".weight_scale_2")):
                plan[key] = {"source": "nvfp4", "file": nvfp4_wm[key], "category": "nvfp4_fallback"}

    # Add BF16 expert tensors for BF16_EXPERT_LAYERS from original model
    for key in bf16_wm:
        layer = get_layer_num(key)
        if layer is not None and layer in BF16_EXPERT_LAYERS:
            if ".mlp.experts." in key and ".shared_expert" not in key:
                plan[key] = {"source": "bf16", "file": bf16_wm[key], "category": "bf16_expert"}

    # Summary
    by_cat = defaultdict(int)
    by_src = defaultdict(int)
    for info in plan.values():
        by_cat[info["category"]] += 1
        by_src[info["source"]] += 1
    logger.info("By source: %s", dict(by_src))
    logger.info("By category: %s", dict(by_cat))
    logger.info("Total tensors: %d", len(plan))

    nvfp4_files = {info["file"] for key, info in plan.items() if info["source"] == "nvfp4"}
    bf16_files = {info["file"] for key, info in plan.items() if info["source"] == "bf16"}
    logger.info("Need %d NVFP4 shards, %d BF16 shards", len(nvfp4_files), len(bf16_files))

    # Assemble with ~5GB output shards
    MAX_SHARD_BYTES = 5 * 1024**3
    all_tensors = {}
    weight_map = {}
    shard_idx = 0
    current_shard_bytes = 0

    def flush_shard():
        nonlocal all_tensors, shard_idx, current_shard_bytes
        if not all_tensors:
            return
        shard_name = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        shard_path = OUTPUT_DIR / shard_name
        logger.info("Writing shard %s (%d tensors, %.2f GB)...",
                     shard_name, len(all_tensors), current_shard_bytes / 1e9)
        save_file(all_tensors, str(shard_path))
        for k in all_tensors:
            weight_map[k] = shard_name
        all_tensors = {}
        current_shard_bytes = 0
        shard_idx += 1

    for nvf in sorted(nvfp4_files):
        logger.info("Processing NVFP4 shard: %s", nvf)
        local = hf_hub_download(NVFP4_MODEL, nvf, token=HF_TOKEN)
        with safe_open(local, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in plan and plan[key]["source"] == "nvfp4":
                    tensor = f.get_tensor(key)
                    all_tensors[key] = tensor
                    current_shard_bytes += tensor.nbytes
                    if current_shard_bytes >= MAX_SHARD_BYTES:
                        flush_shard()

    for bf in sorted(bf16_files):
        logger.info("Processing BF16 shard: %s", bf)
        local = hf_hub_download(BF16_MODEL, bf, token=HF_TOKEN)
        with safe_open(local, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in plan and plan[key]["source"] == "bf16":
                    tensor = f.get_tensor(key)
                    all_tensors[key] = tensor
                    current_shard_bytes += tensor.nbytes
                    if current_shard_bytes >= MAX_SHARD_BYTES:
                        flush_shard()

    flush_shard()

    # Fix shard names
    total_shards = shard_idx
    final_wm = {}
    for key, sn in weight_map.items():
        final_wm[key] = sn.replace("XXXXX", f"{total_shards:05d}")
    for i in range(total_shards):
        old = OUTPUT_DIR / f"model-{i:05d}-of-XXXXX.safetensors"
        new = OUTPUT_DIR / f"model-{i:05d}-of-{total_shards:05d}.safetensors"
        if old.exists():
            old.rename(new)

    with open(OUTPUT_DIR / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {}, "weight_map": final_wm}, f, indent=2)

    # Copy config files from NVFP4 checkpoint
    for cfg in ["config.json", "generation_config.json", "tokenizer.json",
                "tokenizer_config.json", "vocab.json", "preprocessor_config.json",
                "processor_config.json", "video_preprocessor_config.json",
                "hf_quant_config.json"]:
        try:
            src = hf_hub_download(NVFP4_MODEL, cfg, token=HF_TOKEN)
            shutil.copy2(src, OUTPUT_DIR / cfg)
        except Exception:
            pass

    logger.info("=== Assembly complete: %s ===", OUTPUT_DIR)
    logger.info("Shards: %d, Total keys: %d", total_shards, len(final_wm))


if __name__ == "__main__":
    main()
```

Run:

```bash
HF_TOKEN=hf_your_token \
NVFP4_MODEL=lukealonso/Qwen3.5-397B-A17B-NVFP4 \
OUTPUT_DIR=./hybrid-nvfp4 \
python3 assemble_hybrid.py
```

Takes ~2 minutes if both models are already in HF cache.

## Step 2: Add Ignore Patterns to config.json

After assembly, add the BF16 layer patterns to `config.json`'s `quantization_config.ignore` list. This tells SGLang to load these layers as unquantized BF16 instead of NVFP4.

```bash
python3 << 'PYEOF'
import json

with open("./hybrid-nvfp4/config.json") as f:
    config = json.load(f)

qc = config["quantization_config"]
ignore = list(qc.get("ignore", []))

# Shared expert: glob pattern matches all layers
# Regex: .*\.mlp\.shared_expert\..* → matches model.layers.X.mlp.shared_expert.gate_proj etc.
if "*.mlp.shared_expert.*" not in ignore:
    ignore.insert(ignore.index("lm_head") + 1 if "lm_head" in ignore else 0,
                  "*.mlp.shared_expert.*")

# Layer 0 routed experts: matches the FusedMoE prefix model.layers.0.mlp.experts
# When is_layer_excluded returns True for FusedMoE, it falls back to UnquantizedFusedMoEMethod
if "*.layers.0.mlp.experts*" not in ignore:
    ignore.insert(ignore.index("*.mlp.shared_expert.*") + 1,
                  "*.layers.0.mlp.experts*")

qc["ignore"] = ignore
config["quantization_config"] = qc

with open("./hybrid-nvfp4/config.json", "w") as f:
    json.dump(config, f, indent=2)

# Also update hf_quant_config.json for consistency
import os
hf_path = "./hybrid-nvfp4/hf_quant_config.json"
if os.path.exists(hf_path):
    with open(hf_path) as f:
        hf_config = json.load(f)
    hf_ignore = list(hf_config.get("ignore", []))
    for pat in ["*.mlp.shared_expert.*", "*.layers.0.mlp.experts*"]:
        if pat not in hf_ignore:
            hf_ignore.insert(0, pat)
    hf_config["ignore"] = hf_ignore
    with open(hf_path, "w") as f:
        json.dump(hf_config, f, indent=2)

print("Done. Ignore patterns added.")
PYEOF
```

### Verify the config

The `quantization_config.ignore` list should start with:

```json
{
  "quantization_config": {
    "ignore": [
      "lm_head",
      "*.mlp.shared_expert.*",
      "*.layers.0.mlp.experts*",
      "model.language_model.layers.0.linear_attn*",
      "..."
    ],
    "quant_algo": "NVFP4",
    "kv_cache_scheme": { "dynamic": false, "num_bits": 8, "type": "float" }
  }
}
```

## Step 3: Launch SGLang

No patches needed. SGLang reads `config.json` → `quantization_config` → `ignore` list and handles everything natively.

```bash
SGLANG_ENABLE_SPEC_V2=True python3 -m sglang.launch_server \
  --model /path/to/hybrid-nvfp4 \
  --served-model-name Qwen3.5 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --tensor-parallel-size 4 \
  --quantization modelopt_fp4 \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code \
  --attention-backend triton \
  --moe-runner-backend cutlass \
  --fp4-gemm-backend flashinfer_cudnn \
  --cuda-graph-max-bs 4 \
  --max-running-requests 4 \
  --context-length 262144 \
  --chunked-prefill-size 32768 \
  --speculative-algo NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 5 \
  --mamba-scheduler-strategy extra_buffer \
  --page-size 64 \
  --mem-fraction-static 0.85 \
  --host 0.0.0.0 --port 5000 \
  --disable-custom-all-reduce \
  --sleep-on-idle
```

You will see "not found in params_dict" warnings during loading — these are normal (TP sharding: each rank only loads 1/4 of experts, warnings appear for experts assigned to other ranks).

### Requires

```bash
pip install nvidia-cudnn-cu13==9.19.1.2  # for flashinfer_cudnn backend
```

## How SGLang Handles Mixed BF16/NVFP4 Layers

### Shared expert (LinearBase)

Pattern `*.mlp.shared_expert.*` matches prefix `model.layers.X.mlp.shared_expert.gate_proj` via `re.fullmatch`. `_get_quant_method` returns `UnquantizedLinearMethod()` → allocates standard BF16 weight buffers → loads BF16 `.weight` tensors directly (NVFP4 scale tensors are absent and skipped).

### Layer 0 routed experts (FusedMoE)

Pattern `*.layers.0.mlp.experts*` matches prefix `model.layers.0.mlp.experts` via `re.fullmatch`. `_get_quant_method` returns `None` for this FusedMoE layer → SGLang falls back to `UnquantizedFusedMoEMethod` which allocates full BF16 `w13_weight` and `w2_weight` buffers → per-expert BF16 weight tensors load via the standard `weight_loader` with correct TP sharding.

### Config file priority

SGLang reads config in this order:
1. `config.json` → `quantization_config` section (preferred, via HuggingFace `AutoConfig`)
2. `hf_quant_config.json` (legacy fallback, only if `quantization_config` is absent from `config.json`)

Both files should have matching ignore lists for consistency, but only `config.json` matters at runtime.

## Choosing What to Keep in BF16

The assembly script uses `BF16_EXPERT_LAYERS = {0}` by default. You can customize:

| Configuration | `BF16_EXPERT_LAYERS` | Pattern to add | VRAM delta |
|---------------|---------------------|----------------|-----------|
| Shared expert only | `set()` | `*.mlp.shared_expert.*` | +1 GB |
| + Layer 0 experts | `{0}` | + `*.layers.0.mlp.experts*` | +4 GB |
| + Layer 59 experts | `{0, 59}` | + `*.layers.59.mlp.experts*` | +7 GB |

Layer 59 (last layer, directly affects logits) is the next most impactful after layer 0, but adds another ~3 GB.

## Why Not BF16 Routed Experts on All Layers?

512 experts × 60 layers × 3 projections × BF16 would require the full ~752 GB BF16 model. The point of NVFP4 is to fit on 4 GPUs (~96 GB each). Keeping 1-2 layers in BF16 is the sweet spot: minimal VRAM cost, maximum quality impact on the most sensitive positions in the network.

## NVIDIA vs lukealonso Source Checkpoint

Both work as the NVFP4 source. Key difference:

| Property | nvidia | lukealonso |
|----------|--------|------------|
| Shared expert | NVFP4 | **BF16 already** (no assembly needed for shared expert) |
| KV cache scales | Calibrated FP8 | Calibrated FP8 |
| `config.json` format | `hf_quant_config.json` only | Both `config.json` + `hf_quant_config.json` |
| Ignore list | Minimal | Full per-layer patterns (linear_attn, self_attn, shared_expert_gate) |

**Recommendation:** Use `lukealonso/Qwen3.5-397B-A17B-NVFP4` — it already has shared expert in BF16 and proper ignore patterns in `config.json`. You only need to add `*.layers.0.mlp.experts*` for layer 0 BF16 and assemble the layer 0 weights from the original model.
