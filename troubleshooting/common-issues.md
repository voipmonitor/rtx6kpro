# Troubleshooting -- Common Issues on RTX 6000 Pro Blackwell

## Table of Contents

- [NCCL Issues](#nccl-issues)
  - [NCCL Deadlock on Startup](#nccl-deadlock-on-startup)
  - [NCCL Graph OOM](#nccl-graph-oom)
  - [NCCL Bandwidth Underestimate on AMD Turin](#nccl-bandwidth-underestimate-on-amd-turin)
- [CUDA Errors](#cuda-errors)
  - [CUDA Illegal Memory Access (GDN/Speculative Decoding)](#cuda-illegal-memory-access)
  - [CUDA Device-Side Assert (MTP + Radix Cache)](#cuda-device-side-assert-mtp--radix-cache)
  - [CUDA OOM](#cuda-oom)
- [NaN and Output Corruption](#nan-and-output-corruption)
  - [FlashInfer CUTLASS Race Condition (NaN)](#flashinfer-cutlass-race-condition)
  - [FP8 KV Cache Garbled Output (GLM-5)](#fp8-kv-cache-garbled-output-glm-5)
  - [FP8 KV Cache Slow on Kimi K2.5 (SGLang)](#fp8-kv-cache-slow-on-kimi-k25-sglang)
  - [DeepGemm Scale Format NaN (NVFP4 on SM120)](#deepgemm-scale-format-nan)
- [Model Loading Errors](#model-loading-errors)
  - [Weight Size Mismatch (Qwen3.5 NVFP4)](#weight-size-mismatch-qwen35-nvfp4)
  - [MTP Weights Missing](#mtp-weights-missing)
  - [fastsafetensors Crash](#fastsafetensors-crash)
  - [Missing MoE Triton Kernel Configs](#missing-moe-triton-kernel-configs)
- [SM120 Compatibility Issues](#sm120-compatibility-issues)
  - [vLLM: No Valid Attention Backend for GLM-5](#vllm-no-valid-attention-backend-for-glm-5)
  - [NSA Backend Unsupported Architecture](#nsa-backend-unsupported-architecture)
  - [DeepGemm Import Error (SM120)](#deepgemm-import-error)
- [Speculative Decoding Failures](#speculative-decoding-failures)
  - [MTP Breaks Tool Calls](#mtp-breaks-tool-calls)
  - [SGLANG_ENABLE_SPEC_V2 Missing (OOM)](#sglang_enable_spec_v2-missing-oom)
  - [MTP > 3 Instability](#mtp--3-instability)
- [Docker and Environment Issues](#docker-and-environment-issues)
  - [CUDA Compat Library Mismatch](#cuda-compat-library-mismatch)
  - [Docker Build OOM / System Hang](#docker-build-oom)
  - [Custom Allreduce Breaks GLM-5](#custom-allreduce-breaks-glm-5)
- [PCIe and Stability Issues](#pcie-and-stability-issues)
  - [Surprise Link Down (PCIe ASPM)](#surprise-link-down)
  - [NCCL P2P Lockups (IOMMU/UVM)](#nccl-p2p-lockups)
  - [ZFS System Freezes](#zfs-system-freezes)
- [Miscellaneous](#miscellaneous)
  - [endoftext Token in Output](#endoftext-token-in-output)
  - [Open WebUI Stop Token Issue](#open-webui-stop-token-issue)
  - [Developer Role Trap](#developer-role-trap)

---

## NCCL Issues

### NCCL Deadlock on Startup

**Error**: GPUs at 100% utilization, ~140W power, no VRAM growth. Last log message: `vLLM is using nccl==2.28.9`. Process hangs indefinitely.

**Cause**: Incorrect `NCCL_P2P_LEVEL` for the PCIe topology, or IOMMU interfering with P2P.

**Fix**:
1. Change from `NCCL_P2P_LEVEL=4` to `NCCL_P2P_LEVEL=2` or `NCCL_P2P_LEVEL=PHB`
2. Or remove entirely and set `NCCL_P2P_DISABLE=0` to let NCCL auto-negotiate
3. Verify topology: `nvidia-smi topo -m`
4. Enable debug logging: `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL`
5. Test with `--tensor-parallel-size 1` first to isolate

**Also check**:
- Docker mount order (file mount after directory mount can cause issues)
- Add `iommu=off` to kernel boot params
- Add `options nvidia_uvm uvm_disable_hmm=1` to `/etc/modprobe.d/uvm.conf`

---

### NCCL Graph OOM

**Error**: `NCCL error: Failed to CUDA calloc 10485760 bytes`

**Cause**: NCCL graph XML allocates additional GPU memory buffers. Combined with high `--gpu-memory-utilization`, this exceeds available VRAM.

**Fix**: Reduce `--gpu-memory-utilization` from 0.95 to **0.93** or lower.

---

### NCCL Bandwidth Underestimate on AMD Turin

**Problem**: NCCL v2.28.3 hardcodes bandwidth for ALL AMD processors:
```c
// src/graph/topo.h
#define AMD_BW 16.0  // Single value for every AMD CPU
```
AMD EPYC 9575F Turin has 192-256 GB/s actual bandwidth, but NCCL thinks 16 GB/s. This causes graph search to fail, resulting in 2 channels and SIMPLE protocol instead of LL.

**Impact**: 1.5-1.9x slower AllReduce on small messages (32-256 KB range critical for inference).

**Fix (recommended)**: Use NCCL graph XML:
```bash
wget https://www.voipmonitor.org/nccl_graph_opt.xml -O /mnt/nccl_graph_opt.xml
export NCCL_GRAPH_FILE=/mnt/nccl_graph_opt.xml
```

**Fix (simpler)**: Force LL protocol:
```bash
export NCCL_P2P_LEVEL=SYS
export NCCL_PROTO=LL
```

---

## CUDA Errors

### CUDA Illegal Memory Access

**Error**:
```
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

**Context**: Occurs in GDN (Gated Delta Network) attention backend during speculative decoding under load. Scheduler dump shows `all spec tokens rejected: [-1, -1, -1, -1, -1]`.

**Workarounds**:
1. Reduce MTP tokens to 2 (MTP>3 unstable)
2. Disable speculative decoding entirely
3. Apply vLLM PR [#35219](https://github.com/vllm-project/vllm/pull/35219) (zeros freed SSM cache blocks)
4. Avoid high concurrent load with spec decode (crashes at ~48 concurrent benchmark requests)

**Open issue**: https://github.com/vllm-project/vllm/issues/34948

---

### CUDA Device-Side Assert (MTP + Radix Cache)

**Error**:
```
eagle_worker_v2.py:510 _zero_fill_draft_kv_for_cached_prefix
torch.AcceleratorError: CUDA error: device-side assert triggered
```

**Cause**: Eagle V2 speculative decoding crashes when a request hits the radix cache prefix (`#cached-token > 0`). NaN in logits propagates to the verify step. Root cause is the FlashInfer CUTLASS race condition (see below).

**Fix**: SGLang PR [#19897](https://github.com/sgl-project/sglang/pull/19897)

**Bug report**: https://github.com/sgl-project/sglang/issues/20043

---

### CUDA OOM

**Error**: `torch.cuda.OutOfMemoryError: CUDA out of memory`

**Common causes and fixes**:

| Cause | Fix |
|-------|-----|
| Kimi K2.5 on 4 GPUs | Impossible. Model needs 8x GPUs minimum (already INT4). |
| SGLANG_ENABLE_SPEC_V2 not set | Without it, NEXTN loads model twice. Set `SGLANG_ENABLE_SPEC_V2=True`. |
| NCCL graph + high gpu-util | Reduce `--gpu-memory-utilization` to 0.93 or lower. |
| fastsafetensors GPU 0 imbalance | Switch to default safetensors or reduce gpu-util. |
| GLM-5 AWQ (QuantTrio) | OOM during weight loading on 8x. Use NVFP4 instead. |

---

## NaN and Output Corruption

### FlashInfer CUTLASS Race Condition

**Error**:
```
/pytorch/aten/src/ATen/native/cuda/TensorCompare.cu:112: _assert_async_cuda_kernel:
Assertion `probability tensor contains either `inf`, `nan` or element < 0` failed.
```

**Cause**: Race condition in FlashInfer CUTLASS FP4 GEMM kernel silently corrupts memory, producing NaN values.

**Root cause fix**: FlashInfer PR [#2716](https://github.com/flashinfer-ai/flashinfer/pull/2716)

**Issue**: https://github.com/flashinfer-ai/flashinfer/issues/2708

**Workarounds** (in order of preference):
1. Use cuDNN backend: `--fp4-gemm-backend flashinfer_cudnn`
   ```bash
   pip install nvidia-cudnn-cu13==9.19.1.2
   ```
2. Upgrade to CUTLASS 4.4.1 and **wipe JIT cache**: `rm -rf /cache/jit/*`
3. Use `--enable-nan-detection` (prevents crash but may produce garbage tokens)
4. Apply luke's sampler patch (validates probabilities before multinomial sampling)

**Important**: After upgrading Docker images, old JIT cache must be wiped:
```bash
rm -rf /cache/jit/*
```

---

### FP8 KV Cache Garbled Output (GLM-5)

**Symptoms**: `--kv-cache-dtype fp8_e4m3` produces garbled output or emits 1 token and stops.

**Cause**: Missing FP8 dequantization scales in the FlashInfer backend. The ragged+paged split path reads back cached KV without undoing scale division.

**Status**: BF16 KV cache is the **only working option** for GLM-5 on SM120 as of 2026-03-08.

**Fix**: Always use `--kv-cache-dtype bf16` for GLM-5.

---

### FP8 KV Cache Slow on Kimi K2.5 (SGLang)

**Symptoms**: Using `--kv-cache-dtype fp8` with original `moonshotai/Kimi-K2.5` in SGLang drops to 16 tok/s (unusable).

**Workarounds**:
1. Use BF16 KV cache at 90 tok/s (limited context ~170K-232K)
2. Use NVFP4 checkpoint (`nvidia/Kimi-K2.5-NVFP4`) which supports FP8 KV at 55 tok/s
3. Use vLLM with TRITON_MLA + FP8 KV (requires PRs #34597 and #34795)

---

### DeepGemm Scale Format NaN

**Symptoms**: NaN in model outputs when using NVFP4 models on SM120 with DeepGemm enabled.

**Cause**: DeepGemm scale format detection hardcodes `DEEPGEMM_SCALE_UE8M0 = True` for Blackwell, but NVFP4 uses `float8_e4m3fn` scales, not `ue8m0`.

**Fix**:
```bash
sed -i "s/DEEPGEMM_SCALE_UE8M0 = DEEPGEMM_BLACKWELL/DEEPGEMM_SCALE_UE8M0 = False/" \
    /sgl-workspace/sglang/python/sglang/srt/layers/deep_gemm_wrapper/configurer.py
```

Or disable DeepGemm entirely:
```bash
export SGLANG_ENABLE_JIT_DEEPGEMM=0
export SGLANG_ENABLE_DEEP_GEMM=0
```

**SGLang PR**: [#19948](https://github.com/sgl-project/sglang/pull/19948)

---

## Model Loading Errors

### Weight Size Mismatch (Qwen3.5 NVFP4)

**Error**:
```
ValueError: Unsupported model when in features size is not multiple of 16
```
or
```
AssertionError: Tried to load weights of size torch.Size([512, 4096])
to a parameter of size torch.Size([512, 2048])
```

**Fix**: Add entries to `quantization_config.ignore` in the model's `config.json`:
```json
"ignore": [
    "...existing entries...",
    "mtp.fc"
]
```

Also add `"model.language_model.layers..mlp.gate"` to both `config.json` AND `hf_quant_config.json`.

**Relevant PRs**:
- [#35156](https://github.com/vllm-project/vllm/pull/35156): Hardcode mlp.gate as not quantizable
- [#35675](https://github.com/vllm-project/vllm/pull/35675): Fix Qwen3.5-nvfp4 MTP fc layer shape mismatch

---

### MTP Weights Missing

**Error** (vLLM):
```
ValueError: MTP speculative decoding layer 78 weights missing from checkpoint.
```

**Cause**: The model checkpoint does not include MTP layers.

**Fix**:
- For GLM-5: Use `festr2/GLM-5-NVFP4-MTP` (includes MTP layer 78 in BF16) instead of `lukealonso/GLM-5-NVFP4`
- For Qwen3.5: Use `nvidia/Qwen3.5-397B-A17B-NVFP4` (includes MTP) or `osoleve/Qwen3.5-27B-NVFP4-MTP`

---

### fastsafetensors Crash

**Error**: vLLM crashes at startup when `--load-format fastsafetensors` is used.

**Fix**: Install the package:
```bash
pip install fastsafetensors
```
Or remove the `--load-format fastsafetensors` flag.

---

### Missing MoE Triton Kernel Configs

**Error**:
```
Config file not found at .../E=257,N=256,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition.json
```

**Impact**: Sub-optimal MoE kernel performance (falls back to default configs).

**Fix**: Generate configs using the SGLang benchmark tool:
```
https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton
```

Or use `voipmonitor/llm-pytorch-blackwell:nightly` which includes pre-generated configs.

---

## SM120 Compatibility Issues

### vLLM: No Valid Attention Backend for GLM-5

**Error**:
```
ValueError: No valid attention backend found for cuda with
AttentionSelectorConfig(head_size=576, dtype=torch.bfloat16, kv_cache_dtype=auto,
use_mla=True, use_sparse=True, ...)
```

**Cause**: vLLM has no SM120-compatible attention backend that supports both MLA and sparse attention simultaneously. GLM-5 uses `qk_nope_head_dim == 192` (FlashInfer MLA requires 128).

**Status**: **Not fixable** as of 2026-03-08. GLM-5 does not run on vLLM for SM120. Use SGLang instead.

---

### NSA Backend Unsupported Architecture

**Error**:
```
RuntimeError: Assertion error (attention.hpp:159): Unsupported architecture
```

**Cause**: GLM-5 NSA prefill/decode backends default to `flashmla_sparse` / `trtllm` which are SM90/SM100 only.

**Fix**: Override to FlashInfer backend. In SGLang, apply patches that set:
```python
nsa_prefill_backend = "flashinfer"
nsa_decode_backend = "flashinfer"
```

Full patch scripts are available in the GLM-5 deployment guide or use `voipmonitor/llm-pytorch-blackwell:nightly` which includes these patches.

---

### DeepGemm Import Error

**Error**:
```
AttributeError: 'ImportError' object has no attribute 'get_num_sms'
```

**Cause**: DeepGemm import fails silently on SM120 (it requires WGMMA for SM90, TCGEN05 for SM100).

**Fix**:
```bash
export SGLANG_ENABLE_JIT_DEEPGEMM=0
export SGLANG_ENABLE_DEEP_GEMM=0
```

---

## Speculative Decoding Failures

### MTP Breaks Tool Calls

**Symptom**: Model outputs XML tool calls instead of JSON when `tool_choice='required'` and MTP is enabled. 50-70% of tool calls fail.

```xml
<tool_call>
  <function>ask_wiki</function>
  <parameters>{"question": "..."}</parameters>
</tool_call>
```

**Fix**: vLLM PR [#35936](https://github.com/vllm-project/vllm/pull/35936). With `tool_choice='auto'`, the parser handles both XML and JSON.

**Note**: "If thinking is false, even with MTP there is no problem." The combination of thinking mode + MTP causes the most tool call issues.

GLM-5 has the same issue with MTP tool calls: https://github.com/vllm-project/vllm/issues/34449

---

### SGLANG_ENABLE_SPEC_V2 Missing (OOM)

**Symptom**: Instant OOM when launching with NEXTN speculative decoding flags.

**Cause**: Without `SGLANG_ENABLE_SPEC_V2=True`, SGLang silently converts NEXTN to EAGLE and loads the full model a second time as a draft model. For GLM-5: 57 GB x 2 = 114 GB per GPU on 96 GB cards.

**Fix**:
```bash
export SGLANG_ENABLE_SPEC_V2=True
```

---

### MTP > 3 Instability

**Symptom**: Crashes with illegal memory access at high concurrency when using `num_speculative_tokens > 3`.

**Cause**: Bug in vLLM MTP implementation. PR [#35615](https://github.com/vllm-project/vllm/pull/35615) partially fixes this.

**Fix**: Use `num_speculative_tokens: 2` (the recommended sweet spot) or at most 3.

---

## Docker and Environment Issues

### CUDA Compat Library Mismatch

**Symptom**: Docker container crashes initializing CUDA when host has CUDA 13.1 and container has CUDA 13.0.

**Fix**: Mount an empty tmpfs over the compat directory:
```bash
--mount type=tmpfs,destination=/usr/local/cuda-13.0/compat
```

---

### Docker Build OOM

**Symptom**: System becomes unresponsive during Docker builds (e.g., `./build-blackwell-transformers-v5.sh`).

**Fix**: Limit max CPU cores during build to avoid running out of memory:
```bash
./build_vllm_venv.sh /root/vllmvenv 64   # limit to 64 threads
```

---

### Custom Allreduce Breaks GLM-5

**Symptom**: "Only first token is generated and the model produces nothing" when custom allreduce is enabled for GLM-5.

**Fix**: Always use `--disable-custom-all-reduce` for GLM-5 on PCIe setups.

Custom allreduce is optimized for NVLink or PCIe switch topologies. On dual-CPU systems without switches, it is slower than NCCL even for models that do work with it.

---

## PCIe and Stability Issues

### Surprise Link Down

**Symptom**: System lockups with AER error `aer_uncor_status: 0x00000020` (Surprise Link Down). GPU DynamicPowerManagement=3 causes the root port to suspend during GPU link retrain Gen1<->Gen5.

**Fix**: Add to `/etc/default/grub`:
```bash
GRUB_CMDLINE_LINUX_DEFAULT="pcie_aspm=off pcie_port_pm=off"
```

- `pcie_aspm=off` disables Active State Power Management on all PCIe links
- `pcie_port_pm=off` disables PCIe port runtime power management (CRITICAL)

Run `update-grub` and reboot.

---

### NCCL P2P Lockups

**Symptom**: NCCL hangs during initialization with P2P enabled.

**Fix (kernel params)**:
```
iommu=off
```

**Fix (modprobe)**:
```
# /etc/modprobe.d/uvm.conf
options nvidia_uvm uvm_disable_hmm=1
```

---

### ZFS System Freezes

**Symptom**: System freezes under GPU load when using ZFS filesystem.

**Fix**: Use EXT4 for the OS filesystem. ZFS caused system freezes for multiple users.

---

## Miscellaneous

### endoftext Token in Output

**Symptom**: Random `<|endoftext|>` tokens appear in output around 100K context.

**Status**: Known issue with early NVFP4 setups. More stable with later patches and updated Docker images.

---

### Open WebUI Stop Token Issue

**Symptom**: Model keeps generating after completing response in Open WebUI. Stop button appears but must be manually clicked.

**Status**: Suspected chat template or stop token handling issue in SGLang. No confirmed fix.

---

### Developer Role Trap

**Symptom**: Internal errors when using the `developer` role in API requests.

**Cause**: The `developer` role (Anthropic-specific) is not in most chat templates. In Qwen3.5 it causes internal errors. In MiniMax M2.5 it silently drops those messages.

**Fix**: Do not use the `developer` role. Use `system` or `user` roles instead. Cline does not use the developer role.

---

## Quick Reference: Essential Environment Variables

```bash
# NCCL (pick one P2P strategy)
NCCL_P2P_LEVEL=SYS              # or PHB, or 4, or NCCL_P2P_DISABLE=0
NCCL_IB_DISABLE=1
NCCL_MIN_NCHANNELS=8
NCCL_GRAPH_FILE=/mnt/nccl_graph_opt.xml   # for AMD Turin

# SGLang SM120
SGLANG_ENABLE_SPEC_V2=True       # MANDATORY for MTP
SGLANG_ENABLE_JIT_DEEPGEMM=0     # DeepGemm unsupported on SM120
SGLANG_ENABLE_DEEP_GEMM=0

# vLLM Kimi K2.5
VLLM_TEST_FORCE_FP8_MARLIN=1
VLLM_MARLIN_USE_ATOMIC_ADD=1
VLLM_MARLIN_INPUT_DTYPE=fp8

# General
SAFETENSORS_FAST_GPU=1
OMP_NUM_THREADS=8

# Kernel boot params
pcie_aspm=off pcie_port_pm=off iommu=off

# Modprobe
options nvidia_uvm uvm_disable_hmm=1
```

---

## Key GitHub References

| Link | Description |
|------|-------------|
| [vLLM #34948](https://github.com/vllm-project/vllm/issues/34948) | GDN illegal memory access with spec decode |
| [vLLM #34449](https://github.com/vllm-project/vllm/issues/34449) | GLM-5 MTP tool call issue |
| [vLLM #34597](https://github.com/vllm-project/vllm/pull/34597) | FP8 KV cache for Triton MLA (SM120) |
| [vLLM #34795](https://github.com/vllm-project/vllm/pull/34795) | FP8 KV cache with DCP for MLA |
| [vLLM #35219](https://github.com/vllm-project/vllm/pull/35219) | FlashInfer Blackwell accuracy fix |
| [vLLM #35615](https://github.com/vllm-project/vllm/pull/35615) | Fix MTP>1 tool call streaming |
| [vLLM #35675](https://github.com/vllm-project/vllm/pull/35675) | Fix Qwen3.5 MTP fc layer shape mismatch |
| [vLLM #35936](https://github.com/vllm-project/vllm/pull/35936) | Fix tool_choice='required' with MTP |
| [SGLang #19897](https://github.com/sgl-project/sglang/pull/19897) | Fix radix cache + spec decode crash |
| [SGLang #19948](https://github.com/sgl-project/sglang/pull/19948) | DeepGemm SCALE_UE8M0 fix |
| [SGLang #20043](https://github.com/sgl-project/sglang/issues/20043) | NaN crash with spec decode |
| [FlashInfer #2708](https://github.com/flashinfer-ai/flashinfer/issues/2708) | CUTLASS FP4 GEMM race condition |
| [FlashInfer #2716](https://github.com/flashinfer-ai/flashinfer/pull/2716) | Fix for CUTLASS race condition |
