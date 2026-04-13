# NCCL Tuning Guide

NCCL (NVIDIA Collective Communications Library) configuration is one of the most impactful optimizations for multi-GPU inference on RTX PRO 6000 Blackwell. Proper tuning can improve AllReduce performance by 70%+ and directly increase token generation speed.

## Table of Contents

- [Essential Environment Variables](#essential-environment-variables)
- [NCCL P2P Levels](#nccl-p2p-levels)
- [NCCL Protocol Settings](#nccl-protocol-settings)
- [NCCL Channel Configuration](#nccl-channel-configuration)
- [NCCL Graph XML Fix for AMD Turin](#nccl-graph-xml-fix-for-amd-turin)
- [nvidia_uvm Fix](#nvidia_uvm-fix)
- [High-Concurrency Configuration](#high-concurrency-configuration)
- [Performance Comparison Tables](#performance-comparison-tables)
- [Debugging NCCL Issues](#debugging-nccl-issues)
- [Complete Configuration Examples](#complete-configuration-examples)
- [NCCL Tuner Plugin (voipmonitor/nccl-tuner-amd)](#nccl-tuner-plugin)

---

## Essential Environment Variables

### Core NCCL Variables

```bash
NCCL_P2P_LEVEL=SYS          # Enable P2P across entire system (cross-NUMA)
NCCL_IB_DISABLE=1            # Disable InfiniBand (required for non-IB setups)
NCCL_NET_GDR_LEVEL=SYS       # GPU Direct RDMA level
NCCL_MIN_NCHANNELS=8         # Increase from default 2 channels (major bandwidth gain)
NCCL_ALLOC_P2P_NET_LL_BUFFERS=1  # Allocate Low Latency P2P buffers
```

### Additional Useful Variables

```bash
SAFETENSORS_FAST_GPU=1       # Faster safetensors weight loading
OMP_NUM_THREADS=8            # Limit OpenMP threads to avoid contention
NVIDIA_TF32_OVERRIDE=1       # Force TF32 for applicable operations
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Better CUDA memory allocation
```

### Inference Engine Variables

```bash
# vLLM
VLLM_LOG_STATS_INTERVAL=1           # Log throughput stats every second
VLLM_WORKER_MULTIPROC_METHOD=spawn  # Required for multi-GPU vLLM
VLLM_SLEEP_WHEN_IDLE=1              # Save power when no requests
VLLM_NVFP4_GEMM_BACKEND=cutlass     # Control FP4 GEMM backend
VLLM_TEST_FORCE_FP8_MARLIN=1        # Force FP8 Marlin kernels (Kimi K2.5)
VLLM_MARLIN_USE_ATOMIC_ADD=1        # Atomic add for Marlin (Kimi K2.5)
VLLM_MARLIN_INPUT_DTYPE=fp8          # FP8 input for Marlin (Kimi K2.5)

# SGLang
SGLANG_ENABLE_SPEC_V2=True          # Enable speculative decode v2 (required for MTP)
SGLANG_DISABLE_DEEP_GEMM=1          # Disable DeepGemm (SM120 not supported)
SGLANG_ENABLE_JIT_DEEPGEMM=0        # Disable DeepGemm JIT
SGLANG_SET_CPU_AFFINITY=1           # Bind to CPU cores
HF_HUB_OFFLINE=1                    # Prevent model phoning home after download
```

---

## NCCL P2P Levels

The `NCCL_P2P_LEVEL` variable controls which interconnect paths NCCL will use for peer-to-peer communication.

| Level | Name | Meaning |
|-------|------|---------|
| 0 | LOC | Same GPU only (P2P disabled) |
| 1 | NVL | NVLink only |
| 2 | PIX | Same PCIe switch |
| 3 | PHB | Same PCIe Host Bridge (same NUMA node) |
| 4 | SYS | Across system (cross-NUMA, xGMI fabric) |

### Which Level to Use

| Setup | Recommended | Why |
|-------|-------------|-----|
| Single CPU + switches | `NCCL_P2P_LEVEL=PHB` or `SYS` | All GPUs on same NUMA |
| Dual CPU, same-NUMA GPUs only | `NCCL_P2P_LEVEL=PHB` | Avoid cross-NUMA |
| Dual CPU, cross-NUMA needed | `NCCL_P2P_LEVEL=SYS` | Enable cross-socket P2P |
| Dual CPU, all GPUs | `NCCL_P2P_LEVEL=SYS` | Required for 8-GPU TP |

### Common Pitfalls

- `NCCL_P2P_LEVEL=4` is equivalent to `NCCL_P2P_LEVEL=SYS`
- Setting level too high for the topology causes **NCCL deadlocks** (GPUs at 100% utilization, ~140W, no VRAM growth)
- If deadlock occurs, try `NCCL_P2P_LEVEL=2` or remove entirely and use `NCCL_P2P_DISABLE=0`

---

## NCCL Protocol Settings

NCCL uses different protocols based on estimated interconnect speed:

| Protocol | Description | Best For |
|----------|-------------|----------|
| **LL (Low Latency)** | 8-byte inline data in completion flags | Small messages (<64 KB), inference |
| **LL128** | 128-byte Low Latency variant | Medium messages |
| **SIMPLE** | Standard buffered protocol | Large messages (>256 KB), training |

### Forcing LL Protocol

```bash
NCCL_PROTO=LL
```

The LL protocol is the **key driver for inference speedup**. NCCL only selects LL when it believes the interconnect is fast enough. On AMD systems, NCCL's bandwidth detection is broken (see [Graph XML Fix](#nccl-graph-xml-fix-for-amd-turin)), causing it to choose SIMPLE instead of LL for small messages.

**Trade-off:** Forcing LL works well for inference (small messages) but may not be optimal for all message sizes. The Graph XML approach is more nuanced.

---

## NCCL Channel Configuration

Channels determine the parallelism of collective operations. More channels = more bandwidth, but also more GPU memory and CPU overhead.

```bash
NCCL_MIN_NCHANNELS=8     # Minimum channels (default: 2)
```

### Impact of Channel Count

| Setting | Bus BW (8 GPUs, 32M) | Notes |
|---------|----------------------|-------|
| Default (2 channels) | 22.2 GB/s | Festr, dual Turin |
| MIN_NCHANNELS=8 | 37.6 GB/s | Festr, dual Turin |
| MIN_NCHANNELS=8 | 41.1 GB/s | luke, switches |

Increasing from 2 to 8 channels provides a **70% bandwidth improvement** on dual-CPU systems.

### For High-Concurrency Workloads

When disabling P2P for high-concurrency DRAM routing:

```bash
NCCL_P2P_DISABLE=1
NCCL_MIN_NCHANNELS=16          # or 32
NCCL_MAX_NCHANNELS=32          # or 64
NCCL_BUFFSIZE=33554432         # 32 MB buffer (or 67108864 for 64 MB)
CUDA_DEVICE_MAX_CONNECTIONS=32  # or 64
```

These larger values are needed because DRAM routing uses more channels to saturate available memory bandwidth.

---

## NCCL Graph XML Fix for AMD Turin

### The Problem

NCCL v2.28.3 has a hardcoded bandwidth constant for ALL AMD processors:

```c
// src/graph/topo.h
#define AMD_BW 16.0  // Single value for every AMD CPU ever made
```

AMD EPYC 9575F Turin (xGMI3) has **192-256 GB/s actual bandwidth**, but NCCL thinks it is 16 GB/s. This **12-16x underestimate** causes the NCCL graph search to fail, resulting in:

- Wrong protocol selection (SIMPLE instead of LL)
- Wrong connection type classification (SYS instead of PHB/PIX)
- Minimum channel count (2 instead of 4+)
- Drastically reduced AllReduce performance

### Without XML Fix

| Parameter | Value |
|-----------|-------|
| Bandwidth estimate | 0.1 GB/s |
| Connection type | SYS/SYS (worst-case) |
| Channels | 2 (minimum) |
| Small message protocol | SIMPLE (higher latency) |

### With XML Fix

| Parameter | Value |
|-----------|-------|
| Bandwidth estimate | 24 GB/s (realistic) |
| Connection type | PHB/PIX (correct topology) |
| Channels | 4 |
| Small message protocol | LL (Low Latency) |

### Applying the Fix

Download the optimized NCCL graph file:

```bash
wget https://www.voipmonitor.org/nccl_graph_opt.xml -O /mnt/nccl_graph_opt.xml
```

Set the environment variable:

```bash
export NCCL_GRAPH_FILE=/mnt/nccl_graph_opt.xml
```

### Graph XML Content

```xml
<graph id="0" pattern="4" nchannels="2" speedintra="24" speedinter="24"
       typeintra="PHB" typeinter="PIX" samechannels="1">
```

### Performance Impact (AllReduce on Inference-Sized Tensors)

| Message Size | Without XML | With XML | Speedup |
|-------------|-------------|----------|---------|
| 32 KB | 48.16 us | 26.20 us | **1.84x** |
| 64 KB | 48.69 us | 25.59 us | **1.90x** |
| 128 KB | 51.56 us | 32.09 us | **1.61x** |
| 256 KB | 56.48 us | 37.26 us | **1.52x** |

### Simpler Alternative

If you do not want to use the XML file:

```bash
NCCL_P2P_LEVEL=SYS NCCL_PROTO=LL
```

This forces LL protocol for all message sizes. Festr achieved matching inference speed (~65-68 tok/s) with Turin using this approach without the XML file.

### When to Use Each Approach

| Approach | Pros | Cons |
|----------|------|------|
| Graph XML | Optimal protocol per message size | Requires file distribution |
| NCCL_PROTO=LL | Simple, no files needed | May be suboptimal for large messages |
| Neither | No configuration | 1.5-1.9x slower for inference |

---

## nvidia_uvm Fix

### The Problem

Without this fix, NCCL P2P operations lock up on multi-GPU setups.

### The Fix

Create or edit `/etc/modprobe.d/uvm.conf`:

```bash
# /etc/modprobe.d/uvm.conf
options nvidia_uvm uvm_disable_hmm=1
```

Then reload the module or reboot:

```bash
sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm
# Or simply reboot
```

### Also Required

Add `iommu=off` as a kernel boot parameter, otherwise NCCL hangs:

```bash
# In /etc/default/grub, add to GRUB_CMDLINE_LINUX:
iommu=off amd_iommu=off
```

Then `update-grub` and reboot.

Reference: [Level1Techs P2P NCCL Fix](https://forum.level1techs.com/t/dual-rtx-pro-6000-blackwell-max-q-how-to-make-p2p-nccl-work/242403/8)

---

## High-Concurrency Configuration

For high-concurrency serving (many simultaneous requests), disabling P2P and routing through DRAM can be faster.

### P2P Enabled (Low Latency, Single User)

```bash
NCCL_P2P_LEVEL=SYS
NCCL_GRAPH_FILE=/mnt/nccl_graph_opt.xml
NCCL_MIN_NCHANNELS=8
```

### P2P Disabled (High Throughput, Many Users)

```bash
NCCL_P2P_DISABLE=1
NCCL_MIN_NCHANNELS=32
NCCL_MAX_NCHANNELS=64
NCCL_BUFFSIZE=67108864
CUDA_DEVICE_MAX_CONNECTIONS=64
```

**Lighter variant:**

```bash
NCCL_P2P_DISABLE=1
NCCL_MIN_NCHANNELS=16
NCCL_MAX_NCHANNELS=32
NCCL_BUFFSIZE=33554432
CUDA_DEVICE_MAX_CONNECTIONS=32
```

**Important:** `NCCL_P2P_DISABLE=1` alone is not enough -- the channel and buffer settings are also required. This approach assumes fully populated DRAM channels.

### P2P vs No-P2P Throughput Comparison

| Concurrency | P2P Enabled | P2P Disabled | Winner |
|-------------|-------------|-------------|--------|
| 1 request | 90 tok/s | 70 tok/s | P2P |
| 100 requests | 5000 tok/s | 10000 tok/s | No-P2P |

---

## PCIe Oneshot AllReduce (Bypass NCCL for Small Messages)

For PCIe topologies (without NVLink), luke's PCIe oneshot allreduce kernel replaces NCCL for small messages (<512 KB), achieving **1.4–6× lower AllReduce latency** and **5–11% faster end-to-end decode throughput**.

See **[PCIe Oneshot AllReduce Guide](pcie-oneshot-allreduce.md)** for setup, benchmarks, and patch instructions.

Quick summary — consistent +7% across models:

| Model | Config (conc=1) | PCIe Oneshot | NCCL Only | Improvement |
|---|---|---|---|---|
| Qwen3.5-397B TP=4 | cutlass | 76.1 tok/s | 70.6 tok/s | +7.8% |
| GLM-5 TP=8 | b12x | 56.6 tok/s | 52.8 tok/s | +7.2% |

---

## Performance Comparison Tables

### AllReduce Bus Bandwidth (8 GPUs)

| Setup | NCCL Config | Bus BW (GB/s) |
|-------|-------------|---------------|
| luke (3x switches) | MIN_NCHANNELS=8 | **41.1** |
| Grimulkan (4x switches) | MIN_NCHANNELS=8 | 39.4 |
| Festr (dual Turin) | Graph XML + MIN_NCHANNELS=8 | 37.6 |
| Festr (dual Turin) | Default NCCL | 22.2 |

### Before/After NCCL Tuning on Dual Turin

| Metric | Before Tuning | After Tuning | Improvement |
|--------|--------------|--------------|-------------|
| AllReduce bus BW | 22.2 GB/s | 37.6 GB/s | +69% |
| 32 KB AllReduce latency | 48.16 us | 26.20 us | -46% |
| Inference tok/s (Kimi K2.5) | ~40 tok/s | ~65-68 tok/s | +63% |

### Effect of NCCL Graph XML on AllReduce Latency

| Message Size | Without XML (us) | With XML (us) | Improvement |
|-------------|------------------|---------------|-------------|
| 32 KB | 48.16 | 26.20 | 1.84x |
| 64 KB | 48.69 | 25.59 | 1.90x |
| 128 KB | 51.56 | 32.09 | 1.61x |
| 256 KB | 56.48 | 37.26 | 1.52x |

---

## Debugging NCCL Issues

### NCCL Deadlock Diagnosis

**Symptoms:** GPUs at 100% utilization, ~140W power, no VRAM growth. Last log: `vLLM is using nccl==2.28.9`.

**Steps:**

1. Check topology:
   ```bash
   nvidia-smi topo -m
   ```

2. Enable NCCL debug logging:
   ```bash
   NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL
   ```

3. Try reducing P2P level:
   ```bash
   NCCL_P2P_LEVEL=2  # Instead of 4/SYS
   ```

4. Or disable explicit P2P level entirely:
   ```bash
   NCCL_P2P_DISABLE=0  # Let NCCL auto-negotiate
   ```

5. Test with single GPU first:
   ```bash
   --tensor-parallel-size 1
   ```

### NCCL Graph OOM

If you see `NCCL error: Failed to CUDA calloc 10485760 bytes` when using the graph XML file:

```bash
# Reduce gpu-memory-utilization from 0.95 to 0.93
--gpu-memory-utilization 0.93
```

### NCCL Topology Notes

- NCCL uses ring "snake" topology under the hood for PCIe
- Having more natural PCIe lanes to host does **not** help (proven by Grimulkan spanning 4 switches vs 2-per-switch)
- Cross-NUMA fabric link adds ~14 us latency vs P2P on same NUMA (without Turin/NCCL tuning)
- Turin xGMI reduces this gap significantly vs Genoa

---

## Complete Configuration Examples

### Dual Turin, 8 GPUs, Low Latency (Inference)

```bash
export NCCL_P2P_LEVEL=SYS
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=SYS
export NCCL_GRAPH_FILE=/mnt/nccl_graph_opt.xml
export NCCL_MIN_NCHANNELS=8
export NCCL_ALLOC_P2P_NET_LL_BUFFERS=1
export OMP_NUM_THREADS=8
export SAFETENSORS_FAST_GPU=1
```

### Single CPU with Switches, 8 GPUs

```bash
export NCCL_P2P_LEVEL=PHB
export NCCL_IB_DISABLE=1
export NCCL_MIN_NCHANNELS=8
export NCCL_ALLOC_P2P_NET_LL_BUFFERS=1
export OMP_NUM_THREADS=8
export SAFETENSORS_FAST_GPU=1
```

### High-Concurrency Serving (Dual CPU, DRAM Routing)

```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_MIN_NCHANNELS=32
export NCCL_MAX_NCHANNELS=64
export NCCL_BUFFSIZE=67108864
export CUDA_DEVICE_MAX_CONNECTIONS=64
export OMP_NUM_THREADS=8
export SAFETENSORS_FAST_GPU=1
```

### CPU Performance Tuning (Host OS)

```bash
# Set CPU governor to performance
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable swap
sysctl -w vm.swappiness=0

# Disable NUMA balancing
sysctl -w kernel.numa_balancing=0

# Increase scheduler migration cost
sysctl -w kernel.sched_migration_cost_ns=50000

# Set CPU affinity for SGLang
export SGLANG_SET_CPU_AFFINITY=1
```

---

## NCCL Tuner Plugin

**Repository:** [voipmonitor/nccl-tuner-amd](https://github.com/voipmonitor/nccl-tuner-amd)

A tuner plugin (API v5) that optimizes NCCL AllReduce protocol selection on AMD EPYC systems with PCIe-connected GPUs. NCCL's default cost model is tuned for NVLink and makes suboptimal choices on PCIe — specifically, LL128 is completely disabled for PHB/SYS paths, creating a latency gap in the 512K–2M range.

### How It Works

The plugin overrides NCCL's protocol selection:

| GPUs | LL | LL128 | Simple |
|------|-----|-------|--------|
| ≤4 (same NUMA) | 0–576K | — | >576K |
| >4 (cross-socket) | 0–448K | 448K–3M | >3M |

Thresholds are tunable via env vars: `NCCL_TUNER_LL_MAX_4GPU`, `NCCL_TUNER_LL_MAX_8GPU`, `NCCL_TUNER_LL128_MAX_8GPU`.

### Setup

```bash
git clone https://github.com/voipmonitor/nccl-tuner-amd.git
cd nccl-tuner-amd && make    # produces libnccl-tuner.so

# Add to your launch environment:
export NCCL_TUNER_PLUGIN=/path/to/libnccl-tuner.so
export NCCL_PROTO=LL,LL128,Simple    # required: enables all 3 protocols
```

Zero dependencies — builds with just gcc, no NCCL source or CUDA SDK needed.

### AllReduce Benchmark Results

Tested on 8× RTX PRO 6000 Blackwell, AMD EPYC 9575F (Turin), dual-socket. NCCL 2.28.3, nccl-tests.

#### 4 GPU (same NUMA socket)

```
Size      Baseline   Plugin     Δ
8K        1.10       1.09      -1%
64K       7.70       8.46     +10%
256K     13.96      14.01       0%
512K     17.39      17.50      +1%
1M       25.13      25.17       0%
4M       39.09      39.81      +2%
16M      45.58      45.34       0%
Avg BW   14.38      14.45    +0.5%
```

Minimal impact — expected. Plugin only extends LL range slightly, no LL128 for ≤4 GPUs.

#### 8 GPU (cross-socket: 4+4 across two NUMA nodes)

```
Size      Baseline   Plugin     Δ
64K       4.96       4.91      -1%
256K      9.36       8.99      -4%
512K     13.06      14.43     +10%    ← LL128 zone starts
1M       16.47      21.32     +29%    ← LL128
2M       19.10      28.67     +50%    ← LL128 peak
4M       33.46      33.34       0%
16M      40.39      40.19       0%
Avg BW   11.05      11.94     +8%
```

**The LL128 zone (512K–2M) delivers 10–50% improvement.** This is where LL saturates but Simple isn't yet efficient.

### LLM Inference Impact — NCCL Tuner Plugin

Tested GLM-5 NVFP4 (TP=8, b12x MoE) and Qwen3.5-397B NVFP4 (TP=4, cutlass MoE). Full decode benchmark across all concurrency levels and context lengths (0–128K).

**Result: zero measurable impact on decode throughput** — allreduce message sizes are too small for the LL128 zone (see below).

### Better Alternative: PCIe Oneshot AllReduce + Fusion

Instead of the NCCL tuner plugin, use lukealonso's **PCIe oneshot allreduce** with fused RMSNorm, which directly replaces NCCL for small messages where it's 1.3–1.8× faster. Auto crossover: **120 KB on 4 GPUs**, **48 KB on 8 GPUs**.

```
--enable-pcie-oneshot-allreduce            # PCIe oneshot for small messages
```

> **Important:** On direct-attach topologies (no PCIe switch), you **must** configure the nvidia driver for direct P2P BAR1 mapping, otherwise PCIe oneshot allreduce will be ~15× slower than NCCL and the auto-crossover will silently disable it. See [Critical Prerequisite](pcie-oneshot-allreduce.md#critical-prerequisite-nvidia-p2p-driver-config). `NCCL_P2P_LEVEL=SYS` does NOT help — it only affects NCCL, not the custom allreduce kernel.

See [PCIe Oneshot AllReduce](pcie-oneshot-allreduce.md) for details. Measured a consistent **+7% decode throughput** across both Qwen3.5 and GLM-5.

#### PCIe Oneshot Benchmark Results (conc=1, context=0)

| Model | Config | Without PCIe Oneshot | With PCIe Oneshot | Improvement |
|---|---|---|---|---|
| Qwen3.5-397B TP=4 | cutlass, no MTP | 70.6 tok/s | 76.1 tok/s | **+7.8%** |
| Qwen3.5-397B TP=4 | b12x, no MTP | — | 98.4 tok/s | (b12x baseline N/A) |
| Qwen3.5-397B TP=4 | b12x + MTP | — | 165.9 tok/s | (b12x+MTP baseline N/A) |
| GLM-5 TP=8 | b12x, no MTP | 52.8 tok/s | 56.6 tok/s | **+7.2%** |

> **Do NOT use** `--disable-custom-all-reduce` — it disables PCIe oneshot too.
> **Do NOT use** `--enable-flashinfer-allreduce-fusion` — it does not work on SM120.

#### Why: AllReduce Message Sizes During Inference

NCCL call trace from Qwen3.5-397B TP=4 (PyNCCL instrumentation during CUDA graph capture):

```
AllReduce size distribution (per forward pass = ~363 calls/layer):

Size       Bytes      Calls    %      Protocol Zone
  8 KB       8,192      363    8.3%   LL
 16 KB      16,384      363    8.3%   LL
 32 KB      32,768      363    8.3%   LL
 64 KB      65,536      363    8.3%   LL
 96 KB      98,304      363    8.3%   LL
128 KB     131,072      363    8.3%   LL
192 KB     196,608      363    8.3%   LL
256 KB     262,144      363    8.3%   LL
320 KB     327,680      363    8.3%   LL
384 KB     393,216      363    8.3%   LL
448 KB     458,752      363    8.3%   LL
512 KB     524,288      363    8.3%   LL128 ←

LL (≤448K):    91.7%
LL128 (>448K):  8.3%   (only bs=64)
Simple (>3M):   0.0%
```

The sizes correspond to `batch_size × hidden_size_per_shard × 2B (bf16)`. With Qwen3.5 hidden=4096 sharded across TP=4:
- bs=1 → 8 KB, bs=2 → 16 KB, ..., bs=64 → 512 KB

**91.7% of AllReduce calls fall in the LL zone where the plugin changes nothing.** Only the maximum batch size (64) reaches 512 KB — barely into LL128 territory. At typical serving concurrency (1–16 requests), all AllReduce calls are ≤256 KB, fully in the LL zone.

#### When Would the Plugin Help?

The plugin would benefit workloads with **larger AllReduce payloads** (>512K per call):

- **TP=2 instead of TP=4** — doubles the per-GPU hidden size, doubling AllReduce payload
- **Dense models** (not MoE) — larger hidden states
- **Expert Parallelism (EP)** — all-to-all/allgather with larger payloads
- **Higher TP with larger models** — models with hidden_size >8K
- **Non-CUDA-graph workloads** — prefill, dynamic batching with large batch sizes

### FlashInfer AllReduce Fusion: Not Available on SM120

A related finding: `--enable-flashinfer-allreduce-fusion` does **not** work on RTX PRO 6000 (SM120). The code checks:

```python
# communicator.py:100
(_is_sm90_supported or _is_sm100_supported)
```

SM120 (capability 12.0) satisfies neither condition. All allreduce operations go through NCCL regardless of this flag. This has been verified:

```
>>> is_sm90_supported() → False
>>> is_sm100_supported() → False
>>> torch.cuda.get_device_capability(0) → (12, 0)
```

### Verification

Confirm the plugin is loaded:
```bash
NCCL_DEBUG=INFO your_app 2>&1 | grep TUNER
# TUNER/Plugin: Using AMD-Turin-Optimal (v5)
# Successfully loaded external tuner plugin
```
