# GPU Configurations & Rig Builds

Practical hardware configurations for RTX PRO 6000 Blackwell (96 GB GDDR7) inference rigs, including model sizing, power management, and specific community builds.

## Table of Contents

- [Model Sizing Guide](#model-sizing-guide)
- [4-GPU Configurations](#4-gpu-configurations)
- [8-GPU Configurations](#8-gpu-configurations)
- [16-GPU Configurations](#16-gpu-configurations)
- [GPU Variants](#gpu-variants)
- [Community Rig Builds](#community-rig-builds)
- [Power Consumption & Electrical](#power-consumption--electrical)
- [Thermal Management](#thermal-management)
- [Cases & Physical Layout](#cases--physical-layout)

---

## Model Sizing Guide

### VRAM Requirements by GPU Count

| GPUs | NVFP4 Models That Fit | FP8 Models That Fit |
|------|----------------------|---------------------|
| 1x 96GB | Qwen3.5-27B, smaller models | -- |
| 2x 96GB | MiniMax-M2.5 NVFP4, Qwen3.5-122B NVFP4 | -- |
| 4x 96GB | Qwen3.5-397B NVFP4, GLM-4.7 NVFP4, MiniMax-M2.5 FP8 | MiniMax-M2.5 FP8 |
| 6x 96GB | GLM-5 NVFP4 (TP2 PP3) | -- |
| 8x 96GB | All current models incl. Kimi K2.5, GLM-5 | GLM-4.7 FP8, Qwen3.5-397B FP8 |
| 16x 96GB | All models with massive KV cache headroom | GLM-5 FP8 |

### Per-GPU VRAM Breakdown Examples

| Model | Quant | GPUs | Weights/GPU | KV Cache/GPU | Total/GPU |
|-------|-------|------|-------------|-------------|-----------|
| Qwen3.5-397B | NVFP4 | 4x | ~82 GB | varies | ~82 GB at max ctx |
| MiniMax-M2.5 | FP8 | 4x | ~90 GB | varies | ~90 GB |
| GLM-5 (744B) | NVFP4 | 8x | 57.06 GB | 29.32 GB (bf16) | ~86.38 GB |
| Kimi K2.5 (530B) | INT4 | 8x | ~60 GB | varies | depends on KV dtype |

### Tensor Parallel Constraints

- TP must be power-of-2 for vLLM/SGLang: 2, 4, 8, 16
- For odd GPU counts, use Pipeline Parallel: e.g., TP2 PP3 = 6 GPUs
- ExLlama v2/v3 supports arbitrary GPU counts
- 16 cards: only slightly slower single-batch than 8 cards due to comms overhead, but ~6x more KV cache tokens

---

## 4-GPU Configurations

The most common setup for running large MoE models at NVFP4 quantization.

### What Fits on 4x 96GB

| Model | Quant | Status | Notes |
|-------|-------|--------|-------|
| Qwen3.5-397B-A17B | NVFP4 | Works well | ~82 GB/GPU, TP=4 |
| MiniMax-M2.5 | FP8 | Works well | ~90 GB/GPU, TP=4 |
| MiniMax-M2.5 | NVFP4 | Works well | Fits on 2x, TP=2 |
| GLM-4.7 | NVFP4 | Works well | TP=4 |
| GLM-4.7 | FP8 | Works | Tight on VRAM |
| Qwen3.5-122B-A10B | NVFP4 | Works | Fits on 2x |

### What Does NOT Fit on 4x 96GB

| Model | Quant | Why |
|-------|-------|-----|
| Kimi K2.5 (530B) | INT4 (native) | Already INT4 quantized, still needs 8 cards |
| GLM-5 (744B) | NVFP4 | ~440 GB weights alone exceeds 384 GB |
| GLM-5 (744B) | FP8 | Far too large |
| Qwen3.5-397B | FP8 | Needs 8x |

---

## 8-GPU Configurations

Required for the largest models and for FP8 precision on 397B-class models.

### What Fits on 8x 96GB (768 GB total)

| Model | Quant | KV Cache Dtype | KV Cache Capacity | Notes |
|-------|-------|---------------|-------------------|-------|
| Kimi K2.5 | INT4 (native) | BF16 | ~190K tokens | Fastest decode (90 tok/s) |
| Kimi K2.5 | INT4 (native) | FP8 | ~450K tokens | Slightly slower, much more context |
| Kimi K2.5 | INT4 + DCP=8 | FP8 | ~3.6M tokens | Best for long context |
| GLM-5 | NVFP4 + MTP | BF16 | ~314K tokens | FP8 KV broken on SM120 |
| Qwen3.5-397B | FP8 | auto | large | TP=8 |
| Qwen3.5-397B | NVFP4 | FP8 | large | TP=4 + DP=2 possible |

### Multi-Node

- 2 nodes x 4x RTX 5090 connected via 2x 10GbE running Qwen3-235B-A22B-NVFP4 via vLLM Ray
- Occasional instability every 2-3 days requiring restart
- Required building Ray from source with PR #58866 patch

---

## 16-GPU Configurations

For maximum KV cache capacity and running the largest models in higher precision.

- Grimulkan runs 16x RTX 6000 Pro on 4x PCIe switches on a single CPU
- Only slightly slower single-batch than 8 cards due to communication overhead
- ~6x more KV cache tokens than 8 cards (no weight replication overhead in TP)
- GPU display mode must be set to headless for 16 GPUs (BAR1 allocation)

---

## GPU Variants

### RTX PRO 6000 Blackwell Variants

| Variant | TDP | Form Factor | Notes |
|---------|-----|-------------|-------|
| **Workstation Edition** | 600W | Dual-slot, blower | Standard desktop/workstation card |
| **Server Edition** | 600W | Passive cooling | Requires chassis airflow |
| **MaxQ** | 300W | Compact | Designed for dense packing |

### MaxQ vs Workstation Edition Performance

| Metric | MaxQ (300W) | Workstation (600W) | Delta |
|--------|-------------|-------------------|-------|
| Prefill speed | Baseline | ~20% faster | Compute-bound |
| Decode speed (single user) | ~96% of 600W | Baseline | VRAM/PCIe limited |
| Decode speed (64 concurrent) | ~70% of 600W | Baseline | Significant loss |

### Power Limit Scaling (MiniMax-M2.5 NVFP4, 4x cards)

| Power Limit | 4 Concurrent | 16 Concurrent | 32 Concurrent | 64 Concurrent |
|-------------|-------------|---------------|---------------|---------------|
| 300W | Baseline | Baseline | Baseline | 1206 tok/s |
| 400W | ~+2% | ~+10% | ~+18% | ~+22% |
| 500W | ~+3% | ~+16% | ~+25% | 1558 tok/s (+29%) |
| 600W | ~+4% | ~+17% | ~+26% | ~+30% |

**Key findings:**
- Power-limiting 600W to 300W loses only ~4% single-user performance
- Loss increases to ~30% at 64 concurrent users
- 500W performs nearly identically to 600W
- Performance drop from 400W to 300W is significant at high concurrency

Detailed wattage-performance analysis: [shihanqu.github.io/Blackwell-Wattage-Performance](https://shihanqu.github.io/Blackwell-Wattage-Performance/)

### Memory Overclocking (MaxQ Only)

MaxQ cards accept memory clock offset. Server Edition cards return an error.

```python
import pynvml
pynvml.nvmlInit()
count = pynvml.nvmlDeviceGetCount()
for i in range(count):
    h = pynvml.nvmlDeviceGetHandleByIndex(i)
    pynvml.nvmlDeviceSetMemClkVfOffset(h, 4000)
pynvml.nvmlShutdown()
```

> "fwiw - overclocking my GDDR7 got me a few percentage points, it's what got me over the hill to 101." -- luke

---

## Community Rig Builds

### Festr -- Dual Turin Server (8x Server Edition)

| Component | Spec |
|-----------|------|
| CPUs | 2x AMD EPYC 9575F 64-Core (5 GHz boost) |
| Motherboard | K15PG-D24 Series, 60SB0D94-SB0A01 |
| RAM | 24x96 GB Samsung DDR5-6400 (2.2 TB total, all 12 channels/CPU) |
| GPUs | 8x RTX PRO 6000 Blackwell Server Edition |
| Topology | Direct-attach, 4 GPUs per NUMA node |
| xGMI | 3x links (192 GB/s fabric) |
| OS | Ubuntu 24.04.3 LTS, Kernel 6.17.0-14-generic |
| CUDA | 13.1.1, Driver 590.48.01 |

### luke -- PCIe Switch Setup (8x MaxQ)

| Component | Spec |
|-----------|------|
| CPU | AMD Threadripper Pro (single socket) |
| Motherboard | ASUS WRX90E (7 PCIe slots) |
| Switches | 3x c-payne Microchip Switchtec PM50100 (100-lane Gen5) |
| GPUs | 8x RTX 6000 Pro MaxQ |
| Topology | 2 leaf switches (partitioned, 2x x16 uplinks each), 1 root switch |
| Case | Open-air mining chassis |
| Special | Overclocked GDDR7 memory (+4000 offset) |

### Grimulkan -- 16-GPU Switch Setup

| Component | Spec |
|-----------|------|
| CPU | AMD Turin (single socket) |
| Switches | 4x c-payne PCIe Gen5 switches (star topology) |
| GPUs | 16x RTX PRO 6000 Blackwell |
| Notes | Highest total GPU count in community |

### orangezed -- Budget Dual EPYC (8x MaxQ)

| Component | Spec |
|-----------|------|
| CPUs | 2x AMD EPYC 9374F 32-Core (128 threads) |
| Motherboard | ASRockRack TURIN2D24G-2L+/500W |
| RAM | 10x48 GB DDR5-4800 (472 GB, only 5 channels/CPU) |
| GPUs | 8x RTX PRO 6000 Blackwell MaxQ Workstation Edition |
| xGMI | 2x links only |
| Notes | Under-populated DRAM caused performance issues |

### Ixtrix -- Desktop 4-GPU Build

| Component | Spec |
|-----------|------|
| Motherboard | ASUS PRO WS WRX90E-SAGE SE |
| GPUs | 4x RTX 6000 Pro MaxQ |
| Virtualization | Proxmox |
| Notes | Provided critical BIOS/GRUB stability settings |

### Qu (shihanqu) -- 4-GPU Workstation

| Component | Spec |
|-----------|------|
| GPUs | 4x RTX 6000 Pro Workstation (600W) |
| Notes | Wattage-performance benchmark creator |

---

## Power Consumption & Electrical

### Per-Card Power by Workload

| Phase | Power per Card | Notes |
|-------|---------------|-------|
| Idle | ~30-50W | With `VLLM_SLEEP_WHEN_IDLE=1` |
| Decode | ~300W | Memory/PCIe bound |
| Prefill | 400-600W | Compute bound |
| Peak (GLM-5 prefill) | **640W observed** | All 8 cards simultaneously |

### Electrical Requirements

- **Circuit:** 220V 30A recommended for 8-GPU rigs
- **Power distribution:** Boards with 14x 12VHPWR connections (1000A @ 12V)
- **8x 600W cards:** 4800W GPU power alone, plus CPU/RAM/fans/overhead = ~5500-6000W total
- **8x 300W cards (MaxQ):** 2400W GPU power, ~3000-3500W total

### Power Optimization

- Power-limit 600W cards to 500W: nearly identical performance, saves 800W across 8 cards
- Power-limit to 300W: only ~4% single-user loss, saves 2400W across 8 cards
- Use `nvidia-smi -pl 500` or `pynvml` to set power limits

---

## Thermal Management

### Temperature Targets

| Threshold | Impact |
|-----------|--------|
| < 80C | Optimal for longevity |
| 80-85C | Acceptable for sustained operation |
| 88C | PCIe slot temps can cause system instability |
| 95C | Thermal throttle point (per PNY) |

### Cooling Solutions

- **Noctua IndustrialPPC fans:** Standard choice for air cooling, 6x keeps cards under 90C
- **Watercooling:** For single-slot density configurations
- **uCoustic 24U active cooling cabinet:** Model 9210i for silent rackmount cooling

### GPU Fan Control

Despite PNY's claim that fan control requires a GUI/X server, headless fan control is possible:

**pynvml library (no GUI required):**

```python
import pynvml
# Fan control via pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
# Set fan speed as needed
```

**LACT tool (fan curves via config file):**

```yaml
# /etc/lact/config.yaml
profiles:
  Max:
    gpus:
      10DE:2BB1-10DE:204B-0000:01:00.0:
        fan_control_enabled: true
        fan_control_settings:
          mode: curve
          static_speed: 0.5
          temperature_key: gpu
          interval_ms: 1000
          curve:
            70: 0.6
            75: 0.7
            80: 0.75
            82: 0.80
            85: 1.0
          spindown_delay_ms: 10000
          change_threshold: 3
          auto_threshold: 70
        power_cap: 600.0
```

---

## Cases & Physical Layout

### Rackmount Options

- **Supermicro AS-4124GS-TNR:** 8-GPU rackmount server
- **ASUS ESC8000A-E13P:** 8-GPU rackmount (reported working with GLM-5)

### Desktop / Open-Air Options

- **Chinese 8/10/12/16 GPU cases from Alibaba:**
  - LXYD brand recommended: [lxyd.en.alibaba.com](https://lxyd.en.alibaba.com/)
  - GPU baseboards: [serverhome.en.alibaba.com](https://serverhome.en.alibaba.com/productgrouplist-952507615/GPU_baseboard.html)
- **Open-air mining chassis:** Used by luke for 8x MaxQ with switches
- **Custom aluminum rigs:** Built by purplepow3r for mixed GPU setups

### Motherboard Selection

| Motherboard | Slots | CPU | Notes |
|-------------|-------|-----|-------|
| ASUS PRO WS WRX90E-SAGE SE | 7x PCIe | Threadripper | Slot 6 limited to Gen5 x8 |
| Supermicro AS-4124GS-TNR | 8x PCIe | Dual EPYC | Rackmount |
| ASRockRack TURIN2D24G-2L+ | 8x PCIe | Dual Turin EPYC | Budget option, only 2x xGMI |
| K15PG-D24 Series | 8x PCIe | Dual Turin EPYC | Festr's board, 3x xGMI |

### CPU Selection

| CPU | Sockets | PCIe Lanes | xGMI BW | Best For |
|-----|---------|-----------|---------|----------|
| AMD EPYC 9575F Turin | Dual | 128/socket | 192-256 GB/s | Best dual-CPU direct-attach |
| AMD EPYC 9374F Genoa | Dual | 128/socket | ~96 GB/s | Budget dual-CPU |
| AMD Threadripper 9985WX | Single | 128 | N/A | Single-CPU + switches |
| AMD Threadripper 5975WX | Single | 128 | N/A | Budget single-CPU |

### RAM Recommendations

- **Turin:** DDR5-6400, populate all 12 channels per CPU
- **Genoa:** DDR5-4800, populate all 12 channels per CPU
- **Minimum:** 256 GB for 4-GPU rigs, 512 GB+ for 8-GPU rigs
- **Festr's recommendation:** 2+ TB for KV cache offload experiments
- **Warning:** Under-populating DRAM channels causes severe cross-NUMA performance degradation
