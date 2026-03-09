# PCIe Bandwidth & P2P Performance

Measured PCIe peer-to-peer bandwidth, latency, and NCCL AllReduce performance on RTX PRO 6000 Blackwell systems. These numbers are the foundation for understanding inference throughput limits.

## Table of Contents

- [P2P Bandwidth Measurements](#p2p-bandwidth-measurements)
- [P2P Latency Measurements](#p2p-latency-measurements)
- [NCCL AllReduce Bus Bandwidth](#nccl-allreduce-bus-bandwidth)
- [BAR1 Configuration](#bar1-configuration)
- [How PCIe Bandwidth Affects Inference](#how-pcie-bandwidth-affects-inference)
- [GRUB Kernel Parameters for PCIe Stability](#grub-kernel-parameters-for-pcie-stability)
- [Debugging Tools](#debugging-tools)

---

## P2P Bandwidth Measurements

All measurements taken using the CUDA `p2pBandwidthLatencyTest` sample or luke's `p2pmark` tool.

### Unidirectional P2P Bandwidth

Measured with P2P Writes, PCIe Gen5 x16 links.

| Source | Setup | Same-NUMA | Cross-NUMA |
|--------|-------|-----------|------------|
| purplepow3r | Dual Turin, 4x 6000 Pro WS + 2x 5090 | ~55-56 GB/s | ~51 GB/s |
| orangezed | Dual EPYC 9374F, 8x 6000 Pro MaxQ | ~54 GB/s | ~39 GB/s |
| Festr | Dual Turin, 8x 6000 Pro Server | ~53 GB/s | -- |
| luke | Switches, 8x 6000 Pro MaxQ | ~54 GB/s | N/A (single CPU) |

**Theoretical maximum:** PCIe Gen5 x16 = 63 GB/s unidirectional. The ~56 GB/s measured represents ~89% efficiency.

### Bidirectional P2P Bandwidth

| Source | Setup | Same-NUMA | Cross-NUMA |
|--------|-------|-----------|------------|
| purplepow3r | Dual Turin | ~111 GB/s | ~99 GB/s |
| orangezed | Dual EPYC 9374F | ~103 GB/s | ~64 GB/s |

### Full P2P Bandwidth Matrix Example

From purplepow3r's dual Turin system (4x RTX 6000 Pro WS + 2x RTX 5090):

```
Unidirectional P2P=Enabled Bandwidth (P2P Writes) Matrix (GB/s)
   D\D     0      1      2      3      4      5
     0 1488.10  56.57  56.57  50.97  55.61  55.59
     1   56.57 1416.77  56.57  51.01  55.60  55.64
     2   56.57  56.57 1415.11  50.97  55.62  55.64
     3   50.97  50.97  50.97 1375.60  50.16  50.45
     4   55.61  55.60  55.62  50.16 1408.87  55.60
     5   55.60  55.64  55.64  50.45  55.61 1489.91

Bidirectional P2P=Enabled Bandwidth Matrix (GB/s)
   D\D     0      1      2      3      4      5
     0 1485.22 111.38 111.39  99.42 111.02 111.10
     1  111.38 1416.86 111.38  99.59 111.06 111.15
     2  111.39 111.39 1415.11  99.53 111.04 111.12
     3   99.42  99.59  99.53 1377.73  99.09  99.43
     4  111.02 111.06 111.04  99.09 1409.15 111.13
     5  111.10 111.15 111.12  99.43 111.13 1487.77
```

Note: Device 3 shows lower bandwidth (~51/99 GB/s) -- this is the cross-NUMA path. Devices 0-2 are on NUMA0, devices 3-5 are on NUMA1.

### p2pmark Scores (8 GPUs)

luke's [p2pmark](https://github.com/lukealonso/p2pmark) tool provides a standardized comparison:

| System | PCIe Link Score | Dense Interconnect Score | Effective Latency |
|--------|----------------|------------------------|-------------------|
| luke (3x switches, single CPU) | 0.86 (54.3 GB/s) | 0.44 (191.8 / 434.7 GB/s) | 6.79 us |
| Festr (dual Turin, direct-attach) | 0.84 (52.7 GB/s) | 0.41 (173.1 / 421.3 GB/s) | 6.03 us |
| Grimulkan (4x switches, single CPU) | 0.86 (53.9 GB/s) | 0.38 (164.3 / 431.2 GB/s) | 7.04 us |

### p2pmark Scores (4 GPUs)

| System | PCIe Link Score | Dense Interconnect Score | Effective Latency |
|--------|----------------|------------------------|-------------------|
| luke (1 switch) | 0.86 | 0.64 (138.3 / 217.7 GB/s) | 4.10 us |
| Festr (Turin, same NUMA) | 0.88 | 0.59 (129.7 / 220.6 GB/s) | 2.28 us |

---

## P2P Latency Measurements

### P2P Enabled vs Disabled

| Condition | Cross-GPU Latency |
|-----------|-------------------|
| P2P Enabled (same NUMA) | 0.36-0.45 us |
| P2P Enabled (cross-NUMA, Turin) | 0.44 us |
| P2P Disabled | ~14 us |

P2P enablement provides a **30x latency reduction**. This is why the `nvidia_uvm uvm_disable_hmm=1` fix and proper NCCL P2P configuration are critical.

### Full P2P Latency Matrix Example

From purplepow3r's system:

```
P2P=Enabled Latency (P2P Writes) Matrix (us)
   GPU     0      1      2      3      4      5
     0   2.07   0.37   0.36   0.38   0.44   0.36
     1   0.37   2.07   0.36   0.38   0.44   0.36
     2   0.37   0.37   2.07   0.38   0.44   0.36
     3   0.38   0.38   0.38   2.07   0.38   0.38
     4   0.44   0.44   0.44   0.38   2.07   0.36
     5   0.36   0.36   0.36   0.38   0.36   2.07
```

From luke's 8x RTX 6000 Pro on switches:

```
P2P=Enabled Latency (P2P Writes) Matrix (us)
   GPU     0      1      2      3      4      5      6      7
     0   2.07   0.45   0.51   0.45   0.45   0.45   0.45   0.44
     1   0.52   2.07   0.44   0.45   0.52   0.45   0.45   0.44
     ...
```

---

## NCCL AllReduce Bus Bandwidth

AllReduce is the dominant collective operation in tensor-parallel inference. Higher bus bandwidth means faster decode.

### 8-GPU Results

| Setup | NCCL Config | Message Size | Bus BW (GB/s) |
|-------|-------------|-------------|---------------|
| luke (3x switches) | MIN_NCHANNELS=8 | Sweep 8M-2G | **41.1** |
| Grimulkan (4x switches) | MIN_NCHANNELS=8 | Sweep 8M-2G | 39.4 |
| Grimulkan (4x switches) | Default | 32M fixed | 40.1 |
| Festr (dual Turin) | MIN_NCHANNELS=8 | Sweep 8M-2G | 37.6 |
| Festr (dual Turin) | Default | 32M fixed | 22.2 |
| purplepow3r (7 GPUs, dual Turin) | P2P_LEVEL=SYS | 4G fixed | 41.3-41.7 |

### NCCL Test Commands

```bash
# Basic test (32M message, 8 GPUs)
NCCL_P2P_LEVEL=SYS NCCL_NET_GDR_LEVEL=SYS ./all_reduce_perf -b 32M -g 8 -c 0

# Sweep test with tuned channels
NCCL_NET_GDR_LEVEL=SYS NCCL_MIN_NCHANNELS=8 ./all_reduce_perf -b 8M -e 2G -f 2 -g 8 -n 50

# Large message test
NCCL_P2P_LEVEL=SYS NCCL_IB_DISABLE=1 ./build/all_reduce_perf -b 4G -e 4G -f 2 -g 7 -n 20 -N 100
```

NCCL tests are located at `/usr/src/nccl-tests` in NVIDIA containers.

### Custom Allreduce vs NCCL (Small Messages)

luke's custom allreduce kernel (for PCIe switch topologies) compared to NCCL:

| Size | Custom (us) | NCCL (us) | Winner |
|------|-------------|-----------|--------|
| 256 B | 7.5 | 24.6 | Custom 3.3x |
| 1 KB | 7.5 | 24.1 | Custom 3.2x |
| 8 KB | 9.2 | 24.2 | Custom 2.6x |
| 32 KB | 16.5 | 24.5 | Custom 1.5x |
| 64 KB | 25.9 | 24.1 | NCCL 1.1x |
| 256 KB | 73.6 | 28.0 | NCCL 2.6x |

Custom allreduce wins big for inference-relevant message sizes (<32 KB) but loses at larger sizes. This kernel is only effective on densely-interconnected PCIe switch topologies -- it was actually **slower** than NCCL on dual-CPU systems without switches.

Custom allreduce repo: [github.com/lukealonso/sglang/commits/custom_ar/](https://github.com/lukealonso/sglang/commits/custom_ar/)

---

## BAR1 Configuration

BAR1 (Base Address Register 1) maps GPU memory into the CPU's address space for P2P transfers.

- **Resizable BAR:** Must be enabled in BIOS (default on most boards)
- **Expected size:** 96 GB (matching VRAM) per GPU
- **Common issue:** Some BIOS configurations default to 256 MB BAR1, which cripples P2P performance
- **GPU Display Mode:** Set to headless via [NVIDIA Display Mode Selector](https://developer.nvidia.com/displaymodeselector) for largest BAR1 allocation. Required for 16-GPU setups.

Verify BAR1 in `nvidia-smi`:

```
$ nvidia-smi -q | grep "BAR1"
    BAR1 Memory Usage
        Total                             : 98304 MiB
```

---

## How PCIe Bandwidth Affects Inference

### Tensor Parallel Decode

During decode (token generation), each TP step requires:
1. **AllReduce** after attention layer (~32-256 KB per layer)
2. **AllReduce** after MoE/FFN layer (~32-256 KB per layer)

For a 397B MoE model with ~60 layers, that is ~120 AllReduce operations per token. At 25 us per AllReduce, that is **3 ms of pure communication overhead per token**, limiting decode to ~333 tok/s even with zero compute time.

### Why Small-Message Latency Dominates

Inference AllReduce messages are small (32-256 KB). At these sizes:
- **Bandwidth** is irrelevant (56 GB/s can transfer 256 KB in 4.5 us)
- **Latency** dominates (NCCL ring setup, protocol negotiation, synchronization)

This is why:
- NCCL's LL (Low Latency) protocol gives 1.5-1.9x speedup
- Custom allreduce kernels that bypass NCCL overhead give 2-3x speedup
- PCIe switch cut-through latency (~100 ns) matters more than raw bandwidth

### P2P vs No-P2P for High Concurrency

For high-concurrency workloads (many simultaneous requests), disabling P2P and routing through DRAM can actually be faster:

| Workload | P2P Enabled | P2P Disabled | Winner |
|----------|-------------|-------------|--------|
| Single batch (low latency) | 90 tok/s | 70 tok/s | P2P |
| 100 concurrent requests | 5000 tok/s | 10000 tok/s | No-P2P |

This is because DRAM routing can use more channels and higher aggregate bandwidth for large batched operations.

---

## GRUB Kernel Parameters for PCIe Stability

### Critical Parameters

Add to `GRUB_CMDLINE_LINUX_DEFAULT` in `/etc/default/grub`:

```bash
pcie_aspm=off pcie_port_pm=off
```

| Parameter | Purpose |
|-----------|---------|
| `pcie_aspm=off` | Disables Active State Power Management on all PCIe links |
| `pcie_port_pm=off` | Disables PCIe port runtime power management. **CRITICAL:** prevents root port from suspending during GPU link retrain Gen1<->Gen5 |

Without `pcie_port_pm=off`, GPU DynamicPowerManagement=3 causes "Surprise Link Down" errors (`aer_uncor_status: 0x00000020`) leading to **system lockups**.

### Additional Recommended Parameters

```bash
# Full recommended GRUB line (Festr's Turin system):
GRUB_CMDLINE_LINUX="rd.auto=1 rd.md=1 rd.md.conf=1 mitigations=off spectre_v2=off spec_store_bypass_disable=off l1tf=off mds=off tsx_async_abort=off srbds=off mmio_stale_data=off retbleed=off amd_iommu=off iommu=off"
```

| Parameter | Purpose |
|-----------|---------|
| `iommu=off` | Prevents NCCL hangs. Without this, NCCL P2P may deadlock. |
| `amd_iommu=off` | AMD-specific IOMMU disable |
| `mitigations=off` | Disable CPU security mitigations for maximum performance |
| `nvme_core.default_ps_max_latency_us=0` | Prevent NVMe power state transitions (stability) |

After editing, run:

```bash
update-grub
reboot
# Verify:
cat /proc/cmdline
```

### Modprobe Configuration

```bash
# /etc/modprobe.d/uvm.conf
options nvidia_uvm uvm_disable_hmm=1
```

Without this, NCCL P2P operations lock up. This is required on virtually all RTX PRO 6000 multi-GPU setups.

### BIOS Settings (PRO WS WRX90E-SAGE SE)

- **Resizable BAR:** Enabled (default)
- **Above 4G Decoding:** Enabled
- **SR-IOV:** Enabled
- **Slot 6 Warning:** On WRX90E-SAGE SE, slot 6 is limited to Gen5 x8 speed

### Filesystem Warning

ZFS caused system freezes for some users on these GPU workloads. **EXT4 is recommended** for the OS filesystem.

---

## Debugging Tools

| Tool | Purpose |
|------|---------|
| `nvidia-smi topo -m` | Display GPU topology matrix |
| `nvidia-smi -q` | Detailed GPU info including BAR1, power, thermals |
| [p2pmark](https://github.com/lukealonso/p2pmark) | P2P bandwidth, latency, and allreduce benchmarks |
| `p2pBandwidthLatencyTest` | CUDA samples P2P test |
| [amd-epyc-gpu-fabric-monitor](https://github.com/voipmonitor/amd-epyc-gpu-fabric-monitor) | Real-time AMD xGMI fabric transfer monitoring |
| `memtest86` | RAM testing |
| Intel MLC | Memory Latency Checker (works on AMD) |
| `rasdaemon` | AER error monitoring |
| `nvitop` / `nvtop` | GPU monitoring |
| PCIe AER counters | Check `aer_dev_correctable` and `aer_dev_fatal` in sysfs |
