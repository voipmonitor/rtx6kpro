# PCIe Topology & GPU Interconnect

Understanding PCIe topology is critical for multi-GPU inference on RTX PRO 6000 Blackwell systems. The path data takes between GPUs directly determines AllReduce latency, which dominates decode throughput for tensor-parallel inference.

## Table of Contents

- [PCIe Topology Basics](#pcie-topology-basics)
- [Direct-Attach vs Switch Topologies](#direct-attach-vs-switch-topologies)
- [AMD Turin vs Genoa](#amd-turin-vs-genoa)
- [PCIe Switches](#pcie-switches)
- [nvidia-smi Topology Output](#nvidia-smi-topology-output)
- [Impact on Inference Performance](#impact-on-inference-performance)
- [Topology Selection Guide](#topology-selection-guide)

---

## PCIe Topology Basics

In a multi-GPU system, each GPU connects to the CPU(s) through PCIe lanes. The path between any two GPUs determines the communication latency and bandwidth available for tensor-parallel operations like AllReduce.

**Key concepts:**

| Term | Meaning |
|------|---------|
| **Root Complex** | The PCIe controller inside the CPU die. All PCIe lanes originate here. |
| **Root Port** | An individual x16 slot's connection point on the root complex. |
| **PCIe Switch** | An external chip that multiplexes multiple x16 downstream ports onto a single x16 (or wider) upstream link to the CPU. |
| **P2P (Peer-to-Peer)** | Direct GPU-to-GPU data transfer without bouncing through system RAM. |
| **NUMA Node** | A CPU socket and its directly-attached memory. Cross-NUMA transfers traverse the inter-socket fabric (xGMI on AMD). |
| **PHB** | PCIe Host Bridge -- GPUs sharing the same root complex but different root ports. |
| **PIX** | GPUs sharing the same PCIe switch. |
| **SYS** | GPUs separated by the inter-socket (cross-NUMA) fabric link. |

---

## Direct-Attach vs Switch Topologies

### Direct-Attach (Dual-CPU)

Each GPU connects directly to a CPU root port. With dual AMD EPYC CPUs, you get 128 PCIe Gen5 lanes total (64 per socket), supporting 4 GPUs per socket at x16 each.

```
         +---------+     xGMI (192-256 GB/s)     +---------+
         |  CPU 0  |<===========================>|  CPU 1  |
         | (NUMA0) |                              | (NUMA1) |
         +----+----+                              +----+----+
         /  /  \  \                               /  /  \  \
       x16 x16 x16 x16                         x16 x16 x16 x16
        |   |   |   |                            |   |   |   |
      GPU0 GPU1 GPU2 GPU3                      GPU4 GPU5 GPU6 GPU7

      Same-NUMA P2P: ~0.36 us latency, ~56 GB/s unidirectional
      Cross-NUMA P2P: ~0.44 us latency (Turin) / ~14 us (Genoa w/o P2P)
```

**Advantages:**
- 128 total PCIe lanes directly to CPUs and local RAM
- Best for KV cache offload to system RAM (no shared uplink bottleneck)
- Best for training workloads requiring heavy RAM-to-GPU throughput
- Simpler hardware (no switch boards to buy)

**Disadvantages:**
- Cross-NUMA GPU traffic traverses xGMI fabric
- Two CPUs cost more and consume more power
- NCCL may misdetect AMD xGMI bandwidth (requires tuning)

### Switch Topology (Single or Dual CPU)

GPUs connect to PCIe switches, which provide full-bandwidth P2P between GPUs on the same switch without going through the CPU.

```
                     +-------+
                     |  CPU  |
                     +---+---+
                    x16  |  x16
                  +------+------+
                  |             |
             +----+----+  +----+----+
             | Switch  |  | Switch  |
             | (100L)  |  | (100L)  |
             +--+--+--++  ++--+--+--+
              /  |  |  \   /  |  |  \
            x16 x16 x16 x16 x16 x16 x16 x16
             |   |   |   |   |   |   |   |
           GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7

      Same-switch P2P: ~0.45 us, ~56 GB/s (full x16 each, simultaneously)
      Cross-switch P2P: traverses CPU root complex
```

**Advantages:**
- GPU-to-GPU P2P within a switch uses full x16 bandwidth each, simultaneously, without sharing
- ~100 ns cut-through latency on PCIe switches vs 1-2 us on CPU root port
- Single CPU reduces cost, power, and complexity
- Avoids cross-NUMA overhead entirely (single socket)

**Disadvantages:**
- Shared x16 uplink to CPU limits host RAM bandwidth (bottleneck for KV cache offload)
- Switch hardware cost (EUR 2,000-2,400 per 100-lane switch)
- Cross-switch traffic still routes through CPU

### 160-Lane "Fat Switch"

A single large switch with 128 lanes for GPUs + x32 uplink to CPU/RAM.

- Still loses to Turin dual-CPU for memory-heavy workloads (32 shared lanes vs 128 direct lanes)
- Good for pure GPU-to-GPU inference where RAM bandwidth is not critical

---

## AMD Turin vs Genoa

The CPU generation makes a dramatic difference for cross-NUMA GPU communication.

| Feature | AMD EPYC Genoa (9004) | AMD EPYC Turin 9575F (9005) |
|---------|----------------------|----------------------------|
| xGMI Version | xGMI 3.0 | xGMI 3.0 (wider) |
| xGMI Links | 3x | 3x |
| xGMI Bandwidth | ~96 GB/s | 192-256 GB/s actual |
| Cross-NUMA P2P Latency | ~14 us (P2P disabled) | ~0.44 us (P2P enabled) |
| PCIe Lanes per Socket | 128 (Gen5) | 128 (Gen5) |
| DDR5 Speed | 4800 MHz | 6400 MHz |
| DIMM Channels per CPU | 12 | 12 |

**Key finding:** Turin's ultra-fast xGMI3 interconnect drastically lowers cross-socket latency. With proper NCCL tuning, Turin dual-CPU direct-attach setups reach parity with PCIe switch topologies for inference -- something that was impossible on Genoa.

> "Direct-attach on Turin beats 160-lane switch for training and KV cache offload workloads." -- Community consensus

### DRAM Configuration Impact

DRAM channel count and speed affect cross-NUMA bandwidth. Under-populated DRAM reduces available bandwidth:

| System | DIMM Config | xGMI Links | Cross-NUMA BW (bidir) |
|--------|-------------|------------|----------------------|
| Festr (Turin) | 24x96 GB DDR5-6400, all 12 channels/CPU | 3x | ~99+ GB/s |
| orangezed (Genoa) | 10x48 GB DDR5-4800, 5 channels/CPU | 2x | ~64 GB/s |

> **Note:** orangezed initially reported ~9 tok/s at 100K context for Kimi K2.5, but this was a **measurement error** — wall-clock time (including prefill) was divided by tokens generated. Actual decode-only throughput (from vLLM stats log) was **30-35 tok/s**, comparable to other Genoa systems. The DRAM/xGMI differences do impact cross-NUMA bandwidth, but the effect on single-batch decode is smaller than initially believed.

---

## PCIe Switches

### Microchip Switchtec PM50100 (100-Lane, Gen5)

- **Price:** ~EUR 2,420 incl. VAT / EUR 2,000 ex. VAT
- **Source:** [c-payne.com](https://c-payne.com/products/pcie-gen5-mcio-switch-100-lane-microchip-switchtec-pm50100)
- Connects to motherboard via 2x MCIO connectors using a PCIe Gen5 x16 port
- All downstream MCIO ports appear as separate PCIe devices
- Shared uplink cap: x16 Gen5 total to CPU when all devices talk to CPU simultaneously
- GPU-to-GPU traffic within the switch: x16 each, simultaneously, without sharing bandwidth
- Gen4 x16 device only occupies Gen5 x8 to CPU (switch translates signaling)
- Supports 2-partition mode and dual-CPU bridge mode
- Smaller 52-lane version available (cheaper); software-configurable partitions

### Broadcom 144-Lane Switches

- Fits 8 GPUs: 8 x 16 = 128 lanes for GPUs, 16 lanes for CPU uplink
- Excellent for inference (GPU P2P dominant)
- Bottlenecked for training (RAM-to-GPU throughput limited by x16 uplink)

### Guava Systems PCIe Switches

- Products: P5-SW104, P5-SW144 (Gen5)
- URL: [guavasystems.us](https://www.guavasystems.us/products/pcie-switches-2/)
- PCIe Gen6 switch expected June/July 2026

### NVIDIA MGX PCIe Switch Board with ConnectX-8

- 4 PCIe switches, 2 GPUs + 2 NICs per switch = 8 GPUs total
- Each GPU gets its own dedicated 800Gbps NIC (~100GB/s direct GPU-to-GPU across ethernet fabric)
- ConnectX-8 NICs have built-in PCIe switches
- 9th PCIe slot for DPU (ARM cores, can run GPU-only box without host CPU)

---

## nvidia-smi Topology Output

### 4-GPU Direct-Attach (Single NUMA Node)

```
$ nvidia-smi topo -m
        GPU0    GPU1    GPU2    GPU3
GPU0     X      NODE    NODE    NODE
GPU1    NODE     X      NODE    NODE
GPU2    NODE    NODE     X      NODE
GPU3    NODE    NODE    NODE     X
```

All GPUs show `NODE` -- connected via PCIe through the same NUMA node's root complex.

### 8-GPU Dual-CPU (Two NUMA Nodes)

```
$ nvidia-smi topo -m
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7
GPU0     X      PHB     PHB     PHB     SYS     SYS     SYS     SYS
GPU1    PHB      X      PHB     PHB     SYS     SYS     SYS     SYS
GPU2    PHB     PHB      X      PHB     SYS     SYS     SYS     SYS
GPU3    PHB     PHB     PHB      X      SYS     SYS     SYS     SYS
GPU4    SYS     SYS     SYS     SYS      X      PHB     PHB     PHB
GPU5    SYS     SYS     SYS     SYS     PHB      X      PHB     PHB
GPU6    SYS     SYS     SYS     SYS     PHB     PHB      X      PHB
GPU7    SYS     SYS     SYS     SYS     PHB     PHB     PHB      X
```

- `PHB` = Same NUMA node, different PCIe Host Bridge root ports
- `SYS` = Cross-NUMA, traverses xGMI fabric between CPU sockets

### 8-GPU with PCIe Switches (Single CPU)

```
$ nvidia-smi topo -m
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7
GPU0     X      PIX     PIX     PIX     PHB     PHB     PHB     PHB
GPU1    PIX      X      PIX     PIX     PHB     PHB     PHB     PHB
GPU2    PIX     PIX      X      PIX     PHB     PHB     PHB     PHB
GPU3    PIX     PIX     PIX      X      PHB     PHB     PHB     PHB
GPU4    PHB     PHB     PHB     PHB      X      PIX     PIX     PIX
GPU5    PHB     PHB     PHB     PHB     PIX      X      PIX     PIX
GPU6    PHB     PHB     PHB     PHB     PIX     PIX      X      PIX
GPU7    PHB     PHB     PHB     PHB     PIX     PIX     PIX      X
```

- `PIX` = Same PCIe switch (lowest latency P2P)
- `PHB` = Different switches, routed through CPU root complex

### Topology Legend

| Code | Meaning | Typical Latency |
|------|---------|-----------------|
| `PIX` | Same PCIe switch | ~0.45 us |
| `PHB` | Same NUMA, PCIe Host Bridge | ~0.36-0.45 us |
| `NODE` | Same NUMA node | ~0.36-0.45 us |
| `SYS` | Cross-NUMA (xGMI fabric) | ~0.44 us (Turin), ~14 us (no P2P) |

---

## Impact on Inference Performance

### Why Topology Matters for Inference

Tensor-parallel inference performs AllReduce operations after every attention and MoE layer. For models like Qwen3.5-397B or Kimi K2.5, these are small messages (32-256 KB) where **latency dominates over bandwidth**. A 1 us reduction in AllReduce latency translates directly to faster token generation.

### Measured AllReduce Bus Bandwidth (8 GPUs)

| Setup | Bus BW (GB/s) | Notes |
|-------|--------------|-------|
| luke (3x switches, single CPU) | 41.1 | Highest measured |
| Grimulkan (4x switches, single CPU) | 39.4 | With NCCL_MIN_NCHANNELS=8 |
| Festr (dual Turin, no switches) | 37.6 | After NCCL tuning |
| Festr (dual Turin, default NCCL) | 22.2 | Before tuning |

### Expert Parallelism on PCIe

Expert Parallelism (EP) was tested and found to be consistently slower or equivalent on PCIe setups:

> "EP is dead in the water for these setups" -- luke

EP requires massive inter-GPU bandwidth for routing tokens to experts. Without NVLink (~900 GB/s), PCIe (~56 GB/s unidirectional) cannot keep up.

**Exception:** luke achieved ~350 tok/s with EP=8 on Qwen3.5 using a custom allreduce kernel on PCIe switches, but this required significant custom patches.

---

## Topology Selection Guide

| Use Case | Recommended Topology | Why |
|----------|---------------------|-----|
| 4-GPU inference | Direct-attach (any CPU) | Simplest, no switches needed |
| 8-GPU inference (lowest latency) | PCIe switches, single CPU | Avoids cross-NUMA; switch P2P is fastest |
| 8-GPU inference (budget) | Dual Turin direct-attach | Reaches parity with switches after NCCL tuning |
| 8-GPU + KV cache offload to RAM | Dual Turin direct-attach | 128 direct PCIe lanes to RAM vs shared x16 uplink |
| 16-GPU inference | 4x switches, single CPU | Proven by Grimulkan; avoids dual-CPU complexity |
| Training / fine-tuning | Dual Turin direct-attach | RAM bandwidth critical; 128 lanes beats shared switch uplink |

### Key Takeaway

For pure inference (no KV cache offload, no training), **PCIe switches on a single CPU** provide the lowest latency. For workloads that need system RAM bandwidth (KV cache offload, training), **dual Turin direct-attach** is superior. Genoa dual-CPU setups should strongly consider adding switches to compensate for slower xGMI.
