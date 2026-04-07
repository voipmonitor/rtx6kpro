# Dual-Socket EPYC 9124 — 8× RTX PRO 6000 Direct-Attach

PCIe topology analysis and P2P benchmarks for a dual-socket AMD EPYC 9124 system with 8× NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs in a **direct-attach** layout — no PCIe switches. Each GPU connects to its own CPU root port; cross-socket traffic traverses the inter-socket fabric (xGMI).

For comparison with switched topologies on similar GPUs see [WRX90 + 2× c-payne (flat)](https://github.com/voipmonitor/rtx6kpro/blob/master/hardware/wrx90-cpayne-2switch-flat.md) and [WRX90 + 3× c-payne (hierarchy)](https://github.com/voipmonitor/rtx6kpro/blob/master/hardware/wrx90-cpayne-microchip-switches.md).

## Table of Contents

* [System Overview](#system-overview)
* [Physical PCIe Topology](#physical-pcie-topology)
* [P2P Bandwidth Results](#p2p-bandwidth-results)
* [P2P Latency Results](#p2p-latency-results)
* [p2pmark Benchmark Results](#p2pmark-benchmark-results)
* [Multi-Flow Scaling Analysis](#multi-flow-scaling-analysis)
* [Comparison: Direct-Attach vs Switched Topologies](#comparison-direct-attach-vs-switched-topologies)
* [PCIe Oneshot AllReduce Crossover](#pcie-oneshot-allreduce-crossover)
* [GLM-5 Inference Benchmark (TP=8)](#glm-5-inference-benchmark-tp8-b12x-mtp)
* [TODO / Not Yet Measured](#todo--not-yet-measured)

---

## System Overview

| Component | Detail |
| --- | --- |
| **System** | Gigabyte G493-ZB0 |
| **Motherboard** | Gigabyte MZB3-G43-000 |
| **BIOS** | R17_F23 |
| **CPU** | 2× AMD EPYC 9124 16-Core (Genoa, 32 cores total) |
| **NUMA** | 2 nodes (one per socket) |
| **RAM** | 4× 96 GB DDR5 (advertised 6400 MT/s, configured 4800 MT/s) — 384 GB total, **only 4 of 24 memory channels populated** (2 per socket out of 12) |
| **GPUs** | 8× NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB GDDR7) |
| **PCIe Switches** | **None** — all GPUs direct-attached to CPU root ports |
| **Topology** | Direct-attach — 4 GPUs per socket, cross-socket via xGMI |
| **Kernel** | 6.17.9-1-pve (Proxmox VE kernel, bare metal - sorry, intended to run the workloads in a VM, gave up on that and did not do a clean reinstall) |
| **Driver** | NVIDIA 595.58.03 (open) |
| **CUDA** | 13.2 |
| **NIC** | 2× Mellanox ConnectX-4 Lx (mlx5_0/mlx5_1, NUMA 1) + 2× Intel X710-T 10GBase-T (NUMA 0) |

---

## Physical PCIe Topology

Eight GPUs, each on its own CPU root port, distributed across two sockets. There are no PCIe switches in the GPU path — every GPU sits on a `Speed 32GT/s, Width x16` root port (`xx:01.1`) directly off the CPU.

```
graph TD
    subgraph S0["Socket 0 — EPYC 9124 (NUMA 0, cores 0-15)"]
        RP00["Root Port<br/>00:01.1<br/>Gen5 x16"] --> GPU0["GPU0 — 01:00.0"]
        RP20["Root Port<br/>20:01.1<br/>Gen5 x16"] --> GPU1["GPU1 — 21:00.0"]
        RP40["Root Port<br/>40:01.1<br/>Gen5 x16"] --> GPU2["GPU2 — 41:00.0"]
        RP60["Root Port<br/>60:01.1<br/>Gen5 x16"] --> GPU3["GPU3 — 61:00.0"]
    end

    subgraph S1["Socket 1 — EPYC 9124 (NUMA 1, cores 16-31)"]
        RP80["Root Port<br/>80:01.1<br/>Gen5 x16"] --> GPU4["GPU4 — 81:00.0"]
        RPa0["Root Port<br/>a0:01.1<br/>Gen5 x16"] --> GPU5["GPU5 — a1:00.0"]
        RPc0["Root Port<br/>c0:01.1<br/>Gen5 x16"] --> GPU6["GPU6 — c1:00.0"]
        RPe0["Root Port<br/>e0:01.1<br/>Gen5 x16"] --> GPU7["GPU7 — e1:00.0"]
    end

    S0 ---|"xGMI inter-socket fabric"| S1
```

PCIe link state at idle reads `Speed 2.5GT/s, Width x16` on every GPU — this is normal ASPM L1 power management. Under load the links retrain to Gen5 x16 (32 GT/s). All GPU root ports advertise `LnkCap: Speed 32GT/s, Width x16`.

### Key Differences from Switched Topologies

|  | **Direct-Attach (this)** | [WRX90 3-switch hierarchy](https://github.com/voipmonitor/rtx6kpro/blob/master/hardware/wrx90-cpayne-microchip-switches.md) | [WRX90 2-switch flat](https://github.com/voipmonitor/rtx6kpro/blob/master/hardware/wrx90-cpayne-2switch-flat.md) |
| --- | --- | --- | --- |
| **PCIe switches** | None | 3 (1 root + 2 leaf) | 2 (both direct to CPU) |
| **Sockets / NUMA** | 2 / 2 | 1 / 1 | 1 / 1 |
| **Cross-tier path** | GPU → CPU → xGMI → CPU → GPU | GPU → Leaf → Root → Leaf → GPU | GPU → Switch → CPU → Switch → GPU |
| **Cross-tier uses CPU?** | Yes (xGMI) | No (root switch) | Yes (CPU root ports) |
| **GPUs per tier** | 4 per socket | 2 per leaf switch | 4 per switch |
| **nvidia-smi cross-tier** | SYS | NODE (fabric-routed) | NODE |

### nvidia-smi Topology

```
        GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7
GPU0     X    NODE  NODE  NODE  SYS   SYS   SYS   SYS
GPU1    NODE   X    NODE  NODE  SYS   SYS   SYS   SYS
GPU2    NODE  NODE   X    NODE  SYS   SYS   SYS   SYS
GPU3    NODE  NODE  NODE   X    SYS   SYS   SYS   SYS
GPU4    SYS   SYS   SYS   SYS    X    NODE  NODE  NODE
GPU5    SYS   SYS   SYS   SYS   NODE   X    NODE  NODE
GPU6    SYS   SYS   SYS   SYS   NODE  NODE   X    NODE
GPU7    SYS   SYS   SYS   SYS   NODE  NODE  NODE   X
```

GPU0–3 are **NODE** to each other (same socket, traversing the CPU's internal PCIe host bridges within one NUMA node). GPU4–7 likewise. Cross-socket is **SYS** — traffic crosses the xGMI inter-socket fabric. There is no `PIX` tier in this system because there are no switches.

---

## P2P Bandwidth Results

Measured with NVIDIA `p2pBandwidthLatencyTest` from `cuda-samples`.

### Bidirectional P2P=Enabled Matrix (GB/s)

```
   D\D     0      1      2      3      4      5      6      7
     0     —    112.3  112.3  112.3   84.0   82.8   75.8   75.9
     1   112.3    —    112.3  112.3   76.6   76.5   75.9   88.6
     2   112.3  112.3    —    112.3   76.4   75.0   84.1   84.0
     3   112.2  112.3  112.3    —     75.9   82.8   76.6   76.5
     4    84.0   76.5   76.4   75.9    —    112.3  112.4  112.3
     5    82.8   76.5   75.0   82.7  112.3    —    112.3  112.3
     6    75.9   75.9   84.0   76.6  112.3  112.3    —    112.3
     7    75.5   88.2   84.0   76.5  112.3  112.3  112.3    —
```

Two clear tiers:

* **Same-socket pairs**: ~112 GB/s bidirectional, uniform across all NODE pairs
* **Cross-socket pairs**: ~75–88 GB/s bidirectional, with visible variation depending on which xGMI link the traffic happens to take

### Bandwidth by Tier

| Tier | Unidirectional (GB/s) | Bidirectional (GB/s) | Notes |
| --- | --- | --- | --- |
| NODE (same socket, Gen5 x16) | 57.2 | 112.3 | Saturates Gen5 x16 |
| SYS (cross socket, via xGMI) | 42–44 | 75–88 | Limited by inter-socket xGMI link |

The same-socket number (57.2 GB/s unidirectional) is essentially line-rate Gen5 x16 (~63 GB/s theoretical), and a touch higher than the switched WRX90 systems' 53–54 GB/s — direct-attach with no switch ASIC in the path is the cleanest possible link.

The cross-socket number is the headline penalty of dual-socket: roughly **25–33% lower** bidirectional bandwidth and visible variation depending on which xGMI links the routing chooses.

### Cross-Socket Pair (GPU0↔GPU4)

```
Unidirectional P2P=Enabled:    42.0 / 44.4 GB/s
Bidirectional P2P=Enabled:     83.7 / 84.0 GB/s
```

### Same-Socket Pair (GPU0↔GPU1)

```
Unidirectional P2P=Enabled:    57.3 GB/s
Bidirectional P2P=Enabled:     112.3 GB/s
```

### Tests That Don't Apply to Direct-Attach

Two bandwidth-probe tests from the reference reports are **not applicable** to this topology:

* **Uplink Degradation Proof** — the switched reports force a switch uplink down to Gen2 to demonstrate that cross-switch traffic collapses through the CPU root port. No switch uplinks exist here; every GPU has its own root port.
* **Posted-Write Collapse** — the switched reports test a specific arbitration bug found on Broadcom PEX890xx switches by driving concurrent writes to multiple destinations through one switch. With no switches in the path, this bug cannot be triggered. The closest cross-socket equivalent (two concurrent writes from GPU0 and GPU1 into GPU4 and GPU6, all crossing xGMI) is listed as a TODO.

---

## P2P Latency Results

### P2P=Enabled Write Latency (µs) — `p2pBandwidthLatencyTest`

```
   GPU     0      1      2      3      4      5      6      7
     0   1.24   0.44   0.44   0.44   0.51   0.44   0.45   0.44
     1   0.52   1.26   0.44   0.45   0.44   0.52   0.45   0.45
     2   0.45   0.44   1.25   0.44   0.44   0.51   0.51   0.44
     3   0.50   0.44   0.44   1.23   0.51   0.44   0.50   0.44
     4   0.44   0.44   0.44   0.44   1.23   0.44   0.50   0.51
     5   0.45   0.45   0.44   0.44   0.52   1.30   0.44   0.44
     6   0.45   0.44   0.44   0.45   0.52   0.45   1.24   0.51
     7   0.44   0.44   0.44   0.44   0.51   0.44   0.43   1.23
```

Same-socket pairs sit at **0.44–0.45 µs**, cross-socket pairs at **0.50–0.52 µs**. The 0.05 µs penalty for crossing the xGMI link is visible but small.

### p2pmark Latency (128-byte remote reads)

| Tier | 1:1 Latency | Notes |
| --- | --- | --- |
| Same socket (NODE) | 0.83 µs | Average across all 12 same-socket ordered pairs |
| Cross socket (SYS) | 0.91 µs | Average across all 16 cross-socket ordered pairs |
| **Best pair** | **0.83 µs** | |

Under concurrent load (all 8 GPUs × 7 peers, 56 simultaneous transfers): **5.83 µs effective latency**.

---

## p2pmark Benchmark Results

Measured with [p2pmark](https://github.com/lukealonso/p2pmark) commit `3c39f36`.

### Scores

| Score | Value | Notes |
| --- | --- | --- |
| **PCIe Link Score** | **0.88** | 55.70 GB/s avg / 63.0 GB/s PCIe 5.0 x16 theoretical |
| **Dense Interconnect Score** | **0.41** | 184.58 GB/s measured / 445.57 GB/s ideal |

The PCIe Link Score is the highest of any system in this comparison family — direct attach gives clean per-link performance because there's no switch ASIC in the path.

### 8-GPU Topology Probe (staggered distance)

```
+1: 52.27 GB/s avg   ← adjacent in ring (mostly same socket)
+2: 37.90 GB/s avg
+3: 26.76 GB/s avg
+4: 19.32 GB/s avg   ← maximum distance (cross-socket pair)
+5: 39.56 GB/s avg
+6: 41.19 GB/s avg
+7: 40.82 GB/s avg
```

The probe shows the same shape as switched systems (clear minimum at +4 where each GPU pairs with its "antipode" across the topology) but with a different absolute pattern: the +5/+6/+7 wrap-around values are *lower* than +1, indicating that even when nominally "wrapping back" the staggered pattern hits a mix of cross-socket pairs.

The +4 distance value (19.3 GB/s) is the most diagnostic single number for this topology — every transfer in this round crosses the socket boundary.

### Single Reader, All 7 Peers Concurrent

```
GPU 0: 55.16 GB/s    GPU 4: 54.85 GB/s
GPU 1: 55.54 GB/s    GPU 5: 54.82 GB/s
GPU 2: 56.61 GB/s    GPU 6: 56.59 GB/s
GPU 3: 55.99 GB/s    GPU 7: 56.00 GB/s
```

All GPUs pull ~55 GB/s when reading from all peers concurrently — close to single-link Gen5 x16 saturation, since the bottleneck is the reader's own x16 uplink.

### All GPUs Read All Peers Simultaneously (56 concurrent transfers)

```
GPU 0: 24.25 GB/s    GPU 4: 22.80 GB/s
GPU 1: 23.20 GB/s    GPU 5: 22.71 GB/s
GPU 2: 22.81 GB/s    GPU 6: 22.82 GB/s
GPU 3: 22.65 GB/s    GPU 7: 23.34 GB/s

Total system bandwidth: 184.58 GB/s
```

---

## Multi-Flow Scaling Analysis

| Flows | Pattern | Total BW (GB/s) | Per-flow (GB/s) |
| --- | --- | --- | --- |
| 1 | NODE (same socket) | 57.2 | 57.2 |
| 1 | SYS (cross socket) | 42.0–44.4 | 42.0–44.4 |
| 8 | Single-reader, 7 peers concurrent | ~55 | ~7.9 (limited by reader uplink) |
| 56 | All-to-all (every GPU × 7 peers) | 184.6 | 3.30 |

Cross-socket bandwidth has no aggregate uplink to share (unlike switched systems where one x16 uplink feeds all GPUs on a switch) — instead each pair is bounded by xGMI link capacity, which gives variable per-pair numbers (75–88 GB/s bidirectional) depending on which fabric route the traffic picks.

---

## Comparison: Direct-Attach vs Switched Topologies

| Metric | **EPYC 9124 Direct (this)** | [WRX90 + 3× c-payne](https://github.com/voipmonitor/rtx6kpro/blob/master/hardware/wrx90-cpayne-microchip-switches.md) | [WRX90 + 2× c-payne (flat)](https://github.com/voipmonitor/rtx6kpro/blob/master/hardware/wrx90-cpayne-2switch-flat.md) |
| --- | --- | --- | --- |
| **Sockets / NUMA** | 2 / 2 | 1 / 1 | 1 / 1 |
| **PCIe switches** | 0 | 3 | 2 |
| **Same-tier unidir BW** | **57.2 GB/s** | 54.1 GB/s | 53.4 GB/s |
| **Same-tier bidir BW** | **112.3 GB/s** | ~108 GB/s | ~107 GB/s |
| **Cross-tier bidir BW** | 75–88 GB/s | 54.2 GB/s (cross-switch) | 53.1 GB/s (cross-switch) |
| **Best-pair latency** | 0.44 µs (same sock) | 1.14 µs cross-switch | 1.40 µs cross-switch |
| **8-GPU all-to-all** | **184.6 GB/s** | **196 GB/s** | 162 GB/s |
| **PCIe Link Score** | **0.88** | 0.86 | 0.86 |
| **Interconnect Score** | 0.41 | **0.45** | 0.38 |
| **Effective latency (loaded)** | **5.83 µs** | 6.56 µs | 7.48 µs |
| **+4 distance probe** | 19.3 GB/s | **25.6 GB/s** | 12.6 GB/s |

### Key Takeaways

1. **Per-link BW is highest** on direct-attach (57 vs ~54 GB/s) — no switch ASIC overhead, cleanest path
2. **Latency under load is best** on direct-attach (5.83 vs 6.56 / 7.48 µs) — no switch hops to traverse
3. **Aggregate 8-GPU bandwidth sits between** the 2-switch flat (162) and 3-switch hierarchy (196). The NUMA boundary costs roughly what the flat 2-switch CPU-routed cross path costs, but the within-socket NODE traffic is faster than within-switch PIX traffic
4. **Cross-tier bandwidth is the trade-off**: cross-socket xGMI is faster than cross-switch via CPU routing (75–88 vs 53 GB/s bidirectional), but slower than the 3-switch hierarchical fabric for full all-to-all workloads
5. **The xGMI variation** (76–88 GB/s for nominally identical cross-socket pairs) is unique to the dual-socket layout — switched topologies show much more uniform cross-tier numbers

This direct-attach dual-socket topology is competitive for workloads that mostly stay within a socket (e.g., TP=4 sharded across one half of the GPUs) and for any workload sensitive to point-to-point latency. It loses to the 3-switch hierarchy for symmetric all-to-all because the xGMI fabric is more constrained than a dedicated PCIe switch fabric for the cross-half traffic pattern.

### Cross-System Comparison

Reference numbers sourced from the respective voipmonitor/rtx6kpro hardware reports:

| System | 8-GPU all-to-all | Interconnect Score | Cross-tier latency |
| --- | --- | --- | --- |
| WRX90 + 3× c-payne (hierarchy) | **196 GB/s** | **0.45** | 1.14 µs |
| Turin direct-attach | 190 GB/s | 0.41 | 0.84 µs |
| **EPYC 9124 direct-attach (this)** | **184.6 GB/s** | **0.41** | **0.91 µs** |
| WRX90 + 2× c-payne (flat) | 162 GB/s | 0.38 | 1.40 µs |
| ASUS ESC8000A-E13P (Broadcom) | 52 GB/s | 0.12 | 1.34 µs |

This system sits in the top cluster with the other direct-attach and switched hierarchy topologies, closely matching the single-socket Turin direct-attach on both total aggregate bandwidth and interconnect score. The cross-tier latency is better than either switched system because there's no switch ASIC hop to traverse.

---

## PCIe Oneshot AllReduce Crossover

Captured at SGLang startup with `--enable-pcie-oneshot-allreduce --enable-pcie-oneshot-allreduce-fusion`, **TP=8, bf16, 8 GPUs crossing the NUMA boundary**:

| Size | Custom (µs) | NCCL (µs) | Winner |
| --- | --- | --- | --- |
| 1 KB | 7.5 | 18.0 | Custom 2.4× |
| 2 KB | 7.7 | 18.3 | Custom 2.4× |
| 4 KB | 8.2 | 28.2 | Custom 3.4× |
| 8 KB | 8.9 | 18.6 | Custom 2.1× |
| 16 KB | 12.0 | 25.7 | Custom 2.1× |
| 32 KB | 16.4 | 24.9 | Custom 1.5× |
| 64 KB | 25.6 | 27.0 | Custom 1.05× |
| 72 KB | 26.9 | 31.7 | Custom 1.2× |
| 80 KB | 30.1 | 23.3 | **NCCL wins** |
| 96 KB | 35.2 | 30.4 | NCCL wins |
| 120 KB | 41.0 | 33.9 | NCCL wins |
| 128 KB | 44.5 | 30.9 | NCCL wins |
| 256 KB | 80.8 | 55.4 | NCCL wins |
| 512 KB | 149.1 | 79.6 | NCCL wins |
| 1 MB | 288.6 | 129.7 | NCCL wins |

Auto-tuner selected **`max_size = 72 KB`** as the last point where the custom PCIe oneshot kernel beats NCCL.

### Comparison to Reference

| Topology | TP | Crossover | Notes |
| --- | --- | --- | --- |
| **EPYC 9124 direct-attach (this)** | **TP=8** | **72 KB** | Crosses NUMA boundary (xGMI hop) |
| WRX90 + 3× c-payne (hierarchy) | TP=4 | 120 KB | Single switch, no cross-tier hop |
| WRX90 + 2× c-payne (flat) | TP=4 | 120 KB | Single switch, no cross-tier hop |

Not directly comparable — TP=4 on a single switch is a much friendlier workload for the custom kernel than TP=8 across NUMA. The custom kernel coordinates across twice as many GPUs and the coordination traffic has to traverse the xGMI fabric, so NCCL catches up at a smaller message size (72 KB vs 120 KB). At small sizes (1–64 KB) the custom kernel is still 1.05–3.4× faster than NCCL, which is where most latency-sensitive token-generation allreduce calls fall.

---

## GLM-5 Inference Benchmark (TP=8, b12x, MTP)

End-to-end inference benchmark on GLM-5 (744B MoE NVFP4) with TP=8, b12x MoE backend, speculative decoding (MTP), and the PCIe oneshot allreduce kernel enabled. Measured with [llm-inference-bench](https://github.com/voipmonitor/llm-inference-bench) v0.3.0.

The benchmark was run twice: **once without `--cap-add SYS_NICE`** (SGLang cannot pin TP workers to their GPU's local NUMA node) and **once with it** (SGLang wraps each worker in `numactl --cpunodebind=N --membind=N`, correctly mapping TP0–3 → NUMA 0 and TP4–7 → NUMA 1).

### Launch Configuration

```bash
# Docker (the SYS_NICE version)
docker run -it --rm \
    --cap-add SYS_NICE \
    --entrypoint /bin/bash \
    --gpus all \
    --ipc=host --shm-size=8g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --network host \
    -v /hf:/root/.cache/huggingface \
    -v /hf/nccl_graph_opt.xml:/mnt/nccl_graph_opt.xml \
    -v sglang-nightly-jit130:/cache/jit \
    voipmonitor/sglang:cu130

# Server
SGLANG_ENABLE_SPEC_V2=True SGLANG_ENABLE_JIT_DEEPGEMM=0 SGLANG_ENABLE_DEEP_GEMM=0 \
NCCL_GRAPH_FILE=/mnt/nccl_graph_opt.xml NCCL_IB_DISABLE=1 NCCL_P2P_LEVEL=SYS \
NCCL_ALLOC_P2P_NET_LL_BUFFERS=1 NCCL_MIN_NCHANNELS=8 OMP_NUM_THREADS=8 SAFETENSORS_FAST_GPU=1 \
python3 -m sglang.launch_server \
  --model-path festr2/GLM-5-NVFP4-MTP \
  --tp 8 --quantization modelopt_fp4 --kv-cache-dtype bf16 \
  --enable-pcie-oneshot-allreduce --enable-pcie-oneshot-allreduce-fusion \
  --mem-fraction-static 0.85 --cuda-graph-max-bs 32 \
  --chunked-prefill-size 16384 --attention-backend flashinfer \
  --fp4-gemm-backend b12x --moe-runner-backend b12x \
  --max-running-requests 64 --host 0.0.0.0 --port 5000 \
  --served-model-name glm-5
```

Server config (both runs): KV cache budget **199,616 tokens**, max running requests 64, weights ~60.9 GB/GPU, KV cache 18.6 GB/GPU.

With SYS_NICE, SGLang logs confirm correct pinning:

```
mp.set_executable ... exec numactl --cpunodebind=0 --membind=0 ... [TP0-TP3]
mp.set_executable ... exec numactl --cpunodebind=1 --membind=1 ... [TP4-TP7]
```

### Prefill Speed (C=1)

| Context | Tokens | no-SYS_NICE (tok/s) | SYS_NICE (tok/s) | Δ |
| --- | --- | --- | --- | --- |
| 8k | 8,198 | 5,568 | 5,566 | −0.04% |
| 16k | 16,233 | 5,176 | 5,180 | +0.08% |
| 32k | 32,342 | 3,731 | **4,543** | **+21.8%** ⚠ |
| 64k | 64,553 | 3,613 | 3,755 | +3.9% |
| 128k | 125,213 | 2,784 | 2,826 | +1.5% |

The 32k jump is likely a statistical artifact — N=2 samples for both runs, and every other context size is within 4%. Worth re-running with more samples to confirm.

### Aggregate Decode Throughput (tok/s) — SYS_NICE

```
xychart-beta
    title "GLM-5 Aggregate Throughput — EPYC 9124 Direct, TP=8, b12x+MTP (SYS_NICE)"
    x-axis ["c=1", "c=2", "c=4", "c=8", "c=16", "c=32", "c=64"]
    y-axis "Tokens/sec" 0 --> 800
    bar [58.4, 106.5, 181.6, 300.7, 436.8, 640.2, 605.6]
```

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **0** | 58.4 | 106.5 | 181.6 | 300.7 | 436.8 | **640.2** | 605.6 |
| **16k** | 48.6 | 89.3 | 143.3 | 221.9 | — | — | — |
| **32k** | 45.2 | 77.4 | 126.7 | — | — | — | — |
| **64k** | 38.3 | 66.8 | — | — | — | — | — |
| **128k** | 32.4 | — | — | — | — | — | — |

### Per-Request Avg Throughput (tok/s) — SYS_NICE

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **0** | 58.4 | 53.3 | 45.4 | 37.6 | 27.3 | 20.0 | 9.5 |
| **16k** | 48.6 | 44.6 | 35.8 | 27.7 | — | — | — |
| **32k** | 45.2 | 38.7 | 31.7 | — | — | — | — |
| **64k** | 38.3 | 33.4 | — | — | — | — | — |
| **128k** | 32.4 | — | — | — | — | — | — |

### SYS_NICE Does Not Matter For This Workload

Decode throughput at every concurrency × context cell is within **±1.1%** between the two runs — statistical noise:

| Decode c=0 | no-SYS_NICE | SYS_NICE | Δ |
| --- | --- | --- | --- |
| c=1 | 58.6 | 58.4 | −0.3% |
| c=2 | 106.8 | 106.5 | −0.3% |
| c=4 | 181.4 | 181.6 | +0.1% |
| c=8 | 297.4 | 300.7 | +1.1% |
| c=16 | 436.2 | 436.8 | +0.1% |
| c=32 | 640.7 | 640.2 | −0.1% |
| c=64 | 603.0 | 605.6 | +0.4% |

**The c=64 regression (below c=32) persists across both runs**, confirming it is not a NUMA affinity issue. With a KV budget of 199,616 tokens and 64 concurrent requests, that's only ~3,100 tokens per slot — the scheduler is fighting for cache space rather than computing.

This null result is useful: it establishes empirically that **GLM-5 decode on this configuration is GPU-bound, not host-bound.** Weights live on GPU, KV cache lives on GPU, and only the small PCIe oneshot allreduce (≤72 KB) crosses the fabric — which doesn't touch host DRAM at all. There's simply nothing for CPU-side NUMA locality to improve.

Prefill also shows no systematic improvement (4 of 5 sizes within ±4%), though the 32k outlier warrants a re-run for more samples.

### Comparison to Reference Systems

Using the SYS_NICE numbers (the pinned, correctly-configured run):

| Metric | **EPYC 9124 Direct (SYS_NICE)** | [WRX90 2-switch flat](https://github.com/voipmonitor/rtx6kpro/blob/master/hardware/wrx90-cpayne-2switch-flat.md) | Delta |
| --- | --- | --- | --- |
| Decode c=1 | **58.4** | 57.5 | +1.6% |
| Decode c=2 | **106.5** | 102.1 | +4.3% |
| Decode c=4 | **181.6** | 177.3 | +2.4% |
| Decode c=8 | 300.7 | 305.0 | −1.4% |
| Decode c=16 | 436.8 | 473.6 | −7.8% |
| Decode c=32 | 640.2 | 692.4 | −7.5% |
| Decode c=64 | 605.6 ⚠ | 744.3 | −18.6% |
| Prefill 8k (tok/s) | 5,566 | 6,490 | −14.2% |
| Prefill 16k | 5,180 | 6,220 | −16.7% |
| Prefill 32k | 4,543 | 4,542 | **±0%** |
| Prefill 64k | 3,755 | 4,465 | −15.9% |
| Prefill 128k | 2,826 | 3,502 | −19.3% |

**Crossover:** direct-attach wins at c=1–4 (low-concurrency latency advantage from clean per-link), switched reference pulls ahead from c=8 onward and widens with load. The 15–20% prefill gap is consistent across context sizes except 32k (which is likely a noise outlier given N=2 samples).

Since NUMA pinning doesn't explain the gap, the remaining candidates are:

1. **Cross-socket xGMI bottleneck** for TP=8 collective traffic — allreduce participants are split across two sockets, and even with per-link P2P bandwidth at 75–88 GB/s cross-socket, the aggregate pattern for large collective operations may be bound by xGMI capacity
2. **Half-populated memory** (4 of 24 channels) hurts the host-touched portions of prefill (tokenizer, scheduling, weight staging buffers)
3. **Platform/BIOS differences** we haven't characterized (Gigabyte MZB3-G43 vs ASRock WRX90 WS EVO PCIe routing, NPS settings, etc.)
4. **Driver minor-version differences** (595.58.03 here vs 595.45.04 in the reference)

The c=64 collapse is almost certainly KV-budget-limited and would need either fewer concurrent requests, a bigger KV budget, or MTP configuration tuning to fix.

---

## TODO / Not Yet Measured

* **Populated DIMM re-run** — if possible, measure with all 24 channels populated to isolate the memory bandwidth contribution to the prefill gap
* **Posted-Write Collapse Test** — less directly relevant without switches, but the cross-socket equivalent (concurrent writes through xGMI) would be informative
* **32k prefill re-run with more samples** — to confirm the SYS_NICE 21.8% jump was noise
* **NPS4 / sub-NUMA experiments** — the EPYC 9124 supports NPS4 partitioning which might help localize the 4-GPUs-per-socket layout further
* **c=48 / c=56 datapoints** — to find exactly where the KV-budget-induced collapse begins

---

## Hardware Configuration Notes

### ACS

All ACS-capable devices have all `ACSCtl` flags cleared:

```
ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
```

This is the ideal state for GPU P2P — `ReqRedir-` and `CmpltRedir-` mean P2P transactions are not forced through the IOMMU. No manual ACS-disable was needed.

### MaxReadReq

| Device class | MaxReadReq |
| --- | --- |
| GPU endpoints (`xx:00.0`) | 512 bytes |
| GPU root ports (`xx:01.1`) | 512 bytes |
| RCEC (`xx:00.3`) | 128 bytes |

GPU and root port settings match the switched WRX90 reference systems (512 bytes throughout). No ASPEED/X710/Mellanox device limits the GPU path because each GPU sits on its own dedicated root port.

### PCIe Links

* All GPU root ports: `LnkCap Speed 32GT/s, Width x16` (Gen5 x16)
* All GPUs at idle: `LnkSta Speed 2.5GT/s` (Gen1, ASPM L1 power saving — normal)
* All GPUs under load: Gen5 x16, verified with `lspci` during a `p2pBandwidthLatencyTest` run:

```
GPU 01:00.0  LnkSta: Speed 32GT/s, Width x16
GPU 21:00.0  LnkSta: Speed 32GT/s, Width x16
GPU 41:00.0  LnkSta: Speed 32GT/s, Width x16
GPU 61:00.0  LnkSta: Speed 32GT/s, Width x16
GPU 81:00.0  LnkSta: Speed 32GT/s, Width x16
GPU a1:00.0  LnkSta: Speed 32GT/s, Width x16
GPU c1:00.0  LnkSta: Speed 32GT/s, Width x16
GPU e1:00.0  LnkSta: Speed 32GT/s, Width x16
```

### NUMA / CPU

* 2 sockets, 2 NUMA nodes, no NPS sub-NUMA partitioning
* GPU0–3 affined to NUMA 0 (cores 0–15)
* GPU4–7 affined to NUMA 1 (cores 16–31)
* Inter-node distance reported as 32 (vs 10 local)

### Memory — Severely Underpopulated

This system has **4 of 24 memory channels populated** — 2 DIMMs per socket out of a possible 12. This is the most significant non-GPU configuration issue on the box:

| Metric | Configured | Platform max |
| --- | --- | --- |
| DIMMs per socket | 2 | 12 |
| Channels per socket | 2 | 12 |
| Effective speed | DDR5-4800 | DDR5-4800 (CPU-limited on 9124) |
| Per-socket peak BW (theoretical) | ~76 GB/s | ~460 GB/s |
| **Per-socket bandwidth available** | **~17%** | of platform peak |

The DDR5-6400 modules are downclocked to 4800 MT/s because the EPYC 9124 tops out at DDR5-4800. The bigger problem is channel population: with only 2 channels active per socket, host memory bandwidth is roughly 1/6 of what a fully populated 12-DIMM-per-socket configuration would deliver.

This is **largely irrelevant for the GPU-to-GPU P2P benchmarks above** (which travel over PCIe and never touch host DRAM), but it will hurt:

* Model loading from disk into GPU memory (host DRAM is the staging buffer)
* Any workload that pages tensors through host memory
* CPU-side data preprocessing for inference
* NCCL host-staged collectives if P2P is ever disabled

Worth populating the missing channels before any production use.

### NUMA Affinity (SGLang Container)

During the first SGLang run, every TP worker logged:

```
User lacks permission to set NUMA affinity, skipping NUMA node configuration for GPU.
If using docker, try adding --cap-add SYS_NICE to your docker run command.
```

Rerunning with `--cap-add SYS_NICE` does enable correct pinning — SGLang wraps each TP worker in `numactl --cpunodebind=N --membind=N` (TP0–3 → NUMA 0, TP4–7 → NUMA 1) and the log message changes to `NUMA affinity is already constrained for process`.

**However, the inference benchmark results were essentially identical between the two runs** (see [GLM-5 Inference Benchmark](#glm-5-inference-benchmark-tp8-b12x-mtp) section). For this workload — GLM-5 TP=8 with weights and KV cache fully resident on GPU — NUMA pinning does not change measurable throughput. Still recommended for production since it's free and the fix is correct in principle, but not a silver bullet for the ~15% prefill gap versus the switched reference.
