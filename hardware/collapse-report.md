# AMD CPU Posted-Write Collapse — Reproducible Report

**A focused, self-contained report on a PCIe peer-to-peer write bandwidth collapse observed on AMD Threadripper Pro 7955WX (Storm Peak / Genoa-derived sIOD) when the traffic pattern is *one source PCIe switch dispatching to multiple destination CPU root complexes*.**

This document is intended to be readable on its own, without context from the rest of the [rtx6kpro wiki](https://github.com/voipmonitor/rtx6kpro). All measurements are reproducible with the scripts below.

---

## TL;DR

On AMD Threadripper Pro 7000 (and likely all Genoa-derived AMD Server I/O Dies), GPU-to-GPU peer-to-peer **WRITE** bandwidth between PCIe switches collapses by ~75% when **the same source PCIe switch is concurrently writing to GPUs sitting behind two or more different CPU root complexes**.

* **WRITE** drops from ~52 GB/s per pair to ~6–7 GB/s per pair (~85% loss).
* **READ** is unaffected (stays at full ~53 GB/s).
* Single-pair, same-destination-root, and independent-source-uplink patterns are unaffected — full bandwidth.

The collapse is at the **CPU silicon arbitration layer** (AMD I/O Die scalable-data-fabric arbitration of posted writes). It is **not fixed** by:
* newer kernel (tested 6.8 → 6.17 → 6.18.24)
* newer NVIDIA driver (tested 575 → 580 → 595.58.03)
* `iommu=off` or `iommu=pt`
* swapping motherboard slots so each PCIe switch has its own root port

It **is masked** by `iommu=on` (full translation), but that comes with a separate ~15% single-flow bandwidth penalty and reproducible NCCL all-reduce hangs at 8+ GPUs.

---

## System Under Test

| Component | Detail |
|-----------|--------|
| CPU | AMD Ryzen Threadripper PRO 7955WX (Storm Peak, 16C/32T, Zen 4, single sIOD) |
| Motherboard | ASRock WRX90 WS EVO, BIOS v12.09 (2026-02-04) |
| Memory | 256 GB DDR5 RDIMM ECC (8-channel) |
| GPUs | 16× NVIDIA RTX PRO 6000 Blackwell Workstation Edition (96 GB GDDR7, SM120) |
| PCIe switches | 4× **c-payne PCIe Gen5 switch** (Microchip Switchtec PM50100) |
| Kernel | Linux 6.18.24-061824-generic |
| NVIDIA Driver | 595.58.03 (CUDA 13.2) |
| IOMMU | `amd_iommu=off iommu=off` on kernel cmdline |
| ACS | Disabled at boot via `setpci` (Request-Redirect cleared) on every PCIe bridge with ACS capability |

### Topology (each c-payne switch on its own CPU root complex, 4 GPUs each)

```
CPU Threadripper Pro 7955WX (1 IOD, 4 quadrants, 8× x16 Gen5 root ports)
│
├─ root pci0000:00 (Q0, port 00:01.1)
│    └─ c-payne SW1 → GPU 0, 1, 2, 3   (bus 03–06)
│
├─ root pci0000:20 (Q1, port 20:01.1)
│    └─ c-payne SW2 → GPU 4, 5, 6, 7   (bus 23–26)
│
├─ root pci0000:40 (Q2, port 40:01.1)
│    └─ c-payne SW3 → GPU 8, 9, 10, 11 (bus 43–46)
│
└─ root pci0000:e0 (Q3, port e0:03.1)
     └─ c-payne SW4 → GPU 12,13,14,15  (bus E3–E6)
```

All four switches train at PCIe Gen5 x16. P2P confirmed working between all GPU pairs (`nvidia-smi topo -p2p w` reports OK across all). `nvidia-smi topo -m` reports `PIX` for same-switch pairs and `SYS` for cross-switch pairs.

---

## How to identify your topology before reproducing

```bash
# 1) Confirm IOMMU mode in cmdline
cat /proc/cmdline | grep -oE 'iommu=[a-z]+|amd_iommu=[a-z]+'

# 2) GPU bus IDs
nvidia-smi --query-gpu=index,gpu_bus_id --format=csv,noheader

# 3) Walk each GPU up the PCIe tree to its CPU root bus
for gpu in $(nvidia-smi --query-gpu=gpu_bus_id --format=csv,noheader | sed 's/00000000://'); do
  root=$(readlink -f /sys/bus/pci/devices/0000:${gpu,,}/../.. 2>/dev/null | grep -oE 'pci[0-9]+:[0-9a-f]+' | head -1)
  echo "GPU bus $gpu → $root"
done
```

You need at least two physical PCIe switches whose upstream ports terminate on **different** CPU root complexes (different top-level `pci0000:XX` buses) to reproduce.

---

## Reproduction — Method 1: Direct Python (PyTorch)

The cleanest, smallest reproduction. Requires `torch>=2.0` and the GPU indices for one source switch and two destination switches on different roots.

`collapse_repro.py`:
```python
import torch, time

SIZE = 256 * 1024 * 1024  # 256 MB per buffer
ITERS = 15

def concurrent(pairs, reads=False):
    bufs, streams = {}, {}
    for s, d in pairs:
        if reads:
            bufs[(s,d)] = (torch.randn(SIZE//4, device=f'cuda:{d}'),
                           torch.empty(SIZE//4, device=f'cuda:{s}'))
        else:
            bufs[(s,d)] = (torch.randn(SIZE//4, device=f'cuda:{s}'),
                           torch.empty(SIZE//4, device=f'cuda:{d}'))
        torch.cuda.set_device(s)
        streams[(s,d)] = torch.cuda.Stream(torch.device(f'cuda:{s}'))
    for s, d in pairs:                                  # warm-up
        with torch.cuda.stream(streams[(s,d)]):
            bufs[(s,d)][1].copy_(bufs[(s,d)][0])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        for s, d in pairs:
            with torch.cuda.stream(streams[(s,d)]):
                bufs[(s,d)][1].copy_(bufs[(s,d)][0])
    torch.cuda.synchronize()
    return SIZE * ITERS * len(pairs) / (time.perf_counter() - t0) / 1e9

# === ADJUST THESE TO YOUR TOPOLOGY ===
# SW1 GPUs:  0,1,2,3   (root 00, Q0)
# SW2 GPUs:  4,5,6,7   (root 20, Q1)
# SW3 GPUs:  8,9,10,11 (root 40, Q2)
# SW4 GPUs: 12,13,14,15(root e0, Q3)

tests = [
    # Healthy: single pair across switches
    ("BASELINE   1 pair  SW1->SW2 [1 dst root]",                [(0, 4)]),
    # Healthy: same source switch, BOTH destinations on the SAME dst root
    ("CONTROL    SW1->SW2 only, 2 GPUs [1 dst root]",           [(0, 4), (1, 5)]),
    # ── COLLAPSE TRIGGER ──
    # Same source switch, destinations on TWO different dst roots
    ("COLLAPSE-2 SW1 -> SW2 + SW3 [2 dst roots]",               [(0, 4), (1, 8)]),
    # Same source switch, three different dst roots
    ("COLLAPSE-3 SW1 -> SW2 + SW3 + SW4 [3 dst roots]",         [(0, 4), (1, 8), (2, 12)]),
    # Healthy: DIFFERENT source switches, multi dst roots — no per-source dispatch
    ("HEALTHY    indep src: SW1->SW2 + SW2->SW3 + SW3->SW4 + SW4->SW1",
        [(0, 4), (4, 8), (8, 12), (12, 0)]),
]

print(f"{'Test':<60s}  {'WRITE':>10s}  {'READ':>10s}")
print("-" * 84)
for label, pairs in tests:
    w = concurrent(pairs, reads=False)
    r = concurrent(pairs, reads=True)
    n = len(pairs)
    print(f"{label:<60s}  {w:6.1f} GB/s  {r:6.1f} GB/s  ({n} pair{'s'*(n>1)}, "
          f"{w/n:.1f}/{r/n:.1f} per pair)")
```

### Observed output on the test system (kernel 6.18, NVIDIA 595, `iommu=off`)

```
Test                                                              WRITE        READ
------------------------------------------------------------------------------------
BASELINE   1 pair  SW1->SW2 [1 dst root]                          52.5 GB/s    53.4 GB/s
CONTROL    SW1->SW2 only, 2 GPUs [1 dst root]                     52.1 GB/s    53.4 GB/s   ← src uplink saturated, healthy
COLLAPSE-2 SW1 -> SW2 + SW3 [2 dst roots]                         13.5 GB/s    53.2 GB/s   ← writes collapse, reads OK
COLLAPSE-3 SW1 -> SW2 + SW3 + SW4 [3 dst roots]                   12.6 GB/s    54.3 GB/s   ← writes collapse, reads OK
HEALTHY    indep src: SW1->SW2 + SW2->SW3 + SW3->SW4 + SW4->SW1  197.8 GB/s   ~200 GB/s    ← 4× full bandwidth, no collapse
```

**Per-pair WRITE in the COLLAPSE rows is ~6 GB/s**, vs ~52 GB/s in the BASELINE/CONTROL rows. **READ in the same rows is unaffected at ~53 GB/s**, confirming this is a posted-write-only effect.

---

## Reproduction — Method 2: `p2pmark`

Independent reproduction using the public `p2pmark` GPU benchmark. Source: <https://github.com/voipmonitor/p2pmark>

```bash
git clone https://github.com/voipmonitor/p2pmark
cd p2pmark
make            # needs nvcc + nccl
./p2pmark
```

The interesting block is **"Topology probe: staggered writes by peer distance"**, which schedules 8 concurrent transfers and varies how far each one reaches in the GPU index space:

```
+1  0->1 1->2 2->3 3->4 4->5 5->6 6->7 7->0   50.13 avg   401.07 total
+2  0->2 1->3 2->4 3->5 4->6 5->7 6->0 7->1   38.15 avg   305.23 total
+3  0->3 1->4 2->5 3->6 4->7 5->0 6->1 7->2   25.67 avg   205.33 total
+4  0->4 1->5 2->6 3->7 4->0 5->1 6->2 7->3   12.75 avg   102.00 total   ← all 8 streams cross SW1↔SW2 simultaneously, each src switch dispatches to 2 dst roots → collapse
+5  0->5 1->6 2->7 3->0 4->1 5->2 6->3 7->4   25.57 avg   204.58 total
+6  0->6 1->7 2->0 3->1 4->2 5->3 6->4 7->5   37.96 avg   303.67 total
+7  0->7 1->0 2->1 3->2 4->3 5->4 6->5 7->6   50.04 avg   400.28 total
```

`p2pmark` is clamped to 8 GPUs by the CUDA per-process P2P peer-mapping limit, but those 8 GPUs already span two switches on different root complexes (GPU 0–3 on SW1/root 00, GPU 4–7 on SW2/root 20), which is enough to expose the bug.

The **all-to-all stress test** in the same run reports **~20 GB/s per GPU / 159 GB/s total** vs an expected ~50 × 8 ≈ 400 GB/s — the collapse is acting on every cross-switch flow.

---

## What triggers the collapse — precise rule

**Collapse trigger:** Two or more concurrent peer-to-peer **WRITE** flows where:

1. The flows originate on **the same source PCIe switch** (i.e. they share one upstream x16 link to one CPU root port), AND
2. The destinations sit behind **two or more different CPU root complexes**.

If either condition is broken — different source switches *or* a single common destination root — bandwidth is healthy.

### Quick truth table from our measurements

| Source switches | Destination root complexes | Result |
|-----------------|----------------------------|--------|
| 1 (e.g. SW1)    | 1 (all dsts behind same root) | ✓ full bandwidth, uplink-saturated |
| 1 (e.g. SW1)    | **2 or more** (different roots) | **✗ collapse, ~6 GB/s/pair** |
| 2+ different    | 2+ different               | ✓ full bandwidth |
| 2+ different    | 1 common                   | ✓ full bandwidth |

### Reads vs writes

The collapse is **only on PCIe posted writes**. Pulling the data the opposite way (so each transfer is a READ from the perspective of the source GPU's PCIe link) gives full ~53 GB/s on every pattern, including the trigger pattern. This is why the script measures both — the WRITE/READ asymmetry is itself a strong fingerprint of the bug.

---

## Things ruled out

These were tested and **do not fix the collapse**:

* Linux kernel: 6.8 (Ubuntu) → 6.17 → 6.18.24 (latest mainline) — same behavior on all
* NVIDIA driver: 575 → 580 → 595.58.03 — same behavior on all
* CUDA: 12.x → 13.2 — same behavior
* `iommu=off` and `iommu=pt` (passthrough) — collapse on both
* Disabling all CPU mitigations (`mitigations=off spectre_v2=off …`) — no effect
* Disabling ACS Request-Redirect on every PCIe bridge — required for P2P at all, but does not affect collapse magnitude
* NCCL env tuning (`NCCL_P2P_LEVEL=SYS`, custom XML graph) — does not avoid the collapse, only reroutes around it
* **Moving each PCIe switch to its own dedicated CPU root port** (the topology used in this report). The previous test layout had two of the four c-payne switches sharing a single root (Q3); moving them to four independent root ports did not change the collapse magnitude or trigger pattern. This rules out "two switches sharing one root complex" as the cause.

These were tested and **do mask the collapse**, with caveats:

* `iommu=on` (full translated DMA) — restores ~52 GB/s on the collapse pattern, **but**:
  * ~15 % single-flow PCIe bandwidth drop on every transfer
  * 8-GPU NCCL allreduce hangs reproducibly
  * Bandwidth reporting in `p2pmark` produces several "ghost" numbers higher than line rate (likely IOTLB caching artifacts)

These were tested and **fully avoid the collapse** by changing the traffic pattern:

* Hierarchical PCIe-switch fabric with a *root switch* (e.g. 3-stage Microchip PM50100 setup): cross-switch traffic is forwarded fabric-to-fabric and never reaches a CPU root port. Full bandwidth on every pattern.
* Application-level avoidance: NCCL ring all-reduce ordering keeps each switch's outgoing traffic targeted at a single next-hop root complex per moment, so the trigger never fires. Tree all-reduce, all-to-all, and one-to-many broadcast do trigger it.

---

## What this looks like at the hardware level (hypothesis, not confirmed by AMD)

The single source PCIe root port on the source switch's quadrant has to forward every outgoing posted write into the IOD's scalable data fabric (SDF), targeted at one of four destination quadrants. When the destinations are all in one quadrant, the SDF arbiter holds steady credit flow in one direction. When they alternate between two or more destination quadrants, the arbiter has to interleave credits, drain ack queues for both targets, and switch routing tables per TLP. Empirically this drops effective throughput to roughly 1/8 of the line rate of the source x16 link.

Reads are unaffected because the read response path uses non-posted completion TLPs, which take a different arbitration path inside the IOD.

This matches public AMD documentation only loosely — there is no public errata describing this specifically. We have not been able to find a Genoa/Storm Peak PPR section that admits the issue. If anyone with AMD-internal access reads this, an errata pointer or a workaround flag would be greatly appreciated.

---

## Why this matters for c-payne users

c-payne sells PCIe Gen5 switches that the community uses to build 4×, 8×, and 16× GPU rigs without NVLink. The intended use case is precisely the pattern that triggers this collapse: many GPUs behind one switch, talking peer-to-peer to GPUs behind another switch. With Threadripper Pro / EPYC Genoa as the host CPU, the AMD IOD's SDF arbitration silently caps cross-switch GPU-GPU **write** throughput at roughly ⅛ of the link rate as soon as more than one destination root complex is touched concurrently.

In practical workloads:

* **Tensor-parallel inference within a single switch:** unaffected.
* **Tensor-parallel inference across switches with NCCL ring all-reduce:** mostly unaffected (ring keeps trigger off).
* **NCCL tree all-reduce, all-gather, all-to-all, one-to-many broadcast across switches:** **collapse-bound**.
* **Context parallelism / DCP across switches:** likely collapse-bound during cross-switch chunks.

For the c-payne *3-stage hierarchical* configuration (root switch + leaf switches), this collapse does not occur because cross-switch traffic never crosses a CPU root complex. That observation is consistent with everything in this report.

---

## Files in this repo for further context

* [`pcie-posted-write-collapse.md`](pcie-posted-write-collapse.md) — extended history, alternative reproductions, and per-platform results across multiple test rigs.
* [`wrx90-cpayne-microchip-switches.md`](wrx90-cpayne-microchip-switches.md) — 3-switch hierarchical setup that does NOT exhibit the collapse.
* [`wrx90-cpayne-2switch-flat.md`](wrx90-cpayne-2switch-flat.md) — 2-switch flat setup (no collapse, only 2 root complexes involved).
* [`wrx90-cpayne-16gpu-4switch.md`](wrx90-cpayne-16gpu-4switch.md) — 16-GPU 4-switch setup where collapse was originally discovered.
* [`asus-esc8000a-e13p-broadcom-switches.md`](asus-esc8000a-e13p-broadcom-switches.md) — independent reproduction on dual-socket EPYC + Broadcom PEX890xx switches.

---

## Contact

This report was assembled from work on the [voipmonitor/rtx6kpro](https://github.com/voipmonitor/rtx6kpro) wiki. If you have access to:

* Granite Rapids / Xeon 6 platforms with c-payne switches
* EPYC Turin (9005-series) platforms with c-payne switches
* AMD-internal documentation or errata covering IOD posted-write arbitration

…we'd very much like to hear whether the collapse trigger fires on those configurations or not. Open an issue on the repo or PR an additional results table.
