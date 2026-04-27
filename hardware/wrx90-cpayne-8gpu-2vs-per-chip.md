# 8-GPU c-payne with 2 Virtual Switches per Physical Chip — Topology, Bandwidth, Collapse Analysis

A specific 8-GPU c-payne configuration where **two physical c-payne (Microchip Switchtec PM50100) chips** are each partitioned into **two Virtual Switches (VS)**, presenting **four "logical" PCIe switches** to the OS while only consuming two physical c-payne packages. Each physical chip has two independent x16 Gen5 upstream ports landing on two different CPU root ports; intra-chip traffic between the two VS partitions stays inside the switch fabric and never traverses the CPU.

The same chassis was tested in **three different physical wirings** of the four upstream ports against the CPU's root ports. This page documents all three side-by-side, including the bandwidth signatures that uniquely identify each variant, the collapse-trigger behaviour on each, and the inference workload that wins on each.

---

## TL;DR

* **Three rewirings tested** (variants V1, V2, V3) — same chips, same GPUs, only the upstream-port mapping to CPU root ports differs.
* **No collapse on any wiring.** The AMD posted-write collapse trigger pattern that fires catastrophically on Broadcom + Turin (`asus-esc8000a-e13p-broadcom-switches.md`) does not fire on Microchip + TR Pro at the 2-GPU-per-VS scale we tested, on any of the three wirings.
* **For inference, V1 (original 4-quadrant) is the recommended wiring.** It has the highest aggregate fabric throughput (all-to-all 254 GB/s vs 226 in V2 vs 213 in V3), the cleanest lspci mapping, NUMA-aligned memory bandwidth across all four CPU quadrants, and no penalty on any other workload.
* V2 and V3 are interesting science but trade ~10–16 % aggregate fabric for nothing structurally useful.

---

## System under test

| Item | Value |
|------|-------|
| Host | ASRock WRX90 WS EVO + AMD Threadripper Pro 7955WX (single socket, 4 IOD quadrants Q0–Q3) |
| Driver | NVIDIA 595.58.03 |
| CUDA Driver API | 13.2 |
| Kernel | Linux 6.18.24 (Ubuntu 24.04) |
| GPUs | 8× NVIDIA RTX PRO 6000 Blackwell **Server Edition** (96 GB GDDR7, SM120) |
| PCIe switches | **2× physical c-payne** (Microchip Switchtec PM50100, vendor 0x1f18 device 0x0101), each partitioned into 2 Virtual Switches |
| `iommu` | `off` |
| ACS Request-Redirect | disabled at boot via setpci (`/etc/systemd/system/disable-acs.service`) |

OS-visible PCIe switches across all variants: **SW1, SW2, SW3, SW4**, each with 2 GPUs. The OS sees four independent switches at four root buses (or two root buses × two root ports each, depending on the wiring).

---

## Topology overview — three wirings compared

The chip-to-quadrant assignment is what differs between variants. The chip-mapping (which two VS share a physical chip) is determined by which downstream ports the c-payne firmware groups together; the upstream-port destinations are determined by which physical PCIe slots the cables go into. All three variants below were achieved by rewiring the cables between the two physical chips and the WRX90 motherboard's PCIe slots.

| | **V1 — Original 4-quadrant** | **V2 — Split 2-quadrant** | **V3 — Concentrated** |
|---|---|---|---|
| chip A upstreams land on | Q0 + Q2 | Q0 + Q3 | **both on Q0** |
| chip B upstreams land on | Q1 + Q3 | Q0 + Q3 | **both on Q3** |
| Active CPU quadrants | **all 4** (Q0, Q1, Q2, Q3) | 2 (Q0, Q3) | 2 (Q0, Q3) |
| Chip mapping (from BW signature) | A = SW1 + SW3, B = SW2 + SW4 | A = SW1 + SW3, B = SW2 + SW4 | **A = SW1 + SW2, B = SW3 + SW4** |
| Detection: lspci root buses with GPUs | `pci0000:00`, `:20`, `:40`, `:e0` | `pci0000:00`, `:e0` (each with 2 root ports) | `pci0000:00`, `:e0` (each with 2 root ports) |
| Detection: 112 GB/s 2-pair pairs | SW1↔SW3, SW2↔SW4 | SW1↔SW3, SW2↔SW4 | **SW1↔SW2, SW3↔SW4** |

The bandwidth signature is the ground-truth identifier for chip mapping: any pair of "switches" that produce 112 GB/s aggregate on a 2-pair test live on the same physical chip; any pair that saturates at 56 GB/s lives on different physical chips.

For all three variants, traffic between two VS on the same physical chip stays inside that chip's internal fabric and never enters the CPU IOD. The variants differ only in which CPU quadrants the cross-chip traffic ends up using.

```
   Variant 1 — original (recommended for inference):
   ┌──────────────────────────────────┐
   │  CPU IOD: Q0  Q1  Q2  Q3 (all 4) │
   └─┬────┬────┬────┬─────────────────┘
     │    │    │    │
   chip A: ─Q0─┴────Q2─        ← chip A spans Q0+Q2
   chip B:     ─Q1──┴────Q3─   ← chip B spans Q1+Q3
   each VS on its own quadrant

   Variant 2 — split:
   ┌──────────────────────────────────┐
   │  CPU IOD: Q0  Q1  Q2  Q3 (only 2) │
   └─┬────┬─────────────┬────┬────────┘
     │    │             │    │
   chip A: ─Q0──────────┴────Q3─        ← chip A spans Q0+Q3
   chip B: ─Q0──────────┴────Q3─        ← chip B spans Q0+Q3
   each chip on 2 quadrants but Q1/Q2 idle

   Variant 3 — concentrated (current):
   ┌──────────────────────────────────┐
   │  CPU IOD: Q0  Q1  Q2  Q3 (only 2) │
   └─┬─┬─────────────────────┬─┬──────┘
     │ │                     │ │
   chip A: ─Q0─┘             │ │
   chip B:                   └─┴─Q3─
   each chip concentrated on one quadrant
```

---

## Variant 1 — Original 4-quadrant (recommended)

Each VS lands on a separate CPU quadrant:

| OS-visible switch | Upstream bridge | Root bus | CPU quadrant | GPUs |
|---------------------|-----------------|----------|--------------|------|
| SW1 (chip A VS₁) | `0000:01:00.0` | `pci0000:00` | Q0 | GPU 0, 1 |
| SW2 (chip B VS₁) | `0000:21:00.0` | `pci0000:20` | Q1 | GPU 2, 3 |
| SW3 (chip A VS₂) | `0000:41:00.0` | `pci0000:40` | Q2 | GPU 4, 5 |
| SW4 (chip B VS₂) | `0000:e1:00.0` | `pci0000:e0` | Q3 | GPU 6, 7 |

### V1 — single-pair P2P bandwidth (write)

| From → To | Path | BW |
|-----------|------|---:|
| GPU 0 → GPU 1 | intra-VS / intra-chip A | 56.3 GB/s |
| GPU 0 → GPU 4 | cross-VS / **intra-chip A** | 56.3 GB/s |
| GPU 0 → GPU 2 | cross-chip A → B (via CPU) | 56.3 GB/s |
| GPU 0 → GPU 6 | cross-chip A → B (via CPU) | 56.3 GB/s |

### V1 — 2-pair concurrent: same source VS → same destination VS

| | →SW1 (Q0) | →SW2 (Q1) | →SW3 (Q2) | →SW4 (Q3) |
|---|---:|---:|---:|---:|
| **SW1→** | — | 56.4 | **112.5** ✨ | 56.4 |
| **SW2→** | 56.4 | — | 56.4 | **112.5** ✨ |
| **SW3→** | **112.5** ✨ | 56.4 | — | 56.4 |
| **SW4→** | 56.4 | **112.5** ✨ | 56.4 | — |

Chip A = SW1 + SW3 (both intra-chip → 112 GB/s), chip B = SW2 + SW4.

### V1 — 1 src VS → 2 different dst VSs

| Source → 2 destinations | Aggregate | Notes |
|--------------------------|----------:|-------|
| SW1 → SW2 + SW3 | 112.5 | one path intra-chip A |
| SW1 → SW2 + SW4 | 56.4 | both cross-chip |
| SW1 → SW3 + SW4 | 112.5 | one path intra-chip A |
| SW2 → SW1 + SW3 | 56.0 | both cross-chip |
| SW2 → SW1 + SW4 | 112.5 | one intra-chip B |
| SW2 → SW3 + SW4 | 112.5 | one intra-chip B |
| SW3 → SW1 + SW2 | 112.5 | one intra-chip A |
| SW3 → SW1 + SW4 | 112.5 | one intra-chip A |
| SW3 → SW2 + SW4 | 51.4 | both cross-chip (mild contention) |
| SW4 → SW1 + SW2 | 112.5 | one intra-chip B |
| SW4 → SW1 + SW3 | 54.6 | both cross-chip (mild contention) |
| SW4 → SW2 + SW3 | 112.5 | one intra-chip B |

### V1 — aggregate / stress

| Test | V1 |
|------|---:|
| Bidirectional intra-VS (GPU 0 ↔ GPU 1) | 109.4 GB/s |
| 4 source switches → 1 dst switch | 75.4 GB/s |
| **All-to-all 8 GPU (56 pairs)** | **254.2 GB/s** (4.54 GB/s/pair) |

---

## Variant 2 — Split 2-quadrant

Same chip mapping as V1 (chip A = SW1+SW3, chip B = SW2+SW4) but the chips' upstreams now both land on the same two quadrants (Q0 and Q3). Each chip is split across Q0 and Q3.

### V2 — single-pair, 2-pair matrix

Identical to V1 — the chip mapping is the same, so the bandwidth signature is the same:

| | →SW1 | →SW2 | →SW3 | →SW4 |
|---|---:|---:|---:|---:|
| SW1→ | — | 56 | **112** ✨ | 56 |
| SW2→ | 56 | — | 56 | **112** ✨ |
| SW3→ | **112** ✨ | 56 | — | 56 |
| SW4→ | 56 | **112** ✨ | 56 | — |

### V2 — 1 src VS → 2 different dst VSs

| Pattern | V1 | V2 | Δ |
|---------|---:|---:|---:|
| SW3 → SW2+SW4 | 51.4 | **56.4** | +5 |
| SW4 → SW1+SW3 | 54.6 | **56.4** | +2 |
| All other 10 patterns | identical | identical | 0 |

The two cells where V1 sat just below saturation (51.4 and 54.6 GB/s) are now at full saturation.

### V2 — aggregate / stress

| Test | V1 | V2 | Δ |
|------|---:|---:|---:|
| 4 src → 1 dst | 75.4 | 75.2 | 0 |
| **All-to-all 8 GPU** | **254.2** | **225.6** | **−11 %** |
| All-to-all per-pair | 4.54 | 4.03 | −0.51 |

V2 loses ~11 % of all-to-all aggregate. The 8 GPUs now contend for two quadrants' worth of memory and inter-quadrant fabric capacity instead of four; only one Q0 ↔ Q3 IF link carries cross-chip traffic, vs four IF links available in V1.

---

## Variant 3 — Concentrated (current wiring on the rig)

Chip mapping has flipped: chip A is now SW1 + SW2 (both Q0), chip B is SW3 + SW4 (both Q3). Each physical chip has both upstreams on the same CPU quadrant.

| OS-visible switch | Upstream bridge | Root bus / port | CPU quadrant | GPUs |
|---------------------|-----------------|-----------------|--------------|------|
| SW1 (chip A VS₁) | `0000:01:00.0` | `pci0000:00`, port 00:01.1 | Q0 | GPU 0, 1 |
| SW2 (chip A VS₂) | `0000:05:00.0` | `pci0000:00`, port 00:03.1 | Q0 | GPU 2, 3 |
| SW3 (chip B VS₁) | `0000:e1:00.0` | `pci0000:e0`, port e0:01.1 | Q3 | GPU 4, 5 |
| SW4 (chip B VS₂) | `0000:e6:00.0` | `pci0000:e0`, port e0:03.1 | Q3 | GPU 6, 7 |

### V3 — single-pair, 2-pair matrix

Chip mapping has changed, so the matrix flipped:

| | →SW1 | →SW2 | →SW3 | →SW4 |
|---|---:|---:|---:|---:|
| SW1→ | — | **112.5** ✨ | 56.4 | 56.4 |
| SW2→ | **112.5** ✨ | — | 56.4 | 56.4 |
| SW3→ | 56.4 | 56.4 | — | **112.5** ✨ |
| SW4→ | 56.4 | 56.4 | **112.5** ✨ | — |

The 112 GB/s "fast" pair has moved: now SW1 ↔ SW2 and SW3 ↔ SW4. This proves chip A = SW1 + SW2 and chip B = SW3 + SW4.

### V3 — 1 src VS → 2 different dst VSs

| Pattern | V1 | V2 | V3 |
|---------|---:|---:|---:|
| SW1 → SW2 + SW3 | 112.5 | 112.5 | **112.5** |
| SW1 → SW2 + SW4 | 56.4 | 56.0 | **112.5** |
| SW1 → SW3 + SW4 | 112.5 | 112.5 | **56.4** |
| SW2 → SW1 + SW3 | 56.0 | 56.0 | **112.5** |
| SW2 → SW1 + SW4 | 112.5 | 112.5 | 112.5 |
| SW2 → SW3 + SW4 | 112.5 | 112.4 | **56.4** |
| SW3 → SW1 + SW2 | 112.5 | 112.5 | **56.4** |
| SW3 → SW1 + SW4 | 112.5 | 112.5 | 112.5 |
| SW3 → SW2 + SW4 | 51.4 | 56.4 | 112.5 |
| SW4 → SW1 + SW2 | 112.5 | 112.5 | **56.4** |
| SW4 → SW1 + SW3 | 54.6 | 56.4 | 112.5 |
| SW4 → SW2 + SW3 | 112.5 | 112.5 | 112.5 |

The "fast" rows are exactly those where one of the two destinations is the same-chip neighbour: that flow stays inside the chip and the other flow gets the full uplink to itself.

### V3 — aggregate / stress

| Test | V1 | V2 | V3 |
|------|---:|---:|---:|
| 4 src → 1 dst (all to SW1) | 75.4 | 75.2 | **112.0** |
| **All-to-all 8 GPU** | **254.2** | 225.6 | **213.1** |
| All-to-all per-pair | 4.54 | 4.03 | 3.80 |

* V3 has the **worst all-to-all** (−16 % vs V1). Every chip-A → chip-B flow has to traverse the same Q0 ↔ Q3 inter-quadrant fabric link. In V1 there were four IF paths, in V2 there were two parallel paths through chips' split uplinks; in V3 there's exactly one.
* V3 has the **best 4-src → 1-dst** (+49 % vs V1). With chip A = SW1+SW2, the SW2→SW1 flow stays inside chip A and doesn't fight for SW1's uplink.

---

## Side-by-side summary across all three variants

| Test | V1 (4-quadrant) | V2 (split) | V3 (concentrated) | Best |
|------|----------------:|-----------:|------------------:|:---:|
| Single-pair P2P | 56.3 | 56.3 | 56.3 | tie |
| 2-pair intra-chip (SW1↔SW3 in V1/V2, SW1↔SW2 in V3) | 112.5 | 112.5 | 112.5 | tie |
| 2-pair cross-chip (SW1→SW2 in V1/V2, SW1→SW3 in V3) | 56.4 | 56.4 | 56.4 | tie |
| 1 src → 2 cross-chip dsts (worst case) | 51–56 | 56 | 56 | tie |
| Bidirectional intra-VS | 109.4 | 109.4 | 109.4 | tie |
| 4 src → 1 dst | 75.4 | 75.2 | **112.0** | **V3** |
| **All-to-all 8 GPU (aggregate)** | **254.2** | 225.6 | 213.1 | **V1** |
| All-to-all (per-pair) | 4.54 | 4.03 | 3.80 | V1 |
| ASUS COLLAPSE pattern `(0,2)+(1,6)` | 54.0 | 56.0 | 112.5 | n/a — none collapses |
| ASUS 4-flow `(0,2)+(0,6)+(1,3)+(1,7)` | 56.2 | 56.2 | 75.9 | n/a — none collapses |

The only metric where V3 beats V1 is the rare "all four source switches reduce into one destination switch" pattern. Every other interesting workload either ties or favours V1.

---

## Collapse implications

The AMD posted-write collapse trigger documented in [`collapse-report.md`](collapse-report.md) requires:

1. Multiple source GPUs concurrently dispatching writes from the same source PCIe switch (sharing one upstream x16 link to one CPU root port), AND
2. Their destinations sit behind two or more different CPU root complexes.

**On all three variants, the collapse does not fire** at the 2-GPU-per-VS scale of this rig. The closest we come is 51 GB/s on V1's `SW3 → SW2+SW4` pattern, an ~8 % drop from the saturation line — far short of the 4× drop that constitutes a collapse.

Each variant has a different reason for being collapse-resistant:

* **V1**: traffic is spread across four quadrants. Even when 2 source GPUs of one VS dispatch to 2 different dst roots, those flows hit two different IF paths.
* **V2**: same chip mapping as V1, two of the cross-chip dst options lie on the same CPU quadrant, breaking condition (2) for some patterns.
* **V3**: structurally cannot satisfy condition (2) for cross-chip traffic. With chip A entirely on Q0 and chip B entirely on Q3, a 1-src → 2-dst pattern that hits two different roots **must** include at least one intra-chip flow (which bypasses CPU IOD entirely). The "trigger" then has only one flow on the source uplink, not two.

For comparison, the same architectural topology (2 chips × 2 VS × 2 GPUs) on **Broadcom + Turin** (ASUS ESC8000A-E13P, see [`asus-esc8000a-e13p-broadcom-switches.md`](asus-esc8000a-e13p-broadcom-switches.md)) collapses catastrophically (37 → 2.7 GB/s, 93 % drop) at the very same trigger pattern. The Microchip + TR Pro silicon does not.

---

## Inference workload recommendations

For LLM serving workloads, pick the wiring based on the dominant collective in your stack:

| Inference workload | Best variant | Why |
|--------------------|--------------|-----|
| **MoE alltoall** (Mixtral, Qwen-MoE, DeepSeek-MoE) | **V1** | +19 % aggregate fabric vs V3, alltoall is the bottleneck |
| Tensor-parallel (TP=8) ring all-reduce | any | per-hop bandwidth is the same on all variants |
| Tensor-parallel TP=4 × DP=2 | **V1** | each DP instance can use one chip cleanly |
| Pipeline parallel send/recv | any | small messages, latency-equivalent |
| Reduce-to-one (rare) | V3 | +49 % on the 4-src-to-1-dst pattern |
| Model loading from host RAM | **V1** | NUMA-aligned across 4 memory channels' worth of paths |
| CPU offload / pinned-memory dataloader | **V1** | 4 quadrants × 2 DDR5 channels each, cleanest fan-out |

**Default recommendation for production inference: V1.** It wins on MoE, ties on TP, has the cleanest lspci mapping for monitoring, and makes the most of the four CPU quadrants for any host-RAM-touching path (model load, KV-cache offload, dataloader pipelines).

V2 is functionally a slightly worse V1 (same chip mapping, fewer active quadrants, ~11 % less alltoall aggregate). V3 wins one rare workload but loses 16 % alltoall — only worth it if your specific job is reduce-to-one heavy.

NCCL parameters that help on this rig regardless of variant:

```bash
NCCL_P2P_LEVEL=SYS         # default on this rig, keeps cross-NUMA via PCIe peer
NCCL_ALGO=Ring             # ring all-reduce, no risk of triggering collapse
# NCCL_ALGO=Tree           # also fine on TR Pro+Microchip; would risk collapse on Broadcom+Turin
```

---

## Methodology and reproduction

All bandwidth numbers are produced by a single PyTorch test harness:

```python
import torch, time

SIZE = 256 * 1024 * 1024
ITERS = 100

def run(pairs):
    bufs, streams = {}, {}
    for s, d in pairs:
        bufs[(s, d)] = (torch.randn(SIZE//4, device=f'cuda:{s}'),
                        torch.empty(SIZE//4, device=f'cuda:{d}'))
        torch.cuda.set_device(s)
        streams[(s, d)] = torch.cuda.Stream(torch.device(f'cuda:{s}'))
    for s, d in pairs:                                  # warm-up
        with torch.cuda.stream(streams[(s, d)]):
            bufs[(s, d)][1].copy_(bufs[(s, d)][0])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        for s, d in pairs:
            with torch.cuda.stream(streams[(s, d)]):
                bufs[(s, d)][1].copy_(bufs[(s, d)][0])
    torch.cuda.synchronize()
    return SIZE * ITERS * len(pairs) / (time.perf_counter() - t0) / 1e9
```

The test issues `dst.copy_(src)` on a stream owned by the source GPU. With CUDA peer access enabled and ACS Request-Redirect disabled, the runtime selects the direct PCIe peer path — no host-RAM staging. Each `pairs` argument is a list of `(src_gpu, dst_gpu)` tuples; the function returns aggregate write bandwidth across all pairs.

### Test scripts in this repo

* [`scripts/collapse_2gpu_full.py`](../scripts/collapse_2gpu_full.py) — full bandwidth matrix sweep: single-pair P2P, 2-pair single-source-VS → single-destination-VS (4×4 matrix), 1-source → 2-destination patterns (12 combinations), 4-src → 1-dst, and all-to-all 8-GPU.
* [`scripts/asus_replica.py`](../scripts/asus_replica.py) — ASUS-equivalent collapse-trigger patterns with separate WRITE / READ measurements.

Run them as:

```bash
python3 scripts/collapse_2gpu_full.py    # full bandwidth matrix
python3 scripts/asus_replica.py          # ASUS-equivalent collapse patterns
```

Both assume `SW1: GPU 0, 1   SW2: GPU 2, 3   SW3: GPU 4, 5   SW4: GPU 6, 7`. Adjust the lists at the top of each script if your topology assigns differently.

### Topology detection

```bash
# Walk each GPU up the PCIe tree to its CPU root bus
for i in $(seq 0 7); do
  bus=$(nvidia-smi -i $i --query-gpu=gpu_bus_id --format=csv,noheader | sed 's/00000000://')
  root=$(readlink -f /sys/bus/pci/devices/0000:${bus,,}/../.. | grep -oE 'pci[0-9]+:[0-9a-f]+' | head -1)
  echo "GPU $i  bus $bus  -> $root"
done

# Switch upstream port summary
for sw in 01:00.0 05:00.0 21:00.0 41:00.0 e1:00.0 e6:00.0; do
  if [ -d /sys/bus/pci/devices/0000:${sw} ]; then
    parent=$(readlink -f /sys/bus/pci/devices/0000:${sw}/.. | grep -oE '[0-9a-f]+:[0-9a-f]+\.[0-9]+' | tail -1)
    speed=$(lspci -vv -s $sw | grep "LnkSta:" | head -1 | awk '{print $2,$3}')
    echo "$sw upstream parent_root_port=$parent  $speed"
  fi
done
```

Walking the sysfs tree gives the root bus for each GPU. Counting unique root buses tells you which variant you have:

* **4 unique root buses** (`pci0000:00`, `:20`, `:40`, `:e0`) → V1
* **2 unique root buses** with each having 2 distinct root-port BDFs (e.g. `00:01.1` and `00:03.1`) → V2 or V3 — disambiguate by running the bandwidth matrix and reading off which switch pairs hit 112 GB/s.

---

## Cross-references

* [`collapse-report.md`](collapse-report.md) — standalone report on the AMD IOD posted-write collapse, the bug this layout sidesteps.
* [`pcie-posted-write-collapse.md`](pcie-posted-write-collapse.md) — long-form history of the collapse investigation across multiple platforms.
* [`wrx90-cpayne-16gpu-4switch.md`](wrx90-cpayne-16gpu-4switch.md) — the 16-GPU 4-switch (4 GPU/switch) layout where the collapse fired hard on this same rig — i.e., **collapse appears once you have ≥4 GPUs on one upstream port**, not at the 2-GPU-per-VS scale of this page.
* [`asus-esc8000a-e13p-broadcom-switches.md`](asus-esc8000a-e13p-broadcom-switches.md) — same architectural topology (2 chips × 2 VS × 2 GPUs) but with **Broadcom PEX890xx** silicon and EPYC Turin host: collapses catastrophically at 2-GPU-per-VS where Microchip + TR Pro does not.
