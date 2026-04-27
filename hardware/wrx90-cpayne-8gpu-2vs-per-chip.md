# 8-GPU c-payne with 2 Virtual Switches per Physical Chip — Topology, Bandwidth, Collapse Analysis

A specific c-payne configuration where **two physical c-payne (Microchip Switchtec) chips** are each partitioned into **two Virtual Switches (VS)**, presenting **four "logical" PCIe switches** to the OS while only consuming two physical c-payne packages. Each physical chip has two independent x16 Gen5 upstream ports landing on two different CPU root complexes; intra-chip traffic between the two VS partitions stays entirely inside the switch fabric and never traverses the CPU.

This page documents the topology, the surprising bandwidth observations it produces, and what it means for the AMD posted-write collapse trigger we have been chasing on this rig.

---

## System under test

| Item | Value |
|------|-------|
| Host | ASRock WRX90 WS EVO + AMD Threadripper Pro 7955WX (single socket, 4 IOD quadrants Q0–Q3) |
| Driver | NVIDIA 595.58.03 |
| CUDA Driver API | 13.2 |
| Kernel | Linux 6.18.24 (Ubuntu 24.04) |
| GPUs | 8× NVIDIA RTX PRO 6000 Blackwell **Server Edition** (96 GB GDDR7, SM120) |
| PCIe switches | **2× physical c-payne** (Microchip Switchtec PM50100, vendor 0x1f18 device 0x0101) — each partitioned into 2 Virtual Switches |
| `iommu` | `off` |
| ACS Request-Redirect | disabled at boot via setpci (`/etc/systemd/system/disable-acs.service`) |

---

## Topology

The OS sees **four** PCIe switches at four root buses:

| OS-visible "switch" | Upstream bridge | Root bus | CPU quadrant | GPUs |
|---------------------|-----------------|----------|--------------|------|
| SW1 (VS) | `0000:01:00.0` | `pci0000:00` | Q0 | GPU 0, 1 |
| SW2 (VS) | `0000:21:00.0` | `pci0000:20` | Q1 | GPU 2, 3 |
| SW3 (VS) | `0000:41:00.0` | `pci0000:40` | Q2 | GPU 4, 5 |
| SW4 (VS) | `0000:e1:00.0` | `pci0000:e0` | Q3 | GPU 6, 7 |

**But internally there are only two physical c-payne packages**, each carrying two of the above virtual switches:

```
                        ╔══════════════════════════════════════════════╗
                        ║   CPU Threadripper Pro 7955WX  (1 IOD)       ║
                        ║                                              ║
                        ║   Q0    ──── inter-quadrant IF ──── Q1       ║
                        ║   │                                  │       ║
                        ║   │                                  │       ║
                        ║   Q2    ──── inter-quadrant IF ──── Q3       ║
                        ╚════│════════════│════════════│════════│══════╝
                             │            │            │        │
                       x16 Gen5    x16 Gen5    x16 Gen5    x16 Gen5
                       (root 00)   (root 20)   (root 40)   (root e0)
                             │            │            │        │
                             │            │            │        │
            ┌────────────────┼────────────┼────────────┼────────┼─────────────┐
            │  PHYSICAL CHIP A             │            │       PHYSICAL CHIP B │
            │  ┌──────────┐                │            ┌──────────┐         │
            │  │   SW1    │     ┌──────────┐│            │   SW4    │         │
            │  │  (VS A1) │     │   SW2    ││            │  (VS B2) │         │
            │  │ root 00  │     │  (VS B1) ││            │ root e0  │         │
            │  │ Q0 / x16 │     │ root 20  ││            │ Q3 / x16 │         │
            │  └─┬───┬────┘     │ Q1 / x16 ││            └─┬───┬────┘         │
            │    │   │  ╲       └─┬───┬────┘│              │   │  ╲           │
            │    │   │   ↔ chip A │   │  ╲  │              │   │   ↔ chip B  │
            │    │   │ ↕ internal │   │   ↕  │             │   │ ↕ internal  │
            │    │   │   fabric   │   │ chip │             │   │   fabric    │
            │    │   │   ↕        │   │  B   │             │   │   ↕         │
            │    │   │  ╱         │   │ int. │             │   │  ╱          │
            │  ┌─┴───┴────┐       │   │ fab. │           ┌─┴───┴────┐        │
            │  │   SW3    │       │   │      │           │   SW3 ←─── wait, │
            │  │  (VS A2) │       └─┬─┴──────┘           │   that's wrong   │
            │  │ root 40  │         │                    │   redrawing →    │
            │  │ Q2 / x16 │       …continued on chip B…  │                  │
            │  └─┬───┬────┘                              │                  │
            └────┼───┼─────────────────────────────────  └──────────────────┘
                GPU 0,1     GPU 4,5                       GPU 2,3   GPU 6,7
```

Cleaner diagram (the ASCII wraps weirdly above; this is the actual mapping):

```
 Physical c-payne CHIP A                    Physical c-payne CHIP B
 ┌──────────────────────────┐               ┌──────────────────────────┐
 │  ╔════════╗   ╔════════╗ │               │  ╔════════╗   ╔════════╗ │
 │  ║  SW1   ║   ║  SW3   ║ │               │  ║  SW2   ║   ║  SW4   ║ │
 │  ║ (VS A1)║   ║ (VS A2)║ │               │  ║ (VS B1)║   ║ (VS B2)║ │
 │  ║ Q0,x16 ║   ║ Q2,x16 ║ │               │  ║ Q1,x16 ║   ║ Q3,x16 ║ │
 │  ╚═╤════╤═╝   ╚═╤════╤═╝ │               │  ╚═╤════╤═╝   ╚═╤════╤═╝ │
 │   GPU0 GPU1   GPU4 GPU5  │               │   GPU2 GPU3   GPU6 GPU7  │
 │           ╲╲ A-internal ╱╱│               │           ╲╲ B-internal ╱╱│
 │            ╲══fabric══╱   │               │            ╲══fabric══╱   │
 │  ▲ uplinks: 2 separate    │               │  ▲ uplinks: 2 separate    │
 │  ▲ Gen5 x16 to root 00 +  │               │  ▲ Gen5 x16 to root 20 +  │
 │  ▲ root 40 (different     │               │  ▲ root e0 (different     │
 │  ▲ CPU quadrants)         │               │  ▲ CPU quadrants)         │
 └────────────│─────│────────┘               └────────────│─────│────────┘
              │     │                                     │     │
        root 00   root 40                           root 20   root e0
        (Q0)      (Q2)                              (Q1)      (Q3)
              ╲─── CPU IOD inter-quadrant fabric ───╱
                       (Infinity Fabric)
```

So in this configuration:

* **Chip A** carries SW1 (Q0) + SW3 (Q2) — the two virtual switches share the chip's internal fabric. Traffic between SW1 and SW3 stays *inside chip A* and never enters the CPU.
* **Chip B** carries SW2 (Q1) + SW4 (Q3) — same story for traffic between SW2 and SW4.
* **Cross-chip traffic** (e.g. SW1 → SW2, SW1 → SW4, SW3 → SW2, SW3 → SW4) is the only kind that crosses the CPU IOD fabric and uses the upstream PCIe link of the source chip.

This is the c-payne 2-host / dual-VS configuration — the chip's silicon supports it natively because Switchtec PFX-G5 family is multi-host capable.

---

## Bandwidth measurements

All measurements with PyTorch peer-to-peer copy (`tensor.copy_()` with peer access enabled). 256 MB buffers, 100 iterations, mean over 1 Hz telemetry samples. PL = 600 W default on every card. Driver 595.58.03, kernel 6.18.24, IOMMU off.

### Single-pair P2P bandwidth (one direction, write)

Per-pair PCIe x16 Gen5 line-rate ceiling is ~63 GB/s; practical sustained is ~56 GB/s once protocol overhead is counted.

| From → To | Path | Single-pair WRITE | Notes |
|-----------|------|------------------:|-------|
| GPU 0 → GPU 1 | intra-VS / intra-chip A | 56.3 GB/s | stays inside SW1's downstream fabric |
| GPU 0 → GPU 4 | cross-VS / intra-chip A | 56.3 GB/s | SW1 ↔ SW3 inside chip A's internal fabric |
| GPU 0 → GPU 2 | cross-chip A → B (via CPU) | 56.3 GB/s | through SW1 uplink → root 00 → IF → root 20 → SW2 |
| GPU 0 → GPU 6 | cross-chip A → B (via CPU) | 56.3 GB/s | through CPU |

Single-pair on this rig is always uplink-saturated: 56 GB/s regardless of path. The path differences only show up when multiple concurrent flows are in play.

### 2-pair concurrent: same-source-VS → same-destination-VS

Both source GPUs of one VS sending one pair each to the destination VS:

|              | →SW1 (Q0)    | →SW2 (Q1)    | →SW3 (Q2)     | →SW4 (Q3)     |
|--------------|-------------:|-------------:|--------------:|--------------:|
| **SW1 (Q0)→** | —            | 56.4 GB/s    | **112.5 GB/s** ✨ | 56.4 GB/s    |
| **SW2 (Q1)→** | 56.4 GB/s    | —            | 56.4 GB/s     | **112.5 GB/s** ✨ |
| **SW3 (Q2)→** | **112.5 GB/s** ✨ | 56.4 GB/s | —             | 56.4 GB/s    |
| **SW4 (Q3)→** | 56.4 GB/s    | **112.5 GB/s** ✨ | 56.4 GB/s | —             |

**The "diagonal" pairs (highlighted) reach 112 GB/s = full per-pair x16 bandwidth on each pair simultaneously.** All other ("off-diagonal") cross-VS pairs saturate the source uplink at 56 GB/s.

The pattern matches the chip mapping perfectly:
* SW1 ↔ SW3 (both on chip A) → 112 GB/s, traffic stays inside chip A
* SW2 ↔ SW4 (both on chip B) → 112 GB/s, traffic stays inside chip B
* Any chip-A ↔ chip-B pairing → 56 GB/s, traffic crosses the CPU IOD fabric and is bottlenecked by the source chip's single x16 uplink

There is **no AMD IOD fabric magic** here — the diagonal speed-up is just intra-chip switching that bypasses the CPU.

### 2-pair concurrent: one source VS → two different destination VSs (collapse-trigger pattern)

| Source → Two destinations | chips touched | Aggregate WRITE | Status |
|---------------------------|--------------|----------------:|--------|
| SW1 → SW2 + SW3 | A→B + A→A    | 112.5 GB/s | one flow stays in chip A (intra) |
| SW1 → SW2 + SW4 | A→B + A→B    | 56.4 GB/s | both flows go through CPU (uplink limit) |
| SW1 → SW3 + SW4 | A→A + A→B    | 112.5 GB/s | one flow stays in chip A |
| SW2 → SW1 + SW3 | B→A + B→A    | 56.0 GB/s | both go through CPU |
| SW2 → SW1 + SW4 | B→A + B→B    | 112.5 GB/s | one stays in chip B |
| SW2 → SW3 + SW4 | B→A + B→B    | 112.5 GB/s | one stays in chip B |
| SW3 → SW1 + SW2 | A→A + A→B    | 112.5 GB/s | one stays in chip A |
| SW3 → SW1 + SW4 | A→A + A→B    | 112.5 GB/s | one stays in chip A |
| SW3 → SW2 + SW4 | A→B + A→B    | 51.4 GB/s | both go through CPU |
| SW4 → SW1 + SW2 | B→A + B→B    | 112.5 GB/s | one stays in chip B |
| SW4 → SW1 + SW3 | B→A + B→A    | 54.6 GB/s | both go through CPU |
| SW4 → SW2 + SW3 | B→B + B→A    | 112.5 GB/s | one stays in chip B |

Pattern: whenever the source VS dispatches to **at least one destination on the same physical chip**, that flow uses internal switching and aggregate bandwidth hits 112 GB/s (one path internal at 56, plus one path through CPU at 56). When both destinations are on the *other* physical chip, both flows fight for the source chip's single x16 uplink and aggregate is 56 GB/s.

The slight under-saturation visible in three rows (51.4, 54.6, 56.0 GB/s) is mild contention — **not** the dramatic 4× drop that the AMD posted-write collapse produces. See "Collapse implications" below.

### Other measurements

| Test | Result |
|------|--------|
| Bidirectional intra-VS (GPU 0 ↔ GPU 1) | 109.4 GB/s aggregate (full duplex utilisation) |
| 4 source switches → 1 destination switch (concurrent) | 75.4 GB/s aggregate (destination uplink limit) |
| All-to-all 8 GPU (56 ordered pairs concurrent) | **254.2 GB/s aggregate** (4.54 GB/s per pair) |

Compare the all-to-all number to the previous **16-GPU 4-switch (4 GPU/VS) topology** measured on the same rig: ~179 GB/s aggregate. The 2-VS-per-chip topology with 2 GPU/VS is **+42 % faster** in raw all-to-all throughput because much of the cross-VS traffic stays inside each physical chip and never burdens the CPU IOD.

---

## Collapse implications

The AMD I/O Die "posted-write collapse" trigger documented in [`collapse-report.md`](collapse-report.md) requires:

1. **Multiple source GPUs concurrently dispatching writes from the same source PCIe switch** (i.e., they share one uplink to one CPU root port), AND
2. Their destinations sit behind **two or more different CPU root complexes**.

In the previous 16-GPU 4-switch topology (4 GPU per switch, 4 separate roots), this trigger fired hard: bandwidth dropped from ~50 GB/s/pair to ~6 GB/s/pair (75 % collapse).

**On this 2-VS-per-chip topology, the trigger does not fire.** No measurement showed a >10 % drop from baseline; the worst observed was 51 / 56 = 92 % efficiency.

Two independent reasons protect this layout:

1. **Intra-chip cross-VS traffic never touches the CPU.** Half of the possible "cross-switch" routing (SW1↔SW3, SW2↔SW4) is internal to one physical c-payne chip. The CPU IOD never sees those TLPs, so its arbitration logic cannot misbehave on them. In our results these flows always reach full 56 GB/s per pair.

2. **Only 2 source GPUs per VS.** Even when traffic goes through the CPU (cross-chip), at most 2 source GPUs from one VS share that uplink concurrently. The collapse trigger empirically needs ≥3 concurrent source flows from one switch dispatching to different dst roots in order to break the IOD's posted-write arbitration. With 2 source GPUs we never reach that threshold.

The combination is the protective property:

```
Previous topology (16 GPU, 4 GPU/VS):
  4 src GPUs on SW1 → multiple dst roots → COLLAPSE (BW × 0.25)

This topology (8 GPU, 2 GPU/VS, chip A=SW1+SW3, chip B=SW2+SW4):
  2 src GPUs on SW1 → at most 2 dst roots
  AND if either dst is SW3 it's intra-chip (CPU IOD never sees it)
  → no collapse, only normal contention
```

So this layout is essentially **collapse-immune by construction** for any pattern that fits inside one physical chip, and **collapse-resistant** even for cross-chip patterns because it caps source-side concurrency at 2.

---

## Methodology and reproduction

All bandwidth numbers are produced by the same small PyTorch script:

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
    # warm
    for s, d in pairs:
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

The script issues `dst.copy_(src)` on a stream owned by the source GPU. With CUDA peer access enabled and ACS Request-Redirect disabled, the runtime selects the direct PCIe peer path — no host-RAM staging. Each `pairs` argument is a list of `(src_gpu, dst_gpu)` tuples; the function returns aggregate write bandwidth across all pairs.

Topology identification:

```bash
# Walk each GPU up the PCIe tree to its CPU root bus
for i in $(seq 0 7); do
  bus=$(nvidia-smi -i $i --query-gpu=gpu_bus_id --format=csv,noheader | sed 's/00000000://')
  root=$(readlink -f /sys/bus/pci/devices/0000:${bus,,}/../.. | grep -oE 'pci[0-9]+:[0-9a-f]+' | head -1)
  echo "GPU $i  bus $bus  -> $root"
done

# Switch upstream port summary
for sw in 01:00.0 21:00.0 41:00.0 e1:00.0; do
  lspci -vv -s $sw | grep -E "LnkCap:|LnkSta:" | head -2
done
```

That `lspci` output gives the line widths and Gen5 speeds of every upstream link. Walking up `/sys/bus/pci/devices/.../..` gives the root bus for each GPU. The 4-root layout is what makes this look like "4 switches" to the OS.

Detecting that two of the OS-visible "switches" are actually the same physical chip is *not* directly visible from `lspci`. The signature is in the bandwidth pattern: any two "switches" that produce 112 GB/s on a 2-pair test are sharing a physical chip, and any pair that produces 56 GB/s saturated is on different physical chips. The bandwidth matrix in this page reads off the pairing immediately:

* 112 GB/s pairs: SW1↔SW3 and SW2↔SW4 → these live on chip A and chip B respectively.
* 56 GB/s pairs: everything else → these are cross-chip and uplink-bound.

The full sweep script (drives all 12 source-VS / dest-VS combinations and the 12 collapse-trigger combinations) is checked into the repo at [`scripts/collapse_2gpu_full.py`](../scripts/collapse_2gpu_full.py). The ASUS-equivalent collapse pattern test (with WRITE/READ split) is at [`scripts/asus_replica.py`](../scripts/asus_replica.py).

Run them as:

```bash
python3 scripts/collapse_2gpu_full.py    # full bandwidth matrix + 1-src→2-dst patterns + all-to-all
python3 scripts/asus_replica.py          # ASUS-equivalent COLLAPSE / OK / 4-flow patterns, write+read
```

Both assume GPU index → switch mapping is `SW1: 0,1   SW2: 2,3   SW3: 4,5   SW4: 6,7`. Adjust the lists at the top of each script if your topology assigns differently.

---

## Operational guidance

* **For workloads dominated by GPU-to-GPU P2P** within this 8-GPU layout, prefer to keep traffic *inside one physical c-payne chip* whenever the algorithm allows. Specifically: a tensor-parallel group of 4 GPUs aligned to one chip (e.g., GPU 0,1,4,5 on chip A) gets all cross-VS traffic at 112 GB/s with no CPU involvement.
* **For NCCL ring all-reduce** spanning all 8 GPUs, the ring naturally alternates between intra-chip (cheap, 56 GB/s per hop) and cross-chip (also 56 GB/s per hop, but goes through CPU). Both hops are uplink-saturated so the ring throughput is similar to a flat 4-switch topology — but with no collapse risk.
* **For all-to-all** (MoE token routing), this topology delivers 254 GB/s aggregate vs 179 GB/s on the previous 16-GPU 4-switch topology. ~42 % more useful aggregate, half from the intra-chip routing not paying for the IOD round-trip.
* **Mapping a real workload to physical chips** matters: if you assign cards to hosts/processes ignoring the chip mapping, you can accidentally place every cross-process flow on the cross-chip path and miss the intra-chip win. Use the bandwidth matrix above (or measure with the script) to determine the chip assignment of each VS, then plan accordingly.
* **The collapse risk reappears** if you ever scale this to 4 GPU per VS (e.g., the previous 16-GPU layout): in that case 4 source GPUs can dispatch to 3 distinct destination roots concurrently and the IOD arbitration breaks. Sticking to 2 GPU per VS is a structural workaround.

---

## Variant: 2 active root complexes (both chips on same quadrant pairs)

A second wiring of the same two physical c-payne chips was tested on the same rig. The chip-to-VS mapping (i.e. which two VSs share a physical chip) is unchanged — bandwidth signature still cleanly shows chip A = SW1+SW3, chip B = SW2+SW4. What changed is **which CPU root ports the two upstreams of each chip land on**:

| Wiring | Chip A upstreams land on | Chip B upstreams land on | Active CPU quadrants |
|--------|---------------------------|---------------------------|----------------------|
| Original (this page's main results) | Q0 (SW1) + Q2 (SW3) | Q1 (SW2) + Q3 (SW4) | **all 4** (Q0, Q1, Q2, Q3) |
| Variant | Q0 (SW1) + Q0 (SW3) | Q3 (SW2) + Q3 (SW4) | **only 2** (Q0 and Q3) |

Concretely, in the variant: `00:01.1` and `00:03.1` are *both* root ports on the Q0 root complex, each carrying one upstream of chip A. `e0:01.1` and `e0:03.1` are both on Q3, each carrying one upstream of chip B. The other two quadrants (Q1, Q3) host nothing.

### Variant — measured results

Driven by the exact same scripts referenced above. Numbers are aggregate write bandwidth.

**2-pair single src → single dst** (4×4 matrix, GB/s aggregate):

| | →SW1 | →SW2 | →SW3 | →SW4 |
|---|---:|---:|---:|---:|
| SW1→ | — | 56 | **112** ✨ | 56 |
| SW2→ | 56 | — | 56 | **112** ✨ |
| SW3→ | **112** ✨ | 56 | — | 56 |
| SW4→ | 56 | **112** ✨ | 56 | — |

**Identical to the original wiring.** Diagonal pairs hit 112 GB/s, others saturate at 56. Chip mapping unchanged: bandwidth signature confirms chip A = SW1+SW3, chip B = SW2+SW4.

**1 src switch → 2 different dst switches** (12 patterns):

| Pattern | Original wiring | Variant | Δ |
|---------|----------------:|--------:|--:|
| SW1 → SW2+SW3 | 112.5 | 112.5 | 0 |
| SW1 → SW2+SW4 | 56.4 | 56.0 | −0.4 |
| SW1 → SW3+SW4 | 112.5 | 112.5 | 0 |
| SW2 → SW1+SW3 | 56.0 | 56.0 | 0 |
| SW2 → SW1+SW4 | 112.5 | 112.5 | 0 |
| SW2 → SW3+SW4 | 112.5 | 112.4 | 0 |
| SW3 → SW1+SW2 | 112.5 | 112.5 | 0 |
| SW3 → SW1+SW4 | 112.5 | 112.5 | 0 |
| SW3 → SW2+SW4 | **51.4** | **56.4** | **+5** |
| SW4 → SW1+SW2 | 112.5 | 112.5 | 0 |
| SW4 → SW1+SW3 | **54.6** | **56.4** | **+2** |
| SW4 → SW2+SW3 | 112.5 | 112.5 | 0 |

The two cells where the original wiring sat just below the saturation line (51.4 and 54.6 GB/s) are now at full saturation (~56 GB/s). All other cells unchanged.

**Aggregate / stress tests**:

| Test | Original wiring | Variant | Δ |
|------|----------------:|--------:|--:|
| 4 src switches → 1 dst (SW1) | 75.4 GB/s | 75.2 GB/s | 0 |
| **All-to-all 8 GPU (56 pairs)** | **254.2 GB/s** | **225.6 GB/s** | **−28.6 GB/s (−11 %)** |
| All-to-all per-pair | 4.54 GB/s | 4.03 GB/s | −0.51 |

**ASUS-equivalent collapse-trigger patterns** (write):

| Pattern | Original wiring | Variant |
|---------|----------------:|--------:|
| `(0,2)+(1,6)` (SW1→SW2+SW4, "ASUS COLLAPSE") | 54.0 | 56.0 |
| `(0,2)+(4,6)` (different src VS → SW2+SW4, "ASUS OK") | 112.0 | 112.5 |
| `(0,2)+(1,3)` (1 dst root) | 56.4 | 56.4 |
| `(0,4)+(1,6)` (SW1→SW3+SW4) | 111.9 | 111.2 |
| `(0,2)+(0,6)+(1,3)+(1,7)` (4-flow ASUS pattern) | 56.2 | 56.2 |

**No collapse on either wiring.** The ASUS pattern that catastrophically collapses on Broadcom + Turin (~2.7 GB/s) saturates the source-uplink x16 at ~56 GB/s on Microchip + TR Pro in *both* wirings.

### What changed and what didn't

* **Chip mapping is unchanged** — both wirings produce the same bandwidth signature (SW1↔SW3 fast, SW2↔SW4 fast, others saturated). The two physical c-payne packages still pair the same VSs internally.
* **Cross-quadrant fabric utilisation moved**. Original wiring spreads the four upstream ports across all four IOD quadrants (Q0, Q1, Q2, Q3) so each quadrant carries traffic for one switch. Variant wiring concentrates all chip-A traffic on Q0 and all chip-B traffic on Q3 — only one cross-quadrant IF link (Q0 ↔ Q3) carries cross-chip traffic, and Q1/Q2 are idle.
* **All-to-all loses ~11 %**. The eight GPUs now contend for two quadrants' worth of memory and IF capacity instead of four. Spreading the GPUs across four quadrants in the original wiring gives more aggregate bandwidth when the workload is fan-out / fan-in heavy.
* **The per-flow saturation values cleaned up** (the 51 and 55 GB/s cells became 56). The variant has fewer concurrent cross-quadrant IF transitions to schedule, so the IOD reaches per-flow saturation more cleanly.
* **Collapse trigger behaviour is identical**. Neither wiring exposes the AMD IOD posted-write collapse; both saturate at uplink limit on the ASUS-trigger patterns.

### Practical take-away on this variant

* If you have control over which root ports the chips' upstreams land on, the **original 4-quadrant wiring is preferable** for all-to-all-style workloads (~11 % more aggregate fabric throughput). For most tensor-parallel / ring-allreduce workloads the difference is negligible because per-uplink saturation is identical.
* The variant gives a useful scientific data point: **moving from 4 to 2 active root complexes does not introduce the collapse**, on this Microchip + TR Pro silicon. (On Broadcom + Turin, the same two-root-complex pattern would collapse — see [`asus-esc8000a-e13p-broadcom-switches.md`](asus-esc8000a-e13p-broadcom-switches.md).)
* Topology detection: the original wiring shows GPUs distributed across `pci0000:00`, `pci0000:20`, `pci0000:40`, `pci0000:e0`. The variant shows GPUs only on `pci0000:00` and `pci0000:e0` (two GPUs share a root bus when their chip's other upstream lands on a different port of the same root complex).

---

## Variant 3 — chips concentrated on one quadrant each ("crossed" wiring)

A third wiring of the same two physical c-payne chips. Both physical chips still partition into 2 VS each, still all 8 GPUs, but the chip-to-quadrant assignment is now different:

| Wiring | chip A upstreams land on | chip B upstreams land on | Active quadrants | Chip mapping (from BW signature) |
|--------|---------------------------|---------------------------|------------------|----------------------------------|
| Original (variant 1) | Q0 (SW1) + Q2 (SW3) | Q1 (SW2) + Q3 (SW4) | all 4 | A=SW1+SW3, B=SW2+SW4 |
| Variant 2 | Q0 (SW1) + Q3 (SW3) — split | Q0 (SW2) + Q3 (SW4) — split | 2 (Q0, Q3) | A=SW1+SW3, B=SW2+SW4 |
| **Variant 3** | **Q0 (SW1) + Q0 (SW2) — both Q0** | **Q3 (SW3) + Q3 (SW4) — both Q3** | **2 (Q0, Q3)** | **A=SW1+SW2, B=SW3+SW4** |

In variant 3, each physical chip has **both uplinks on the same CPU quadrant**. Chip A's two virtual switches both land on root 00 (different root ports of Q0); chip B's two on root e0 (Q3).

The bandwidth signature confirms the new chip mapping: SW1↔SW2 is now the diagonal "fast" pair (intra-chip A) and SW3↔SW4 is the second fast pair (intra-chip B). Previously these were SW1↔SW3 and SW2↔SW4.

### Variant 3 — measured results

**2-pair single src → single dst** (4×4, GB/s aggregate):

| | →SW1 | →SW2 | →SW3 | →SW4 |
|---|---:|---:|---:|---:|
| SW1→ | — | **112.5** ✨ | 56.4 | 56.4 |
| SW2→ | **112.5** ✨ | — | 56.4 | 56.4 |
| SW3→ | 56.4 | 56.4 | — | **112.5** ✨ |
| SW4→ | 56.4 | 56.4 | **112.5** ✨ | — |

The intra-chip pair has *moved*: now SW1+SW2 and SW3+SW4 give the doubled bandwidth (vs SW1+SW3 / SW2+SW4 in variants 1 and 2). Same chip-internal-fabric phenomenon, just paired differently.

**1 src → 2 dst** (12 patterns):

| Pattern | V1 | V2 | V3 | Notes |
|---------|---:|---:|---:|-------|
| SW1 → SW2+SW3 | 112.5 | 112.5 | 112.5 | one path intra-chip A in V3 |
| SW1 → SW2+SW4 | 56.4 | 56.0 | **112.5** | one path intra-chip A in V3 |
| SW1 → SW3+SW4 | 112.5 | 112.5 | **56.4** | both cross-chip in V3 |
| SW2 → SW1+SW3 | 56.0 | 56.0 | **112.5** | one intra in V3 |
| SW2 → SW1+SW4 | 112.5 | 112.5 | 112.5 | one intra in V3 |
| SW2 → SW3+SW4 | 112.5 | 112.4 | **56.4** | both cross-chip |
| SW3 → SW1+SW2 | 112.5 | 112.5 | **56.4** | both cross-chip |
| SW3 → SW1+SW4 | 112.5 | 112.5 | 112.5 | one intra-chip B |
| SW3 → SW2+SW4 | 51.4 | 56.4 | 112.5 | one intra in V3 |
| SW4 → SW1+SW2 | 112.5 | 112.5 | **56.4** | both cross-chip |
| SW4 → SW1+SW3 | 54.6 | 56.4 | 112.5 | one intra in V3 |
| SW4 → SW2+SW3 | 112.5 | 112.5 | 112.5 | one intra-chip B |

The "fast" rows in V3 (112.5) are exactly those where one of the two destinations is the same-chip neighbour: that flow stays inside the chip and the other flow gets the full uplink to itself.

**Aggregate / stress tests**:

| Test | V1 (4-quad) | V2 (split) | **V3 (concentrated)** |
|------|------------:|-----------:|----------------------:|
| 4 src switches → 1 dst (SW1) | 75.4 | 75.2 | **112.0** |
| **All-to-all 8 GPU (56 pairs)** | **254.2** | 225.6 | **213.1** |
| All-to-all per-pair | 4.54 | 4.03 | 3.80 |

* V3 has the worst all-to-all (213 GB/s, ~16 % below V1).
* V3 has the **best** 4-src-to-1-dst aggregate (112 GB/s vs 75 in V1/V2). When all four source switches hit a single destination switch, two of those flows are intra-chip (SW2→SW1 stays in chip A) so they don't fight for any uplink.

**ASUS-equivalent collapse-trigger patterns** (write):

| Pattern | V1 | V2 | **V3** | Notes |
|---------|---:|---:|-------:|-------|
| `(0,2)+(1,6)` (SW1→SW2+SW4) | 54.0 | 56.0 | **112.5** | SW1→SW2 is now intra-chip in V3 |
| `(0,2)+(4,6)` (different src VS → SW2+SW4) | 112.0 | 112.5 | 112.5 | |
| `(0,2)+(1,3)` (SW1→SW2 only, 1 dst root) | 56.4 | 56.4 | **112.5** | both intra-chip in V3 |
| `(0,4)+(1,6)` (SW1→SW3+SW4) | 111.9 | 111.2 | **56.4** | both cross-chip in V3 |
| `(0,2)+(0,6)+(1,3)+(1,7)` (4-flow) | 56.2 | 56.2 | **75.9** | mix of intra+cross |

**Still no collapse on any pattern**, even at the ASUS-trigger conditions. The chip mapping change just shifts which pairs land in the "fast intra-chip" lane vs the "slow cross-chip" lane.

### Why V3 has different numbers

* **All-to-all drops further**: with both chips concentrated on one quadrant each, *every* cross-chip flow has to traverse the same Q0 ↔ Q3 inter-quadrant fabric link. In V2, chip-A traffic had two paths into Q3 (one intra-chip via the chip A's own Q3 upstream, one cross-quadrant via Q0↔Q3 IF) so contention was lower. V3 has no chip-A path to Q3 at all — every chip A → chip B flow crosses the same IF.
* **4-src → 1-dst SW1 jumps to 112 GB/s**: of the four source GPUs (one per switch), the two on chip A (SW2's two GPUs) reach SW1 entirely through chip A's internal fabric. Only the two flows from chip B (SW3, SW4) actually hit SW1's upstream port. So SW1's uplink only carries half the traffic.
* **The collapse trigger truth-table is structurally different**: with chip A entirely on Q0 and chip B entirely on Q3, **a 1-src → 2-dst pattern that hits "two different destination roots" is structurally impossible without including an intra-chip path** (because to hit two different roots from one chip, one destination must be on the chip's own quadrant — which is intra-chip — and the other on the opposite quadrant — which is cross-chip). The intra-chip flow bypasses CPU IOD entirely, so the "trigger" pattern never has both flows hitting CPU arbitration.

### Picking between the variants

| Workload pattern | Best variant |
|------------------|--------------|
| All-to-all (MoE token routing, NCCL alltoall) | **V1** (4-quadrant): 254 GB/s aggregate |
| Reduce-to-one-GPU (NCCL reduce) | **V3** (concentrated): 112 GB/s into a single switch |
| Tensor-parallel within one chip's GPUs | any (chip-internal fabric is the same in all variants) |
| Tensor-parallel across chips | V1 marginally best, V2/V3 within ±5 % |
| Avoiding collapse | all three variants on TR Pro + Microchip — none collapses on AMD-host with this 2-VS-per-chip layout |

For our specific chassis on a TR Pro 7955WX, the **default recommendation is V1 (4-quadrant, original wiring)**: it's the only variant with no aggregate-throughput penalty, and it gives the cleanest mapping for monitoring (each VS on its own root makes lspci traceable). V2 and V3 are interesting science but trade ~10–16 % aggregate fabric for nothing structural.

---

## Cross-references

* [`collapse-report.md`](collapse-report.md) — standalone report on the AMD IOD posted-write collapse, the bug this layout sidesteps.
* [`pcie-posted-write-collapse.md`](pcie-posted-write-collapse.md) — long-form history of the collapse investigation across multiple platforms.
* [`wrx90-cpayne-16gpu-4switch.md`](wrx90-cpayne-16gpu-4switch.md) — the previous 16-GPU 4-switch (4 GPU/VS) layout where the collapse fired hard.
* [`wrx90-cpayne-8gpu-root-topology-comparison.md`](wrx90-cpayne-8gpu-root-topology-comparison.md) — earlier comparison of two 8-GPU groupings (separate roots vs shared root) on a previous topology.
