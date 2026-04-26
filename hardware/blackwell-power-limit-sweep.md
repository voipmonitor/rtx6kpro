# RTX PRO 6000 Blackwell — Power-Limit Sweep: Workstation vs Server vs Max-Q

Power-limit sweep on three variants of the same Blackwell silicon (NVIDIA RTX PRO 6000 generation) running `gpu_burn -tc` (tensor-core compare burn). Power limit stepped in 50 W intervals from each card's minimum upward, ~10 s ramp + ~30 s telemetry at 1 Hz, averages over the steady-state window.

Two Workstation cards and two Server cards on the same rig were each swept independently to verify reproducibility. Community-contributed datapoints from a third Workstation card (10 W resolution) and a Max-Q variant were also incorporated.

Originally requested by Luke (`lukeRole`, Quant Creators) for a clock-vs-power-limit characterisation across SKUs.

> **Cooling caveat:** the test rig is a workstation tower, *not* a server-grade chassis. The Server-edition card is designed for forced-air server enclosures; in this tower it cannot dissipate above ~500 W and starts thermal-throttling at ~85 °C. The 550 W and 600 W Server datapoints are **clamped by chassis cooling, not by the card itself**, and have been excluded from the cross-SKU comparisons and conclusions below. In a properly cooled server chassis the Server SKU would scale further, possibly to its 600 W TDP.

---

## System under test

| Item | Value |
|------|-------|
| Host | ASRock WRX90 WS EVO + AMD Threadripper Pro 7955WX |
| Driver | 595.58.03 |
| CUDA Driver API | 13.2 |
| Kernel | Linux 6.18.24 (Ubuntu 24.04) |
| GPUs in chassis | 16× RTX PRO 6000 Blackwell (8× WS + 8× Server, all 96 GB) |
| Chassis | Workstation tower, stock fans (NOT a server chassis) |
| Burn workload | `gpu_burn -i N -tc <duration>` (tensor-core compare) |
| Sampling | `nvidia-smi --query-gpu=...` 1 Hz, mean over last 25 of 30 s |
| Cards swept locally | GPU 3 (WS, SW1), GPU 14 (WS, SW4), GPU 5 (Server, SW2), GPU 8 (Server, SW3) |
| Community contributions | 1× WS card (10 W resolution, 250-360 W range), 1× Max-Q card (4 datapoints, 250-325 W) |
| Persistence mode | Enabled |

Two WS GPUs on different PCIe switches in our rig were swept to verify per-card reproducibility. Two Server GPUs likewise to confirm the throttle behaviour is the SKU's, not a per-card defect.

---

## Variant overview

| Variant | Min PL | Max PL | TDP rating | Cooling design | pstate at burn | Memory clock |
|---------|---:|---:|---:|---|:-:|---:|
| **Workstation Edition** | 150 W | 600 W | 600 W | Active blower, 1-slot | **P1** | 13 365 MHz |
| **Server Edition** | 300 W | 600 W | 600 W | Passive (relies on chassis CFM) | **P0** | 12 481 MHz |
| **Max-Q variant** | ~250 W | ~325 W tested | ~300 W | Active dual-slot | **P1** | 15 865 MHz |

Same silicon but **three distinct boost-table / pstate / memory-clock profiles** baked into the SKU. Memory clock varies between variants and is invariant within each variant.

---

## Workstation Edition — two cards

### GPU 3 (WS, on switch SW1)

| PL (W) | Actual (W) | pstate | GFX/SM (MHz) | Mem (MHz) | Temp (°C) | Gflop/s |
|---:|---:|:-:|---:|---:|---:|---:|
| 150 | 150.0 | P1 |  606 | 13365 | 31.3 |  37 691 |
| 200 | 200.0 | P1 |  657 | 13365 | 37.3 |  59 045 |
| 250 | 250.0 | P1 |  858 | 13365 | 42.0 |  76 315 |
| 300 | 300.0 | P1 | 1031 | 13365 | 46.3 |  91 076 |
| 350 | 342.6 | P1 | 1274 | 13365 | 50.4 | 106 441 |
| 400 | 400.0 | P1 | 1375 | 13365 | 54.1 | 120 203 |
| 450 | 441.7 | P1 | 1577 | 13365 | 57.2 | 132 732 |
| 500 | 499.6 | P1 | 1708 | 13365 | 60.4 | 144 235 |
| 550 | 550.3 | P1 | 1804 | 13365 | 64.3 | 155 836 |
| 600 | 596.3 | P1 | 1948 | 13365 | 67.4 | **165 682** |

### GPU 14 (WS, on switch SW4 — independent verification)

| PL (W) | Actual (W) | pstate | GFX/SM (MHz) | Mem (MHz) | Temp (°C) | Gflop/s |
|---:|---:|:-:|---:|---:|---:|---:|
| 150 | 150.0 | P1 |  626 | 13365 | 30.4 |  37 490 |
| 200 | 189.8 | P1 |  735 | 13365 | 36.8 |  59 454 |
| 250 | 250.0 | P1 |  840 | 13365 | 41.4 |  74 764 |
| 300 | 300.0 | P1 | 1007 | 13365 | 46.5 |  89 344 |
| 350 | 350.0 | P1 | 1219 | 13365 | 50.9 | 104 451 |
| 400 | 399.9 | P1 | 1355 | 13365 | 54.8 | 117 469 |
| 450 | 450.1 | P1 | 1470 | 13365 | 58.4 | 129 521 |
| 500 | 500.2 | P1 | 1626 | 13365 | 61.9 | 140 619 |
| 550 | 549.8 | P1 | 1766 | 13365 | 65.2 | 151 328 |
| 600 | 598.5 | P1 | 1872 | 13365 | 68.6 | **161 408** |

### Cross-validation: GPU 3 vs GPU 14 vs community

WS Gflop/s comparison (the headline metric):

| PL (W) | GPU 3 | GPU 14 | Community card |
|---:|---:|---:|:-:|
| 250 |  76 315 |  74 764 | (only Gflops not reported) |
| 300 |  91 076 |  89 344 | — |
| 350 | 106 441 | 104 451 | — |
| 400 | 120 203 | 117 469 | — |
| 450 | 132 732 | 129 521 | — |
| 500 | 144 235 | 140 619 | — |
| 550 | 155 836 | 151 328 | — |
| 600 | 165 682 | 161 408 | — |

GPU 14 is consistently **~2 % below GPU 3** across the entire sweep — a typical silicon-binning offset. Both produce the same shape of curve, and GPU 3 happens to have undershot the configured PL at 350 W and 450 W (boost-table step) while GPU 14 hit the configured limits cleanly at all stops.

WS clock comparison vs the community-contributed datapoints (10 W resolution, 250-360 W range, also pstate P1, also memory @ 13 365 MHz):

| PL (W) | GPU 3 (MHz) | GPU 14 (MHz) | Community (MHz) |
|---:|---:|---:|---:|
| 250 |  858 |  840 |  885 |
| 290 | — | — | 1012 |
| 300 | 1031 | 1007 | 1050 |
| 310 | — | — | 1095 |
| 320 | — | — | 1132 |
| 325 | — | — | 1140 |
| 330 | — | — | 1162 |
| 340 | — | — | 1192 |
| 350 | 1274 (@343W) | 1219 | 1215 |
| 360 | — | — | 1252 |

The community card's curve falls **right between** GPU 3 and GPU 14. Across three independent WS cards the binning spread is about ±5 % — far smaller than the difference between SKUs. **WS-edition behaviour is reproducible.** The community 10 W resolution data also confirms the curve is smooth (no hidden boost-table steps inside the 50 W intervals our local sweep used).

---

## Server Edition — two cards (clean range 300 → 500 W)

| PL (W) | GPU 5 actual / clock / temp / Gflop/s | GPU 8 actual / clock / temp / Gflop/s |
|---:|:--|:--|
| 300 | 300.0 W / 1438 MHz / 45.7 °C / 125 076 | 300.1 W / 1395 MHz / 45.5 °C / 123 371 |
| 350 | 349.8 W / 1599 MHz / 57.9 °C / 135 845 | 343.3 W / 1611 MHz / 57.4 °C / 133 575 |
| 400 | 400.2 W / 1749 MHz / 66.9 °C / 144 326 | 400.2 W / 1699 MHz / 66.0 °C / 143 211 |
| 450 | 450.5 W / 1802 MHz / 73.7 °C / 152 369 | 442.7 W / 1896 MHz / 73.4 °C / 150 458 |
| 500 | 492.6 W / 1909 MHz / 80.7 °C / **158 331** | 499.0 W / 1880 MHz / 80.4 °C / **156 755** |

* Server cards run at **pstate P0** (vs WS P1) — distinct boost-table behaviour.
* **Memory clock pinned at 12 481 MHz** — *lower* than WS's 13 365 MHz.
* Two-card cross-validation: per-pair difference between GPU 5 and GPU 8 is ~1 %, matching normal binning noise.
* PL settings ≥ 550 W are **excluded** from the cross-SKU comparison because the chassis cooling tops out: temperature pegs at 84-85 °C and the firmware throttles clock to keep it there. With proper server-grade airflow the curve would extrapolate forward, not regress.

---

## Max-Q variant — community datapoints

Four datapoints from a community contributor's RTX PRO 6000 Blackwell **Max-Q** desktop variant:

| PL (W) | pstate | GFX/SM (MHz) | Mem (MHz) | Power (W) | Temp (°C) |
|---:|:-:|---:|---:|---:|---:|
| 250 | P1 | 1252 | 15865 | 250.01 | 69 |
| 275 | P1 | 1365 | 15865 | 275.03 | 69 |
| 300 | P1 | 1417 | 15865 | 300.02 | 74 |
| 325 | P1 | 1500 | 15865 | 325.01 | 75 |

Notable:
* **Memory clock 15 865 MHz** — substantially higher than both WS (13 365) and Server (12 481).
* Operates in **pstate P1** like the WS card.
* Tested range 250 → 325 W only (Max-Q variant tops out near 325 W TDP, hence the name).
* At 300 W the Max-Q clocks at 1417 MHz — much closer to Server (1395-1438 MHz) than to WS (1007-1031 MHz) at the same wattage. Same silicon, but Max-Q uses a Server-style aggressive V/F curve combined with the highest memory clock of the three SKUs.

Throughput numbers were not reported by the Max-Q contributor.

---

## Side-by-side comparison (clean data only)

### GFX/SM clock vs power limit

| PL (W) | WS mean (GPU 3 + 14) | WS community | Server mean (GPU 5 + 8) | Max-Q (community) |
|---:|---:|---:|---:|---:|
| 150 |  616 | — | — | — |
| 200 |  696 | — | — | — |
| 250 |  849 |  885 | — | 1252 |
| 275 | — | — | — | 1365 |
| 300 | 1019 | 1050 | 1417 | 1417 |
| 325 | — | 1140 | — | 1500 |
| 350 | 1247 | 1215 | 1605 | — |
| 400 | 1365 | — | 1724 | — |
| 450 | 1524 | — | 1849 | — |
| 500 | 1667 | — | 1895 | — |
| 550 | 1785 | — | (cooling limit) | — |
| 600 | 1910 | — | (cooling limit) | — |

### Tensor-core throughput at each power-limit (Gflop/s)

| PL (W) | WS mean | Server mean | Δ Server vs WS |
|---:|---:|---:|---:|
| 300 |  90 210 | 124 224 | **+38 %** |
| 350 | 105 446 | 134 710 | +28 % |
| 400 | 118 836 | 143 769 | +21 % |
| 450 | 131 127 | 151 414 | +15 % |
| 500 | 142 427 | **157 543** | **+11 %** |
| 550 | 153 582 | (cooling limit) | — |
| 600 | **163 545** | (cooling limit) | — |

* **Server card delivers more Gflop/s than WS at every cleanly-measured PL setting** (300-500 W), gap narrowing from +38 % at 300 W to +11 % at 500 W.
* **Best WS Gflop/s (clean): 163 545 @ 600 W** — full thermal headroom available with active blower.
* **Best Server Gflop/s (clean): 157 543 @ 500 W** — extrapolating, the curve is still climbing; with server cooling it would likely cross 165-170 k at 600 W.
* In *this* (workstation-tower) chassis, the two SKUs reach a near-tie at 500-600 W. In a server chassis the Server SKU would clearly win at high PL too.

### Efficiency (Gflop/s per Watt)

| PL (W) | WS Gflop/s/W | Server Gflop/s/W | Δ |
|---:|---:|---:|---:|
| 150 | 251 | — | — |
| 200 | 304 | — | — |
| 250 | 304 | — | — |
| 300 | 301 | **414** | **+38 %** |
| 350 | 305 | 388 | +27 % |
| 400 | 297 | 359 | +21 % |
| 450 | 295 | 339 | +15 % |
| 500 | 285 | **314** | +10 % |
| 550 | 280 | (clamped) | — |
| 600 | 274 | (clamped) | — |

The **Server card is dramatically more efficient at low/mid power**: at its 300 W floor it delivers **414 Gflop/s/W** — 38 % more than the WS card at the same wattage. This is its single best operating point.

WS efficiency is **flat in the 200-350 W range** at ~300-305 Gflop/s/W and drops slowly above that. The "default 600 W" setting on a WS card costs 11 % efficiency vs running it at ~250-350 W.

---

## ASCII charts

### GFX/SM clock vs power limit (mean of 2 cards each variant)

```
   GFX MHz
    2000 │                                                       *  WS @ 600W
    1900 │                                       SS  S      *
    1700 │                              S       *
    1500 │                    S             *
    1400 │  M    M    M  S         *
    1200 │  M    M       *  
    1000 │           *                       Legend:
     800 │       *                            *  WS  (GPU 3 + 14 mean)
     600 │  *                                 S  Server (GPU 5 + 8 mean)
                                              M  Max-Q (community)
       0 └───────────────────────────────────────────────────
         150  200  250  300  350  400  450  500  550  600   PL (W)
```

### Tensor-core Gflop/s vs power limit

```
   Gflop/s
   170k │                                                       *  WS @ 600W
   160k │                                                  *
   150k │                              S    *      ← Server peak measurable
   140k │                       S       *
   130k │                     S    *  
   120k │              S      *      
   100k │      S      *               Server +38% at 300W
    80k │       *
    60k │   *
    40k │  *
       └───────────────────────────────────────────────────
         150  200  250  300  350  400  450  500  550  600   PL (W)
```

### Efficiency (Gflop/s per Watt) vs power limit

```
   Gflop/s/W
       420 │   S
       400 │            
       380 │      S            Server peak efficiency
       360 │        S
       340 │            S
       320 │              S
       310 │  *  *  *  *  *  S
       290 │              *  *
       280 │                    *  *  *  *
       260 │
           └────────────────────────────────────────────
             150 200 250 300 350 400 450 500 550 600   PL (W)
```

---

## Conclusions (cooling-independent)

1. **Same silicon, three very different SKUs.** Memory clock (WS 13 365 / Server 12 481 / Max-Q 15 865 MHz), pstate (WS P1, Server P0, Max-Q P1), and V/F curve all differ by SKU. None of these are user-tunable — they are fixed in firmware.

2. **WS-edition behaviour is highly reproducible.** Two WS cards on this rig agreed to ~2 %, and a third WS card from the community fell within the same ±5 % silicon-binning band. The clock-vs-power curve is smooth at 10 W resolution.

3. **At any clean-measured power limit ≤ 500 W, the Server SKU delivers more Gflop/s than the WS SKU.** The gap is +38 % at 300 W and tapers to +11 % at 500 W. This is independent of cooling — it's a boost-table choice baked into the SKUs.

4. **Server SKU has dramatically better Gflop/s per Watt.** At 300 W: 414 vs WS's 301 (+38 %). For energy-constrained inference, capping Server cards at ~300 W is by far the most efficient operating point of any setting on any of the variants tested.

5. **Memory clock is invariant within each variant** regardless of power limit. Power limit gates only GFX/SM clock — memory always runs at its variant's fixed clock. This will affect memory-bound LLM kernels (KV-cache reads, attention) very differently from the compute-bound `gpu_burn` workload here:
   * WS at 13.4 GHz vs Server at 12.5 GHz vs Max-Q at 15.9 GHz on the SAME silicon
   * For decode-heavy LLM serving, **Max-Q's higher memory clock could matter more than its lower GFX clock**, but we have no Gflop/s data from the community contributor to confirm this.

6. **WS-edition card has actual headroom at 600 W.** With its own active blower it stays at 67 °C drawing 596 W and produces 165 k Gflop/s with no throttle. WS at 600 W is the highest sustained throughput we measured on any card.

7. **Server-edition card is cooling-limited in this chassis at 500 W.** The card itself isn't the bottleneck — the workstation-tower chassis can't move enough air for the Server card's passive design. With proper server-chassis CFM the Server card almost certainly continues scaling past 500 W and would beat WS at 600 W as well. We can't measure that here.

8. **Max-Q at 300 W ≈ Server at 300 W on clock**, but Max-Q has the highest memory clock of all three. Hard to know what the equivalent throughput is without Gflop/s numbers.

---

## Operational guidance

| Constraint | Best variant | Best power-limit setting |
|------------|--------------|--------------------------|
| Energy-efficient inference at moderate density | **Server**, capped at 300 W | 300 W = 414 Gflop/s/W (best perf/W of any SKU at any setting) |
| Max raw Gflop/s in a workstation chassis | **Workstation** | 600 W default, 165 k Gflop/s |
| Max raw Gflop/s in a server chassis | **Server** | 600 W (in proper CFM, would exceed WS) |
| Quiet workstation, modest perf | Workstation | 300-350 W, 47-51 °C, low fan noise |
| Memory-bandwidth-bound inference (decode) | Possibly **Max-Q** if available | ~325 W, highest mem clock — needs validation |

**Server SKU + workstation chassis (this rig):** cap power limit at 500 W (`nvidia-smi -pl 500`) to stop wasting the 100 W of "phantom" budget that would just produce more heat without more throughput.

**WS SKU at 350 W:** very close to peak efficiency, almost half the power of 600 W default for ~64 % of throughput. Good operating point for energy-budget-constrained deployments that need flat per-card power.

---

## Raw data files

* [`data/blackwell-ws-pl-sweep-summary.csv`](../data/blackwell-ws-pl-sweep-summary.csv) — WS GPU 3 steady-state averages
* [`data/blackwell-ws-pl-sweep.csv`](../data/blackwell-ws-pl-sweep.csv) — WS GPU 3 1 Hz telemetry
* [`data/blackwell-ws-pl-sweep-gpu14-summary.csv`](../data/blackwell-ws-pl-sweep-gpu14-summary.csv) — WS GPU 14 verification, summary
* [`data/blackwell-ws-pl-sweep-gpu14.csv`](../data/blackwell-ws-pl-sweep-gpu14.csv) — WS GPU 14 verification, full 1 Hz telemetry
* [`data/blackwell-server-pl-sweep-summary.csv`](../data/blackwell-server-pl-sweep-summary.csv) — Server GPU 8 summary
* [`data/blackwell-server-pl-sweep.csv`](../data/blackwell-server-pl-sweep.csv) — Server GPU 8 1 Hz telemetry (550 W & 600 W rows clamped by chassis cooling)

Reproduction script: `/tmp/luke_pl_sweep.sh` (WS, 150-600 W) and `/tmp/luke_pl_sweep_server.sh` (Server, 300-600 W) on the test rig. Each just sweeps PL with `nvidia-smi -pl` and runs `gpu_burn -i N -tc <duration>` while sampling `nvidia-smi --query-gpu=...` at 1 Hz.

---

## What this does NOT measure

* **Real LLM inference throughput** — `gpu_burn -tc` is a synthetic compute-bound stress. Memory-bound kernels (attention, KV cache) will respond very differently to power limit since memory clock is invariant.
* **Server SKU above 500 W in proper cooling** — would require a server chassis to obtain.
* **Max-Q throughput numbers** — community contributor reported only telemetry, not Gflop/s.
* **Sustained multi-hour behaviour** — each level was 30 s steady state. Long sustained loads may shift thermal equilibrium upward.
* **Different chassis / airflow** — these results are specific to this 4 U workstation tower with stock fans.

---

## Addendum — Same workload simultaneously on WS and Server (no PL cap)

The previous sweep capped each card at fixed power limits. A different question: with PL at the default 600 W on both cards, run **the same** workload simultaneously on a WS GPU (14) and a Server GPU (8) and see how each card chooses to clock. Hypothesis under test: "if both cards naturally draw ~300 W from a moderate workload, do they clock the same?" — i.e. is the SKU difference purely a function of power-budget headroom, or is the boost decision itself SKU-specific?

Five workloads, 25 s sustained, both GPUs at PL=600 W default, single host process running both as concurrent threads.

### Results (steady-state mid-run sample)

| Workload | WS (GPU 14): clock / power / iter/s | Server (GPU 8): clock / power / iter/s |
|----------|-------------------------------------|----------------------------------------|
| BF16 8192² matmul (max compute) | **1762 MHz** / 600 W / 292 it/s | **2062 MHz** / 600 W / **341 it/s** |
| BF16 2048² matmul (medium compute) | **2497 MHz** / 600 W / 3318 it/s | **2370 MHz** / 542 W / 3138 it/s |
| BF16 1024² matmul (light compute) | **2805 MHz** / 600 W / 75 125 it/s | **2415 MHz** / **390 W** / 64 949 it/s |
| FP32 1024² matmul | **2227 MHz** / 600 W / 21 536 it/s | **2137 MHz** / 499 W (throttle 85 °C) / 20 966 it/s |
| 4 GB memcopy (memory-bound) | **2842 MHz** / **401 W** / 682 it/s | **2422 MHz** / **365 W** / 680 it/s |

### What this shows

* **WS card aggressively uses all 600 W** even on light workloads. BF16 1024² matmul saturates a WS card at the full 600 W power limit and 2805 MHz; the same kernel on a Server card draws only **390 W** at 2415 MHz. Same workload, ≈1.54 × the power on WS.

* **Server card is fundamentally more conservative.** With no PL cap it still chooses to clock and draw less than WS for any non-saturating workload. This is a firmware/SKU choice, not a temperature decision (Server card was 70 °C at 390 W, plenty of thermal headroom).

* **At the closest-to-equal power point (memcopy)** — WS naturally draws 401 W, Server 365 W — **WS still clocks 17 % higher** (2842 vs 2422 MHz). They are not equivalent at equal power. The boost decision is itself SKU-specific.

* **Throughput is workload-dependent:**
  * Compute-saturating workloads: Server's higher boost-curve wins (BF16 8192² → +17 % iter/s on Server at the same 600 W).
  * Light/memory-bound workloads: WS's "burn the watts" approach wins (BF16 1024² → +16 % iter/s on WS, but at 1.54 × the power).
  * Memory-bound (memcopy): both deliver essentially identical iter/s (682 vs 680) — the workload doesn't care about boost decisions because the memory clock is invariant within each SKU.

### Refuting the hypothesis

The original hypothesis was: "if we let both cards run at their natural draw without PL cap and pick a workload that produces ~300 W, will the cores clock similarly?"

**No.** Even at *closer-to-equal* power draws (401 W WS / 365 W Server during memcopy), the WS card still clocks ~17 % higher than the Server card. The two SKUs do not converge to the same clock at the same power; they have different V/F decisions baked into firmware.

Even more pointedly: the *same* workload (BF16 1024² matmul) lands at very different operating points:
* WS: 2805 MHz / 600 W
* Server: 2415 MHz / 390 W

The Server card, given identical work, simply chooses a less aggressive boost. Whether you cap the WS to 390 W or let it run free, it will not behave like the Server card — they have different boost philosophies.

This is also why the previous PL=300 W sweep showed Server *winning* by +38 % at 300 W: at a forced 300 W cap, Server's V/F is already tuned for low/mid power and produces high clock there; WS's V/F is tuned for high-power operation and produces a lower clock when starved.

### Practical implication

* If you have a moderate compute workload (LLM inference at typical batch sizes, where compute is not saturating), **a WS card will draw substantially more power than a Server card on the same kernel** — by 1.5–2 × at light load. This is not because the WS card is "less efficient" but because its firmware deliberately races-to-idle-with-headroom, while the Server card prefers low-power operation by default.

* For energy-constrained inference, **either pin the Server SKU at PL=300 W (best Gflop/s/W = 414)**, or trust the Server's natural boost behavior to spend less power on partial-utilisation workloads. Do **not** assume a WS card with no PL cap will behave like a Server card on the same kernel; it will use the available watts to push clocks higher.

Raw output: [`data/blackwell-natural-draw-output.txt`](../data/blackwell-natural-draw-output.txt)
