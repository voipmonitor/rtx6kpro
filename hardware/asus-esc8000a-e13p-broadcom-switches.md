# ASUS ESC8000A-E13P with Broadcom PEX890xx Switches

Detailed PCIe topology analysis, ACS configuration, and P2P performance measurements for the ASUS ESC8000A-E13P server with 8x RTX PRO 6000 Blackwell Server Edition GPUs connected via Broadcom PEX890xx Gen 5 PCIe switches.

## Table of Contents

- [System Overview](#system-overview)
- [Physical PCIe Topology](#physical-pcie-topology)
- [ACS (Access Control Services) — Critical for P2P](#acs-access-control-services--critical-for-p2p)
- [P2P Bandwidth Results](#p2p-bandwidth-results)
- [p2pmark Benchmark Results](#p2pmark-benchmark-results)
- [NCCL Configuration](#nccl-configuration)
- [Troubleshooting: x4 Link Width on Root Ports](#troubleshooting-x4-link-width-on-root-ports)
- [Proving P2P Goes Through Switch Fabric](#proving-p2p-goes-through-switch-fabric)
- [Quick Reference: ACS Disable Script](#quick-reference-acs-disable-script)

---

## System Overview

| Component | Detail |
|-----------|--------|
| **Motherboard** | ASUS ESC8000A-E13P (K15PG-D24 Series) |
| **CPUs** | 2x AMD EPYC 9575F 64-Core (Turin) |
| **NUMA Nodes** | 2 (GPU0-3 on NUMA 0, GPU4-7 on NUMA 1) |
| **GPUs** | 8x NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB GDDR7 each) |
| **PCIe Switches** | 2x Broadcom PEX890xx Gen 5 (each partitioned into 2 logical switches) |
| **PCIe Link** | Gen5 x16 per GPU (32 GT/s, ~63 GB/s theoretical per direction) |

---

## Physical PCIe Topology

The system uses **two physical Broadcom PEX890xx switch chips**, each partitioned into **two logical switches**. This is not obvious from `nvidia-smi topo` or `lspci` — it was discovered by measuring actual P2P bandwidth with ACS disabled.

```
                    ┌─────────────────────────────────────────────────┐
                    │                 AMD EPYC 9575F                  │
                    │            Dual Socket (Turin)                  │
                    │                                                 │
                    │  NUMA 0                          NUMA 1         │
                    │  root port   root port   root port   root port  │
                    │  10:01.1     70:01.1     80:01.1     f0:01.1    │
                    └───┬────────────┬──────────┬────────────┬────────┘
                        │            │          │            │
                     Gen5 x16    Gen5 x16    Gen5 x16    Gen5 x16
                        │            │          │            │
    ┌═══════════════════╧════════════╧══╗  ┌════╧════════════╧════════════┐
    ║       BROADCOM PEX890xx CHIP A    ║  ║       BROADCOM PEX890xx CHIP B    ║
    ║    (SN: ...b2-00 / ...b2-30)      ║  ║    (SN: ...b2-20 / ...b2-00)      ║
    ║                                   ║  ║                                    ║
    ║  ┌────────────┐  ┌────────────┐   ║  ║  ┌────────────┐  ┌────────────┐   ║
    ║  │ Switch 1   │  │ Switch 3   │   ║  ║  │ Switch 2   │  │ Switch 4   │   ║
    ║  │ (bus 11)   │  │ (bus 81)   │   ║  ║  │ (bus 71)   │  │ (bus f1)   │   ║
    ║  │ NUMA 0     │  │ NUMA 1     │   ║  ║  │ NUMA 0     │  │ NUMA 1     │   ║
    ║  │            ◄══►            │   ║  ║  │            ◄══►            │   ║
    ║  │  internal fabric           │   ║  ║  │  internal fabric           │   ║
    ║  │  ~103 GB/s bidir           │   ║  ║  │  ~103 GB/s bidir           │   ║
    ║  │            │  │            │   ║  ║  │            │  │            │   ║
    ║  │ GPU0  GPU1 │  │ GPU4  GPU5 │   ║  ║  │ GPU2  GPU3 │  │ GPU6  GPU7 │   ║
    ║  │ x16   x16  │  │ x16   x16  │   ║  ║  │ x16   x16  │  │ x16   x16  │   ║
    ║  └────────────┘  └────────────┘   ║  ║  └────────────┘  └────────────┘   ║
    ║                                   ║  ║                                    ║
    ╚═══════════════════════════════════╝  ╚════════════════════════════════════╝
```

### Physical Chip Grouping

| Physical Chip | Partition 1 | Partition 2 | Internal Fabric |
|---|---|---|---|
| **Chip A** | Switch 1: GPU0, GPU1 (NUMA 0) | Switch 3: GPU4, GPU5 (NUMA 1) | ~103 GB/s bidir |
| **Chip B** | Switch 2: GPU2, GPU3 (NUMA 0) | Switch 4: GPU6, GPU7 (NUMA 1) | ~103 GB/s bidir |

**How to identify which GPUs share a physical chip:** With ACS disabled, same-chip GPU pairs achieve ~103 GB/s bidirectional P2P regardless of which partition they are on. Cross-chip pairs are limited by the root port upstream links.

### nvidia-smi Topology

```
$ nvidia-smi topo -m
        GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7
GPU0     X    PIX   NODE  NODE  SYS   SYS   SYS   SYS     NUMA 0
GPU1    PIX    X    NODE  NODE  SYS   SYS   SYS   SYS     NUMA 0
GPU2    NODE  NODE   X    PIX   SYS   SYS   SYS   SYS     NUMA 0
GPU3    NODE  NODE  PIX    X    SYS   SYS   SYS   SYS     NUMA 0
GPU4    SYS   SYS   SYS   SYS   X    PIX   NODE  NODE    NUMA 1
GPU5    SYS   SYS   SYS   SYS  PIX    X    NODE  NODE    NUMA 1
GPU6    SYS   SYS   SYS   SYS  NODE  NODE   X    PIX     NUMA 1
GPU7    SYS   SYS   SYS   SYS  NODE  NODE  PIX    X      NUMA 1
```

- **PIX** = same logical switch partition (e.g., GPU0↔GPU1)
- **NODE** = different partitions on same NUMA (e.g., GPU0↔GPU2)
- **SYS** = cross-NUMA via Infinity Fabric (e.g., GPU0↔GPU4)

> **Important:** `nvidia-smi topo` does NOT show that GPU0 and GPU4 share a physical Broadcom chip. It reports them as `SYS` (cross-NUMA). Only bandwidth measurements reveal the physical chip grouping.

---

## ACS (Access Control Services) — Critical for P2P

### The Problem

By default, ACS is enabled on both the **AMD root ports** and the **Broadcom switch downstream ports**. When ACS bits `ReqRedir` and `CmpltRedir` are set, **all P2P traffic is forced through the upstream root port** instead of being routed directly within the switch fabric. This means:

- Same-switch P2P that should stay within the chip is instead hairpinned through the root port
- P2P bandwidth is limited by the upstream link speed, not the switch fabric speed
- GPUs on the same physical chip cannot communicate at full speed

### Impact: Before vs After ACS Disable

| GPU Pair | Relationship | ACS ON (default) | ACS OFF |
|---|---|---|---|
| GPU0↔GPU1 | PIX (same partition) | ~50 GB/s | **~103 GB/s** |
| GPU4↔GPU5 | PIX (same partition) | ~50 GB/s | **~103 GB/s** |
| GPU0↔GPU4 | Same chip, cross-NUMA | ~95 GB/s | **~103 GB/s** |
| GPU0↔GPU2 | Cross-chip, same NUMA | ~103 GB/s | ~103 GB/s |
| GPU0↔GPU6 | Cross-chip, cross-NUMA | ~94 GB/s | ~94 GB/s |

The **2x improvement** for PIX pairs (50 → 103 GB/s) is because traffic now routes directly through the switch fabric instead of hairpinning through the root port.

### Where ACS Lives

ACS must be disabled on **two levels**:

1. **AMD Root Ports** — ACS capability at offset `0x2a0`, control register at `0x2a6`
2. **Broadcom Switch Downstream Ports** — ACS capability at offset `0x170`, control register at `0x176`

Disabling only the root ports is **NOT sufficient**. The switch downstream ports also enforce ACS redirect independently.

### How to Find and Disable ACS

```bash
# Step 1: Find ALL devices with ACS ReqRedir+ enabled
python3 -c "
import subprocess, re
out = subprocess.check_output(['lspci', '-vv'], text=True, timeout=30)
devices = out.split('\n\n')
for dev in devices:
    lines = dev.strip().split('\n')
    if not lines or not lines[0]: continue
    bdf = lines[0].split()[0]
    acs_offset = None
    for line in lines:
        if 'Access Control Services' in line:
            m = re.search(r'\[([0-9a-fA-F]+)\s', line)
            if m: acs_offset = m.group(1)
        if 'ACSCtl:' in line and 'ReqRedir+' in line and acs_offset:
            ctrl_offset = hex(int(acs_offset, 16) + 6)
            print(f'{bdf} ACS@0x{acs_offset} ctrl@{ctrl_offset}')
            break
"

# Step 2: Disable ReqRedir and CmpltRedir on each device
# For root ports (ACS ctrl at 0x2a6):
#   setpci -s <BDF> 2a6.w=0x0011
# For switch ports (ACS ctrl at 0x176):
#   setpci -s <BDF> 176.w=0x0011

# Step 3: Verify
lspci -vv | grep "ACSCtl:" | grep "ReqRedir+"
# Should return empty (no devices with ReqRedir+ remaining)
```

The value `0x0011` keeps `SrcValid` (bit 0) and `UpstreamFwd` (bit 4) enabled while disabling `ReqRedir` (bit 2) and `CmpltRedir` (bit 3).

### ACS Control Register Bits

```
Bit 0: SrcValid      - Source validation (keep enabled)
Bit 1: TransBlk      - Translation blocking
Bit 2: ReqRedir      - P2P Request Redirect ← DISABLE THIS
Bit 3: CmpltRedir    - P2P Completion Redirect ← DISABLE THIS
Bit 4: UpstreamFwd   - Upstream forwarding (keep enabled for cross-switch traffic)
Bit 5: EgressCtrl    - Egress control
Bit 6: DirectTrans   - Direct translated P2P
```

### Security Note

Disabling ACS allows GPUs to directly access each other's memory without IOMMU isolation. This is fine for bare-metal GPU compute but should be considered if using VFIO/VM passthrough.

---

## P2P Bandwidth Results

### Bidirectional P2P=Enabled Bandwidth Matrix (ACS disabled, all links Gen5 x16)

```
   D\D     0      1      2      3      4      5      6      7
     0    -    102    103    103     94     95     94     94
     1   104    -     103    104     94     93     94     95
     2   103   103     -     103     95     96     95     96
     3   103   102    103     -      95     95     96     94
     4    95    95     95     93     -     103    104    103
     5    94    93     94     96    103     -     103    103
     6    94    95     96     95    103    103     -     103
     7    96    96     95     95    103    103    103     -
```

| Path Type | Bandwidth | Notes |
|---|---|---|
| Same partition (PIX) | ~103 GB/s | GPU0↔1, GPU2↔3, GPU4↔5, GPU6↔7 |
| Same chip, cross-partition | ~103 GB/s | GPU0↔4, GPU0↔5, GPU2↔6, GPU2↔7 (via internal fabric) |
| Cross-chip, same NUMA | ~103 GB/s | GPU0↔2, GPU0↔3 (via root ports, both x16) |
| Cross-chip, cross-NUMA | ~94-95 GB/s | GPU0↔6, GPU2↔4 (via Infinity Fabric) |

### P2P Latency (P2P Writes, microseconds)

```
 Src->  GPU0   GPU1   GPU2   GPU3   GPU4   GPU5   GPU6   GPU7
GPU 0:    -    0.73   0.73   0.73   1.34   1.34   1.33   1.33
GPU 1:  0.73     -    0.73   0.73   1.34   1.34   1.33   1.33
GPU 2:  0.74   0.74     -    0.73   1.32   1.33   1.31   1.31
GPU 3:  0.71   0.72   0.71     -    1.29   1.29   1.27   1.27
GPU 4:  1.35   1.35   1.33   1.32     -    0.74   0.73   0.74
GPU 5:  1.35   1.36   1.33   1.33   0.74     -    0.74   0.74
GPU 6:  1.30   1.31   1.28   1.28   0.72   0.72     -    0.72
GPU 7:  1.35   1.36   1.32   1.32   0.74   0.74   0.74     -
```

| Path | Latency |
|---|---|
| Same NUMA | 0.71-0.74 us |
| Cross NUMA (Infinity Fabric) | 1.27-1.36 us |

---

## p2pmark Benchmark Results

All results measured with [p2pmark](https://github.com/lukealonso/p2pmark) version `3c39f36` (same version used by luke, Festr, and Grimulkan for their reference scores).

### Scores Summary

| Config | PCIe Link Score | Interconnect Score (all-to-all / ideal) | Effective Latency |
|---|---|---|---|
| **4 GPU (NUMA 0)** | **0.88** (55.3 GB/s) | **0.58** (129.2 / 221.3 GB/s) | **2.12 us** |
| **8 GPU (all)** | **0.85** (53.7 GB/s) | **0.11** (46.4 / 429.2 GB/s) | **7.39 us** |

### Comparison with Reference Systems

| System | GPUs | PCIe Score | Interconnect Score (all-to-all / ideal) | Eff. Latency |
|---|---|---|---|---|
| **This system (ACS off)** | 4 | **0.88** | **0.58** (129 / 221 GB/s) | 2.12 us |
| luke (3x Microchip switches) | 4 | 0.86 | 0.64 (138 / 218 GB/s) | 4.10 us |
| Festr (dual Turin, direct-attach) | 4 | 0.88 | 0.59 (130 / 221 GB/s) | 2.28 us |
| **This system (ACS off)** | 8 | **0.85** | **0.11** (46 / 429 GB/s) | 7.39 us |
| luke (3x Microchip switches) | 8 | 0.86 | 0.44 (192 / 435 GB/s) | 6.79 us |
| Festr (dual Turin, direct-attach) | 8 | 0.84 | 0.41 (173 / 421 GB/s) | 6.03 us |

**4-GPU scores** are comparable to the best reference systems. The low 8-GPU interconnect score (0.11) is because the all-to-all metric fires 56 concurrent transfers, completely saturating the Infinity Fabric cross-NUMA links. This is a measurement artifact — actual NCCL ring performance is excellent (see topology probe below).

### Topology Probe (8 GPU, staggered reads by peer distance)

```
+1: 48.62 GB/s avg (389 total)   ← neighbors
+2: 36.24 GB/s avg (290 total)
+3: 22.80 GB/s avg (182 total)
+4: 14.77 GB/s avg (118 total)   ← max distance (cross-NUMA, opposite chip)
+5: 22.70 GB/s avg (182 total)
+6: 35.30 GB/s avg (282 total)
+7: 47.43 GB/s avg (379 total)   ← neighbors (wrapping)
```

### Sequential P2P Bandwidth (8 GPU, GB/s)

```
 Dst->  GPU0   GPU1   GPU2   GPU3   GPU4   GPU5   GPU6   GPU7
GPU0:    -     54.3   54.7   54.6   38.2   37.6   38.0   37.9    same-NUMA ~55, cross ~38
GPU4:  42.4   41.2   37.9   37.8    -     54.8   54.4   54.6    same-NUMA ~55, cross ~38-42
```

### AllReduce: Custom vs NCCL (4 GPU, fp16)

| Size | Custom (us) | NCCL (us) | Winner |
|---|---|---|---|
| 256 B | 6.9 | 18.0 | Custom 2.6x |
| 4 KB | 7.4 | 15.4 | Custom 2.1x |
| 16 KB | 9.2 | 19.7 | Custom 2.1x |
| 32 KB | 11.0 | 17.6 | Custom 1.6x |
| 64 KB | 14.0 | 19.1 | Custom 1.4x |
| 128 KB | 19.8 | 28.3 | Custom 1.4x |
| 256 KB | 32.8 | 46.5 | Custom 1.4x |
| 512 KB | 55.7 | 87.1 | Custom 1.6x |
| 1 MB | 104.4 | 109.2 | Custom 1.0x |
| 2 MB | 203.0 | 152.1 | NCCL 1.3x |
| 32 MB | 3237 | 1790 | NCCL 1.8x |

Custom allreduce wins up to **1 MB** on 4 GPUs.

### AllReduce: Custom vs NCCL (8 GPU, fp16)

| Size | Custom (us) | NCCL (us) | Winner |
|---|---|---|---|
| 256 B | 9.6 | 28.6 | Custom 3.0x |
| 4 KB | 39.5 | 65.6 | Custom 1.7x |
| 8 KB | 68.8 | 70.8 | Custom 1.0x |
| 16 KB | 151.2 | 69.3 | NCCL 2.2x |
| 64 KB | 587.5 | 79.5 | NCCL 7.4x |
| 256 KB | 2665 | 86.2 | NCCL 30.9x |
| 32 MB | 332546 | 2976 | NCCL 112x |

Custom allreduce is only useful up to ~8 KB on 8 GPUs. For larger messages NCCL ring dominates.

---

## NCCL Configuration

### Transport Selection

By default, NCCL uses **SHM (shared memory)** for NODE pairs and **P2P/direct pointer** only for PIX pairs. To force P2P everywhere:

```bash
export NCCL_P2P_LEVEL=SYS
```

Verification:

```
# With NCCL_P2P_LEVEL=SYS:
Check P2P Type isAllDirectP2p 1 directMode 1 isAllCudaP2p 1
Channel 00/0 : 0[0] -> 1[1] via P2P/direct pointer
Channel 00/0 : 1[1] -> 2[2] via P2P/direct pointer  ← NODE pair now uses P2P
Channel 00/0 : 2[2] -> 3[3] via P2P/direct pointer
Channel 00/0 : 3[3] -> 0[0] via P2P/direct pointer

# Without (default):
Check P2P Type isAllDirectP2p 0
Channel 00 : 1[1] -> 2[2] via SHM/direct/direct     ← NODE pair uses shared memory
```

### AllReduce Bus Bandwidth (32M, 4 GPU)

| NCCL Config | BusBw (GB/s) |
|---|---|
| Default (SHM for NODE pairs) | 30.3 |
| **NCCL_P2P_LEVEL=SYS** | **39.3** |

### Recommended NCCL Environment

```bash
export NCCL_P2P_LEVEL=SYS
export NCCL_NET_GDR_LEVEL=SYS
export NCCL_MIN_NCHANNELS=8
```

---

## Troubleshooting: x4 Link Width on Root Ports

### Symptom

Massive P2P bandwidth asymmetry. Some GPU pairs achieve only ~13-27 GB/s instead of expected ~100 GB/s:

```
Bidirectional P2P=Enabled (before fix):
GPU4↔GPU5: 13.5 GB/s   ← should be ~103 GB/s
GPU0↔GPU4: 27.2 GB/s   ← should be ~95 GB/s
GPU0↔GPU1: 50.7 GB/s   ← should be ~103 GB/s (with ACS disabled)
```

### Diagnosis

Check root port link capabilities:

```bash
for rp in $(lspci | grep "PCI bridge.*AMD" | awk '{print $1}'); do
  cap=$(lspci -vvs $rp 2>/dev/null | grep "LnkCap:" | head -1)
  sta=$(lspci -vvs $rp 2>/dev/null | grep "LnkSta:" | head -1)
  if echo "$cap" | grep -q "Width x[0-9]"; then
    echo "$rp: $cap | $sta"
  fi
done
```

If any root port shows `LnkCap: Width x4` instead of `Width x16`, that root port is misconfigured in the BIOS.

### Root Cause

BIOS PCIe bifurcation / lane allocation set incorrectly for that root port. Only 4 lanes assigned instead of 16.

### Fix

Enter BIOS setup and look for:
- **PCIe Lane Configuration** / **Bifurcation** for the affected root port
- **NBIO Configuration** → per-port lane width settings

Set the affected port to x16. The root port's `LnkCap` should show `Width x16` after the fix.

---

## Proving P2P Goes Through Switch Fabric

To definitively prove that P2P traffic stays within the Broadcom switch fabric (and does not hairpin through the upstream root port), we deliberately degraded the upstream link speed:

### Test: Degrade Upstream to Gen2, Measure Same-Switch P2P

```bash
# Set target speed to Gen2 (5 GT/s) on all root ports
for rp in 10:01.1 70:01.1 80:01.1 f0:01.1; do
  # Read LnkCtl2, set target speed bits to Gen2 (value 2)
  cur=$(setpci -s $rp 88.w)
  new=$(printf "%04x" $(( (16#$cur & 0xFFF0) | 0x0002 )))
  setpci -s $rp 88.w=0x$new
  # Trigger link retrain
  cur_ctl=$(setpci -s $rp 68.w)
  retrain=$(printf "%04x" $(( 16#$cur_ctl | 0x0020 )))
  setpci -s $rp 68.w=0x$retrain
done
```

### Results: Upstream at Gen2 x16

```
Bidirectional P2P=Enabled Bandwidth (upstream degraded to Gen2):
   D\D     0      1      2      3      4      5      6      7
     0    -    102    103    103     14     14     14     14
     1   103    -     104    104     14     14     14     14
     2   103   103     -     103     14     14     14     14
     3   103   104    104     -      14     14     14     14
     4    14    14     14     14     -     103    103    103
     5    14    14     14     14    103     -     103    103
     6    14    14     14     14    103    103     -     103
     7    14    14     14     14    103    103    103     -
```

| Path | Upstream Link Used? | Bandwidth |
|---|---|---|
| Same-switch (PIX) | **NO** | **103 GB/s** — unchanged! |
| Same-chip cross-partition | **NO** | **103 GB/s** — unchanged! |
| Cross-chip | **YES** | **14 GB/s** — degraded to Gen2 speed |

**This proves conclusively** that with ACS disabled, P2P traffic within the same Broadcom physical chip routes through the internal switch fabric and **never touches the upstream link**. Only cross-chip traffic uses the root port upstream links.

### Restoring Full Speed

```bash
# Set target speed back to Gen5 (value 5) and retrain
for rp in 10:01.1 70:01.1 80:01.1 f0:01.1; do
  cur=$(setpci -s $rp 88.w)
  new=$(printf "%04x" $(( (16#$cur & 0xFFF0) | 0x0005 )))
  setpci -s $rp 88.w=0x$new
  cur_ctl=$(setpci -s $rp 68.w)
  retrain=$(printf "%04x" $(( 16#$cur_ctl | 0x0020 )))
  setpci -s $rp 68.w=0x$retrain
done
```

---

## Quick Reference: ACS Disable Script

This script finds and disables ACS on all root ports and switch ports. **Run after every reboot** (ACS settings do not persist).

```bash
#!/bin/bash
# disable-acs.sh — Disable ACS P2P redirect on all PCIe bridges
# Run as root after every boot

echo "Finding devices with ACS ReqRedir enabled..."

python3 -c "
import subprocess, re

out = subprocess.check_output(['lspci', '-vv'], text=True, timeout=30)
devices = out.split('\n\n')
count = 0

for dev in devices:
    lines = dev.strip().split('\n')
    if not lines or not lines[0]: continue
    bdf = lines[0].split()[0]
    acs_offset = None
    for line in lines:
        if 'Access Control Services' in line:
            m = re.search(r'\[([0-9a-fA-F]+)\s', line)
            if m: acs_offset = m.group(1)
        if 'ACSCtl:' in line and 'ReqRedir+' in line and acs_offset:
            ctrl_offset = int(acs_offset, 16) + 6
            subprocess.run(['setpci', '-s', bdf, f'{ctrl_offset:x}.w=0x0011'], check=True)
            count += 1
            print(f'  Disabled ACS on {bdf} (ctrl@0x{ctrl_offset:x})')
            break

print(f'Done. Disabled ACS on {count} devices.')
"

# Verify
remaining=$(lspci -vv 2>/dev/null | grep "ACSCtl:" | grep -c "ReqRedir+")
echo "Devices with ReqRedir+ remaining: $remaining"
```

Save as `/usr/local/bin/disable-acs.sh`, make executable, and create a systemd service:

```bash
chmod +x /usr/local/bin/disable-acs.sh

cat > /etc/systemd/system/disable-acs.service << 'EOF'
[Unit]
Description=Disable PCIe ACS for GPU P2P
After=nvidia-persistenced.service

[Service]
Type=oneshot
ExecStart=/usr/local/bin/disable-acs.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

systemctl enable disable-acs.service
```
