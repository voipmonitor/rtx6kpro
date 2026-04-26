# PCIe AER "Replay Timer Timeout" During Posted-Write Collapse

A secondary symptom of the AMD posted-write collapse: when the collapse trigger pattern is active, the system also accumulates **PCIe correctable AER errors of type "Replay Timer Timeout"** (status bit 12), which appear in the BMC IPMI SEL as `PCI PERR Asserted` events. Other heavy-traffic patterns (including ones with higher aggregate bandwidth than the collapse pattern itself) do **not** trigger these errors.

These are **correctable** errors — the PCIe link auto-retransmits the TLP and continues. There is no data loss and no observable application impact. They are, however, a useful diagnostic fingerprint: if you see "PCI PERR Asserted" events in IPMI on a c-payne / Broadcom / similar PCIe Gen5 GPU rig with AMD CPU and the timestamps cluster around heavy GPU-to-GPU traffic, the collapse trigger is almost certainly the cause.

## What you see in IPMI

```
ID  | TimeStamp           | Sensor | Event              | Description
----|---------------------|--------|--------------------| --------------------------------
22  | 04/25/2026 17:57:23 | BIOS   | critical_interrupt | PCIe SEL Log - Asserted
                                                          Data1: PCI PERR
                                                          Data2: PCI bus number for failed device: 0x03
                                                          Data3: PCI device number: 0x00 PCI function number: 0x00
21  | 04/25/2026 17:56:48 | BIOS   | critical_interrupt | PCIe SEL Log - Asserted
                                                          Data1: PCI PERR
                                                          Data2: PCI bus number for failed device: 0x25
…
```

The PCI bus numbers in `Data2` are the GPU device buses (e.g. 0x03 = GPU 0, 0x46 = GPU 7, 0xec = GPU 13 in our 16-GPU layout). The IPMI label "PCI PERR" is the BMC's legacy mapping for what AER actually classifies as a Data Link Layer correctable error.

## What you see in `dmesg` / sysfs

The same events surface in the kernel as APEI (Generic Hardware Error Source) entries:

```
{8}[Hardware Error]: Hardware error from APEI Generic Hardware Error Source: 514
{8}[Hardware Error]: It has been corrected by h/w and requires no further action
{8}[Hardware Error]: event severity: corrected
{8}[Hardware Error]:  Error 0, type: corrected
{8}[Hardware Error]:  fru_text: PcieError
{8}[Hardware Error]:   section_type: PCIe error
{8}[Hardware Error]:   port_type: 1, legacy PCI end point
{8}[Hardware Error]:   command: 0x0406, status: 0x0010
{8}[Hardware Error]:   device_id: 0000:e3:00.0
{8}[Hardware Error]:   vendor_id: 0x10de, device_id: 0x2bb5
{8}[Hardware Error]:   aer_cor_status: 0x00001000, aer_cor_mask: 0x00000000
{8}[Hardware Error]:   aer_uncor_status: 0x00000000, aer_uncor_mask: 0x00100000
nvidia 0000:e3:00.0: aer_status: 0x00001000, aer_mask: 0x00000000
nvidia 0000:e3:00.0: aer_layer=Data Link Layer, aer_agent=Transmitter ID
```

The key bits are:
* `aer_cor_status: 0x00001000` — bit 12 set = **Replay Timer Timeout**
* `aer_layer=Data Link Layer, aer_agent=Transmitter ID` — the **GPU's PCIe transmitter** sent a TLP and did not receive an Ack from its parent switch within the replay timer window. The transmitter then retried the TLP and the link continued normally.
* `event severity: corrected` — not fatal, no driver intervention required.

The same data is visible per-device in sysfs:

```
$ cat /sys/bus/pci/devices/0000:03:00.0/aer_dev_correctable
RxErr            0
BadTLP           0
BadDLLP          0
Rollover         0
Timeout          2
NonFatalErr      0
CorrIntErr       0
HeaderOF         0
TOTAL_ERR_COR    2
```

`Timeout` is the per-device count of Replay Timer Timeouts since boot.

## Reproduction matrix

System: ASRock WRX90 WS EVO + AMD Threadripper Pro 7955WX, 16 × NVIDIA RTX PRO 6000 Blackwell, 4 × c-payne PCIe Gen5 switches, kernel 6.18.24, NVIDIA driver 595.58.03, `iommu=off`. Same rig as [`collapse-report.md`](collapse-report.md).

Topology: SW1 (GPU 0–3) on root 00, SW2 (GPU 4–7) on root 40, SW3 (GPU 8–11) on root e0:01.1, SW4 (GPU 12–15) on root e0:03.1.

Each test was a 60-second sustained-traffic loop hammering a specific pattern. Counters were snapshotted every ~5 s; "errors" below means new Replay Timer Timeout events that appeared on any GPU in the system during the run.

| Test | Pattern | Aggregate BW | Cross-switch dst roots from one src switch | New AER timeouts in 60 s |
|------|---------|-------------:|--------------------------------------------|-------------------------:|
| Idle (no traffic) | — | 0 GB/s | — | 0 |
| SW1 intra-switch heavy | (0,1) + (2,3) | 112.5 GB/s | none (intra) | 0 |
| SW2 intra-switch heavy | (4,5) + (6,7) | 112.5 GB/s | none (intra) | 0 |
| SW3 intra-switch heavy | (8,9) + (10,11) | 112.5 GB/s | none (intra) | 0 |
| SW4 intra-switch heavy | (12,13) + (14,15) | 112.5 GB/s | none (intra) | 0 |
| 4-pair SW1 → SW2 (uplink saturated) | 4 pairs to single dst | 33.7 GB/s | 1 (root 40 only) | 0 |
| 8 independent cross-switch flows | 1 src GPU per pair, 8 different src/dst | 97.7 GB/s | 1 per source | 0 |
| All-to-all Group A (SW1+SW2 only) | 56 pairs across 2 switches | 160.6 GB/s | 1 per source switch | 0 |
| **SW1 → SW2 + SW3 (collapse, 2 dst roots)** | (0,4) + (1,8) | 13.5 GB/s | **2** | sporadic, ~1 per 30 s |
| **SW1 → SW2 + SW3 + SW4 (collapse, 3 dst roots)** | (0,4) + (1,8) + (2,12) | 51.6 GB/s | **3** | reliable, ~3 per 60 s |

### Key signal

The trigger condition for AER "Replay Timer Timeout" is **the same as the trigger for the bandwidth collapse**:

* one source PCIe switch dispatching outbound writes to ≥ 2 different CPU root complexes simultaneously.

It is **not** triggered by:

* high aggregate bandwidth on its own (the all-to-all test pushed 3× the bandwidth of the collapse test with zero errors)
* saturating a single switch's uplink (4-pair to one dst root = 0 errors)
* multiple independent source switches each going to multiple dst roots (each individual source switch only dispatches to one dst root in that case)
* intra-switch traffic at any volume

### Where the errors land

The Replay Timer Timeouts are recorded on the **GPU side** of the PCIe link:

| Run | Pattern | Errors on which GPUs |
|-----|---------|----------------------|
| 1 | (0,4)+(1,8)+(2,12) | GPU 0 (SW1), GPU 7 (SW2), GPU 13 (SW4) |
| 2 | (0,4)+(1,8)+(2,12) | GPU 0 (SW1) |
| 3 | (0,4)+(1,8)+(2,12) | GPU 9 (SW3) |
| 4 | (0,4)+(1,8) | GPU 0 (SW1), GPU 1 (SW1) |

Active source GPUs (those actually transmitting in the test) are the most frequent victims. But errors also occasionally appear on GPUs that are *not* active in the test, sitting on either the source switch or on a destination switch. Our hypothesis is that the collapse-induced congestion in the source root port and the destination switches creates back-pressure that delays Ack DLLPs flowing back to all GPUs sharing those switches; whichever GPU happens to issue a small TLP at the wrong moment sees its replay timer expire.

The errors are very rare (single digits per minute under sustained collapse traffic), and PCIe Gen 5 links replay them automatically. **There is no observed application-level impact** — bandwidth and NCCL allreduce throughput numbers are exactly the same on a "clean" run as on a run that accumulated several Replay Timeouts.

## Reproduction script

`aer_monitor.sh`:

```bash
#!/bin/bash
# Snapshot AER counters across the full c-payne tree (4 switches, 16 GPUs)
paths=(
  0000:00:01.1 0000:01:00.0 0000:02:00.0 0000:02:01.0 0000:02:02.0 0000:02:03.0 0000:02:04.0 0000:02:05.0
  0000:03:00.0 0000:04:00.0 0000:05:00.0 0000:06:00.0
  0000:40:01.1 0000:41:00.0 0000:42:00.0 0000:42:01.0 0000:42:02.0 0000:42:03.0 0000:42:04.0 0000:42:05.0
  0000:43:00.0 0000:44:00.0 0000:45:00.0 0000:46:00.0
  0000:e0:01.1 0000:e1:00.0 0000:e2:00.0 0000:e2:01.0 0000:e2:02.0 0000:e2:03.0 0000:e2:04.0 0000:e2:05.0
  0000:e3:00.0 0000:e4:00.0 0000:e5:00.0 0000:e6:00.0
  0000:e0:03.1 0000:e9:00.0 0000:ea:00.0 0000:ea:01.0 0000:ea:02.0 0000:ea:03.0 0000:ea:04.0 0000:ea:05.0
  0000:eb:00.0 0000:ec:00.0 0000:ed:00.0 0000:ee:00.0
)
for bdf in "${paths[@]}"; do
  f=/sys/bus/pci/devices/$bdf/aer_dev_correctable
  if [ -f "$f" ]; then
    total=$(grep TOTAL_ERR_COR $f | awk '{print $2}')
    timeout=$(grep "^Timeout" $f | awk '{print $2}')
    echo "$bdf total=$total timeout=$timeout"
  fi
done
```

Driver script (PyTorch):

```python
import torch, time

SIZE = 256 * 1024 * 1024
DURATION = 60
pairs = [(0, 4), (1, 8), (2, 12)]   # SW1 dispatch to SW2, SW3, SW4 = 3 dst roots

bufs, streams = {}, {}
for s, d in pairs:
    bufs[(s,d)] = (torch.randn(SIZE//4, device=f'cuda:{s}'),
                   torch.empty(SIZE//4, device=f'cuda:{d}'))
    torch.cuda.set_device(s)
    streams[(s,d)] = torch.cuda.Stream(torch.device(f'cuda:{s}'))

t0 = time.perf_counter()
n = 0
while time.perf_counter() - t0 < DURATION:
    for s, d in pairs:
        with torch.cuda.stream(streams[(s,d)]):
            bufs[(s,d)][1].copy_(bufs[(s,d)][0])
    n += 1
torch.cuda.synchronize()
print(f"{n} iterations in {DURATION} s")
```

Usage:

```bash
./aer_monitor.sh > before.txt
python3 stress.py
./aer_monitor.sh > after.txt
diff before.txt after.txt
```

A clean run shows no diff. A collapse-trigger run almost always shows one or more `Timeout` counters incremented on at least one GPU.

## What this implies for c-payne / Broadcom diagnostics

A widely-deployed monitoring approach for these PCIe Gen5 GPU rigs is to alert on `PCI PERR` events in IPMI. On AMD-host platforms running cross-switch GPU workloads, those alerts will fire whenever the user's NCCL or custom kernel happens to hit the collapse trigger pattern.

* The errors are **not a c-payne or Broadcom switch fault.** No errors are recorded on the switch ports themselves at the silicon level — only the GPU PCIe transmitters see Replay Timer Timeouts, and only under the specific traffic pattern that the [collapse report](collapse-report.md) documents.
* They are also **not a GPU fault** in the sense that the same GPU runs cleanly under all other patterns (intra-switch traffic, single-dst-root cross-switch traffic, even heavy all-to-all). The errors are a downstream symptom of the collapse-induced arbitration stall in the AMD CPU's I/O die scalable data fabric.
* The errors are correctable and there is no observed performance impact beyond the ~85 % WRITE bandwidth collapse that the collapse report already covers.

For a server vendor diagnostic policy, the practical implication is that an `AER Replay Timer Timeout` rate of "a handful per minute" under specific user workloads is **normal behaviour** on AMD Genoa-family / Turin-family hosts with multi-switch fabrics, and is not by itself an indicator of a faulty cable, switch, or GPU. The *workload pattern* is the underlying cause.

To further isolate: run the controlled test above with ```NCCL_ALGO=Ring``` (no trigger) versus the explicit collapse pattern (trigger). Ring all-reduce produces zero AER timeouts; the explicit collapse pattern reproduces them within seconds.

## Cross-references

* [`collapse-report.md`](collapse-report.md) — the standalone report on the underlying bandwidth collapse, which has the same trigger pattern as these AER errors.
* [`pcie-posted-write-collapse.md`](pcie-posted-write-collapse.md) — long-form history of the collapse investigation across multiple platforms.
* [`wrx90-cpayne-8gpu-root-topology-comparison.md`](wrx90-cpayne-8gpu-root-topology-comparison.md) — controlled test showing that within a 2-switch / 8-GPU group the trigger cannot fire and AER stays clean regardless of root layout.
