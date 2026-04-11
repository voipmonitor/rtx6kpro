# NVMe RAID I/O Optimization for Docker GPU Servers

Comprehensive benchmark results and optimization guide for md RAID arrays with NVMe SSDs running Docker containers on RunPod GPU infrastructure (53 bare-metal servers, 8x RTX 5090 each).

**Test environment:** Inferno-10 (I10), AMD EPYC 256-core, 1 TB RAM, Ubuntu 22.04, kernel 6.8.0-106-generic

## Table of Contents

- [The Problem](#the-problem)
- [Root Cause Analysis](#root-cause-analysis)
- [Solution: nosync overlay + libnosync](#solution-nosync-overlay--libnosync)
- [Benchmark Results](#benchmark-results)
  - [I/O Scheduler Comparison](#io-scheduler-comparison)
  - [Docker Create/Stop Under Load](#docker-createstop-under-load)
  - [RAID5 vs RAID10](#raid5-vs-raid10)
  - [Enterprise vs Consumer NVMe](#enterprise-vs-consumer-nvme)
  - [Dirty Pages Tuning](#dirty-pages-tuning)
  - [Bitmap Impact](#bitmap-impact)
  - [RAID5 Tuning](#raid5-tuning)
- [Build: overlay-nosync.ko](#build-overlay-nosyncko)
- [Build: libnosync.so](#build-libnosyncso)
- [Deployment](#deployment)
- [Production Configuration](#production-configuration)

---

## The Problem

Docker container lifecycle operations (create, start, stop, rm) hang for **2-5 minutes** on servers under I/O load. The hang manifests as `ovl_sync_fs` in kernel stack traces — Docker's overlay filesystem calls `syncfs()` which flushes ALL dirty pages on the underlying XFS, not just the overlay's own data.

With 6 containers actively writing and default dirty page settings (40% of 1 TB RAM = 400 GB), a single `docker stop` can trigger a flush of hundreds of gigabytes.

### Symptoms
- `docker stop` takes 60-300+ seconds
- `docker create` takes 120-1600+ seconds
- `dockerd` shows `futex_wait_queue` in `/proc/<pid>/stack`
- `D` state (uninterruptible sleep) processes in `top`
- kern.log: `task dockerd blocked for more than 120 seconds`

### Who is affected
Any server running Docker with overlay2 storage driver on md RAID with significant I/O load from containers.

---

## Root Cause Analysis

Two independent mechanisms cause the hang:

### 1. Kernel: `ovl_sync_fs` (overlay filesystem sync)

When Docker performs container lifecycle operations, it calls `syncfs()` on the overlay mount. The kernel function `ovl_sync_fs` in `fs/overlayfs/super.c` then calls `sync_filesystem()` on the **entire underlying XFS filesystem**, flushing ALL dirty pages — not just the overlay's data.

```
dockerd → syncfs(overlay_fd) → ovl_sync_fs() → sync_filesystem(upper_sb) → flush ALL dirty pages on XFS
```

This is a kernel design choice: overlay sync flushes the entire backing filesystem.

### 2. Userspace: Docker's `fsync()` on metadata

Docker daemon (`dockerd`) calls `fsync()` and `fdatasync()` on container metadata files (config.json, layer links, network config). Under I/O load, each `fsync()` blocks waiting for the disk to confirm the write. With hundreds of gigabytes of dirty pages competing for I/O bandwidth, individual `fsync()` calls can take minutes.

### Why both matter

| Fix applied | Create time | Stop time |
|---|---|---|
| Neither (stock) | >120s (timeout) | 71-120s |
| nosync overlay only | 190-1655s (worse!) | N/A |
| eatmydata/libnosync only | >8 min (hangs) | N/A |
| **Both nosync + libnosync** | **1.9s** | **1.3s** |

- **nosync overlay alone** is worse than stock because dirty pages accumulate without being flushed, making subsequent fsync calls even slower.
- **libnosync alone** intercepts userspace fsync but cannot intercept kernel `sync_filesystem()` called by `ovl_sync_fs`.
- **Both together** eliminate sync at both layers.

---

## Solution: nosync overlay + libnosync

Two components are required:

### 1. overlay-nosync.ko — Kernel module

A patched overlay kernel module where `ovl_sync_fs()` returns 0 immediately instead of calling `sync_filesystem()`. Must be compiled per kernel version.

### 2. libnosync.so — Userspace LD_PRELOAD library

A shared library that intercepts `fsync()`, `fdatasync()`, `syncfs()`, `sync()`, and `sync_file_range()` from dockerd, making them return 0 immediately. Applied via `LD_PRELOAD` in the dockerd systemd unit.

### Data safety

With both patches applied:
- Data sits in kernel page cache (RAM) for up to 30 seconds before kernel background writeback flushes it to disk
- XFS journal still protects filesystem metadata integrity
- Risk: hard power loss (no UPS) within 30 seconds of a write = data loss for that write
- Docker containers are stateless (images re-pulled on restart), so overlay data loss is acceptable
- Persistent data should be on volumes (not overlay), which are unaffected

**Suitable for:** Datacenter servers with UPS. **Not suitable for:** Servers without UPS or power protection.

---

## Benchmark Results

All benchmarks on I10: 3x Samsung PM9A3 3.84TB (enterprise) + 1x WD BLACK SN850X 4TB (consumer), kernel 6.8.0-106-generic.

### I/O Scheduler Comparison

RAID10 4x NVMe, 4K random write, 60s sustained, bitmap=none:

| Scheduler | IOPS | Latency | vs none |
|---|---|---|---|
| **none** | **91,800** | 697 us | baseline |
| kyber | 63,900 | 1,001 us | -30% |
| mq-deadline | 63,300 | 1,010 us | -31% |
| BFQ | 48,600 | 1,315 us | -47% |
| BFQ (tuned: slice_idle=0, low_latency=0) | 46,000 | 1,391 us | -50% |

**BFQ kills ~47% of IOPS on NVMe.** Designed for rotational disks where seek time matters. On NVMe with zero seek latency, it adds pure overhead. BFQ tuning does not help — the overhead is architectural (per-request accounting, budget computation, weight-based dispatch).

**cgroup v2 I/O fairness:** Docker's `--blkio-weight` sets `io.bfq.weight` which requires BFQ. Without BFQ, proportional weight fairness does not work. However, `io.max` (hard IOPS/bandwidth limits) works with any scheduler. In practice, kernel dirty page throttling (`balance_dirty_pages`) provides per-process fairness regardless of scheduler.

**Recommendation:** `scheduler=none` on all NVMe drives under RAID arrays. BFQ only on boot disk if needed.

### Docker Create/Stop Under Load

6 containers continuously writing (`dd if=/dev/urandom`), then measuring `docker create` and `docker stop` of additional containers.

#### On enterprise disks only (3x Samsung PM9A3, RAID5)

| Overlay | Dirty limit | Create | Stop |
|---|---|---|---|
| Stock | 16 GB | >120s (timeout) | 71-120s |
| Stock | default (400 GB) | hangs (>8 min) | hangs |
| Stock + libnosync (no overlay patch) | default | hangs (>8 min) | hangs |
| Nosync overlay (no libnosync) | default | 190-1655s | N/A |
| **Nosync + libnosync** | **16 GB** | **5.7s** | **2.6s** |
| **Nosync + libnosync** | **default** | **1.9s** | **1.3s** |

#### On mixed disks (2x Samsung + 1x WD SN850X, RAID5)

| Overlay | Dirty limit | Create | Stop |
|---|---|---|---|
| Nosync + libnosync | 16 GB | 5.6s | 2.5s |

#### Key finding: RAID type does not matter

| Config (nosync+libnosync, dirty=16G) | RAID5 Create | RAID10 Create |
|---|---|---|
| 3x Samsung PM9A3 | 5.7s | 5.4s |

RAID5 and RAID10 perform identically for Docker lifecycle operations.

### Enterprise vs Consumer NVMe

Sustained 30-second sequential write (after SLC cache exhaustion):

| Disk | Sustained Write | Type |
|---|---|---|
| Samsung PM9A3 (MZQL23T8) | **4,140 MB/s** | Enterprise |
| WD BLACK SN850X 4TB | **886 MB/s** | Consumer |

The WD SN850X is **4.7x slower** after SLC cache exhaustion. In RAID10 mirror pairs, the entire pair runs at the speed of the slowest disk.

| RAID10 config | Docker create (nosync+libnosync, dirty=16G) |
|---|---|
| 3x Samsung + WD (4 disk RAID10) | 250s |
| 3x Samsung only (degraded RAID10) | 5.4s |
| 2x Samsung RAID0 | 2.8s |
| Samsung+WD RAID0 | 5.6s |

**Consumer NVMe disks (WD SN850X) in RAID10 mirror = catastrophic performance.** The 250s create was caused by WD's slow sustained write holding RAID10 mirror locks under dirty throttling, creating cascading contention.

#### WD SN850X 4TB SLC Cache

| Parameter | Value |
|---|---|
| Static SLC cache | ~12-18 GB |
| Dynamic SLC (empty disk) | ~600 GB |
| Dynamic SLC (50% full) | ~300 GB |
| Burst write (in SLC) | 6,200-6,400 MB/s |
| After SLC exhaustion (fresh) | 1,500-1,800 MB/s |
| After SLC exhaustion (75%+ full) | 800-1,200 MB/s |
| Cache recovery | 10-15 min idle |
| Transition | Cliff (abrupt drop) |

### Dirty Pages Tuning

`vm.dirty_bytes` controls how much data can sit in memory before the kernel throttles writing processes.

| Setting | Max dirty | Effect |
|---|---|---|
| `dirty_bytes=16GB` | 16 GB | Fast sync (<16 GB to flush), but throttles ALL writes when 6+ writers saturate budget |
| Default (`dirty_ratio=40%`) | ~400 GB (on 1TB RAM) | No throttling, but sync operations flush hundreds of GB |

**With nosync+libnosync (sync eliminated):**

| Dirty setting | Create | Stop |
|---|---|---|
| 16 GB | 5.7s | 2.6s |
| default | **1.9s** | **1.3s** |

Default dirty is better with nosync+libnosync because there is no sync to flush, and processes are never throttled.

**Without nosync+libnosync:** `dirty_bytes=16GB` is required to limit sync flush time. But even then, stock overlay hangs for >120s.

**Practical impact of dirty_bytes=16GB on GPU workloads:** Minimal. GPU training writes small checkpoints (1-10 GB), model loading is reads. Image pulls may slow by a few seconds after 16 GB buffered. The 16 GB limit prevents RAM waste on dirty pages.

### Bitmap Impact

RAID10, 4K random write, scheduler=none:

| Bitmap | IOPS | vs none |
|---|---|---|
| bitmap=none | 91,800 | baseline |
| bitmap=internal (traditional) | 87,700 | -5% |

Bitmap cost is only ~5%. Safe to leave enabled for crash recovery (partial resync instead of full).

**Kernel 6.18 lockless bitmap (llbitmap):** Achieves 97% of no-bitmap performance (87,600 IOPS), but requires kernel 6.18+ and mdadm 4.6+. Not needed on kernel 6.8 where traditional bitmap is only 5% overhead.

### RAID5 Tuning

RAID5 3x Samsung PM9A3, effect of `group_thread_cnt` and `stripe_cache_size`:

| Parameter | Seq Write | Rand Write 4K |
|---|---|---|
| Default (threads=0, cache=256) | 1,380 MB/s | 64,700 IOPS |
| Tuned (threads=8, cache=32768) | 1,770 MB/s | 80,800 IOPS |
| **Optimal (threads=2, NUMA pin, cache=32768)** | **1,690 MB/s** | **109,000 IOPS** |

- `group_thread_cnt=2` is optimal (1 = slightly less random perf, 8+ = contention)
- `stripe_cache_size=32768` is the kernel maximum
- NUMA pinning of `raid5d` thread to the same node as NVMe devices: +32% random write
- RAID5 sequential write caps at ~1.7 GB/s due to single-threaded `raid5d` stripe dispatch (kernel architecture limit)

#### RAID5 vs RAID10 (3x Samsung PM9A3)

| Test | RAID10 | RAID5 |
|---|---|---|
| Sequential Write 1M | **6.7 GB/s** | 1.77 GB/s |
| Sequential Read 1M | 8.4 GB/s | 8.5 GB/s |
| Random Write 4K | 96,600 IOPS | **109,000 IOPS** |
| Random Read 4K | 126,000 IOPS | 134,000 IOPS |
| Docker create (nosync+libnosync) | 5.4s | 5.7s |
| Docker stop (nosync+libnosync) | 2.6s | 2.6s |
| Usable capacity (4 disks) | 50% | 67% |

RAID10 advantage is sequential write throughput (3.8x). RAID5 gives more capacity and slightly better random IOPS (NUMA-tuned). Docker lifecycle is identical.

---

## Build: overlay-nosync.ko

Must be compiled for each kernel version running on the servers.

### Prerequisites

```bash
apt-get install -y build-essential linux-headers-$(uname -r)
```

### Download overlay source

```bash
KVER=$(uname -r | grep -oP '^\d+\.\d+')  # e.g., "6.8"
mkdir -p /tmp/overlay-build && cd /tmp/overlay-build

for f in super.c inode.c dir.c readdir.c copy_up.c export.c namei.c \
         util.c file.c ovl_entry.h params.c params.h xattrs.c \
         overlayfs.h; do
  wget -q "https://raw.githubusercontent.com/torvalds/linux/v${KVER}/fs/overlayfs/${f}" -O "$f"
done

# internal.h needed by namei.c
wget -q "https://raw.githubusercontent.com/torvalds/linux/v${KVER}/fs/internal.h" -O internal.h
sed -i 's|../internal.h|internal.h|g' namei.c
```

### Patch ovl_sync_fs

```bash
python3 << 'PYEOF'
with open("super.c") as f:
    lines = f.readlines()

start = None
brace_count = 0
end = None

for i, line in enumerate(lines):
    if "static int ovl_sync_fs(" in line and start is None:
        start = i
    if start is not None and end is None:
        brace_count += line.count("{") - line.count("}")
        if brace_count == 0 and "}" in line:
            end = i
            break

new_lines = lines[:start] + [
    "static int ovl_sync_fs(struct super_block *sb, int wait)\n",
    "{\n",
    "\t/* nosync: skip sync to eliminate ovl_sync_fs hang */\n",
    "\treturn 0;\n",
    "}\n",
] + lines[end+1:]

with open("super.c", "w") as f:
    f.writelines(new_lines)
print(f"Patched ovl_sync_fs at lines {start+1}-{end+1}")
PYEOF
```

### Compile

```bash
cat > Kbuild << 'EOF'
obj-m += overlay.o
overlay-objs := super.o namei.o util.o inode.o file.o dir.o readdir.o copy_up.o export.o params.o xattrs.o
EOF

make -C /lib/modules/$(uname -r)/build M=$(pwd) modules
```

### Install

```bash
cp overlay.ko /usr/local/lib/modules/overlay-nosync-$(uname -r).ko
```

### Load (runtime, replaces stock overlay)

```bash
# Stop Docker first
systemctl stop docker docker.socket containerd
sleep 3
rmmod overlay
insmod /usr/local/lib/modules/overlay-nosync-$(uname -r).ko
systemctl start docker
```

### Auto-load at boot

```bash
cat > /etc/systemd/system/overlay-nosync.service << 'EOF'
[Unit]
Description=Load nosync overlay module
Before=docker.service containerd.service
After=systemd-modules-load.service

[Service]
Type=oneshot
ExecStart=/bin/bash -c "rmmod overlay 2>/dev/null; insmod /usr/local/lib/modules/overlay-nosync-$(uname -r).ko"
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable overlay-nosync.service
```

---

## Build: libnosync.so

Simple C library that intercepts sync-related syscalls. No external dependencies.

### Source: nosync.c

```c
/*
 * libnosync.so — Intercept fsync/fdatasync/syncfs/sync to eliminate
 * Docker metadata sync overhead. Use with LD_PRELOAD on dockerd.
 *
 * Compile: gcc -shared -fPIC -o libnosync.so nosync.c
 * Usage:   LD_PRELOAD=/usr/local/lib/libnosync.so dockerd ...
 *
 * Safety:  Data remains in kernel page cache and is flushed by
 *          background writeback within ~30 seconds. XFS journal
 *          protects filesystem integrity. Requires UPS.
 */

#define _GNU_SOURCE
#include <unistd.h>
#include <sys/types.h>

int fsync(int fd) {
    (void)fd;
    return 0;
}

int fdatasync(int fd) {
    (void)fd;
    return 0;
}

void sync(void) {
    return;
}

int syncfs(int fd) {
    (void)fd;
    return 0;
}

int sync_file_range(int fd, off_t offset, off_t nbytes, unsigned int flags) {
    (void)fd; (void)offset; (void)nbytes; (void)flags;
    return 0;
}
```

### Compile and install

```bash
gcc -shared -fPIC -O2 -o /usr/local/lib/libnosync.so nosync.c
```

### Apply to dockerd

```bash
mkdir -p /etc/systemd/system/docker.service.d/
cat > /etc/systemd/system/docker.service.d/nosync.conf << 'EOF'
[Service]
Environment=LD_PRELOAD=/usr/local/lib/libnosync.so
EOF

systemctl daemon-reload
systemctl restart docker
```

### Verify

```bash
cat /proc/$(pidof dockerd)/environ | tr '\0' '\n' | grep LD_PRELOAD
# Should show: LD_PRELOAD=/usr/local/lib/libnosync.so
```

---

## Deployment

### Full deployment script (per server)

```bash
#!/bin/bash
set -e

# 1. Compile and install libnosync.so
cat > /tmp/nosync.c << 'CEOF'
#define _GNU_SOURCE
#include <unistd.h>
#include <sys/types.h>
int fsync(int fd) { (void)fd; return 0; }
int fdatasync(int fd) { (void)fd; return 0; }
void sync(void) { return; }
int syncfs(int fd) { (void)fd; return 0; }
int sync_file_range(int fd, off_t offset, off_t nbytes, unsigned int flags) {
    (void)fd; (void)offset; (void)nbytes; (void)flags; return 0;
}
CEOF
gcc -shared -fPIC -O2 -o /usr/local/lib/libnosync.so /tmp/nosync.c

# 2. Configure dockerd LD_PRELOAD
mkdir -p /etc/systemd/system/docker.service.d/
echo -e "[Service]\nEnvironment=LD_PRELOAD=/usr/local/lib/libnosync.so" \
    > /etc/systemd/system/docker.service.d/nosync.conf

# 3. Set I/O scheduler to none on NVMe under RAID
for d in /sys/block/nvme*/queue/scheduler; do
    echo none > $d 2>/dev/null
done

# 4. Apply immediately
systemctl daemon-reload
systemctl restart docker

echo "Deployed: libnosync + scheduler=none"
```

**Note:** overlay-nosync.ko requires per-kernel compilation and is more complex to deploy fleet-wide. The libnosync.so + scheduler=none deployment provides significant improvement and can be done immediately. overlay-nosync.ko can be added later for complete fix.

---

## Production Configuration

### Recommended settings for enterprise NVMe (Samsung PM9A3, Kingston DC, Intel DC)

```
RAID:               RAID5 or RAID10, 128K chunk, no partition (XFS direct on /dev/md0)
Scheduler:          none on all NVMe
Dirty pages:        default (vm.dirty_ratio=40, no dirty_bytes override)
Bitmap:             internal (5% overhead, crash recovery benefit)
Overlay:            nosync (overlay-nosync.ko)
Docker LD_PRELOAD:  libnosync.so
read_ahead_kb:      3072
ps_max_latency_us:  0 (NVMe power management disabled)
```

### Additional RAID5 tuning

```
stripe_cache_size:       32768
group_thread_cnt:        2
preread_bypass_threshold: 0
rq_affinity:             2 (on NVMe devices)
NUMA pin raid5d:         taskset -pc <local_cpus> $(pgrep md0_raid5)
```

### Settings for mixed/consumer NVMe (WD SN850X, Seagate, Kingston FURY)

Same as above, plus:
```
Dirty pages:        vm.dirty_bytes=17179869184 (16 GB) — prevents dirty throttling cascading with slow consumer disk writes
```

**Important:** Consumer NVMe SSDs should NOT be placed in RAID10 mirror pairs with enterprise SSDs. The consumer disk's slow sustained write speed (800-1200 MB/s after SLC cache exhaustion) holds mirror locks and creates cascading contention. Use RAID5 stripe (where slow disk affects only its stripe portion) or replace with enterprise disks.

### rc.local template

```bash
#!/bin/bash
MD=$(ls /sys/block/ | grep "^md[0-9]" | head -1)
if [ -n "$MD" ]; then
    echo 3072 > /sys/block/$MD/queue/read_ahead_kb
    # RAID5 only:
    echo 32768 > /sys/block/$MD/md/stripe_cache_size 2>/dev/null
    echo 2 > /sys/block/$MD/md/group_thread_cnt 2>/dev/null
fi
for d in /sys/block/nvme*/queue/scheduler; do
    echo none > $d 2>/dev/null
done
echo 0 > /sys/module/nvme_core/parameters/default_ps_max_latency_us
exit 0
```

---

## Summary

| Optimization | Impact | Effort |
|---|---|---|
| **nosync overlay + libnosync** | **50-100x faster Docker lifecycle** | Medium (kernel module per version + .so) |
| **scheduler=none on NVMe** | **+90% IOPS** | Low (rc.local + udev) |
| Replace consumer NVMe with enterprise | Eliminates dirty throttling cascade | High (hardware) |
| RAID5 tuning (threads, cache, NUMA) | +25-65% write performance | Low (rc.local) |
| dirty_bytes=16G (consumer disks only) | Limits sync flush time | Low (sysctl) |
| Kernel 6.18 llbitmap | Bitmap at 97% of no-bitmap speed | High (kernel upgrade) |

**The single most important optimization is nosync overlay + libnosync.** Without it, no amount of RAID tuning, scheduler changes, or dirty page settings will fix the Docker lifecycle hang.
