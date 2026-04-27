"""
ASUS-equivalent collapse test (persistent in /root/tests/).
2 src GPUs from same VS → 2 cross-chip dst on different roots.

Mapping (GPU index → switch):
  SW1: GPU 0, 1
  SW2: GPU 2, 3
  SW3: GPU 4, 5
  SW4: GPU 6, 7
"""
import torch, time, subprocess

SIZE = 256 * 1024 * 1024
ITERS = 100

def concurrent_write(pairs, iters=ITERS, size=SIZE):
    bufs, streams = {}, {}
    for s, d in pairs:
        bufs[(s,d)] = (torch.randn(size//4, device=f'cuda:{s}'),
                       torch.empty(size//4, device=f'cuda:{d}'))
        torch.cuda.set_device(s)
        streams[(s,d)] = torch.cuda.Stream(torch.device(f'cuda:{s}'))
    for s, d in pairs:
        with torch.cuda.stream(streams[(s,d)]): bufs[(s,d)][1].copy_(bufs[(s,d)][0])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        for s, d in pairs:
            with torch.cuda.stream(streams[(s,d)]): bufs[(s,d)][1].copy_(bufs[(s,d)][0])
    torch.cuda.synchronize()
    return size * iters * len(pairs) / (time.perf_counter() - t0) / 1e9

def concurrent_read(pairs, iters=ITERS, size=SIZE):
    bufs, streams = {}, {}
    for s, d in pairs:
        bufs[(s,d)] = (torch.randn(size//4, device=f'cuda:{d}'),
                       torch.empty(size//4, device=f'cuda:{s}'))
        torch.cuda.set_device(s)
        streams[(s,d)] = torch.cuda.Stream(torch.device(f'cuda:{s}'))
    for s, d in pairs:
        with torch.cuda.stream(streams[(s,d)]): bufs[(s,d)][1].copy_(bufs[(s,d)][0])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        for s, d in pairs:
            with torch.cuda.stream(streams[(s,d)]): bufs[(s,d)][1].copy_(bufs[(s,d)][0])
    torch.cuda.synchronize()
    return size * iters * len(pairs) / (time.perf_counter() - t0) / 1e9

print("=== ASUS-replica patterns ===")

tests = [
    ("COLLAPSE [(0,2),(1,6)] = SW1 → SW2+SW4 (2 dst, possibly diff roots)",
     [(0, 2), (1, 6)]),
    ("OK-A [(0,2),(4,6)] = SW1+SW3 (diff src VS) → SW2+SW4",
     [(0, 2), (4, 6)]),
    ("OK-B [(0,2),(1,3)] = SW1 → SW2 only (1 dst root)",
     [(0, 2), (1, 3)]),
    ("AS-LIT [(0,4),(1,6)] = SW1 → SW3+SW4 (2 dst)",
     [(0, 4), (1, 6)]),
    ("4FLOW [(0,2),(0,6),(1,3),(1,7)] = SW1 → 2 cross-chip dst each",
     [(0, 2), (0, 6), (1, 3), (1, 7)]),
]

print(f"{'Pattern':<60s}  {'WRITE':>10s}  {'READ':>10s}")
print("-"*82)
for label, pairs in tests:
    w = concurrent_write(pairs)
    r = concurrent_read(pairs)
    print(f"{label:<60s}  {w:7.1f}    {r:7.1f}")
