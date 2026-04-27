import torch, time

SIZE = 256 * 1024 * 1024
ITERS = 100

# 2 GPUs per switch, 4 switches
# Topology mapping (GPU index → switch):
# SW1: GPU 0, 1
# SW2: GPU 2, 3
# SW3: GPU 4, 5
# SW4: GPU 6, 7

def run(pairs):
    bufs = {}; streams = {}
    for s, d in pairs:
        bufs[(s,d)] = (torch.randn(SIZE//4, device=f'cuda:{s}'),
                       torch.empty(SIZE//4, device=f'cuda:{d}'))
        torch.cuda.set_device(s)
        streams[(s,d)] = torch.cuda.Stream(torch.device(f'cuda:{s}'))
    for s, d in pairs:
        with torch.cuda.stream(streams[(s,d)]): bufs[(s,d)][1].copy_(bufs[(s,d)][0])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        for s, d in pairs:
            with torch.cuda.stream(streams[(s,d)]): bufs[(s,d)][1].copy_(bufs[(s,d)][0])
    torch.cuda.synchronize()
    return SIZE * ITERS * len(pairs) / (time.perf_counter() - t0) / 1e9

print("=== 1 src switch → 1 dst switch (2-pair concurrent, single dst root) ===")
for src_sw, src_gpus in [("SW1",[0,1]), ("SW2",[2,3]), ("SW3",[4,5]), ("SW4",[6,7])]:
  for dst_sw, dst_gpus in [("SW1",[0,1]), ("SW2",[2,3]), ("SW3",[4,5]), ("SW4",[6,7])]:
    if src_sw == dst_sw: continue
    bw = run([(src_gpus[0], dst_gpus[0]), (src_gpus[1], dst_gpus[1])])
    print(f"  {src_sw} -> {dst_sw}:  {bw:.1f} GB/s aggregate")

print("\n=== 1 src switch → 2 dst SWITCHES (2 different dst switches) ===")
for src_sw, sg in [("SW1",[0,1]), ("SW2",[2,3]), ("SW3",[4,5]), ("SW4",[6,7])]:
  others = [(s,g) for s,g in [("SW1",[0,1]),("SW2",[2,3]),("SW3",[4,5]),("SW4",[6,7])] if s != src_sw]
  for i in range(len(others)):
    for j in range(i+1, len(others)):
      d1_name, d1_g = others[i]
      d2_name, d2_g = others[j]
      pairs = [(sg[0], d1_g[0]), (sg[1], d2_g[0])]
      bw = run(pairs)
      print(f"  {src_sw} -> {d1_name}+{d2_name}:  {bw:.1f} GB/s aggregate")

print("\n=== 4 source switches all to 1 dst switch ===")
pairs = [(2, 0), (4, 1), (6, 0), (3, 1)]
bw = run(pairs)
print(f"  4 src -> SW1: {bw:.1f} GB/s aggregate")

print("\n=== All-to-all 8 GPU ===")
all_pairs = [(s, d) for s in range(8) for d in range(8) if s != d]
bw = run(all_pairs)
print(f"  56 pairs all-to-all: {bw:.1f} GB/s ({bw/56:.2f} per pair)")
