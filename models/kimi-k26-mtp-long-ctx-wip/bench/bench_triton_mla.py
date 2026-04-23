#!/usr/bin/env python3
"""Microbenchmark the TRITON_MLA decode kernel in isolation.

Goal: reproduce the "slow at long context" symptom without loading the full
model. Calls decode_attention_fwd directly with a synthetic FP8 paged KV
cache and measures per-call latency as a function of seq_len and
num_kv_splits.

Run inside the voipmonitor/vllm:cu130 container. Uses single GPU (no TP)
since attention is per-rank anyway and we only care about kernel latency.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import triton

# vllm paths
sys.path.insert(0, "/opt/vllm")
from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd  # noqa


# Kimi-K2.6 / DeepSeek-V3.2 MLA shapes (per TP rank assumption)
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
Lk_MLA = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
Lv_MLA = KV_LORA_RANK  # 512
Q_HEADS_PER_RANK_DEFAULT = 16  # 128 heads / TP=8. Kimi-K2.6 likely same.


def alloc_paged_kv_cache(num_blocks: int, block_size: int, dtype: torch.dtype):
    """Single-head MLA KV cache: (num_blocks, block_size, 1, KV_LORA+PE)."""
    kv = torch.empty(
        (num_blocks, block_size, Lk_MLA),
        dtype=dtype,
        device="cuda",
    )
    # Random fp8 init is unstable; fill with small values.
    if dtype == torch.float8_e4m3fn:
        tmp = torch.randn(kv.shape, device="cuda", dtype=torch.bfloat16) * 0.1
        kv.copy_(tmp.to(dtype))
    else:
        kv.normal_(mean=0.0, std=0.1)
    return kv  # [num_blocks, block_size, Lk]  (will unsqueeze head later)


def build_block_table(num_reqs: int, seq_len: int, block_size: int):
    blocks_per_req = triton.cdiv(seq_len, block_size)
    # simple contiguous layout: req i uses blocks [i*bp, (i+1)*bp)
    bt = torch.zeros((num_reqs, blocks_per_req), dtype=torch.int32, device="cuda")
    for i in range(num_reqs):
        bt[i] = torch.arange(
            i * blocks_per_req,
            (i + 1) * blocks_per_req,
            dtype=torch.int32,
            device="cuda",
        )
    return bt


def compute_num_kv_splits(
    max_seq_len: int,
    sm_count: int,
    min_work_per_split: int = 512,
    occupancy_multiplier: int = 2,
    sm120_fp8_cap: int = 128,
    sm120_fp8: bool = True,
    batch: int = 1,
):
    ideal_splits = max(1, max_seq_len // min_work_per_split)
    ideal_splits = triton.next_power_of_2(ideal_splits)
    max_splits = sm_count * occupancy_multiplier
    if batch == 1 and sm120_fp8:
        max_splits = min(max_splits, sm120_fp8_cap)
    return min(ideal_splits, max_splits)


def bench_one(
    seq_len: int,
    batch: int,
    q_heads: int,
    block_size: int,
    dtype_str: str,
    num_kv_splits: int,
    warmup: int,
    iters: int,
):
    assert dtype_str in ("fp8", "bf16")
    kv_dtype = torch.float8_e4m3fn if dtype_str == "fp8" else torch.bfloat16
    q_dtype = torch.bfloat16  # Triton kernel uses BF16 queries with FP8 KV

    blocks_per_req = triton.cdiv(seq_len, block_size)
    num_blocks = batch * blocks_per_req + 4
    kv = alloc_paged_kv_cache(num_blocks, block_size, kv_dtype)  # [N, B, Lk]

    # Match triton_mla.py: unsqueeze head dim to 1
    kv_c_and_k_pe_cache = kv.unsqueeze(2)  # [N, B, 1, Lk]
    kv_c_cache = kv_c_and_k_pe_cache[..., :KV_LORA_RANK]  # [N, B, 1, 512]

    q = torch.randn((batch, q_heads, Lk_MLA), dtype=q_dtype, device="cuda") * 0.1
    o = torch.zeros(
        (batch, q_heads, KV_LORA_RANK), dtype=q_dtype, device="cuda"
    )
    lse = torch.zeros((batch, q_heads), dtype=q_dtype, device="cuda")

    block_table = build_block_table(batch, seq_len, block_size)
    seq_lens = torch.full((batch,), seq_len, dtype=torch.int32, device="cuda")

    # attn_logits: [B, Hq, splits, kv_lora_rank+1]
    attn_logits = torch.empty(
        (batch, q_heads, num_kv_splits, KV_LORA_RANK + 1),
        dtype=torch.float32,
        device="cuda",
    )

    sm_scale = 1.0 / (Lk_MLA ** 0.5)
    k_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    def call():
        decode_attention_fwd(
            q,
            kv_c_and_k_pe_cache,
            kv_c_cache,
            o,
            lse,
            block_table,
            seq_lens,
            attn_logits,
            num_kv_splits,
            sm_scale,
            block_size,
            k_scale=k_scale,
            v_scale=k_scale,
            is_mla=True,
        )

    # warmup
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()

    evts_start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    evts_end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        evts_start[i].record()
        call()
        evts_end[i].record()
    torch.cuda.synchronize()
    times_ms = [s.elapsed_time(e) for s, e in zip(evts_start, evts_end)]
    return times_ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-lens", type=str, default="1000,5000,10000,20000,30000,50000,100000")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--q-heads", type=int, default=Q_HEADS_PER_RANK_DEFAULT)
    p.add_argument("--block-size", type=int, default=64)  # vLLM Kimi default
    p.add_argument("--dtype", type=str, default="fp8", choices=["fp8", "bf16"])
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--splits", type=str, default="auto",
                   help="'auto' uses heuristic, else comma-separated ints")
    p.add_argument("--json-out", type=str, default="")
    args = p.parse_args()

    torch.cuda.init()
    torch.manual_seed(0)

    seq_lens = [int(s) for s in args.seq_lens.split(",") if s]

    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    device_name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    sm120 = (cc[0] * 10 + cc[1]) >= 120

    print(f"# device: {device_name} (sm{cc[0]*10+cc[1]}, {sm_count} SMs)")
    print(f"# batch={args.batch} q_heads={args.q_heads} block_size={args.block_size} dtype={args.dtype}")
    print(
        f"# {'seq':>7}  {'splits':>6}  {'median_ms':>10}  {'min_ms':>10}  "
        f"{'p95_ms':>10}  {'hbm_est_GB/s':>13}"
    )

    rows = []
    for seq in seq_lens:
        if args.splits == "auto":
            splits_list = [compute_num_kv_splits(seq, sm_count, batch=args.batch, sm120_fp8=sm120)]
        else:
            splits_list = [int(s) for s in args.splits.split(",")]
        for splits in splits_list:
            try:
                times = bench_one(
                    seq_len=seq,
                    batch=args.batch,
                    q_heads=args.q_heads,
                    block_size=args.block_size,
                    dtype_str=args.dtype,
                    num_kv_splits=splits,
                    warmup=args.warmup,
                    iters=args.iters,
                )
            except Exception as e:
                print(f"  seq={seq} splits={splits} FAILED: {type(e).__name__}: {e}")
                continue
            times_sorted = sorted(times)
            median = times_sorted[len(times_sorted) // 2]
            mn = times_sorted[0]
            p95 = times_sorted[int(len(times_sorted) * 0.95)]
            bytes_per_call = seq * args.batch * Lk_MLA * (1 if args.dtype == "fp8" else 2)
            hbm = bytes_per_call / (median * 1e-3) / 1e9
            print(f"  {seq:>7}  {splits:>6}  {median:>10.3f}  {mn:>10.3f}  {p95:>10.3f}  {hbm:>13.1f}")
            rows.append(
                dict(seq=seq, splits=splits, median_ms=median, min_ms=mn, p95_ms=p95, hbm_GBps=hbm)
            )

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({
                "device": device_name,
                "sm": cc[0] * 10 + cc[1],
                "sm_count": sm_count,
                "batch": args.batch,
                "q_heads": args.q_heads,
                "block_size": args.block_size,
                "dtype": args.dtype,
                "rows": rows,
            }, f, indent=2)


if __name__ == "__main__":
    main()
