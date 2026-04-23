#!/usr/bin/env python3
"""End-to-end streaming benchmark that measures per-token interarrival time
and aggregate tok/s across a generation against a vLLM OpenAI-compatible
server.  Designed to surface eagle3/MTP overhead at long context.

Usage:
  python e2e_bench.py --port 5002 -c 30000 --max-tokens 500 --label mtp-on
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass

import requests


DEFAULT_PROMPT = (
    "Write a detailed explanation of how TCP fast retransmit works. "
    "Keep it technical and concise, with specific RFC references."
)


def make_padding(n_tokens: int) -> str:
    if n_tokens <= 0:
        return ""
    # rough english-token ratio, good enough since server returns precise usage
    return ("lorem ipsum dolor sit amet consectetur adipiscing elit " * 1000)[
        : n_tokens * 4
    ]


def run_once(args):
    url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    padding = make_padding(args.context_tokens)
    messages = []
    if padding:
        messages.append(
            {
                "role": "user",
                "content": f"Context to keep in mind:\n{padding}\n\nEnd of context.",
            }
        )
        messages.append(
            {"role": "assistant", "content": "Understood, I will keep that in mind."}
        )
    messages.append({"role": "user", "content": args.prompt or DEFAULT_PROMPT})

    payload = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
    }

    headers = {"Content-Type": "application/json"}
    t_start = time.perf_counter()
    t_first = None
    last_server_tokens = 0
    last_wall = None
    inter_deltas_ms = []
    server_step_deltas = []
    resp = requests.post(url, headers=headers, json=payload, stream=True, timeout=600)
    resp.raise_for_status()
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data:"):
            line = line[len("data:") :].strip()
        if line == "[DONE]":
            break
        try:
            obj = json.loads(line)
        except Exception:
            continue
        now = time.perf_counter()
        choices = obj.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {}) or {}
            if (
                delta.get("content")
                or delta.get("reasoning_content")
                or delta.get("reasoning")
            ):
                if t_first is None:
                    t_first = now
                if last_wall is not None:
                    inter_deltas_ms.append((now - last_wall) * 1000.0)
                last_wall = now
        usage = obj.get("usage") or {}
        completion_tokens = usage.get("completion_tokens")
        if completion_tokens is not None:
            dt = completion_tokens - last_server_tokens
            if dt > 0 and last_wall is not None:
                server_step_deltas.append((now - last_wall, dt))
            last_server_tokens = completion_tokens
    t_end = time.perf_counter()

    total_tokens = last_server_tokens
    ttft = (t_first - t_start) if t_first else float("nan")
    decode_time = (t_end - t_first) if t_first else float("nan")
    tok_s = total_tokens / decode_time if decode_time > 0 else float("nan")
    ttft_ms = ttft * 1000.0
    inter_deltas_ms.sort()
    p50 = statistics.median(inter_deltas_ms) if inter_deltas_ms else float("nan")
    p90 = (
        inter_deltas_ms[int(len(inter_deltas_ms) * 0.9)]
        if inter_deltas_ms
        else float("nan")
    )
    p99 = (
        inter_deltas_ms[int(len(inter_deltas_ms) * 0.99)]
        if inter_deltas_ms
        else float("nan")
    )

    line_out = (
        f"{args.label:>20s}  ctx={args.context_tokens:>6d}  max={args.max_tokens:>5d}  "
        f"total={total_tokens:>5d}  "
        f"ttft={ttft_ms:>8.1f}ms  decode={decode_time:>7.2f}s  "
        f"tok/s={tok_s:>6.1f}  "
        f"interarrival p50/p90/p99 ms = {p50:>6.2f}/{p90:>6.2f}/{p99:>6.2f}"
    )
    print(line_out)
    return {
        "label": args.label,
        "ctx": args.context_tokens,
        "max_tokens": args.max_tokens,
        "total_tokens": total_tokens,
        "ttft_ms": ttft_ms,
        "decode_s": decode_time,
        "tok_s": tok_s,
        "interarrival_ms_p50": p50,
        "interarrival_ms_p90": p90,
        "interarrival_ms_p99": p99,
        "interarrival_ms_all": inter_deltas_ms,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=5002)
    p.add_argument("-c", "--context-tokens", type=int, default=30000)
    p.add_argument("--max-tokens", type=int, default=500)
    p.add_argument("--model", default="Kimi-K2.5")
    p.add_argument("--label", default="run")
    p.add_argument("--prompt", default=None)
    p.add_argument("--json-out", default="")
    args = p.parse_args()
    result = run_once(args)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
