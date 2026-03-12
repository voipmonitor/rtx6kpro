#!/usr/bin/env python3
"""
sglang Inference Benchmark with Rich TUI Dashboard.

Measures decode throughput across a matrix of concurrency levels and context lengths.
Connects to a running sglang server via OpenAI-compatible API.

Usage:
    python3 benchmark_sglang.py
    python3 benchmark_sglang.py --concurrency 1,2,4 --contexts 0,16384 --duration 10
    python3 benchmark_sglang.py --port 5001 --max-tokens 4096
"""

import argparse
import asyncio
import json
import random
import re
import string
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from statistics import mean, median
from typing import Optional

import httpx
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHARS_PER_TOKEN = 4

PADDING_SENTENCES = [
    "The history of European architecture spans thousands of years and encompasses a wide variety of styles and movements.",
    "From the ancient Greek temples to the Gothic cathedrals of the Middle Ages, each era has left its distinctive mark on the built environment.",
    "The Renaissance brought a renewed interest in classical forms, while the Baroque period introduced dramatic ornamentation and grandeur.",
    "In the modern era, architects have experimented with new materials such as steel, glass, and reinforced concrete.",
    "The development of skyscrapers in the late 19th century transformed urban landscapes around the world.",
    "Sustainable architecture has become increasingly important as societies grapple with climate change and resource depletion.",
    "The principles of good design include functionality, durability, and aesthetic appeal.",
    "Urban planning plays a crucial role in shaping how cities develop and how their inhabitants experience daily life.",
    "Public spaces such as parks, plazas, and waterfronts contribute significantly to the quality of urban living.",
    "The integration of technology into building design has opened up new possibilities for energy efficiency and comfort.",
    "Historical preservation efforts seek to maintain the cultural heritage embodied in older structures.",
    "The relationship between architecture and nature has been explored by many influential designers throughout history.",
    "Building codes and regulations ensure that structures meet minimum standards for safety and accessibility.",
    "The choice of materials in construction affects not only the appearance of a building but also its environmental impact.",
    "Innovative structural engineering techniques have made it possible to create buildings of unprecedented scale and complexity.",
    "The study of vernacular architecture reveals how different cultures have adapted their building practices to local conditions.",
    "Interior design complements architecture by addressing the arrangement and decoration of interior spaces.",
    "Landscape architecture deals with the design of outdoor areas, landmarks, and structures to achieve environmental or aesthetic outcomes.",
    "The concept of smart cities integrates information technology with urban infrastructure to improve efficiency and quality of life.",
    "Affordable housing remains one of the most pressing challenges facing urban planners and policymakers worldwide.",
]

GENERATION_PROMPT = (
    "Write an extremely detailed, comprehensive encyclopedia article about the complete "
    "history of mathematics from ancient Mesopotamia to 2025. Cover every civilization, "
    "every major mathematician, every theorem, proof, and breakthrough. Include detailed "
    "biographical information, historical context, and mathematical explanations. "
    "Do not summarize - provide maximum detail on every topic."
)

METRIC_RE = re.compile(r'^(sglang:\w+)(?:\{([^}]*)\})?\s+([\d.eE+-]+)')


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StreamResult:
    ttft: float = 0.0
    total_tokens: int = 0
    total_time: float = 0.0
    tokens_per_sec: float = 0.0
    error: Optional[str] = None


@dataclass
class CellResult:
    concurrency: int = 0
    context_tokens: int = 0
    aggregate_tps: float = 0.0
    per_request_avg_tps: float = 0.0
    ttft_avg: float = 0.0
    ttft_p50: float = 0.0
    ttft_p99: float = 0.0
    total_tokens: int = 0
    wall_time: float = 0.0
    num_completed: int = 0
    num_errors: int = 0
    server_gen_throughput: float = 0.0
    server_utilization: float = 0.0
    server_spec_accept_rate: float = 0.0


@dataclass
class TUIState:
    # Overall
    model_name: str = ""
    server_url: str = ""
    total_tests: int = 0
    completed_tests: int = 0
    overall_start: float = 0.0
    # Current cell
    current_concurrency: int = 0
    current_context: int = 0
    cell_start: float = 0.0
    cell_duration: float = 20.0
    cell_tokens: int = 0
    cell_live_tps: float = 0.0
    cell_running: bool = False
    # Server metrics
    srv_gen_throughput: float = 0.0
    srv_running_reqs: int = 0
    srv_queue_reqs: int = 0
    srv_utilization: float = 0.0
    srv_spec_accept_rate: float = 0.0
    srv_spec_accept_length: float = 0.0
    # Results
    results: dict = field(default_factory=dict)  # (ctx, conc) -> aggregate_tps
    errors: dict = field(default_factory=dict)   # (ctx, conc) -> num_errors
    concurrency_levels: list = field(default_factory=list)
    context_lengths: list = field(default_factory=list)
    # Prefill results: ctx -> {ttft, tok_per_sec}
    prefill_results: dict = field(default_factory=dict)
    prefill_contexts: list = field(default_factory=list)
    prefill_phase: bool = False
    # Server limits
    kv_cache_budget: int = 0
    max_running_requests: int = 0
    skipped_cells: int = 0
    max_tokens: int = 0
    # Timing
    cell_times: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_padding_text(target_tokens: int) -> str:
    target_chars = target_tokens * CHARS_PER_TOKEN
    lines = []
    current_chars = 0
    idx = 0
    while current_chars < target_chars:
        sentence = PADDING_SENTENCES[idx % len(PADDING_SENTENCES)]
        lines.append(sentence)
        current_chars += len(sentence) + 1
        idx += 1
    return " ".join(lines)


def build_messages(context_tokens: int, context_text: str) -> list:
    messages = []
    if context_tokens > 0 and context_text:
        messages.append({
            "role": "user",
            "content": (
                "Below is a large reference document. Read it carefully, "
                "then answer the question that follows.\n\n"
                "--- BEGIN REFERENCE DOCUMENT ---\n"
                f"{context_text}\n"
                "--- END REFERENCE DOCUMENT ---"
            )
        })
        messages.append({
            "role": "assistant",
            "content": "I have read the entire reference document. Please ask your question."
        })
    messages.append({"role": "user", "content": GENERATION_PROMPT})
    return messages


def percentile(data: list, p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def format_context(ctx: int) -> str:
    if ctx == 0:
        return "0"
    elif ctx >= 1024:
        return f"{ctx // 1024}k"
    return str(ctx)


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


# ---------------------------------------------------------------------------
# Metrics scraping
# ---------------------------------------------------------------------------

async def scrape_metrics(client: httpx.AsyncClient, base_url: str) -> dict:
    metrics = {}
    try:
        resp = await client.get(f"{base_url}/metrics", timeout=5.0)
        for line in resp.text.splitlines():
            if line.startswith("#"):
                continue
            m = METRIC_RE.match(line)
            if m:
                name, labels, value = m.group(1), m.group(2) or "", float(m.group(3))
                # Only take tp_rank=0 metrics to avoid duplicates
                if "tp_rank=" in labels and 'tp_rank="0"' not in labels:
                    continue
                key = f"{name}|{labels}" if labels else name
                metrics[key] = value
    except Exception:
        pass
    return metrics


def extract_metric(metrics: dict, name: str, label_filter: str = "") -> float:
    for key, val in metrics.items():
        if key.startswith(name):
            if label_filter and label_filter not in key:
                continue
            return val
    return 0.0


# ---------------------------------------------------------------------------
# Streaming request
# ---------------------------------------------------------------------------

async def stream_one_request(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
    index: int,
    cancel_event: asyncio.Event,
    shared_token_count: list,
) -> StreamResult:
    result = StreamResult()
    t_start = time.monotonic()
    t_first = None
    char_count = 0
    usage_tokens = None

    try:
        async with client.stream("POST", url, json=payload, timeout=httpx.Timeout(600.0, connect=30.0)) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                result.error = f"HTTP {resp.status_code}: {body.decode()[:200]}"
                result.total_time = time.monotonic() - t_start
                return result

            async for line in resp.aiter_lines():
                if cancel_event.is_set():
                    break

                if not line or not line.startswith("data: "):
                    continue

                data_str = line[6:]
                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Check for usage in final chunk (stream_options.include_usage)
                usage = data.get("usage")
                if usage and "completion_tokens" in usage:
                    usage_tokens = usage["completion_tokens"]

                if "choices" not in data or len(data["choices"]) == 0:
                    continue

                delta = data["choices"][0].get("delta", {})
                text = ""

                reasoning = delta.get("reasoning") or delta.get("reasoning_content")
                if reasoning:
                    text += reasoning

                content = delta.get("content")
                if content:
                    text += content

                if text:
                    if t_first is None:
                        t_first = time.monotonic()
                    chars = len(text)
                    char_count += chars
                    # Estimate tokens from chars for live display
                    # (MTP batches multiple tokens per SSE event)
                    estimated_new = max(1, round(chars / CHARS_PER_TOKEN))
                    shared_token_count[0] += estimated_new

    except httpx.ReadTimeout:
        result.error = "ReadTimeout"
    except httpx.ConnectError as e:
        result.error = f"ConnectError: {e}"
    except httpx.RemoteProtocolError as e:
        result.error = f"ProtocolError: {e}"
    except asyncio.CancelledError:
        pass
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"

    t_end = time.monotonic()
    # Use server-reported usage if available, else estimate from chars
    if usage_tokens is not None:
        result.total_tokens = usage_tokens
    else:
        result.total_tokens = max(1, round(char_count / CHARS_PER_TOKEN)) if char_count > 0 else 0
    result.total_time = t_end - t_start
    if t_first is not None:
        result.ttft = t_first - t_start
    if result.total_tokens > 0 and result.total_time > 0:
        result.tokens_per_sec = result.total_tokens / result.total_time

    return result


# ---------------------------------------------------------------------------
# Run one cell
# ---------------------------------------------------------------------------

async def run_one_cell(
    client: httpx.AsyncClient,
    base_url: str,
    concurrency: int,
    context_tokens: int,
    context_text: str,
    duration: float,
    max_tokens: int,
    model: str,
    state: TUIState,
    live: Live,
) -> CellResult:
    messages = build_messages(context_tokens, context_text)
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": max_tokens,
        "stream_options": {"include_usage": True},
    }

    url = f"{base_url}/v1/chat/completions"
    cancel_event = asyncio.Event()
    shared_token_count = [0]

    # Update TUI state
    state.current_concurrency = concurrency
    state.current_context = context_tokens
    state.cell_start = time.monotonic()
    state.cell_duration = duration
    state.cell_tokens = 0
    state.cell_live_tps = 0.0
    state.cell_running = True

    # Launch all streams
    tasks = [
        asyncio.create_task(
            stream_one_request(client, url, payload, i, cancel_event, shared_token_count)
        )
        for i in range(concurrency)
    ]

    # Monitor loop — collect server gen_throughput samples for accurate measurement
    metrics_interval = 1.0
    warmup_seconds = 4.0  # skip early samples (CUDA graph warmup)
    last_metrics_time = 0.0
    gen_throughput_samples = []

    while True:
        await asyncio.sleep(0.5)
        now = time.monotonic()
        elapsed = now - state.cell_start

        # Update token counts from client-side estimate (for TUI only)
        state.cell_tokens = shared_token_count[0]

        # Scrape server metrics periodically
        if now - last_metrics_time > metrics_interval:
            metrics = await scrape_metrics(client, base_url)
            state.srv_gen_throughput = extract_metric(metrics, "sglang:gen_throughput")
            state.srv_running_reqs = int(extract_metric(metrics, "sglang:num_running_reqs"))
            state.srv_queue_reqs = int(extract_metric(metrics, "sglang:num_queue_reqs"))
            state.srv_utilization = extract_metric(metrics, "sglang:utilization")
            state.srv_spec_accept_rate = extract_metric(metrics, "sglang:spec_accept_rate")
            state.srv_spec_accept_length = extract_metric(metrics, "sglang:spec_accept_length")
            # Collect gen_throughput samples after warmup (skip ramp-up period)
            if state.srv_gen_throughput > 0 and elapsed > warmup_seconds:
                gen_throughput_samples.append(state.srv_gen_throughput)
            last_metrics_time = now

        # Use server gen_throughput for live display
        state.cell_live_tps = state.srv_gen_throughput

        # Update TUI
        live.update(build_display(state))

        # Check duration
        if elapsed >= duration:
            cancel_event.set()
            break

        # Check if all tasks already done
        if all(t.done() for t in tasks):
            break

    # Wait for tasks to finish (grace period)
    done, pending = await asyncio.wait(tasks, timeout=30.0)
    for t in pending:
        t.cancel()
    if pending:
        await asyncio.wait(pending, timeout=5.0)

    # Collect results
    wall_time = time.monotonic() - state.cell_start
    stream_results = []
    for t in tasks:
        try:
            stream_results.append(t.result())
        except (asyncio.CancelledError, Exception):
            stream_results.append(StreamResult(error="cancelled", total_time=wall_time))

    # Final metrics scrape
    metrics = await scrape_metrics(client, base_url)
    final_gen_throughput = extract_metric(metrics, "sglang:gen_throughput")
    if final_gen_throughput > 0:
        gen_throughput_samples.append(final_gen_throughput)

    # Use server-side gen_throughput as the primary metric (median for robustness)
    avg_gen_throughput = median(gen_throughput_samples) if gen_throughput_samples else 0.0

    # Client-side stats
    successful = [r for r in stream_results if r.error is None]
    total_tokens = sum(r.total_tokens for r in stream_results)
    # Derive per-request from server metric for consistency (aggregate / concurrency)
    per_req_tps = avg_gen_throughput / concurrency if concurrency > 0 else 0.0
    ttfts = [r.ttft for r in successful if r.ttft > 0]

    cell = CellResult(
        concurrency=concurrency,
        context_tokens=context_tokens,
        aggregate_tps=avg_gen_throughput,
        per_request_avg_tps=per_req_tps,
        ttft_avg=mean(ttfts) if ttfts else 0.0,
        ttft_p50=percentile(ttfts, 50) if ttfts else 0.0,
        ttft_p99=percentile(ttfts, 99) if ttfts else 0.0,
        total_tokens=total_tokens,
        wall_time=wall_time,
        num_completed=len(successful),
        num_errors=len(stream_results) - len(successful),
        server_gen_throughput=avg_gen_throughput,
        server_utilization=extract_metric(metrics, "sglang:utilization"),
        server_spec_accept_rate=extract_metric(metrics, "sglang:spec_accept_rate"),
    )

    state.cell_running = False
    state.results[(context_tokens, concurrency)] = cell.aggregate_tps
    state.errors[(context_tokens, concurrency)] = cell.num_errors

    return cell


# ---------------------------------------------------------------------------
# TUI rendering
# ---------------------------------------------------------------------------

def build_display(state: TUIState) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="middle", size=10),
        Layout(name="results", ratio=1, minimum_size=8),
        Layout(name="footer", size=3),
    )
    layout["middle"].split_row(
        Layout(name="current_test", ratio=1),
        Layout(name="server_metrics", ratio=1),
    )

    # Header
    header_text = Text()
    header_text.append("sglang Benchmark", style="bold cyan")
    header_text.append(f"  {state.model_name} @ {state.server_url}")
    header_text.append(f"  |  {state.total_tests} tests  |  {state.cell_duration:.0f}s each")
    if state.kv_cache_budget > 0 or state.max_running_requests > 0:
        header_text.append("  |  ", style="dim")
        if state.kv_cache_budget > 0:
            header_text.append(f"KV: {state.kv_cache_budget:,}", style="cyan")
        if state.max_running_requests > 0:
            if state.kv_cache_budget > 0:
                header_text.append("  ", style="dim")
            header_text.append(f"MaxReqs: {state.max_running_requests}", style="cyan")
        if state.skipped_cells > 0:
            header_text.append(f"  ({state.skipped_cells} skipped)", style="yellow")
    layout["header"].update(Panel(header_text, style="bold"))

    # Current test panel
    if state.cell_running:
        elapsed = time.monotonic() - state.cell_start
        if state.prefill_phase:
            cell_text = (
                f"[bold magenta]PREFILL[/bold magenta]  ctx={format_context(state.current_context)}\n"
                f"  Elapsed: {elapsed:.1f}s\n"
                f"  Populating radix cache...\n"
                f"  Test [bold]{state.completed_tests + 1}[/bold] of {state.total_tests}"
            )
        else:
            pct = min(elapsed / state.cell_duration, 1.0) if state.cell_duration > 0 else 0
            bar_width = 30
            filled = int(pct * bar_width)
            bar = "[green]" + "=" * filled + ">" + "[/green]" + " " * (bar_width - filled - 1)

            cell_text = (
                f"[bold]DECODE[/bold]  C={state.current_concurrency}, ctx={format_context(state.current_context)}\n"
                f"  [{bar}] {elapsed:.0f}/{state.cell_duration:.0f}s\n"
                f"  Server: [bold yellow]{state.cell_live_tps:.1f}[/bold yellow] tok/s (gen_throughput)\n"
                f"  Test [bold]{state.completed_tests + 1}[/bold] of {state.total_tests}"
            )
    else:
        cell_text = "[dim]Waiting...[/dim]"
    layout["current_test"].update(Panel(cell_text, title="Current Test", border_style="cyan"))

    # Server metrics panel
    srv_table = Table(show_header=False, box=None, padding=(0, 1))
    srv_table.add_column("Metric", style="dim")
    srv_table.add_column("Value", style="bold")
    srv_table.add_row("gen_throughput", f"{state.srv_gen_throughput:.1f} tok/s")
    srv_table.add_row("running_reqs", str(state.srv_running_reqs))
    srv_table.add_row("queue_reqs", str(state.srv_queue_reqs))
    srv_table.add_row("utilization", f"{state.srv_utilization:.2%}")
    srv_table.add_row("spec_accept_rate", f"{state.srv_spec_accept_rate:.2%}")
    srv_table.add_row("spec_accept_len", f"{state.srv_spec_accept_length:.1f}")
    layout["server_metrics"].update(Panel(srv_table, title="Server Metrics", border_style="magenta"))

    # Results table
    results_table = Table(title="Aggregate Throughput (tok/s)", border_style="green", expand=True)
    results_table.add_column("ctx \\ conc", style="bold cyan", min_width=8)
    for conc in state.concurrency_levels:
        results_table.add_column(str(conc), justify="right", min_width=7)

    # Determine color thresholds from existing results (exclude skipped=-1)
    all_values = [v for v in state.results.values() if v > 0]
    if all_values:
        p25 = percentile(all_values, 25)
        p75 = percentile(all_values, 75)
    else:
        p25, p75 = 0, 0

    for ctx in state.context_lengths:
        row = [format_context(ctx)]
        for conc in state.concurrency_levels:
            key = (ctx, conc)
            if key in state.results:
                val = state.results[key]
                if val < 0:
                    needed = conc * (ctx + state.max_tokens)
                    row.append(f"[dim]N/A ({needed // 1024}k)[/dim]")
                    continue
                errs = state.errors.get(key, 0)
                if val > p75 and p75 > 0:
                    style = "bold green"
                elif val < p25 and p25 > 0:
                    style = "red"
                else:
                    style = "yellow"
                cell = f"{val:.1f}"
                if errs > 0:
                    cell += f" [red]({errs}e)[/red]"
                row.append(f"[{style}]{cell}[/{style}]")
            else:
                row.append("[dim]...[/dim]")
        results_table.add_row(*row)

    # Prefill table (shown alongside decode results)
    if state.prefill_contexts:
        prefill_table = Table(title="Prefill Speed (C=1)", border_style="magenta", expand=True)
        prefill_table.add_column("Context", style="bold cyan", min_width=6)
        prefill_table.add_column("TTFT", justify="right", min_width=6)
        prefill_table.add_column("Prefill", justify="right", min_width=6)
        prefill_table.add_column("tok/s", justify="right", min_width=8)
        for ctx in state.prefill_contexts:
            if ctx in state.prefill_results:
                pr = state.prefill_results[ctx]
                prefill_table.add_row(
                    format_context(ctx),
                    f"{pr['ttft']:.2f}s",
                    f"{pr.get('prefill_time', pr['ttft']):.2f}s",
                    f"[bold green]{pr['tok_per_sec']:,.0f}[/bold green]",
                )
            else:
                prefill_table.add_row(format_context(ctx), "[dim]...[/dim]", "[dim]...[/dim]", "[dim]...[/dim]")

        results_layout = Layout()
        results_layout.split_row(
            Layout(Panel(prefill_table), ratio=1),
            Layout(Panel(results_table), ratio=3),
        )
        layout["results"].update(results_layout)
    else:
        layout["results"].update(Panel(results_table))

    # Footer - overall progress
    if state.total_tests > 0:
        overall_pct = state.completed_tests / state.total_tests
        elapsed_total = time.monotonic() - state.overall_start if state.overall_start > 0 else 0
        if state.cell_times:
            avg_cell = mean(state.cell_times)
            remaining = (state.total_tests - state.completed_tests) * avg_cell
            eta_str = format_time(remaining)
        else:
            eta_str = "calculating..."

        bar_width = 50
        filled = int(overall_pct * bar_width)
        bar = "[green]" + "=" * filled + ">" + "[/green]" + " " * max(0, bar_width - filled - 1)
        footer_text = (
            f"  [{bar}]  "
            f"{state.completed_tests}/{state.total_tests}  "
            f"Elapsed: {format_time(elapsed_total)}  "
            f"ETA: {eta_str}"
        )
    else:
        footer_text = "Initializing..."
    layout["footer"].update(Panel(footer_text, style="bold"))

    return layout


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

async def run_benchmark(args):
    concurrency_levels = [int(x) for x in args.concurrency.split(",")]
    context_lengths = [int(x) for x in args.contexts.split(",")]
    base_url = f"http://{args.host}:{args.port}"
    console = Console()

    # --- Step 1: Connect to server and read limits ---
    server_context_length = 0
    max_running = None
    async with httpx.AsyncClient() as check_client:
        try:
            resp = await check_client.get(f"{base_url}/v1/models", timeout=10.0)
            models = resp.json()
            console.print(f"[green]Server OK.[/green] Models: {[m['id'] for m in models.get('data', [])]}")
        except Exception as e:
            console.print(f"[red]Cannot connect to server at {base_url}: {e}[/red]")
            console.print("Make sure sglang is running and the port is correct.")
            return [], {}

        try:
            resp = await check_client.get(f"{base_url}/get_server_info", timeout=10.0)
            server_info = resp.json()
            if args.max_total_tokens == 0:
                kv_budget = server_info.get("max_total_num_tokens", 0)
                if kv_budget:
                    args.max_total_tokens = int(kv_budget)
                    console.print(f"[cyan]KV cache budget:[/cyan] {args.max_total_tokens:,} tokens")
            max_running = server_info.get("max_running_requests")
            if max_running:
                max_running = int(max_running)
                over = [c for c in concurrency_levels if c > max_running]
                if over:
                    concurrency_levels = [c for c in concurrency_levels if c <= max_running]
                    console.print(f"[cyan]Max running requests:[/cyan] {max_running} (dropped C={','.join(str(c) for c in over)})")
            server_context_length = server_info.get("context_length") or 0
            if server_context_length:
                console.print(f"[cyan]Model context length:[/cyan] {server_context_length:,} tokens")
        except Exception:
            console.print("[yellow]Could not read /get_server_info. No auto-detection of limits.[/yellow]")

    # --- Step 2: Build prefill context list (up to 200k or model limit) ---
    PREFILL_CANDIDATES = [8192, 16384, 32768, 65536, 131072]
    max_prefill = min(131072, server_context_length - 64) if server_context_length > 0 else 131072
    prefill_contexts = [c for c in PREFILL_CANDIDATES if c <= max_prefill]
    if not prefill_contexts and max_prefill > 0:
        prefill_contexts = [max_prefill]
    console.print(f"[cyan]Prefill tests:[/cyan] {[format_context(c) for c in prefill_contexts]}")

    # --- Step 3: Generate unique padding text per context length ---
    # Each context gets a unique prefix so radix cache cannot match across lengths.
    # Within same length, same text is reused → decode phase gets cache hit.
    all_ctx_sizes = sorted(set(prefill_contexts + [c for c in context_lengths if c > 0]))
    max_ctx = max(all_ctx_sizes) if all_ctx_sizes else 0
    context_cache = {}
    run_id = ''.join(random.choices(string.ascii_lowercase, k=12))
    if max_ctx > 0:
        console.print(f"[bold]Generating padding texts (run={run_id}, up to {format_context(max_ctx)})...[/bold]")
        base_text = generate_padding_text(max_ctx)
        for ctx in all_ctx_sizes:
            # Unique prefix per run + context length → no cross-run or cross-length cache hits
            prefix = f"[BENCH_{run_id}_CTX_{ctx}] "
            target_chars = ctx * CHARS_PER_TOKEN
            text = prefix + base_text
            context_cache[ctx] = text[:target_chars]
    context_cache[0] = ""
    console.print("[green]Done.[/green]\n")

    # --- Step 4: Initialize TUI state ---
    state = TUIState(
        model_name=args.model,
        server_url=f"{args.host}:{args.port}",
        total_tests=len(concurrency_levels) * len(context_lengths),
        concurrency_levels=concurrency_levels,
        context_lengths=context_lengths,
        overall_start=time.monotonic(),
    )
    if max_running:
        state.max_running_requests = int(max_running)
    state.max_tokens = args.max_tokens
    state.prefill_contexts = prefill_contexts

    # Mark skipped decode cells
    if args.max_total_tokens > 0:
        state.kv_cache_budget = args.max_total_tokens
        runnable = sum(
            1 for ctx in context_lengths for conc in concurrency_levels
            if conc * (ctx + args.max_tokens) <= args.max_total_tokens
        )
        skipped = state.total_tests - runnable
        state.skipped_cells = skipped
        for ctx in context_lengths:
            for conc in concurrency_levels:
                if conc * (ctx + args.max_tokens) > args.max_total_tokens:
                    state.results[(ctx, conc)] = -1

    # Add prefill tests to total count
    state.total_tests += len(prefill_contexts)

    # Run benchmark
    global _partial_results
    all_results = []
    max_conc = max(concurrency_levels)
    limits = httpx.Limits(max_connections=max_conc + 20, max_keepalive_connections=max_conc + 10)

    async def measure_ttft(client, messages):
        """Send one streaming request with max_tokens=1, return TTFT in seconds."""
        payload = {
            "model": args.model,
            "messages": messages,
            "stream": True,
            "max_tokens": 1,
        }
        t0 = time.monotonic()
        try:
            async with client.stream(
                "POST", f"{base_url}/v1/chat/completions",
                json=payload,
                timeout=httpx.Timeout(600.0, connect=30.0),
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    if "choices" in data and len(data["choices"]) > 0:
                        delta = data["choices"][0].get("delta", {})
                        if delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content"):
                            return time.monotonic() - t0
        except Exception:
            pass
        return time.monotonic() - t0

    async with httpx.AsyncClient(limits=limits) as client:
        with Live(build_display(state), refresh_per_second=2, console=console) as live:

            # === Phase 1: Prefill benchmark (C=1, max_tokens=1) ===
            # First measure baseline TTFT (ctx=0) to subtract overhead
            state.prefill_phase = True
            state.current_context = 0
            state.cell_running = True
            state.cell_start = time.monotonic()
            live.update(build_display(state))

            baseline_msgs = [{"role": "user", "content": "Say OK."}]
            baseline_samples = []
            for _ in range(5):
                t = await measure_ttft(client, baseline_msgs)
                baseline_samples.append(t)
            baseline_ttft = median(baseline_samples)
            state.cell_running = False

            # Now measure each prefill context
            # Small contexts (<16k): repeat 3× and average for stability
            # Large contexts: single measurement (long enough to be accurate)
            REPEAT_THRESHOLD = 8192

            for ctx in prefill_contexts:
                state.current_concurrency = 1
                state.current_context = ctx
                state.cell_running = True
                state.cell_start = time.monotonic()
                state.cell_duration = 0
                live.update(build_display(state))

                repeats = 3 if ctx < REPEAT_THRESHOLD else 1
                ttft_samples = []
                for r in range(repeats):
                    # Each repeat needs unique text to avoid cache
                    if r == 0:
                        msgs = build_messages(ctx, context_cache[ctx])
                    else:
                        # Generate variant with different prefix for repeat runs
                        prefix = f"[BENCH_{run_id}_CTX_{ctx}_R{r}] "
                        orig_prefix_len = len(f"[BENCH_{run_id}_CTX_{ctx}] ")
                        variant_text = prefix + context_cache[ctx][orig_prefix_len:]
                        msgs = build_messages(ctx, variant_text)
                    t = await measure_ttft(client, msgs)
                    ttft_samples.append(t)

                raw_ttft = median(ttft_samples)
                # Subtract baseline to get pure prefill time
                prefill_time = max(raw_ttft - baseline_ttft, 0.001)
                tok_per_sec = ctx / prefill_time

                state.prefill_results[ctx] = {
                    "ttft": raw_ttft,
                    "prefill_time": prefill_time,
                    "tok_per_sec": tok_per_sec,
                    "baseline": baseline_ttft,
                }

                state.cell_running = False
                state.completed_tests += 1
                cell_time = time.monotonic() - state.cell_start
                state.cell_times.append(cell_time)
                live.update(build_display(state))
                await asyncio.sleep(1.0)

            # Re-cache the primary text for decode phase (repeat runs used variants)
            for ctx in prefill_contexts:
                if ctx < REPEAT_THRESHOLD:
                    msgs = build_messages(ctx, context_cache[ctx])
                    await measure_ttft(client, msgs)

            # Warm radix cache for decode contexts not already tested in prefill
            for ctx in context_lengths:
                if ctx > 0 and ctx not in prefill_contexts:
                    warmup_msgs = build_messages(ctx, context_cache[ctx])
                    warmup_payload = {
                        "model": args.model, "messages": warmup_msgs,
                        "stream": False, "max_tokens": 1,
                    }
                    try:
                        await client.post(
                            f"{base_url}/v1/chat/completions",
                            json=warmup_payload,
                            timeout=httpx.Timeout(600.0, connect=30.0),
                        )
                    except Exception:
                        pass
                    await asyncio.sleep(1.0)

            # === Phase 2: Decode benchmark (cached prefill, pure decode speed) ===
            state.prefill_phase = False
            for ctx in context_lengths:
                for conc in concurrency_levels:
                    # Skip cells that exceed token budget
                    cell_total = conc * (ctx + args.max_tokens)
                    if args.max_total_tokens > 0 and cell_total > args.max_total_tokens:
                        state.results[(ctx, conc)] = -1  # mark as skipped
                        cell = CellResult(concurrency=conc, context_tokens=ctx, aggregate_tps=-1)
                        all_results.append(cell)
                        _partial_results = all_results
                        state.completed_tests += 1
                        live.update(build_display(state))
                        continue

                    cell_start = time.monotonic()

                    try:
                        result = await run_one_cell(
                            client=client,
                            base_url=base_url,
                            concurrency=conc,
                            context_tokens=ctx,
                            context_text=context_cache[ctx],
                            duration=args.duration,
                            max_tokens=args.max_tokens,
                            model=args.model,
                            state=state,
                            live=live,
                        )
                        all_results.append(result)
                        _partial_results = all_results
                    except Exception as e:
                        console.print(f"[red]Cell C={conc} ctx={format_context(ctx)} failed: {e}[/red]")
                        cell = CellResult(concurrency=conc, context_tokens=ctx)
                        all_results.append(cell)
                        _partial_results = all_results
                        state.results[(ctx, conc)] = 0.0
                        state.errors[(ctx, conc)] = conc

                    cell_time = time.monotonic() - cell_start
                    state.cell_times.append(cell_time)
                    state.completed_tests += 1
                    live.update(build_display(state))

                    # Brief pause between cells to let server settle
                    await asyncio.sleep(2.0)

    return all_results, state.prefill_results


# ---------------------------------------------------------------------------
# Results output
# ---------------------------------------------------------------------------

def print_final_results(results: list, concurrency_levels: list, context_lengths: list,
                        console: Console, prefill_results: dict = None):
    console.print("\n")

    # Prefill table
    if prefill_results:
        baseline = next(iter(prefill_results.values()), {}).get("baseline", 0)
        pt = Table(title=f"Prefill Speed (C=1, baseline TTFT={baseline:.3f}s subtracted)",
                   border_style="magenta")
        pt.add_column("Context", style="bold cyan")
        pt.add_column("TTFT (s)", justify="right")
        pt.add_column("Prefill (s)", justify="right")
        pt.add_column("Prefill tok/s", justify="right")
        for ctx in sorted(prefill_results.keys()):
            pr = prefill_results[ctx]
            pt.add_row(
                format_context(ctx),
                f"{pr['ttft']:.2f}",
                f"{pr.get('prefill_time', pr['ttft']):.2f}",
                f"{pr['tok_per_sec']:,.0f}",
            )
        console.print(pt)
        console.print()

    # Aggregate throughput table
    table = Table(title="Aggregate Throughput (tok/s)", border_style="green")
    table.add_column("ctx \\ conc", style="bold cyan")
    for conc in concurrency_levels:
        table.add_column(str(conc), justify="right")

    result_map = {(r.context_tokens, r.concurrency): r for r in results}
    for ctx in context_lengths:
        row = [format_context(ctx)]
        for conc in concurrency_levels:
            r = result_map.get((ctx, conc))
            if r and r.aggregate_tps < 0:
                row.append("skip")
            elif r:
                val = f"{r.aggregate_tps:.1f}"
                if r.num_errors > 0:
                    val += f" ({r.num_errors}e)"
                row.append(val)
            else:
                row.append("-")
        table.add_row(*row)

    console.print(table)

    # Per-request avg tok/s table
    table2 = Table(title="Per-Request Avg Throughput (tok/s)", border_style="blue")
    table2.add_column("ctx \\ conc", style="bold cyan")
    for conc in concurrency_levels:
        table2.add_column(str(conc), justify="right")

    for ctx in context_lengths:
        row = [format_context(ctx)]
        for conc in concurrency_levels:
            r = result_map.get((ctx, conc))
            if r and r.aggregate_tps < 0:
                row.append("skip")
            elif r and r.per_request_avg_tps > 0:
                row.append(f"{r.per_request_avg_tps:.1f}")
            else:
                row.append("-")
        table2.add_row(*row)

    console.print(table2)

    # TTFT table
    table3 = Table(title="Avg TTFT (seconds)", border_style="yellow")
    table3.add_column("ctx \\ conc", style="bold cyan")
    for conc in concurrency_levels:
        table3.add_column(str(conc), justify="right")

    for ctx in context_lengths:
        row = [format_context(ctx)]
        for conc in concurrency_levels:
            r = result_map.get((ctx, conc))
            if r and r.aggregate_tps < 0:
                row.append("skip")
            elif r and r.ttft_avg > 0:
                row.append(f"{r.ttft_avg:.2f}")
            else:
                row.append("-")
        table3.add_row(*row)

    console.print(table3)


def save_results(results: list, args, filepath: str, prefill_results: dict = None):
    concurrency_levels = [int(x) for x in args.concurrency.split(",")]
    context_lengths = [int(x) for x in args.contexts.split(",")]

    # Build summary table (exclude skipped)
    summary = {}
    actual_results = [r for r in results if r.aggregate_tps >= 0]
    for r in actual_results:
        ctx_key = str(r.context_tokens)
        if ctx_key not in summary:
            summary[ctx_key] = {}
        summary[ctx_key][str(r.concurrency)] = r.aggregate_tps

    # Prefill summary
    prefill_summary = {}
    if prefill_results:
        for ctx, pr in sorted(prefill_results.items()):
            prefill_summary[str(ctx)] = {
                "ttft_seconds": round(pr["ttft"], 3),
                "tok_per_sec": round(pr["tok_per_sec"], 0),
            }

    output = {
        "metadata": {
            "model": args.model,
            "server": f"{args.host}:{args.port}",
            "timestamp": datetime.now().isoformat(),
            "duration_per_test": args.duration,
            "max_tokens": args.max_tokens,
            "max_total_tokens": args.max_total_tokens,
            "concurrency_levels": concurrency_levels,
            "context_lengths": context_lengths,
        },
        "prefill": prefill_summary,
        "results": [asdict(r) for r in actual_results],
        "summary_table": summary,
    }

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="sglang Inference Benchmark with Rich TUI Dashboard"
    )
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=5000, help="Server port (default: 5000)")
    parser.add_argument(
        "--concurrency", default="1,2,4,8,16,32,64,128",
        help="Comma-separated concurrency levels (default: 1,2,4,8,16,32,64,128)"
    )
    parser.add_argument(
        "--contexts", default="0,16384,32768,65536,131072",
        help="Comma-separated context lengths in tokens (default: 0,16384,32768,65536,131072)"
    )
    parser.add_argument(
        "--duration", type=float, default=30.0,
        help="Duration per test cell in seconds (default: 30)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=8192,
        help="Max tokens to generate per request (default: 8192)"
    )
    parser.add_argument(
        "--output", default="benchmark_results.json",
        help="Output JSON file path (default: benchmark_results.json)"
    )
    parser.add_argument(
        "--model", default="Qwen3.5",
        help="Model name for API requests (default: Qwen3.5)"
    )
    parser.add_argument(
        "--max-total-tokens", type=int, default=0,
        help="Max total tokens budget (concurrency × (context + max_tokens)). "
             "Cells exceeding this are skipped. 0 = no limit (default: 0)"
    )
    return parser.parse_args()


_partial_results: list = []
_prefill_results: dict = {}


def main():
    global _partial_results, _prefill_results
    args = parse_args()
    console = Console()

    concurrency_levels = [int(x) for x in args.concurrency.split(",")]
    context_lengths = [int(x) for x in args.contexts.split(",")]
    decode_count = len(concurrency_levels) * len(context_lengths)

    console.print(Panel(
        f"[bold cyan]sglang Inference Benchmark[/bold cyan]\n"
        f"Model: {args.model} @ {args.host}:{args.port}\n"
        f"Decode concurrency: {concurrency_levels}\n"
        f"Decode contexts: {[format_context(c) for c in context_lengths]}\n"
        f"Duration: {args.duration}s per decode test | Max tokens: {args.max_tokens}\n"
        f"Phase 1: prefill (auto, up to 200k) | Phase 2: {decode_count} decode tests (cached)",
        title="Configuration",
        border_style="cyan",
    ))

    try:
        results, prefill_results = asyncio.run(run_benchmark(args))
        _prefill_results = prefill_results
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user. Saving partial results...[/yellow]")
        results = _partial_results
        prefill_results = _prefill_results

    if results or prefill_results:
        print_final_results(results, concurrency_levels, context_lengths, console, prefill_results)
        save_results(results, args, args.output, prefill_results)
        console.print(f"\n[green]Results saved to {args.output}[/green]")
    else:
        console.print("[red]No results collected.[/red]")


if __name__ == "__main__":
    main()
