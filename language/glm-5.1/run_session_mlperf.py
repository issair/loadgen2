#!/usr/bin/env python3
"""Session-based MLPerf benchmark entry-point (Path B).

Usage::

    python run_session_mlperf.py \
        --input data.json \
        --output-dir results/ \
        --num-triggers 10 \
        --poisson-lam 5 \
        --poisson-seed 42

Workflow
--------
1. ``process_dataset()`` reads conversation JSON files and builds one
   ``LargeModelSession`` per virtual user.
2. A minimal LoadGen QSL is constructed (no dependency replay --- the SUT
   handles that internally).
3. LoadGen runs in **Offline** mode, issuing every user turn as a single
   sample all at once.
4. ``SessionSUT`` maps samples to sessions and launches ``SessionScheduler``.
5. The scheduler dispatches turns across *N* worker triggers, with per-VU
   Poisson pacing provided by ``LargeModelSession.next_execute_request()``.
"""

import argparse
import json as _json
import logging
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import mlperf_loadgen as lg

from data_tools.split_andmapping import LargeModelSession, process_dataset
from mlperf.session_sut import SessionSUT
from utils import get_backend_instance

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Minimal QSL (no dependency replay --- SUT handles chains internally)
# ---------------------------------------------------------------------------


class _MinimalQSL:
    """Thin wrapper around ``lg.ConstructQSL`` for Path B."""

    def __init__(self, total_turns: int, name: str = "SessionQSL") -> None:
        self.count = total_turns
        self.perf_count = total_turns
        self.name = name
        self.qsl = lg.ConstructQSL(
            self.count,
            self.perf_count,
            lambda x: None,  # LoadSamplesToRam
            lambda x: None,  # UnloadSamplesFromRam
            lambda x: [],  # get_dependency_chain (unused, required by API)
        )
        logger.info(f"QSL created: {total_turns} samples (no dependency replay)")

    def __del__(self) -> None:
        if hasattr(self, "qsl") and self.qsl is not None:
            lg.DestroyQSL(self.qsl)


# ---------------------------------------------------------------------------
#  Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MLPerf benchmark",
    )
    p.add_argument("--input", "-i", required=True, help="Trace JSON file/dir")
    p.add_argument(
        "--output-dir",
        default="trace_mlperf_results",
        help="Output directory for LoadGen logs",
    )
    p.add_argument(
        "--max-trigger",
        type=int,
        default=10,
        help="Maximum number of triggers (default 10)",
    )
    p.add_argument(
        "--poisson-lam",
        type=int,
        default=1,
        help="Poisson lambda for inter-turn intervals (default 1)",
    )
    p.add_argument(
        "--poisson-seed",
        type=int,
        default=2,
        help="RNG seed for Poisson generation (default 0)",
    )
    p.add_argument(
        "--poisson-pool-size",
        type=int,
        default=1000,
        help="Number of pre-generated Poisson intervals per trigger (default 1000)",
    )
    p.add_argument(
        "--time-unit",
        choices=["s", "ms"],
        default="s",
        help="Unit of ps_delta / executed_time (default s)",
    )
    p.add_argument(
        "--max-trace",
        type=int,
        default=None,
        help="Maximum number of traces (limits traces, truncates if needed)",
    )
    p.add_argument(
        "--mlperf-conf",
        type=str,
        default="/inference/mlperf.conf",
        help="Path to mlperf.conf",
    )
    p.add_argument(
        "--target-qps",
        type=float,
        default=1,
        help="Override *.Offline.target_qps (or *.Server.target_qps if --scenario Server)",
    )
    p.add_argument(
        "--min-duration",
        type=int,
        default=60000,
        help="Override *.Offline.min_duration (ms)",
    )
    p.add_argument(
        "--min-query-count",
        type=int,
        default=50,
        help="Override *.Offline.min_query_count",
    )
    p.add_argument(
        "--go-to-end",
        action="store_true",
        help="Iterate all requests to trace end (instead of stopping at next user), "
        "sleeping Poisson intervals between users",
    )
    p.add_argument(
        "--tpm-interval",
        type=float,
        default=60.0,
        help="Time interval in seconds for per-interval TPM calculation (default 60.0 = 1 min)",
    )
    p.add_argument(
        "--plot-interval",
        type=float,
        default=10.0,
        help="Time interval in seconds for input-token timeline plot (default 5.0)",
    )
    return p


# ---------------------------------------------------------------------------
#  LoadGen settings
# ---------------------------------------------------------------------------


def _configure_loadgen(
    total_turns: int,
    accuracy: bool,
    mlperf_conf: str | None = None,
    user_conf: str | None = None,
    model_name: str = "glm-5.1",
    scenario: str = "Offline",
) -> lg.TestSettings:
    settings = lg.TestSettings()
    settings.scenario = (
        lg.TestScenario.Offline if scenario == "Offline" else lg.TestScenario.Server
    )
    settings.mode = (
        lg.TestMode.AccuracyOnly if accuracy else lg.TestMode.PerformanceOnly
    )
    settings.use_token_latencies = True
    settings.enable_dependency_replay = False
    settings.server_coalesce_queries = False

    settings.min_query_count = total_turns
    settings.max_query_count = total_turns

    if mlperf_conf and Path(mlperf_conf).exists():
        settings.FromConfig(mlperf_conf, model_name, scenario, 2)
    if user_conf and Path(user_conf).exists():
        settings.FromConfig(user_conf, model_name, scenario, 1)

    return settings


# ---------------------------------------------------------------------------
#  Temporary user.conf generation from CLI overrides
# ---------------------------------------------------------------------------


def _build_temp_user_conf(
    target_qps: Optional[float] = None,
    min_duration: Optional[int] = None,
    min_query_count: Optional[int] = None,
    scenario: str = "Offline",
) -> str | None:
    """If any override is provided, write a temporary user.conf and return its path.

    Returns ``None`` when no overrides are given, so callers can fall back to
    the ``--user-conf`` path normally.
    """
    if target_qps is None and min_duration is None and min_query_count is None:
        return None

    lines: list[str] = []
    if target_qps is not None:
        lines.append(f"*.*.target_qps = {target_qps}")
    if min_duration is not None:
        lines.append(f"*.{scenario}.min_duration = {min_duration}")
    if min_query_count is not None:
        lines.append(f"*.{scenario}.min_query_count = {min_query_count}")

    fd, path = tempfile.mkstemp(suffix=".conf", prefix="user_conf_")
    with open(fd, "w", encoding="utf-8") as f:
        f.write("# Auto-generated user.conf from CLI overrides\n")
        f.write("\n".join(lines))
        f.write("\n")

    logger.info("Generated temporary user.conf at %s with:\n%s", path, "\n".join(lines))
    return path


# ---------------------------------------------------------------------------
#  Summary printer
# ---------------------------------------------------------------------------

PERCENTILE_LABELS = ["min", "max", "avg", "p50", "p75", "p90", "p95", "p99"]
LABEL_WIDTH = 26  # display columns for metric-name column
VAL_WIDTH = 10  # display columns per numeric column


def _display_width(s: str) -> int:
    """Terminal display width: CJK/fullwidth chars = 2, others = 1."""
    w = 0
    for ch in s:
        cp = ord(ch)
        if (
            0x4E00 <= cp <= 0x9FFF  # CJK Unified
            or 0x3000 <= cp <= 0x303F  # CJK Symbols
            or 0xFF00 <= cp <= 0xFFEF
        ):  # Fullwidth forms
            w += 2
        else:
            w += 1
    return w


def _pad(s: str, target: int) -> str:
    """Pad *s* with trailing spaces to *target* display columns."""
    cur = _display_width(s)
    return s + " " * max(target - cur, 0)


def _fmt_val(v: float, fmt: str) -> str:
    """Format a value right-aligned into VAL_WIDTH columns."""
    text = ("%" + fmt) % v
    return text.rjust(VAL_WIDTH)


def _print_stat_line(label: str, value, fmt: str = ".2f") -> None:
    """Print a labeled stat line with right-aligned value.

    Uses a fixed label width of 40 and value width of 10 for consistent
    alignment matching the example output format.
    """
    STAT_LABEL_WIDTH = 40
    STAT_VAL_WIDTH = 10
    text = ("%" + fmt) % value
    logger.info(
        "%s%s",
        _pad(label, STAT_LABEL_WIDTH),
        text.rjust(STAT_VAL_WIDTH),
    )


def _percentiles(vals: List[float]) -> List[float]:
    """Return [min, max, avg, p50, p75, p90, p95, p99] for a list of floats."""
    arr = sorted(vals)
    n = len(arr)
    if n == 0:
        return [0.0] * 8
    return [
        arr[0],
        arr[-1],
        sum(arr) / n,
        arr[int(n * 0.50)] if n > 1 else arr[0],
        arr[int(n * 0.75)] if n > 1 else arr[0],
        arr[int(n * 0.90)] if n > 1 else arr[0],
        arr[int(n * 0.95)] if n > 1 else arr[0],
        arr[int(n * 0.99)] if n > 1 else arr[0],
    ]


def _print_summary(
    results: List[dict],
    num_triggers: int = 0,
    tpm_interval: float = 60.0,
    plot_interval: float = 5.0,
    plot_output: str = "",
) -> None:
    """Print a percentile-based summary table and overall benchmark statistics."""
    if not results:
        logger.info("No results to print.")
        return

    # Separate failed and successful requests
    # Failed = end_time is None (empty)
    successful = [r for r in results if r.get("end_time") is not None]
    failed = [r for r in results if r.get("end_time") is None]

    num_failed = len(failed)
    num_successful = len(successful)

    if not successful:
        logger.info("No successful results to print.")
        return

    # Per-request derived metrics (only for successful requests)
    total_ms: List[float] = []
    ttft_ms: List[float] = []
    tpot_ms: List[float] = []
    cache_pct: List[float] = []
    reuse_pct: List[float] = []
    input_tokens: List[int] = []
    output_tokens: List[int] = []

    # Events for peak concurrent request tracking: (time, delta)
    # delta = +1 for start, -1 for end
    events: List[tuple] = []

    for r in successful:
        ct = r.get("completion_time") or 0.0
        ft = r.get("first_token_time") or 0.0
        out_tok = r.get("completion_tokens", 0) or 0
        in_tok = r.get("prompt_tokens", 0) or 0
        cached = r.get("cached_tokens", 0) or 0

        total_ms.append(ct * 1000.0)
        ttft_ms.append(ft * 1000.0 if ft else 0.0)

        # TPOT = (total - ttft) / (output_tokens - 1), min 1 token
        gen_tokens = max(out_tok - 1, 1)
        tpot_ms.append((ct - ft) / gen_tokens * 1000.0 if ct > ft else 0.0)

        cache_pct.append(cached / in_tok * 100.0 if in_tok > 0 else 0.0)
        input_tokens.append(in_tok)
        output_tokens.append(out_tok)

        # Track start/end events for concurrent request counting
        start_time = r.get("start_time", 0.0)
        end_time = start_time + ct
        events.append((start_time, 1))
        events.append((end_time, -1))

    by_sid: Dict[str, List[dict]] = defaultdict(list)
    for r in successful:
        by_sid[r.get("sid", "")].append(r)
    for sid, reqs in by_sid.items():
        reqs.sort(key=lambda x: x.get("end_at", 0))
        prev_in = 0
        for i, r in enumerate(reqs):
            in_tok = r.get("prompt_tokens", 0) or 0
            if i == 0 or in_tok == 0:
                reuse_pct.append(0.0)
            else:
                rate = prev_in / in_tok  # repetition rate = overlap ratio
                reuse_pct.append(min(100.0, rate * 100.0))  # cap at 100%
            prev_in = in_tok

    # Calculate peak concurrent requests by walking through sorted events
    events.sort(key=lambda x: x[0])
    current = 0
    peak_concurrent = 0
    for _t, delta in events:
        current += delta
        if current > peak_concurrent:
            peak_concurrent = current

    # Benchmark duration = time from first start to last end
    min_start = 0.0
    if events:
        min_start = min(e[0] for e in events if e[1] == 1)
        max_end = max(e[0] for e in events if e[1] == -1)
        benchmark_duration_s = max_end - min_start
    else:
        benchmark_duration_s = 0.0

    # Totals
    total_input_tokens = sum(input_tokens)
    total_output_tokens = sum(output_tokens)

    # Throughput
    request_throughput = (
        num_successful / benchmark_duration_s if benchmark_duration_s > 0 else 0.0
    )
    output_token_throughput = (
        total_output_tokens / benchmark_duration_s if benchmark_duration_s > 0 else 0.0
    )
    total_token_throughput = (
        (total_input_tokens + total_output_tokens) / benchmark_duration_s
        if benchmark_duration_s > 0
        else 0.0
    )

    # ── Net metrics (exclude waiting time) ───────────────────────────
    # Net duration = max(total completion_time per worker), i.e. the
    # busiest worker's cumulative execution time.
    worker_times: Dict[int, float] = defaultdict(float)
    for r in successful:
        wid = r.get("workid", 0)
        ct = r.get("completion_time", 0.0) or 0.0
        worker_times[wid] += ct
    net_benchmark_duration_s = max(worker_times.values()) if worker_times else 0.0

    net_output_token_throughput = (
        total_output_tokens / net_benchmark_duration_s
        if net_benchmark_duration_s > 0
        else 0.0
    )
    net_total_token_throughput = (
        (total_input_tokens + total_output_tokens) / net_benchmark_duration_s
        if net_benchmark_duration_s > 0
        else 0.0
    )

    # Peak output token throughput = max(output_tokens / completion_time) per request
    peak_output_tok_throughput = 0.0
    for r in successful:
        ct = r.get("completion_time", 0.0) or 0.0
        out_tok = r.get("completion_tokens", 0) or 0
        if ct > 0:
            tok_per_s = out_tok / ct
            if tok_per_s > peak_output_tok_throughput:
                peak_output_tok_throughput = tok_per_s

    # ── Per-interval TPM ─────────────────────────────────────────
    tpm_per_interval: List[float] = []
    if benchmark_duration_s > 0 and tpm_interval > 0:
        if benchmark_duration_s < tpm_interval:
            # Total duration shorter than one interval: use total duration
            total_tokens = total_input_tokens + total_output_tokens
            tpm_per_interval.append(total_tokens / (benchmark_duration_s / 60.0))
        else:
            num_buckets = int(benchmark_duration_s / tpm_interval) + 1
            bucket_tokens: Dict[int, int] = defaultdict(int)
            for r in successful:
                start_time = r.get("start_time", 0.0)
                ct = r.get("completion_time", 0.0) or 0.0
                end_time = start_time + ct
                rel_end = end_time - min_start
                bucket_idx = int(rel_end / tpm_interval)
                if bucket_idx >= num_buckets:
                    bucket_idx = num_buckets - 1
                in_tok = r.get("prompt_tokens", 0) or 0
                out_tok = r.get("completion_tokens", 0) or 0
                bucket_tokens[bucket_idx] += in_tok + out_tok
            for idx in range(num_buckets):
                tokens = bucket_tokens.get(idx, 0)
                if tokens > 0:
                    tpm_per_interval.append(tokens / (tpm_interval / 60.0))

    # ── Print percentile table (renamed labels) ───────────────────
    rows = [
        ("End-to-end Latency(ms)", total_ms, "7.1f"),
        ("Time to First Token(ms)", ttft_ms, "7.1f"),
        ("TPOT(ms)", tpot_ms, "7.1f"),
        ("Cache(%)", cache_pct, "7.1f"),
        ("Reuse(%)", reuse_pct, "7.1f"),
        ("Input(tokens)", [float(x) for x in input_tokens], "7.0f"),
        ("Output(tokens)", [float(x) for x in output_tokens], "7.0f"),
    ]
    if tpm_per_interval:
        rows.append(("TPM(TW 1 min)", tpm_per_interval, "7.1f"))

    sep = "=" * (
        LABEL_WIDTH + VAL_WIDTH * len(PERCENTILE_LABELS) + len(PERCENTILE_LABELS)
    )
    logger.info(sep)
    logger.info("SessionSUT Results: %d records", len(results))
    logger.info(sep)

    # Header
    header = _pad("Metric", LABEL_WIDTH)
    for p in PERCENTILE_LABELS:
        header += " " + p.rjust(VAL_WIDTH)
    logger.info(header)
    logger.info("-" * len(sep))

    for label, vals, fmt in rows:
        pcts = _percentiles(vals)
        line = _pad(label, LABEL_WIDTH)
        for v in pcts:
            line += " " + _fmt_val(v, fmt)
        logger.info(line)

    logger.info(sep)

    # ── Print overall benchmark summary ───────────────────────────
    _print_stat_line("Successful requests", num_successful, "d")
    _print_stat_line("Failed requests", num_failed, "d")
    if num_triggers > 0:
        _print_stat_line("Maximum request concurrency", num_triggers, "d")
    _print_stat_line("Benchmark duration (s)", benchmark_duration_s, ".2f")
    _print_stat_line("Total input tokens", total_input_tokens, "d")
    _print_stat_line("Total generated tokens", total_output_tokens, "d")
    _print_stat_line("Request throughput (req/s)", request_throughput, ".2f")
    rpm = (
        num_successful / (benchmark_duration_s / 60.0)
        if benchmark_duration_s > 0
        else 0.0
    )
    net_rpm = (
        num_successful / (net_benchmark_duration_s / 60.0)
        if net_benchmark_duration_s > 0
        else 0.0
    )
    _print_stat_line("RPM (req/min)", rpm, ".2f")
    _print_stat_line("RPM (Net, req/min)", net_rpm, ".2f")
    _print_stat_line(
        "Peak output token throughput (tok/s)", peak_output_tok_throughput, ".2f"
    )
    _print_stat_line("Peak concurrent requests", peak_concurrent, ".2f")
    _print_stat_line("Output token throughput (tok/s)", output_token_throughput, ".2f")
    _print_stat_line(
        "Net output token throughput (tok/s)", net_output_token_throughput, ".2f"
    )
    _print_stat_line("Total token throughput (tok/s)", total_token_throughput, ".2f")
    _print_stat_line(
        "Net total token throughput (tok/s)", net_total_token_throughput, ".2f"
    )

    # TPM = total output tokens / total minutes
    tpm = (
        (total_output_tokens + total_input_tokens) / (benchmark_duration_s / 60.0)
        if benchmark_duration_s > 0
        else 0.0
    )
    net_tpm = (
        (total_output_tokens + total_input_tokens) / (net_benchmark_duration_s / 60.0)
        if net_benchmark_duration_s > 0
        else 0.0
    )
    _print_stat_line("TPM", tpm, ".2f")
    _print_stat_line("Net TPM", net_tpm, ".2f")

    # ── Timeline plot (optional) ─────────────────────────────────
    if plot_output:
        _plot_input_tokens_timeline(
            successful,
            min_start,
            benchmark_duration_s,
            plot_interval,
            plot_output,
        )

    # ── Worker request dot plot ───────────────────────────────────
    dot_plot_output = (
        plot_output.replace("input_tokens_timeline", "worker_request_dots")
        if plot_output
        else ""
    )
    if dot_plot_output:
        _plot_worker_request_dots(
            successful,
            min_start,
            benchmark_duration_s,
            dot_plot_output,
        )

    logger.info(sep)


def _plot_input_tokens_timeline(
    successful: List[dict],
    min_start: float,
    benchmark_duration_s: float,
    plot_interval: float,
    output_path: str,
) -> None:
    """Plot average input tokens per time bucket over the benchmark timeline.

    Divides the benchmark duration into buckets of *plot_interval* seconds,
    computes the average input tokens per bucket, and generates a scatter plot
    using matplotlib.  X-axis uses actual wall-clock time (from the first
    request's start_time).
    """
    if benchmark_duration_s <= 0 or plot_interval <= 0 or not successful:
        logger.warning("Skipping timeline plot: no data or invalid interval.")
        return

    try:
        from datetime import datetime, timezone

        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, skipping timeline plot.")
        return

    num_buckets = int(benchmark_duration_s / plot_interval) + 1
    # Accumulate input tokens and counts per bucket
    bucket_tokens: dict[int, float] = {}
    bucket_counts: dict[int, int] = {}
    for r in successful:
        start_time = r.get("start_time", 0.0)
        ct = r.get("completion_time", 0.0) or 0.0
        end_time = start_time + ct
        rel_end = end_time - min_start
        bucket_idx = int(rel_end / plot_interval)
        if bucket_idx >= num_buckets:
            bucket_idx = num_buckets - 1
        in_tok = r.get("prompt_tokens", 0) or 0
        bucket_tokens[bucket_idx] = bucket_tokens.get(bucket_idx, 0.0) + in_tok
        bucket_counts[bucket_idx] = bucket_counts.get(bucket_idx, 0) + 1

    # Build (datetime, avg_input_tokens) pairs
    x_vals: list[datetime] = []
    y_vals: list[float] = []
    for idx in range(num_buckets):
        count = bucket_counts.get(idx, 0)
        if count > 0:
            bucket_time = datetime.fromtimestamp(
                min_start + (idx + 0.5) * plot_interval, tz=timezone.utc
            )
            avg_tokens = bucket_tokens.get(idx, 0.0) / count
            x_vals.append(bucket_time)
            y_vals.append(avg_tokens)

    if not x_vals:
        logger.warning("No data points for timeline plot.")
        return

    min_start_dt = datetime.fromtimestamp(min_start, tz=timezone.utc)
    date_str = min_start_dt.strftime("%Y-%m-%d")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.scatter(x_vals, y_vals, s=20, alpha=0.7, edgecolors="none")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate(rotation=30)
    ax.set_xlabel("Wall-clock Time (HH:MM:SS)")
    ax.set_ylabel("Avg Input Tokens")
    ax.set_title(
        "Average Input Tokens per {:.0f}s Interval — {}".format(plot_interval, date_str)
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Timeline plot saved to %s", output_path)


# ---------------------------------------------------------------------------
#  Print CLI arguments
# ---------------------------------------------------------------------------


def _plot_worker_request_dots(
    successful: List[dict],
    min_start: float,
    benchmark_duration_s: float,
    output_path: str,
) -> None:
    """Plot each trigger/worker's requests as dots on a timeline.

    Y-axis: worker (trigger) id.
    X-axis: wall-clock time (HH:MM:SS).
    Each dot = one request start time.

    Example output::

        trigger1    .        .
        trigger2    . .  .
        trigger3    .  .     .
        ...              .        .
        triggern    .         .
                  12:00   12:01   13:01
    """
    if benchmark_duration_s <= 0 or not successful:
        logger.warning("Skipping worker dot plot: no data.")
        return

    try:
        from datetime import datetime, timezone

        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, skipping worker dot plot.")
        return

    # Group request start times by worker id
    worker_times: dict[int, list[float]] = {}
    for r in successful:
        wid = r.get("workid", 0)
        st = r.get("start_time", 0.0)
        if wid not in worker_times:
            worker_times[wid] = []
        worker_times[wid].append(st)

    if not worker_times:
        logger.warning("No worker data for dot plot.")
        return

    sorted_wids = sorted(worker_times.keys())
    min_start_dt = datetime.fromtimestamp(min_start, tz=timezone.utc)

    fig, ax = plt.subplots(figsize=(14, max(5, len(sorted_wids) * 0.35)))

    for wid in sorted_wids:
        times = worker_times[wid]
        # Convert to datetime for matplotlib date axis
        dts = [datetime.fromtimestamp(t, tz=timezone.utc) for t in times]
        ax.scatter(
            dts,
            [wid] * len(times),
            s=30,
            alpha=0.6,
            edgecolors="none",
            label=f"trigger{wid + 1}",
        )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate(rotation=30)
    ax.set_yticks(sorted_wids)
    ax.set_yticklabels([f"trigger{wid + 1}" for wid in sorted_wids])
    ax.set_xlabel("Wall-clock Time (HH:MM:SS)")
    ax.set_ylabel("Trigger / Worker")
    ax.set_title(
        "Request Timeline per Trigger — {} ({} workers, {} requests)".format(
            min_start_dt.strftime("%Y-%m-%d"),
            len(sorted_wids),
            len(successful),
        )
    )
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Worker request dot plot saved to %s", output_path)


def _print_args(args: argparse.Namespace) -> None:
    """Print all CLI arguments in a formatted table."""
    sep = "=" * 60
    logger.info(sep)
    logger.info("CLI Arguments")
    logger.info(sep)

    params = [
        ("Input data path", args.input),
        ("Output directory", args.output_dir),
        ("Max concurrent triggers", args.max_trigger),
        ("Max traces", args.max_trace),
        ("Poisson lambda", args.poisson_lam),
        ("Time unit", args.time_unit),
        ("Poisson seed", args.poisson_seed),
        ("Min query count", args.min_query_count),
        ("Go to end", args.go_to_end),
    ]

    STAT_LABEL_WIDTH = 24
    for label, value in params:
        _print_stat_line(
            label,
            value,
            "s"
            if isinstance(value, (str, bool))
            else ".2f"
            if isinstance(value, float)
            else "d",
        )

    logger.info(sep)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _build_parser().parse_args()

    # --- 1. Build LargeModelSessions ------------------------------------
    logger.info("Building traces via process_dataset ...")
    sessions: Dict[str, LargeModelSession] = process_dataset(
        input_path=args.input,
        output_path="",  # vestigial; process_dataset no longer writes output
        poisson_lam=args.poisson_lam,
        poisson_seed=args.poisson_seed,
    )

    if args.max_trace is not None:
        keys = sorted(sessions.keys())[: args.max_trace]
        sessions = {k: sessions[k] for k in keys}
        logger.info(f"Truncated to {args.max_trace} traces (--max-trace)")

    total_turns = sum(len(s.user_conv_request) for s in sessions.values())
    total_requests = sum(len(s.all_conv_request) for s in sessions.values())
    logger.info(
        "Traces: %d traces, %d total user turns, %d total requests",
        len(sessions),
        total_turns,
        total_requests,
    )

    if total_turns == 0:
        logger.error("No user turns found --- check input data")
        sys.exit(1)

    # --- 2. Backend -----------------------------------------------------
    logger.info("Initializing backend: %s", "openai")
    backend = get_backend_instance("openai")

    with backend:
        # --- 3. QSL ----------------------------------------------------
        qsl = _MinimalQSL(total_turns)

        # --- 4. SUT ----------------------------------------------------
        sut = SessionSUT(
            backend=backend,
            sessions=sessions,
            num_triggers=args.max_trigger,
            time_unit=args.time_unit,
            go_to_end=args.go_to_end,
            poisson_seed=args.poisson_seed,
            poisson_lam=args.poisson_lam,
            poisson_pool_size=args.poisson_pool_size,
        )

        # --- 5. Settings & logs -----------------------------------------
        # If CLI overrides are provided, generate a temporary user.conf that
        # takes precedence over (or replaces) the static --user-conf file.
        temp_user_conf = _build_temp_user_conf(
            target_qps=args.target_qps,
            min_duration=args.min_duration,
            min_query_count=args.min_query_count,
            scenario="Offline",
        )
        final_user_conf = temp_user_conf

        settings = _configure_loadgen(
            total_turns=total_turns,
            accuracy=False,
            mlperf_conf=args.mlperf_conf,
            user_conf=final_user_conf,
            scenario="Offline",
        )

        log_dir = Path(args.output_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_settings = lg.LogSettings()
        log_settings.log_output.outdir = str(log_dir)
        log_settings.log_output.copy_summary_to_stdout = False
        log_settings.enable_trace = False

        # --- 6. Run ----------------------------------------------------
        logger.info("Starting SessionSUT ...")
        sut.start()
        try:
            lg.StartTestWithLogSettings(sut.sut, qsl.qsl, settings, log_settings)
        finally:
            sut.stop()
            if qsl.qsl is not None:
                lg.DestroyQSL(qsl.qsl)
                qsl.qsl = None
            # Clean up temporary user.conf if we generated one
            if temp_user_conf is not None:
                Path(temp_user_conf).unlink(missing_ok=True)
                logger.info("Cleaned up temporary user.conf at %s", temp_user_conf)

        # --- 7. Collect & print results --------------------------------
        results = sut.get_results()
        _print_args(args)
        plot_output = str(log_dir / "input_tokens_timeline.png")
        _print_summary(
            results,
            num_triggers=args.max_trigger,
            tpm_interval=args.tpm_interval,
            plot_interval=args.plot_interval,
            plot_output=plot_output,
        )

        # Also write JSON to output dir
        results_path = log_dir / "trace_results.json"
        with open(results_path, "w", encoding="utf-8") as fp:
            _json.dump(results, fp, ensure_ascii=False, indent=2, default=str)
        logger.info("Results written to %s", results_path)

    logger.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
