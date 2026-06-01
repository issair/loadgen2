"""Compute per-session prefix repetition rate from session benchmark results.

Usage::

    python3 calc_reuse.py [session_results.json]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def _sorted_by_session(results: List[dict]) -> List[dict]:
    """Sort results by ``sid``, then by ``end_at`` within each session."""
    by_sid: Dict[str, List[dict]] = defaultdict(list)
    for r in results:
        by_sid[r.get("sid", "")].append(r)
    ordered: List[dict] = []
    for sid in sorted(by_sid):
        ordered.extend(sorted(by_sid[sid], key=lambda x: x.get("end_at", 0)))
    return ordered


def calc_reuse(results: List[dict]) -> List[float]:
    """Calculate repetition rate for each request, grouped by session.

    Within each session (``sid``), requests are sorted by ``end_at``.
    The first request gets rate 0; subsequent requests get::

        rate = prev_prompt_tokens / cur_prompt_tokens

    i.e. the fraction of prompt tokens that **overlap** with the
    previous turn in the same session (prefix reuse ratio).

    Returns rates in the same order as ``results`` (input list order).
    """
    # Build mapping from request index to its session-grouped position
    by_sid: Dict[str, List[dict]] = defaultdict(list)
    for r in results:
        by_sid[r.get("sid", "")].append(r)

    # Calculate rates per session, store in a parallel index map
    rate_by_idx: Dict[int, float] = {}
    for sid, reqs in by_sid.items():
        reqs.sort(key=lambda x: x.get("end_at", 0))
        prev_in = 0
        for i, r in enumerate(reqs):
            idx = r.get("_orig_idx", 0)
            in_tok = r.get("prompt_tokens", 0) or 0
            if i == 0 or in_tok == 0:
                rate_by_idx[idx] = 0.0
            else:
                rate = prev_in / in_tok
                rate_by_idx[idx] = min(100.0, rate * 100.0)
            prev_in = in_tok

    return [rate_by_idx[i] for i in range(len(results))]


def print_results(results: List[dict]) -> None:
    """Print per-request reuse rates and summary statistics."""
    # Tag each record with its original index for rate mapping
    tagged = [{**r, "_orig_idx": i} for i, r in enumerate(results)]
    reuse_pct = calc_reuse(tagged)

    # Print sorted by session
    ordered = _sorted_by_session(results)
    lookup = {(r["sid"], r["end_at"]): i for i, r in enumerate(results)}

    print(
        f"{'sid':<42} {'end_at':>6} {'prompt_tok':>10} {'completion':>10} {'reuse%':>7}"
    )
    print("-" * 80)

    prev_sid = None
    for r in ordered:
        sid = r.get("sid", "")[-36:]
        end_at = r.get("end_at", 0)
        in_tok = r.get("prompt_tokens", 0) or 0
        completion = r.get("completion_tokens", 0) or 0
        idx = lookup[(r["sid"], r["end_at"])]
        pct = reuse_pct[idx]
        if r["sid"] != prev_sid:
            if prev_sid is not None:
                print()  # blank line between sessions
            prev_sid = r["sid"]
        print(f"{sid:<42} {end_at:>6} {in_tok:>10} {completion:>10} {pct:>6.1f}")

    print()
    n = len(reuse_pct)
    if n == 0:
        return

    vals = sorted(reuse_pct)
    mean = sum(vals) / n
    print(f"Reuse(%) stats:  n={n}")
    print(f"  min     {vals[0]:.1f}")
    print(f"  max     {vals[-1]:.1f}")
    print(f"  avg     {mean:.1f}")
    print(f"  median  {vals[n // 2]:.1f}")
    print(f"  P90     {vals[int(n * 0.90)]:.1f}")

    # Per-session summary
    print()
    by_sid: Dict[str, List[dict]] = defaultdict(list)
    for r in results:
        by_sid[r.get("sid", "")].append(r)

    print(f"{'sid':<42} {'turns':>5} {'avg_reuse%':>10}")
    print("-" * 60)
    for sid, reqs in sorted(by_sid.items()):
        s_tagged = [{**r, "_orig_idx": i} for i, r in enumerate(reqs)]
        s_rates = calc_reuse(s_tagged)
        avg = sum(s_rates) / len(s_rates) if s_rates else 0.0
        short = sid[-36:]
        print(f"{short:<42} {len(reqs):>5} {avg:>10.1f}")


def main() -> None:
    path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).resolve().parent
        / "session_mlperf_results"
        / "session_results.json"
    )
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    results = json.loads(path.read_text(encoding="utf-8"))
    # Only successful requests
    results = [r for r in results if r.get("end_time") is not None]
    print_results(results)


if __name__ == "__main__":
    main()
