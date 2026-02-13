"""
Plot distribution of OpenAlex query counts.

Reads JSONL with an `oax_query_counts` list per record and plots a histogram
(useful for deciding bucket thresholds).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, List

# ---------------- CONFIGURATION ----------------
INPUT_FILE = Path(
    "./data/final_old/with_oax/sr4all_full_normalized_boolean_with_year_range_oax_with_counts.jsonl"
)
OUTPUT_FILE = Path("./data/final_old/with_oax/plots/oax_query_counts_hist.png")

# Comma-separated bin edges as a string, or None to auto-generate log bins.
# Example: "0,1,2,5,10,20,50,100,200,500,1000"
BINS = None

# Log-scale y-axis (x-axis is always log-scale to spread large counts)
LOG_Y = False

# Number of bins for auto-generated log bins
AUTO_LOG_BINS = 30
# ----------------------------------------------


def _iter_counts(jsonl_path: Path) -> Iterable[int]:
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            counts = record.get("oax_query_counts")
            if not isinstance(counts, list):
                continue
            for c in counts:
                try:
                    c_int = int(c)
                except (TypeError, ValueError):
                    continue
                if c_int < 0:
                    continue
                yield c_int


def _default_log_bins(values: List[int], bins: int = 30) -> List[float]:
    if not values:
        return [0, 1]
    non_zero = [v for v in values if v > 0]
    if not non_zero:
        return [0, 1]
    min_v = min(non_zero)
    max_v = max(non_zero)
    if min_v == max_v:
        return [0, min_v, max_v + 1]

    log_min = math.log10(min_v)
    log_max = math.log10(max_v)
    step = (log_max - log_min) / bins
    edges = [10 ** (log_min + i * step) for i in range(bins + 1)]
    return [0.0] + edges


def _parse_bins_arg(bins_arg: str | None) -> List[float] | None:
    if not bins_arg:
        return None
    parts = [p.strip() for p in bins_arg.split(",") if p.strip()]
    if not parts:
        return None
    bins: List[float] = []
    for part in parts:
        bins.append(float(part))
    return bins


def main() -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required. Install with: pip install matplotlib"
        ) from exc

    counts = list(_iter_counts(INPUT_FILE))
    if not counts:
        raise SystemExit("No counts found in input file.")

    bins = _parse_bins_arg(BINS)
    if bins is None:
        bins = _default_log_bins(counts, bins=AUTO_LOG_BINS)

    # Prepare output directory
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(counts, bins=bins, edgecolor="white")
    ax.set_xlabel("OpenAlex query result count")
    ax.set_ylabel("Number of queries")
    ax.set_title("Distribution of OpenAlex query counts")

    if LOG_Y:
        ax.set_yscale("log")

    ax.set_xscale("log")
    ax.grid(True, which="both", axis="both", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=200)

    # Simple stats summary for bucket decisions
    counts_sorted = sorted(counts)
    n = len(counts_sorted)

    def _pct(p: float) -> int:
        if n == 0:
            return 0
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return counts_sorted[idx]

    summary = {
        "total_queries": n,
        "min": counts_sorted[0],
        "p50": _pct(0.50),
        "p75": _pct(0.75),
        "p90": _pct(0.90),
        "p95": _pct(0.95),
        "p99": _pct(0.99),
        "max": counts_sorted[-1],
    }

    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"Saved histogram to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
