#!/usr/bin/env python3
"""Plot distribution of abstract_coverage from a JSONL file."""

import argparse
import json
from pathlib import Path
from statistics import mean, median


def extract_coverage(obj):
    """Return coverage ratio if present, else None."""
    if "abstract_coverage" in obj and isinstance(
        obj["abstract_coverage"], (int, float)
    ):
        return float(obj["abstract_coverage"])
    rac = obj.get("references_abstract_coverage")
    if isinstance(rac, dict):
        ratio = rac.get("ratio")
        if isinstance(ratio, (int, float)):
            return float(ratio)
    return None


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot distribution of abstract coverage ratio."
    )
    p.add_argument(
        "--input",
        default="./data/final/sr4all_full.jsonl",
        help="Path to JSONL file",
    )
    p.add_argument(
        "--output",
        default="./data/final/abstract_coverage_hist.png",
        help="Path to output PNG",
    )
    p.add_argument("--bins", type=int, default=50, help="Number of histogram bins")
    p.add_argument("--show", action="store_true", help="Display the plot window")
    return p.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    values = []
    total = 0
    missing = 0

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                missing += 1
                continue
            val = extract_coverage(obj)
            if val is None:
                missing += 1
                continue
            values.append(val)

    if not values:
        raise SystemExit("No abstract coverage values found.")

    # Lazy import so the script can be used for quick stats without matplotlib.
    import matplotlib.pyplot as plt  # noqa: WPS433

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=args.bins, color="#2F6B8F", edgecolor="white")
    plt.title("Abstract Coverage Distribution")
    plt.xlabel("Coverage ratio")
    plt.ylabel("Count")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)

    print(f"Rows: {total}")
    print(f"Values: {len(values)}")
    print(f"Missing/invalid: {missing}")
    print(f"Min: {min(values):.4f}")
    print(f"Max: {max(values):.4f}")
    print(f"Mean: {mean(values):.4f}")
    print(f"Median: {median(values):.4f}")
    print(f"Saved: {output_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
