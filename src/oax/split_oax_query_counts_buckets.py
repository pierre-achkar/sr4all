"""
Split JSONL records into buckets based on oax_query_counts.

Buckets are defined by count edges. Each record is assigned a score using
SCORE_STRATEGY and written to the corresponding bucket file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

# ---------------- CONFIGURATION ----------------
INPUT_FILE = Path(
    "./data/final/with_oax/sr4all_full_normalized_boolean_with_year_range_oax_with_counts.jsonl"
)
OUTPUT_DIR = Path("./data/final/with_oax/oax_count_buckets")

# Bucket edges (8 buckets):
# 0
# 1–5k
# 5k–50k
# 50k–250k
# 250k–1M
# 1M–5M
# 5M–20M
# 20M+
BUCKET_EDGES = [0, 1, 5000, 50000, 250000, 1_000_000, 5_000_000, 20_000_000]

# How to reduce oax_query_counts to a single score per record:
# "max" -> max count in list (default)
# "sum" -> sum of counts
# "min" -> min count
# "first" -> first count
SCORE_STRATEGY = "max"

# If True, attach the computed score and bucket label to each record
# under fields: oax_query_count_score, oax_query_count_bucket
WRITE_SCORE_FIELDS = True
# ----------------------------------------------


def _iter_records(jsonl_path: Path) -> Iterable[dict]:
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _score(counts: List[int]) -> int:
    if not counts:
        return 0
    if SCORE_STRATEGY == "sum":
        return int(sum(counts))
    if SCORE_STRATEGY == "min":
        return int(min(counts))
    if SCORE_STRATEGY == "first":
        return int(counts[0])
    return int(max(counts))


def _bucket_label(score: int) -> str:
    e = BUCKET_EDGES
    if score <= e[0]:
        return "b0_0"
    if e[1] <= score < e[2]:
        return "b1_1_5k"
    if e[2] <= score < e[3]:
        return "b2_5k_50k"
    if e[3] <= score < e[4]:
        return "b3_50k_250k"
    if e[4] <= score < e[5]:
        return "b4_250k_1m"
    if e[5] <= score < e[6]:
        return "b5_1m_5m"
    if e[6] <= score < e[7]:
        return "b6_5m_20m"
    return "b7_20m_plus"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "b0_0": OUTPUT_DIR / "bucket_0_0.jsonl",
        "b1_1_5k": OUTPUT_DIR / "bucket_1_5k.jsonl",
        "b2_5k_50k": OUTPUT_DIR / "bucket_5k_50k.jsonl",
        "b3_50k_250k": OUTPUT_DIR / "bucket_50k_250k.jsonl",
        "b4_250k_1m": OUTPUT_DIR / "bucket_250k_1m.jsonl",
        "b5_1m_5m": OUTPUT_DIR / "bucket_1m_5m.jsonl",
        "b6_5m_20m": OUTPUT_DIR / "bucket_5m_20m.jsonl",
        "b7_20m_plus": OUTPUT_DIR / "bucket_20m_plus.jsonl",
    }

    handles = {
        label: path.open("w", encoding="utf-8") for label, path in output_paths.items()
    }

    counts_written = {label: 0 for label in output_paths}

    try:
        for record in _iter_records(INPUT_FILE):
            counts = record.get("oax_query_counts")
            if not isinstance(counts, list):
                counts = []

            clean_counts: List[int] = []
            for c in counts:
                try:
                    c_int = int(c)
                except (TypeError, ValueError):
                    continue
                if c_int < 0:
                    continue
                clean_counts.append(c_int)

            score = _score(clean_counts)
            label = _bucket_label(score)

            if WRITE_SCORE_FIELDS:
                record["oax_query_count_score"] = score
                record["oax_query_count_bucket"] = label

            handles[label].write(json.dumps(record, ensure_ascii=False) + "\n")
            counts_written[label] += 1
    finally:
        for h in handles.values():
            h.close()

    print("Wrote bucket files to:", OUTPUT_DIR)
    for label, n in counts_written.items():
        print(f"  {label}: {n}")


if __name__ == "__main__":
    main()
