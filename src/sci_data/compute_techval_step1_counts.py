#!/usr/bin/env python3
"""Compute Step-1 baseline counts for Technical Validation.

This script streams the primary extracted-values JSONL file and writes:
1) A machine-readable JSON summary
2) A human-readable Markdown summary

Usage:
  /opt/pyenv/versions/sci_data/bin/python src/compute_techval_step1_counts.py \
    --input /path/to/sr4all_full.jsonl \
    --json-output outputs/techval_step1_baseline_counts.json \
    --md-output outputs/techval_step1_baseline_counts.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_INPUT = "/home/fhg/pie65738/projects/sr4all/data_old/release/sr4all_full.jsonl"
DEFAULT_JSON_OUT = "outputs/techval_step1_baseline_counts.json"
DEFAULT_MD_OUT = "outputs/techval_step1_baseline_counts.md"

# Keys that are generally metadata and not extraction targets.
EXCLUDED_TOP_LEVEL_KEYS = {
    "id",
    "doi",
    "title",
    "abstract",
    "language",
    "publication_year",
    "pdf_url",
    "openalex_id",
    "referenced_works",
    "num_referenced_works",
    "source",
}

# Container names commonly used for extracted/methodological fields.
METHOD_CONTAINER_NAMES = {
    "methodology",
    "extraction",
    "extracted",
    "structured",
    "fields",
    "review_methodology",
}

# Substring patterns to identify methodological extraction fields.
METHOD_PATTERNS = (
    "objective",
    "aim",
    "question",
    "keyword",
    "boolean",
    "inclusion",
    "exclusion",
    "criteria",
    "date_range",
    "search",
    "database",
    "retrieved",
    "included",
    "citation_chasing",
    "query",
)


def has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return True


def looks_method_key(key: str) -> bool:
    k = key.lower()
    return any(pattern in k for pattern in METHOD_PATTERNS)


def iter_method_items(record: dict[str, Any]):
    # Top-level candidates
    for key, value in record.items():
        if key in EXCLUDED_TOP_LEVEL_KEYS:
            continue
        if looks_method_key(key):
            yield f"top.{key}", value

    # Nested container candidates
    for key, value in record.items():
        if key.lower() not in METHOD_CONTAINER_NAMES:
            continue
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if looks_method_key(sub_key):
                    yield f"{key}.{sub_key}", sub_value


def pct(n: int, d: int) -> float:
    return (100.0 * n / d) if d else 0.0


def write_markdown(summary: dict[str, Any], output_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Technical Validation Step 1: Baseline Counts")
    lines.append("")
    lines.append(f"- generated_utc: {summary['generated_utc']}")
    lines.append(f"- input_file: {summary['input_file']}")
    lines.append(f"- total_records: {summary['total_records']:,}")
    lines.append(
        "- records_with_any_method_value: "
        f"{summary['records_with_any_method_value']:,} "
        f"({summary['records_with_any_method_value_pct']:.2f}%)"
    )
    lines.append("")
    lines.append("## Non-empty Counts by Method Field")
    lines.append("")
    lines.append("| Field | Non-empty Count | Percentage of Total |")
    lines.append("|---|---:|---:|")
    for row in summary["field_non_empty"]:
        lines.append(
            f"| {row['field']} | {row['count']:,} | {row['pct_total']:.2f}% |"
        )
    lines.append("")
    lines.append("## Parse Diagnostics")
    lines.append("")
    lines.append(f"- parsed_records: {summary['parsed_records']:,}")
    lines.append(f"- json_decode_errors: {summary['json_decode_errors']:,}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to input JSONL")
    parser.add_argument("--json-output", default=DEFAULT_JSON_OUT, help="Path to JSON output")
    parser.add_argument("--md-output", default=DEFAULT_MD_OUT, help="Path to markdown output")
    args = parser.parse_args()

    input_path = Path(args.input)
    json_output = Path(args.json_output)
    md_output = Path(args.md_output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    total_records = 0
    parsed_records = 0
    json_decode_errors = 0

    field_non_empty_counter: Counter[str] = Counter()
    records_with_any_method_value = 0

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            total_records += 1
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
                parsed_records += 1
            except json.JSONDecodeError:
                json_decode_errors += 1
                continue

            method_items = list(iter_method_items(record))
            any_value = False

            for method_field, method_value in method_items:
                if has_value(method_value):
                    field_non_empty_counter[method_field] += 1
                    any_value = True

            if any_value:
                records_with_any_method_value += 1

    field_rows = [
        {
            "field": field,
            "count": count,
            "pct_total": pct(count, total_records),
        }
        for field, count in field_non_empty_counter.most_common()
    ]

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "total_records": total_records,
        "parsed_records": parsed_records,
        "json_decode_errors": json_decode_errors,
        "records_with_any_method_value": records_with_any_method_value,
        "records_with_any_method_value_pct": pct(records_with_any_method_value, total_records),
        "field_non_empty": field_rows,
    }

    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, md_output)

    print(f"Wrote JSON summary to {json_output}")
    print(f"Wrote Markdown summary to {md_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
