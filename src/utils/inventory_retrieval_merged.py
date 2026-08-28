"""Create a coverage inventory for merged OpenAlex retrieval artifacts."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


WORK_ID_PATTERN = re.compile(r"W\d+")


def is_populated(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return True


def work_id(value: str) -> str | None:
    matches = WORK_ID_PATTERN.findall(value)
    return matches[-1] if matches else None


def markdown_work_ids(markdown_dir: Path) -> set[str]:
    return {
        matched_id
        for path in markdown_dir.rglob("*.md")
        for matched_id in [work_id(str(path))]
        if matched_id is not None
    }


def percentage(numerator: int, denominator: int) -> str:
    return f"{(100 * numerator / denominator) if denominator else 0:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--markdown-dir", type=Path, required=True)
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    markdown_ids = markdown_work_ids(args.markdown_dir)
    full_text_entries = len(markdown_ids)
    field_counts: Counter[str] = Counter()
    primary_field_counts: Counter[str] = Counter()
    total_entries = 0
    malformed_lines = 0
    unidentifiable_entries = 0

    with args.jsonl.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed_lines += 1
                continue

            if not isinstance(record, dict):
                malformed_lines += 1
                continue

            total_entries += 1
            field_counts.update(
                key for key, value in record.items() if is_populated(value)
            )
            primary_field = record.get("field")
            if is_populated(primary_field):
                primary_field_counts[str(primary_field).strip()] += 1
            record_id = work_id(str(record.get("id", "")))
            if record_id is None:
                unidentifiable_entries += 1

    markdown_files = sum(1 for path in args.markdown_dir.rglob("*.md"))
    pdf_files = sum(1 for path in args.pdf_dir.rglob("*") if path.is_file())
    report_time = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "# Merged Retrieval Data Inventory",
        "",
        f"Generated: {report_time}",
        "",
        "## Inputs",
        f"- JSONL: `{args.jsonl}`",
        f"- Parsed Markdown directory: `{args.markdown_dir}`",
        f"- PDF directory: `{args.pdf_dir}`",
        "",
        "## Record Coverage",
        "",
        "| Measure | Count | Percentage of JSONL entries |",
        "| --- | ---: | ---: |",
        f"| Total valid JSONL entries | {total_entries:,} | 100.00% |",
        (
            f"| Works with parsed Markdown full text | {full_text_entries:,} | "
            f"{percentage(full_text_entries, total_entries)} |"
        ),
        (
                f"| Works without parsed Markdown full text | "
            f"{total_entries - full_text_entries:,} | "
            f"{percentage(total_entries - full_text_entries, total_entries)} |"
        ),
        "",
        "## Artifact Counts",
        "",
        f"- Parsed Markdown files found: {markdown_files:,}",
        f"- PDF files found: {pdf_files:,}",
        "",
        "## JSONL Field Coverage",
        "",
        "A field is counted when its value is not null, not an empty string, and not an empty list or object.",
        "",
        "| Field | Populated entries | Coverage |",
        "| --- | ---: | ---: |",
    ]
    lines.extend(
        f"| `{field}` | {count:,} | {percentage(count, total_entries)} |"
        for field, count in sorted(field_counts.items())
    )
    lines.extend(
        [
            "",
            "## Primary Field Distribution",
            "",
            "Percentages use records with a populated `field` value as the denominator.",
            "",
            "| Primary field | Entries | Percentage |",
            "| --- | ---: | ---: |",
        ]
    )
    lines.extend(
        f"| {field} | {count:,} | {percentage(count, field_counts['field'])} |"
        for field, count in sorted(primary_field_counts.items())
    )
    lines.extend(
        [
            "",
            "## Data Quality Notes",
            "",
            f"- Malformed or non-object JSONL lines skipped: {malformed_lines:,}",
            f"- JSONL entries without an identifiable OpenAlex work ID: {unidentifiable_entries:,}",
            "- Full-text work coverage is determined from Markdown files using the final `W<digits>` token in each Markdown path.",
        ]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()