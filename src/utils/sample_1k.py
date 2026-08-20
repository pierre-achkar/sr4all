"""Create a field-restricted SR subset and copy corresponding markdown files.

Default behavior:
1) Read sr4all_full.jsonl as a stream.
2) Keep records with non-empty objective, research_questions, at least one of
    inclusion_criteria or exclusion_criteria, and referenced_works_count > 15.
3) Keep only records whose field is "Computer Science".
4) Write all matching records to rag4report/sr4all_computer_science_subset.jsonl.
5) Copy markdown files (based on OpenAlex W-id from each record id) to rag4report/md.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any


WORK_ID_RE = re.compile(r"(W\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path("data_old/release/sr4all_full.jsonl"),
        help="Input SR JSONL file",
    )
    parser.add_argument(
        "--md-root",
        type=Path,
        default=Path("data_old/sr4all/md"),
        help="Root directory containing markdown files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("rag4report"),
        help="Output directory for subset jsonl and markdown files",
    )
    parser.add_argument(
        "--out-jsonl-name",
        default="sr4all_computer_science_subset.jsonl",
        help="Output jsonl filename inside --out-dir",
    )
    parser.add_argument(
        "--field",
        default="Computer Science",
        help="Field value to keep (case-insensitive exact match)",
    )
    parser.add_argument(
        "--min-refs",
        type=int,
        default=10,
        help="Minimum references count threshold (strictly greater than this value)",
    )
    return parser.parse_args()


def has_content(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set)):
        return any(has_content(v) for v in value)
    if isinstance(value, dict):
        return bool(value)
    return True


def normalize_field(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item.strip()
    return "Unknown"


def get_references_count(record: dict[str, Any]) -> int:
    value = record.get("referenced_works_count")
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0

    coverage = record.get("references_abstract_coverage")
    if isinstance(coverage, dict):
        total_refs = coverage.get("total_refs")
        if isinstance(total_refs, (int, float)):
            return int(total_refs)
        if isinstance(total_refs, str):
            try:
                return int(total_refs)
            except ValueError:
                return 0
    return 0


def extract_work_id(record_id: Any) -> str | None:
    if not isinstance(record_id, str):
        return None
    match = WORK_ID_RE.search(record_id.strip())
    if not match:
        return None
    return match.group(1)


def resolve_md_source_path(md_root: Path, work_id: str) -> Path:
    # Shard layout observed in this dataset: md/W22/W22776/W2277613887.md
    return md_root / work_id[:3] / work_id[:6] / f"{work_id}.md"


def is_eligible(record: dict[str, Any], min_refs: int) -> bool:
    if not has_content(record.get("objective")):
        return False
    if not has_content(record.get("research_questions")):
        return False
    has_inclusion = has_content(record.get("inclusion_criteria"))
    has_exclusion = has_content(record.get("exclusion_criteria"))
    if not (has_inclusion or has_exclusion):
        return False
    return get_references_count(record) > min_refs


def collect_subset(
    input_jsonl: Path, min_refs: int, target_field: str
) -> tuple[list[dict[str, Any]], int, int, int]:
    total = 0
    eligible = 0
    field_matched = 0
    subset: list[dict[str, Any]] = []
    normalized_target = target_field.strip().lower()

    with input_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            total += 1
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not is_eligible(record, min_refs):
                continue

            eligible += 1
            field_value = normalize_field(record.get("field"))
            if field_value.lower() != normalized_target:
                continue

            field_matched += 1
            subset.append(record)

    return subset, total, eligible, field_matched


def write_jsonl(records: list[dict[str, Any]], output_jsonl: Path) -> None:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def copy_markdown_files(
    records: list[dict[str, Any]], md_root: Path, out_md_root: Path
) -> tuple[int, int]:
    copied = 0
    missing = 0
    out_md_root.mkdir(parents=True, exist_ok=True)

    for rec in records:
        work_id = extract_work_id(rec.get("id"))
        if not work_id:
            missing += 1
            continue

        source = resolve_md_source_path(md_root, work_id)
        if not source.exists():
            missing += 1
            continue

        relative_source = source.relative_to(md_root)
        target = out_md_root / relative_source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied += 1

    return copied, missing


def flatten_markdown_dir(out_md_root: Path) -> int:
    """Move all .md files in subdirectories directly into out_md_root, then remove empty dirs."""
    moved = 0
    for md_file in list(out_md_root.rglob("*.md")):
        if md_file.parent == out_md_root:
            continue
        dest = out_md_root / md_file.name
        if not dest.exists():
            md_file.rename(dest)
            moved += 1

    # Remove now-empty subdirectories
    for subdir in sorted(out_md_root.rglob("*"), reverse=True):
        if subdir.is_dir() and not any(subdir.iterdir()):
            subdir.rmdir()

    return moved


def main() -> None:
    args = parse_args()

    subset, total, eligible, field_matched = collect_subset(
        input_jsonl=args.input_jsonl,
        min_refs=args.min_refs,
        target_field=args.field,
    )

    out_jsonl = args.out_dir / args.out_jsonl_name
    write_jsonl(subset, out_jsonl)

    copied, missing = copy_markdown_files(
        records=subset,
        md_root=args.md_root,
        out_md_root=args.out_dir / "md",
    )

    print(f"Input records scanned: {total}")
    print(f"Eligible records found: {eligible}")
    print(f"Field filter: {args.field}")
    print(f"Eligible records in field: {field_matched}")
    print(f"Subset records written: {len(subset)}")
    print(f"Subset JSONL: {out_jsonl}")
    print(f"Markdown copied: {copied}")
    print(f"Markdown missing/unresolved: {missing}")
    out_md_root = args.out_dir / "md"
    moved = flatten_markdown_dir(out_md_root)
    print(f"Markdown output root: {out_md_root}")
    print(f"Markdown files flattened: {moved}")


if __name__ == "__main__":
    main()
