#!/usr/bin/env python3
"""Create a reproducible PDF-backed sample for manual extraction validation."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


SCHEMA_FIELDS = (
    "objective",
    "research_questions",
    "exact_boolean_queries",
    "keywords_used",
    "inclusion_criteria",
    "exclusion_criteria",
    "n_studies_initial",
    "n_studies_final",
    "year_range",
    "databases_used",
    "snowballing",
)
ID_RE = re.compile(r"W\d+$")


def normalize_id(value: Any) -> str | None:
    """Return the OpenAlex W-id from a URL, shard path, or bare id."""
    if not isinstance(value, str):
        return None
    match = ID_RE.search(value.strip().rstrip("/"))
    return match.group(0) if match else None


def resolve_pdf_path(pdf_dir: Path, doc_id: str) -> Path:
    return pdf_dir / doc_id[:2] / f"{doc_id}.pdf"


def has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def build_task(record: dict[str, Any], metadata: dict[str, dict[str, Any]], pdf_path: Path) -> dict[str, Any]:
    doc_id = normalize_id(record.get("doc_id"))
    if not doc_id:
        raise ValueError(f"invalid repaired record id: {record.get('doc_id')!r}")
    extraction = record.get("extraction", {})
    fields = []
    null_fields = []
    for name in SCHEMA_FIELDS:
        if name == "exact_boolean_queries":
            queries = extraction.get(name) or []
            if isinstance(queries, list):
                for index, query in enumerate(queries):
                    if not isinstance(query, dict) or not has_value(query.get("boolean_query_string")):
                        continue
                    fields.append({
                        "name": f"{name}[{index}]",
                        "value": query["boolean_query_string"],
                        "evidence_span": query.get("verbatim_source"),
                    })
            continue
        evidence = extraction.get(name, {})
        value = evidence.get("value") if isinstance(evidence, dict) else evidence
        span = evidence.get("verbatim_source") if isinstance(evidence, dict) else None
        if has_value(value):
            fields.append({"name": name, "value": value, "evidence_span": span})
        else:
            null_fields.append(name)

    meta = metadata.get(doc_id, {})
    return {
        "doc_id": doc_id,
        "title": meta.get("title", ""),
        "field": meta.get("field", "Unknown") or "Unknown",
        "pdf": str(pdf_path),
        "fields": fields,
        "null_fields": null_fields,
    }


def make_length_bins(sizes: list[int], n_bins: int = 4) -> dict[int, str]:
    if not sizes or n_bins < 1:
        raise ValueError("sizes must be non-empty and n_bins must be positive")
    labels = ["short", "long"] if n_bins == 2 else [f"q{i}" for i in range(1, n_bins + 1)]
    ordered = sorted(sizes)
    result = {}
    for rank, size in enumerate(ordered):
        result.setdefault(size, labels[min(rank * n_bins // len(ordered), n_bins - 1)])
    return result


def sample_records(
    records: list[dict[str, Any]], sample_size: int, seed: int, length_bins: dict[int, str]
) -> list[dict[str, Any]]:
    if sample_size > len(records):
        raise ValueError(f"sample size {sample_size} exceeds eligible records {len(records)}")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        length_bin = length_bins[record["pdf_size"]]
        record["length_bin"] = length_bin
        grouped[(record.get("field", "Unknown") or "Unknown", length_bin)].append(record)

    ideal = {key: sample_size * len(value) / len(records) for key, value in grouped.items()}
    allocation = {key: min(len(grouped[key]), int(value)) for key, value in ideal.items()}
    remaining = sample_size - sum(allocation.values())
    ranked = sorted(ideal, key=lambda key: (ideal[key] - int(ideal[key]), key), reverse=True)
    for key in ranked:
        if remaining and allocation[key] < len(grouped[key]):
            allocation[key] += 1
            remaining -= 1
    if remaining:
        raise ValueError("could not allocate requested sample across strata")

    rng = random.Random(seed)
    selected = []
    for key in sorted(grouped):
        selected.extend(rng.sample(grouped[key], allocation[key]))
    rng.shuffle(selected)
    return selected


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction-dir", type=Path, default=Path("data/extraction/repaired_fact_checked"))
    parser.add_argument("--metadata-jsonl", type=Path, default=Path("data/retrieval/merged/oax_slim.jsonl"))
    parser.add_argument("--pdf-dir", type=Path, default=Path("data/retrieval/merged/pdfs"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation_sample"))
    parser.add_argument("--sample-size", type=int, default=385)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--length-bins", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = {}
    for record in load_jsonl(args.metadata_jsonl):
        doc_id = normalize_id(record.get("id"))
        if doc_id:
            metadata[doc_id] = record

    candidates = []
    skipped = defaultdict(int)
    for path in sorted(args.extraction_dir.rglob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        doc_id = normalize_id(record.get("doc_id"))
        if not doc_id:
            skipped["invalid_id"] += 1
            continue
        pdf_path = resolve_pdf_path(args.pdf_dir, doc_id)
        if not pdf_path.exists():
            skipped["missing_pdf"] += 1
            continue
        task = build_task(record, metadata, pdf_path)
        task["pdf_size"] = pdf_path.stat().st_size
        candidates.append(task)

    bins = make_length_bins([r["pdf_size"] for r in candidates], args.length_bins)
    selected = sample_records(candidates, args.sample_size, args.seed, bins)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_path = args.output_dir / f"tasks_{args.sample_size}.jsonl"
    with task_path.open("w", encoding="utf-8") as handle:
        for task in selected:
            task = {k: v for k, v in task.items() if k not in {"pdf_size", "length_bin"}}
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")

    manifest = {
        "seed": args.seed,
        "sample_size": args.sample_size,
        "eligible_with_pdf": len(candidates),
        "skipped": dict(skipped),
        "length_measure": "PDF file size in bytes",
        "length_bins": args.length_bins,
        "strata": {},
        "sampled_doc_ids": [task["doc_id"] for task in selected],
    }
    for task in selected:
        key = f"{task['field']}|{task['length_bin']}"
        manifest["strata"][key] = manifest["strata"].get(key, 0) + 1
    (args.output_dir / "sampling_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"eligible with PDFs: {len(candidates)}")
    print(f"sampled: {len(selected)} -> {task_path}")


if __name__ == "__main__":
    main()