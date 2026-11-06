#!/usr/bin/env python3
"""Build a standalone HTML view of an existing annotation evaluation snapshot."""
import argparse
import csv
import hashlib
import html
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote


def load_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def build_report(report_dir, output=None, tasks_path=None, query_source_audit_path=None):
    report_dir = Path(report_dir).resolve()
    output = Path(output).resolve() if output else report_dir / "dashboard.html"
    if output.suffix.lower() not in (".html", ".htm"):
        raise ValueError("output must be an HTML file, never a source report")
    payload = {}
    for key, name in {
        "summary": "summary.json", "performance": "results_per_field.json",
        "agreement": "fleiss_kappa.json", "pairwise": "pairwise_agreement.json",
        "comments": "comment_analysis.json", "sample": "sample_report.json",
    }.items():
        payload[key] = load_json(report_dir / name)
    payload["audit"] = load_jsonl(report_dir / "final_labels.jsonl")
    payload["errors"] = load_jsonl(report_dir / "error_cases.jsonl")
    for key, name in {"annotatorMetrics": "annotator_metrics.tsv", "errorCounts": "error_counts.tsv"}.items():
        with (report_dir / name).open(encoding="utf-8", newline="") as stream:
            payload[key] = list(csv.DictReader(stream, delimiter="\t"))
    coverage = {row["field"]: row for row in payload["sample"]["task_field_coverage"]}
    for row in payload["performance"] + payload["annotatorMetrics"]:
        if row["field"] == "exact_boolean_queries" and not coverage.get(row["field"], {}).get("null_tasks"):
            for key in ("recall", "rec_lo", "rec_hi", "n_recall", "missed"):
                row[key] = None
    payload["tasks"] = {}
    if tasks_path:
        for task in load_jsonl(Path(tasks_path)):
            if task["doc_id"] in payload["tasks"]:
                raise ValueError(f"duplicate task: {task['doc_id']}")
            payload["tasks"][task["doc_id"]] = task
    payload["querySourceAudit"] = None
    if query_source_audit_path:
        audit_path = Path(query_source_audit_path).resolve()
        with audit_path.open("rb") as stream:
            audit_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        payload["querySourceAudit"] = {
            "report": load_json(audit_path),
            "source": {
                "name": audit_path.name, "bytes": audit_path.stat().st_size,
                "sha256": audit_sha256,
                "href": quote(os.path.relpath(audit_path, output.parent)),
            },
        }
    payload["recallPerformance"] = [dict(row) for row in payload["performance"]]
    if payload["querySourceAudit"]:
        query_audit = payload["querySourceAudit"]["report"]
        query_row = next(row for row in payload["recallPerformance"]
                         if row["field"] == "exact_boolean_queries")
        query_row.update({
            "recall": query_audit["recall"]["estimate"],
            "rec_lo": query_audit["recall"]["ci95"][0],
            "rec_hi": query_audit["recall"]["ci95"][1],
            "n_recall": query_audit["recall"]["denominator"],
            "missed": query_audit["missed"],
            "null_absent": (query_audit["source_decisions"]["absent"]
                            + query_audit["source_decisions"]["unresolved"]),
        })
    payload["snapshot"] = report_dir.name
    payload["generatedAt"] = datetime.now(timezone.utc).isoformat()
    payload["sourceReports"] = []
    for path in sorted(report_dir.iterdir()):
        if path.is_file() and path.suffix in (".json", ".jsonl", ".tsv", ".tex", ".md"):
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            payload["sourceReports"].append({
                "name": path.name, "bytes": path.stat().st_size, "sha256": digest,
                "href": quote(os.path.relpath(path, output.parent)),
            })
    embedded = json.dumps(payload, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    embedded = embedded.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    template = Path(__file__).resolve().parents[1] / "tool/evaluation_report.html"
    text = template.read_text(encoding="utf-8")
    if text.count("__REPORT_DATA__") != 1:
        raise ValueError("HTML template must contain exactly one data placeholder")
    text = text.replace("__REPORT_DATA__", embedded).replace("__SNAPSHOT__", html.escape(report_dir.name))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True, help="complete evaluation report snapshot")
    parser.add_argument("--out", type=Path, help="defaults to <dir>/dashboard.html")
    parser.add_argument("--tasks", type=Path, help="optional master tasks for original extraction and source context")
    parser.add_argument("--query-source-audit", type=Path,
                        help="optional supplemental audit of query presence in empty fields")
    args = parser.parse_args()
    try:
        path = build_report(args.dir, args.out, args.tasks, args.query_source_audit)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    print(f"HTML report: {path}")


if __name__ == "__main__":
    main()