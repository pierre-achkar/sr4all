"""Prepare page-numbered evidence for the supplemental query-presence audit."""

import argparse
import hashlib
import json
import re
import sys
import tempfile
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--doc-ids", nargs="+")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output directory already exists; preserve the prior audit")
    root = Path("data/validation_sample")
    tasks_path = root / "tasks_385.jsonl"
    tasks = list(read_jsonl(tasks_path))
    selected = [task for task in tasks if not any(
        node["name"].startswith("exact_boolean_queries") for node in task["fields"]
    )]
    if args.doc_ids:
        wanted = set(args.doc_ids)
        selected = [task for task in selected if task["doc_id"] in wanted]
        if {task["doc_id"] for task in selected} != wanted:
            parser.error("requested IDs are not all empty-query sampled reviews")
    elif len(selected) != 305:
        parser.error("expected exactly 305 empty-query sampled reviews")
    if len({task["doc_id"] for task in tasks}) != len(tasks):
        parser.error("duplicate master-task identifiers")
    selected.sort(key=lambda task: task["doc_id"])
    source_notes = defaultdict(list)
    inputs = [{"path": str(tasks_path), "sha256": sha256(tasks_path)}]
    for path in sorted((root / "results").glob("annotations_final_*.jsonl")):
        inputs.append({"path": str(path), "sha256": sha256(path)})
        for row in read_jsonl(path):
            for kind in ("field_judgments", "null_field_judgments"):
                for node in row.get(kind, []):
                    comment = node.get("comment") or ""
                    if re.search(r"wrong (?:pdf|paper|article)|checklist|supplement|appendi", comment, re.I):
                        source_notes[row["doc_id"]].append({
                            "annotator": row["annotator"], "field": node["name"],
                            "comment": comment,
                        })
    requirement = next(line.strip() for line in Path("requirements.txt").read_text().splitlines()
                       if line.strip().startswith("pypdf=="))
    package, version = requirement.split("==", 1)
    with urllib.request.urlopen(f"https://pypi.org/pypi/{package}/{version}/json", timeout=30) as response:
        metadata = json.load(response)
    wheel = next(item for item in metadata["urls"] if item["filename"].endswith("-py3-none-any.whl"))
    if urllib.parse.urlparse(wheel["url"]).hostname != "files.pythonhosted.org":
        raise ValueError("unexpected PDF-reader download host")
    with urllib.request.urlopen(wheel["url"], timeout=30) as response:
        content = response.read(2_000_001)
    if len(content) > 2_000_000 or hashlib.sha256(content).hexdigest() != wheel["digests"]["sha256"]:
        raise ValueError("PDF-reader wheel size or checksum mismatch")
    args.out.mkdir(parents=True)
    counts = Counter()
    with tempfile.TemporaryDirectory(prefix="sr4all-query-audit-") as temporary:
        wheel_path = Path(temporary) / wheel["filename"]
        wheel_path.write_bytes(content)
        sys.path.insert(0, str(wheel_path))
        from pypdf import PdfReader

        for batch_index in range(0, len(selected), 39):
            batch_path = args.out / f"batch_{batch_index // 39:02d}.jsonl"
            with batch_path.open("x", encoding="utf-8") as stream:
                for task in selected[batch_index:batch_index + 39]:
                    pdf_path = Path(task["pdf"])
                    record = {key: task[key] for key in ("doc_id", "title", "pdf")}
                    record.update({"pdf_sha256": sha256(pdf_path), "pages": [],
                                   "source_notes": source_notes[task["doc_id"]], "extraction_error": None})
                    try:
                        reader = PdfReader(pdf_path)
                        for page_number, page in enumerate(reader.pages, 1):
                            record["pages"].append({"page": page_number, "text": page.extract_text() or ""})
                    except Exception as error:
                        record["extraction_error"] = f"{type(error).__name__}: {error}"
                        counts["extraction_errors"] += 1
                    stream.write(json.dumps(record, ensure_ascii=True) + "\n")
                    counts["documents"] += 1
                    counts["pages"] += len(record["pages"])
            print(batch_path, flush=True)
    manifest = {"assessment": "post-hoc AI-assisted source-presence audit", "inputs": inputs,
                "pdf_reader": requirement, "wheel_sha256": wheel["digests"]["sha256"],
                "counts": dict(counts), "doc_ids": [task["doc_id"] for task in selected],
                "batch_sha256": {path.name: sha256(path) for path in sorted(args.out.glob("batch_*.jsonl"))}}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(counts)))


if __name__ == "__main__":
    main()