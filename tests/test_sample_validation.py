import json
import subprocess
import zipfile
from pathlib import Path

from src.utils.sample_validation import (
    build_task,
    make_length_bins,
    sample_records,
)


def test_build_task_preserves_evidence_and_null_fields(tmp_path: Path):
    pdf = tmp_path / "W123.pdf"
    pdf.write_bytes(b"pdf")
    record = {
        "doc_id": "W1/W123",
        "extraction": {
            "objective": {"value": "Aim", "verbatim_source": "The aim."},
            "year_range": {"value": None, "verbatim_source": None},
        },
    }

    task = build_task(record, {"W123": {"title": "Example", "field": "Medicine"}}, pdf)

    assert task["doc_id"] == "W123"
    assert task["fields"] == [{"name": "objective", "value": "Aim", "evidence_span": "The aim."}]
    assert "year_range" in task["null_fields"]
    assert "objective" not in task["null_fields"]


def test_build_task_flattens_boolean_query_evidence(tmp_path: Path):
    pdf = tmp_path / "W123.pdf"
    pdf.write_bytes(b"pdf")
    record = {
        "doc_id": "W1/W123",
        "extraction": {
            "exact_boolean_queries": [{
                "verbatim_source": "(burnout OR stress)",
                "boolean_query_string": "(burnout OR stress)",
                "database_source": ["PubMed"],
            }],
        },
    }

    task = build_task(record, {}, pdf)

    assert task["fields"] == [{
        "name": "exact_boolean_queries[0]",
        "value": "(burnout OR stress)",
        "evidence_span": "(burnout OR stress)",
    }]
    assert "exact_boolean_queries" not in task["null_fields"]
    assert task["field"] == "Medicine"
    assert task["pdf"] == str(pdf)


def test_sample_records_is_deterministic_and_exact(tmp_path: Path):
    records = [
        {"doc_id": f"W{i}", "field": "Medicine" if i < 4 else "Engineering", "pdf_size": i + 1}
        for i in range(8)
    ]
    bins = make_length_bins([r["pdf_size"] for r in records], 2)

    first = sample_records(records, 5, seed=17, length_bins=bins)
    second = sample_records(records, 5, seed=17, length_bins=bins)

    assert [r["doc_id"] for r in first] == [r["doc_id"] for r in second]
    assert len(first) == 5
    assert len({r["doc_id"] for r in first}) == 5
    assert {r["length_bin"] for r in first} == {"short", "long"}


def test_validation_package_includes_guidelines_and_panel(tmp_path: Path):
    pdf_dir = tmp_path / "pdfs" / "W1"
    pdf_dir.mkdir(parents=True)
    (pdf_dir / "W123.pdf").write_bytes(b"pdf")
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps({"doc_id": "W123", "pdf": "W123.pdf"}) + "\n", encoding="utf-8")
    out = tmp_path / "packages"
    tool = Path("sr4all-validation-toolkit/sr4all-validation/tool/index.html")

    subprocess.run([
        "python3", "sr4all-validation-toolkit/sr4all-validation/scripts/make_packages.py",
        "--tasks", str(tasks), "--pdf-dir", str(tmp_path / "pdfs"), "--tool", str(tool),
        "--annotators", "test", "--shared", "0", "--out", str(out),
    ], check=True)

    with zipfile.ZipFile(out / "package_test.zip") as package:
        names = set(package.namelist())
        html = package.read("index.html").decode("utf-8")
        assert "ANNOTATOR_GUIDELINES.md" in names
        assert 'id="guidelinesBtn"' in html


def test_validation_tool_explains_incomplete_document_status():
    html = Path("sr4all-validation-toolkit/sr4all-validation/tool/index.html").read_text(
        encoding="utf-8"
    )

    assert "incomplete" in html
    assert "remaining" in html
    assert "a.null_fields[f]?.judgment" in html
    assert "a.null_fields[f.name]?.judgment" not in html