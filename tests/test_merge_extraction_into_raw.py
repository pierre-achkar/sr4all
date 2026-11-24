import json
import re
import subprocess
import sys
from pathlib import Path

import pytest


MERGER = Path(__file__).resolve().parents[1] / "src/final_ds/01_merge_extraction_into_raw.py"
EXTRACTION = Path("data/extraction/repaired_fact_checked/repaired_fact_checked_merged.jsonl")
OPENALEX = Path("data/retrieval/merged/oax_slim.jsonl")
OUTPUT = Path("data/final_ds/oax_slim_with_extraction.jsonl")
SAMPLE = Path("data/final_ds/oax_slim_with_extraction_sample_5.jsonl")
LOG = Path("logs/final_ds/merge_extraction_into_raw.log")


def run_merge(directory):
    return subprocess.run(
        [sys.executable, str(MERGER)],
        cwd=directory, capture_output=True, text=True, timeout=10,
    )


@pytest.fixture
def merge_inputs(tmp_path):
    extractions = [
        {"doc_id": "W1", "extraction": {"objective": {"value": "Earlier aim"}}},
        {"doc_id": "batch/W1", "extraction": {
            "objective": {"value": "Latest aim", "verbatim_source": "Evidence"},
            "snowballing": {"value": False},
            "n_studies_final": {"value": 0},
            "exact_boolean_queries": [
                None,
                {"boolean_query_string": "(café OR tea)", "database_source": ["Scopus"],
                 "verbatim_source": "Query evidence", "verification_score": 0.9},
            ],
        }},
        {"doc_id": "W2", "extraction": None},
        *[
            {"doc_id": f"W{index}", "extraction": {"objective": {"value": f"Aim {index}"}}}
            for index in range(3, 7)
        ],
    ]
    records = [
        {"id": "W0", "title": "Unmatched first", "year": 2024},
        {"id": "https://openalex.org/W1", "title": "Café", "year": 2024},
        *[{"id": f"W{index}", "title": f"Review {index}"} for index in range(2, 8)],
    ]
    for path, rows in [(EXTRACTION, extractions), (OPENALEX, records)]:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    return tmp_path


def test_merge_preserves_rows_ids_and_last_duplicate_and_limits_matched_sample(merge_inputs):
    result = run_merge(merge_inputs)
    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in (merge_inputs / OUTPUT).read_text().splitlines()]
    assert [record["id"] for record in records] == [
        "W0", "https://openalex.org/W1", "W2", "W3", "W4", "W5", "W6", "W7",
    ]
    assert records[0] == {"id": "W0", "title": "Unmatched first", "year": 2024}
    assert records[1] == {
        "id": "https://openalex.org/W1", "title": "Café", "year": 2024,
        "objective": "Latest aim", "snowballing": False, "n_studies_final": 0,
        "exact_boolean_queries": [
            {"boolean_query_string": "(café OR tea)", "database_source": ["Scopus"]},
        ],
    }
    assert records[2] == {"id": "W2", "title": "Review 2"}
    assert records[-1] == {"id": "W7", "title": "Review 7"}
    sample = [json.loads(line) for line in (merge_inputs / SAMPLE).read_text().splitlines()]
    assert [record["id"] for record in sample] == [
        "https://openalex.org/W1", "W2", "W3", "W4", "W5",
    ]
    assert sample == records[1:6]
    for label, count in [
        ("Duplicate extraction IDs", 1), ("Matched records", 6), ("Unmatched OpenAlex", 2),
    ]:
        assert re.search(rf"{label}:\s+{count}\b", result.stderr)


@pytest.mark.parametrize("existing_log", [False, True])
def test_import_does_not_create_outputs_or_create_or_truncate_logs(tmp_path, existing_log):
    log = tmp_path / LOG
    if existing_log:
        log.parent.mkdir(parents=True)
        log.write_text("previous run\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-c", f"import runpy; runpy.run_path({str(MERGER)!r})"],
        cwd=tmp_path, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / OUTPUT).parent.exists()
    if existing_log:
        assert log.read_text() == "previous run\n"
    else:
        assert not log.parent.exists()


@pytest.mark.parametrize("missing", [EXTRACTION, OPENALEX])
def test_missing_input_fails_without_publishing_outputs(merge_inputs, missing):
    (merge_inputs / missing).unlink()
    result = run_merge(merge_inputs)
    assert result.returncode == 1
    assert str(missing) in result.stderr
    assert not (merge_inputs / OUTPUT).exists()
    assert not (merge_inputs / SAMPLE).exists()


@pytest.mark.parametrize("source,invalid,line_number", [
    (EXTRACTION, "{broken", 8),
    (OPENALEX, "{broken", 9),
    (EXTRACTION, "[]", 8),
    (OPENALEX, "[]", 9),
    (EXTRACTION, '{"doc_id": null, "extraction": {}}', 8),
    (OPENALEX, '{"id": " "}', 9),
    (EXTRACTION, '{"doc_id": "W9", "extraction": []}', 8),
    (EXTRACTION, '{"doc_id": "W9", "extraction": {"exact_boolean_queries": ["term"]}}', 8),
    (EXTRACTION, '{"doc_id": "W1", "extraction": {"id": "W99"}}', 8),
])
def test_invalid_input_reports_context_and_preserves_previous_outputs(
    merge_inputs, source, invalid, line_number,
):
    for path in [OUTPUT, SAMPLE]:
        target = merge_inputs / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("previous complete output\n", encoding="utf-8")
    with (merge_inputs / source).open("a", encoding="utf-8") as stream:
        stream.write(invalid + "\n")

    result = run_merge(merge_inputs)
    assert result.returncode == 1, result.stderr
    assert str(source) in result.stderr
    assert f"line {line_number}" in result.stderr
    assert "MERGE COMPLETE" not in result.stderr
    for path in [OUTPUT, SAMPLE]:
        assert (merge_inputs / path).read_text() == "previous complete output\n"
    assert set((merge_inputs / OUTPUT).parent.iterdir()) == {
        merge_inputs / OUTPUT, merge_inputs / SAMPLE,
    }


def test_failed_raw_merge_does_not_leave_partial_outputs(merge_inputs):
    with (merge_inputs / OPENALEX).open("a", encoding="utf-8") as stream:
        stream.write("{broken\n")
    result = run_merge(merge_inputs)
    assert result.returncode == 1
    assert not (merge_inputs / OUTPUT).exists()
    assert not (merge_inputs / SAMPLE).exists()
    parent = (merge_inputs / OUTPUT).parent
    assert not parent.exists() or not list(parent.iterdir())


def test_sample_publication_failure_preserves_main_output_and_cleans_temporary_files(
    merge_inputs,
):
    output = merge_inputs / OUTPUT
    output.parent.mkdir(parents=True)
    output.write_text("previous complete dataset\n", encoding="utf-8")
    (merge_inputs / SAMPLE).mkdir()
    result = run_merge(merge_inputs)
    assert result.returncode == 1
    assert output.read_text() == "previous complete dataset\n"
    assert SAMPLE.name in result.stderr
    assert "MERGE COMPLETE" not in result.stderr
    assert "Traceback" not in result.stderr
    assert set(output.parent.iterdir()) == {output, merge_inputs / SAMPLE}


def test_cli_appends_logs_across_successful_reruns(merge_inputs):
    log = merge_inputs / LOG
    log.parent.mkdir(parents=True)
    log.write_text("previous run\n", encoding="utf-8")
    for _ in range(2):
        result = run_merge(merge_inputs)
        assert result.returncode == 0, result.stderr
    text = log.read_text()
    assert text.startswith("previous run\n")
    assert text.count("MERGE COMPLETE") == 2
