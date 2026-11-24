import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "src/final_ds"
CHECKER = SOURCE / "check_completeness.py"


def run_checker(directory, lines=None):
    if lines is not None:
        source = directory / "data/final_ds/sr4all_final.jsonl"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(CHECKER)],
        cwd=directory, capture_output=True, text=True, timeout=10,
    )


@pytest.fixture
def checker(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location("completeness_checker", CHECKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("existing_log", [False, True])
def test_import_does_not_create_or_truncate_logs(tmp_path, existing_log):
    log = tmp_path / "logs/final_ds/completeness_check_all.log"
    if existing_log:
        log.parent.mkdir(parents=True)
        log.write_text("previous run\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-c", f"import runpy; runpy.run_path({str(CHECKER)!r})"],
        cwd=tmp_path, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    if existing_log:
        assert log.read_text() == "previous run\n"
    else:
        assert not log.parent.exists()


@pytest.mark.parametrize("value", [
    {"value": " \t"},
    {"value": [None, " "]},
    [{"boolean_query_string": None, "database_source": ["Scopus"]}],
    [None, {"value": []}],
    [" ", ""],
])
def test_empty_wrappers_and_ghost_lists_are_not_filled(checker, value):
    assert not checker.is_filled(value)


@pytest.mark.parametrize("value", [{"value": False}, {"value": 0}])
def test_false_and_zero_are_observed_values(checker, value):
    assert checker.is_filled(value)


@pytest.fixture
def methodology_records():
    return [
        {"objective": {"value": "Aim"}, "inclusion_criteria": {"value": ["Adults"]},
         "exact_boolean_queries": [{"boolean_query_string": " "}],
         "keywords_used": {"value": [" "]}},
        {"objective": {"value": "Other aim"}, "exclusion_criteria": {"value": ["Grey literature"]},
         "exact_boolean_queries": [{"boolean_query_string": None}],
         "keywords_used": {"value": ["screening"]}},
        {"objective": {"value": " "}, "inclusion_criteria": {"value": None},
         "exact_boolean_queries": [None, {"boolean_query_string": None}],
         "keywords_used": {"value": [" "]}},
        {"year_range": {"value": [2000, 2020]}},
    ]


@pytest.mark.parametrize("nested", [False, True])
def test_report_uses_consistent_presence_and_counts_empty_documents(
    tmp_path, methodology_records, nested,
):
    records = [
        {"id": f"W{index}", **({"extraction": data} if nested else data)}
        for index, data in enumerate(methodology_records, 1)
    ]
    result = run_checker(tmp_path, [json.dumps(record) for record in records])
    assert result.returncode == 0, result.stderr
    assert "records=4, docs_with_extraction=2, invalid=0" in result.stderr
    metrics = {}
    for line in result.stderr.splitlines():
        match = re.search(r"([^|]+)\|\s*(\d+)\s*\|\s*([\d.]+)%$", line)
        if match:
            metrics[match[1].strip()] = (int(match[2]), float(match[3]))
    assert metrics["objective"] == (2, 50.0)
    assert metrics["exact_boolean_queries"] == (0, 0.0)
    assert metrics["research_questions"] == (0, 0.0)
    assert metrics["year_range"] == (1, 25.0)
    assert metrics["Docs with ALL fields null/empty"] == (1, 25.0)
    assert metrics["Strategy (Queries OR Keywords)"] == (1, 50.0)
    assert metrics["Essentials complete (all 3)"] == (1, 50.0)


@pytest.mark.parametrize("invalid", [
    "{broken", "[]", '{"extraction": []}', '{"exact_boolean_queries": ["term"]}',
])
def test_invalid_records_are_identified_and_fail_the_run(tmp_path, invalid):
    result = run_checker(tmp_path, ['{"objective": "Aim"}', invalid])
    assert result.returncode == 1
    assert "line 2" in result.stderr
    assert "invalid=1" in result.stderr


def test_missing_input_fails_explicitly(tmp_path):
    result = run_checker(tmp_path)
    assert result.returncode == 1
    assert "Input file not found" in result.stderr


def test_cli_preserves_previous_run_logs(tmp_path):
    log = tmp_path / "logs/final_ds/completeness_check_all.log"
    log.parent.mkdir(parents=True)
    log.write_text("previous run\n", encoding="utf-8")
    for _ in range(2):
        result = run_checker(tmp_path, ['{"objective": "Aim"}'])
        assert result.returncode == 0, result.stderr
    text = log.read_text()
    assert text.startswith("previous run\n")
    assert text.count("COMPLETENESS REPORT") == 2


def test_splitter_does_not_route_ghost_queries_into_boolean_cohort(
    tmp_path, methodology_records,
):
    directory = tmp_path / "data/final"
    directory.mkdir(parents=True)
    records = [{"id": f"W{index}", **data} for index, data in enumerate(methodology_records, 1)]
    (directory / "sr4all_full_normalized_year_range.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(SOURCE / "3_split_search_strategy.py")],
        cwd=tmp_path, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert (directory / "sr4all_full_normalized_year_range_search_has_boolean.jsonl").read_text() == ""
    keywords = directory / "sr4all_full_normalized_year_range_search_keywords_only.jsonl"
    assert [json.loads(line)["id"] for line in keywords.read_text().splitlines()] == ["W2"]
