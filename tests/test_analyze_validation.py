import csv
import importlib.util
import json
import tempfile
import unittest
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / (
    "sr4all-validation-toolkit/sr4all-validation/scripts/analyze.py"
)
SPEC = importlib.util.spec_from_file_location("validation_analyze", SCRIPT)
analyze = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyze)


def record(doc_id, label="correct", shared=False, null_label="absent"):
    return {
        "doc_id": doc_id,
        "status": "done",
        "shared": shared,
        "field_judgments": [
            {"name": "objective", "judgment": label, "error_type": None}
        ],
        "null_field_judgments": [
            {"name": "year_range", "judgment": null_label}
        ],
    }


class ValidationAnalysisTests(unittest.TestCase):
    def test_wilson_known_value_and_empty_denominator(self):
        estimate, lower, upper = analyze.wilson(5, 10)
        self.assertEqual(estimate, 0.5)
        self.assertAlmostEqual(lower, 0.236593, places=6)
        self.assertAlmostEqual(upper, 0.763407, places=6)
        self.assertEqual(analyze.wilson(0, 0), (None, None, None))

    def test_fleiss_reference_and_degenerate_cases(self):
        counts = [Counter(correct=2), Counter(partial=2), Counter(correct=1, partial=1)]
        kappa, agreement = analyze.fleiss_kappa(counts, ["correct", "partial"])
        self.assertAlmostEqual(kappa, 1 / 3)
        self.assertAlmostEqual(agreement, 2 / 3)
        self.assertEqual(analyze.fleiss_kappa([Counter(correct=2)], ["correct"]), (None, 1.0))
        self.assertEqual(analyze.fleiss_kappa([Counter(correct=1)], ["correct"]), (None, None))
        with self.assertRaises(ValueError):
            analyze.fleiss_kappa([Counter(correct=2), Counter(correct=3)], ["correct"])

    def test_disagreements_require_adjudication_without_changing_agreement(self):
        data = {
            "alice": {"W1": record("W1", shared=True)},
            "bob": {"W1": record("W1", label="partial", shared=True)},
        }
        result = analyze.evaluate(data)
        self.assertEqual(result["summary"]["unresolved_items"], 1)
        self.assertEqual(result["summary"]["estimate_status"], "provisional")
        objective = next(row for row in result["metrics"] if row["field"] == "objective")
        self.assertEqual(objective["n_precision"], 0)
        decision = [{"doc_id": "W1", "kind": "field", "field": "objective",
                     "judgment": "correct", "comment": "Checked together"}]
        resolved = analyze.evaluate(data, adjudications=decision)
        self.assertEqual(resolved["summary"]["unresolved_items"], 0)
        self.assertEqual(resolved["agreement"], result["agreement"])
        self.assertEqual(resolved["metrics"][0]["n_precision"], 1)

    def test_three_rater_majority_is_not_adjudication(self):
        data = {name: {"W1": record("W1", label=label, shared=True)}
                for name, label in [("alice", "correct"), ("bob", "correct"), ("cara", "partial")]}
        self.assertEqual(analyze.evaluate(data)["summary"]["unresolved_items"], 1)

    def test_pairwise_report_preserves_categories_and_reports_query_concentration(self):
        data = {name: {"W1": record("W1", shared=True), "W2": record("W2", shared=True)}
                for name in ("alice", "bob", "cara")}
        for name in data:
            data[name]["W1"]["field_judgments"] = [
                {"name": f"exact_boolean_queries[{index}]", "judgment": "partial" if name == "alice" else "correct"}
                for index in range(3)]
        result = analyze.evaluate(data)
        report = analyze.pairwise_report(result)
        pair = next(row for row in report["pairs"] if row["annotator_a"] == "alice"
                    and row["annotator_b"] == "bob" and row["scope"] == "all" and row["kind"] == "field")
        self.assertEqual(pair["n_items"], 4)
        self.assertEqual(pair["raw_agreement"], 0.25)
        self.assertEqual(pair["cohen_kappa"], 0)
        self.assertEqual(pair["document_balanced_agreement"], 0.5)
        collapsed = next(row for row in report["pairs"] if row["annotator_a"] == "alice"
                         and row["annotator_b"] == "bob" and row["scope"] == "document_field" and row["kind"] == "field")
        self.assertEqual(collapsed["n_items"], 2)
        self.assertEqual(collapsed["raw_agreement"], 0.5)
        self.assertEqual(report["largest_query_document"], "W1")
        self.assertEqual(report["query_documents"][0]["n_items"], 3)
        decision = [{"doc_id": "W1", "kind": "field", "field": "exact_boolean_queries[0]", "judgment": "correct"}]
        self.assertEqual(analyze.pairwise_report(analyze.evaluate(data, adjudications=decision)), report)

    def test_comment_inventory_includes_all_nonempty_comments_without_duplicate_votes(self):
        data = {name: {"W1": record("W1", shared=True)} for name in ("alice", "bob", "cara")}
        for name in data:
            data[name]["W1"]["field_judgments"][0]["comment"] = "Missing one database"
            data[name]["W1"]["null_field_judgments"][0]["comment"] = "  "
        result = analyze.evaluate(data)
        inventory = analyze.comment_inventory(result)
        self.assertEqual(len(inventory), 1)
        self.assertEqual(len(inventory[0]["comment_key"]), 16)
        self.assertEqual(inventory[0]["comment"], "Missing one database")
        self.assertEqual(len(inventory[0]["observations"]), 3)
        self.assertEqual({row["annotator"] for row in inventory[0]["observations"]}, set(data))
        self.assertEqual(inventory[0]["observations"][0]["judgment"], "correct")
        data["alice"]["W1"]["field_judgments"][0]["comment"] = "Missing two databases"
        self.assertEqual(len(analyze.comment_inventory(analyze.evaluate(data))), 2)

    def test_comment_coding_requires_exhaustive_valid_mapping_and_counts_unique_cases(self):
        data = {name: {"W1": record("W1", shared=True)} for name in ("alice", "bob", "cara")}
        for name in data:
            data[name]["W1"]["field_judgments"][0].update(judgment="partial", comment="Missing one database")
        result = analyze.evaluate(data)
        inventory = analyze.comment_inventory(result)
        codes = [{"comment_key": inventory[0]["comment_key"], "themes": ["incomplete_value"]}]
        report = analyze.coded_comment_report(result, codes)
        self.assertEqual(report["summary"]["comment_observations"], 3)
        self.assertEqual(report["summary"]["distinct_comment_contexts"], 1)
        self.assertEqual(report["counts"][0]["n_document_fields"], 1)
        self.assertEqual(report["counts"][0]["n_observations"], 3)
        self.assertEqual(report["counts"][0]["status"], "resolved_error")
        self.assertEqual(report["coded_comments"][0]["comment"], "Missing one database")
        for invalid in ([], codes + codes, [{"comment_key": "unknown", "themes": ["incomplete_value"]}],
                        [{"comment_key": inventory[0]["comment_key"], "themes": ["invented"]}],
                        [{"comment_key": inventory[0]["comment_key"], "themes": []}]):
            with self.assertRaises(ValueError):
                analyze.coded_comment_report(result, invalid)

    def test_agreement_reports_pairwise_agreement_and_unanimity_separately(self):
        data = {name: {"W1": record("W1", label=label, shared=True)}
                for name, label in [("alice", "correct"), ("bob", "correct"), ("cara", "partial")]}
        result = analyze.evaluate(data)
        self.assertEqual(result["agreement"]["extracted"]["n_unanimous"], 0)
        self.assertEqual(result["agreement"]["null"]["n_unanimous"], 1)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            analyze.write_results(result, out)
            with (out / "agreement_per_field.tsv").open() as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            objective = next(row for row in rows if row["field"] == "objective" and row["kind"] == "field")
            self.assertAlmostEqual(float(objective["raw_agreement"]), 1 / 3)
            self.assertEqual(objective["n_unanimous"], "0")
            table = (out / "agreement_table.tex").read_text()
            self.assertIn("Original independent ratings", table)
            self.assertIn("year range & 0 & -- & -- & 1 & 1.000 & --", table)

    def test_discussion_report_groups_pending_items_and_preserves_original_ratings(self):
        data = {name: {"W1": record("W1", label=label, shared=True)}
                for name, label in [("alice", "correct"), ("bob", "correct"), ("cara", "partial")]}
        data["cara"]["W1"]["field_judgments"][0].update(
            comment="Missing an objective", found_text="Evidence", page="2")
        data["cara"]["W1"]["null_field_judgments"][0]["judgment"] = "present"
        original = json.dumps(data, sort_keys=True)
        result = analyze.evaluate(data, adjudications=[
            {"doc_id": "W1", "kind": "null", "field": "year_range", "judgment": "absent"}])
        tasks = {"W1": {"title": "Review <one>", "fields": [{"name": "objective", "value": "Aim"}],
                        "null_fields": ["year_range"]}}
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            analyze.write_results(result, out, tasks=tasks)
            report = (out / "discussion_report.md").read_text()
            self.assertEqual(report.count("## W1"), 1)
            self.assertIn("1 unresolved items across 1 documents", report)
            self.assertIn("Review <one>", report)
            self.assertIn("Missing an objective", report)
            self.assertIn('"value": "Aim"', report)
            self.assertIn("| alice | correct |", report)
            self.assertIn("| cara | partial |", report)
            self.assertNotIn("### null: year_range", report)
            with (out / "discussion_documents.tsv").open() as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual(rows[0]["pending_items"], "1")
            self.assertEqual(rows[0]["pending_document_fields"], "1")
        self.assertEqual(json.dumps(data, sort_keys=True), original)

    def test_precision_recall_partial_and_hallucinated_span(self):
        docs = {f"W{index}": record(f"W{index}", label=label, null_label="present")
                for index, label in enumerate(["correct", "partial", "incorrect"])}
        docs["W2"]["field_judgments"][0]["error_type"] = "hallucinated span"
        docs["W3"] = record("W3")
        docs["W3"]["field_judgments"] = []
        docs["W3"]["null_field_judgments"] = [{"name": "objective", "judgment": "present"}]
        result = analyze.evaluate({"alice": docs})
        objective, year_range = result["metrics"]
        self.assertEqual(objective["n_precision"], 3)
        self.assertEqual(objective["precision"], 1 / 3)
        self.assertEqual(objective["n_recall"], 4)
        self.assertEqual(objective["recall"], 1 / 4)
        self.assertEqual(year_range["missed"], 3)
        self.assertEqual(year_range["recall"], 0)

    def test_boolean_queries_use_document_level_field_denominator(self):
        entry = record("W1")
        entry["field_judgments"] = [
            {"name": "exact_boolean_queries[0]", "judgment": "correct"},
            {"name": "exact_boolean_queries[1]", "judgment": "partial"},
        ]
        result = analyze.evaluate({"alice": {"W1": entry}})
        queries = result["metrics"][0]
        self.assertEqual(queries["field"], "exact_boolean_queries")
        self.assertEqual(queries["n_precision"], 1)
        self.assertEqual(queries["partial"], 1)
        self.assertEqual(queries["precision"], 0)

    def test_missing_shared_rating_not_treated_as_consensus(self):
        result = analyze.evaluate({"alice": {"W1": record("W1", shared=True)}, "bob": {}})
        self.assertEqual(result["summary"]["unresolved_items"], 2)
        self.assertEqual(result["agreement"]["extracted"]["n_items"], 0)

    def test_task_validation_catches_missing_and_unknown_fields(self):
        tasks = {"W1": {"doc_id": "W1", "fields": [{"name": "objective"}],
                        "null_fields": ["year_range", "snowballing"]}}
        result = analyze.evaluate({"alice": {"W1": record("W1")}}, tasks=tasks)
        self.assertEqual(result["summary"]["unresolved_items"], 1)
        self.assertEqual(result["summary"]["incomplete_documents"], ["W1"])
        bad = record("W1")
        bad["null_field_judgments"].append({"name": "unknown", "judgment": "absent"})
        with self.assertRaises(ValueError):
            analyze.evaluate({"alice": {"W1": bad}}, tasks=tasks)

    def test_invalid_labels_and_adjudication_keys_are_rejected(self):
        with self.assertRaises(ValueError):
            analyze.evaluate({"alice": {"W1": record("W1", label="probably")}})
        with self.assertRaises(ValueError):
            analyze.evaluate({"alice": {"W1": record("W1")}}, adjudications=[
                {"doc_id": "unknown", "kind": "field", "field": "objective", "judgment": "correct"}
            ])

    def test_duplicate_exports_are_rejected_instead_of_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            entry = record("W1")
            entry["annotator"] = "alice"
            for filename in ["annotations_first.jsonl", "annotations_second.jsonl"]:
                (Path(directory) / filename).write_text(json.dumps(entry) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                analyze.load_exports(directory)

    def test_reports_and_adjudication_round_trip_preserve_template(self):
        data = {"alice": {"W1": record("W1", shared=True)},
                "bob": {"W1": record("W1", label="partial", shared=True)}}
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            result = analyze.evaluate(data)
            analyze.write_results(result, out)
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["unresolved_items"], 1)
            self.assertIn("conditional recall", (out / "results_table.tex").read_text())
            self.assertTrue((out / "error_analysis.tsv").exists())
            template = out / "adjudication_template.jsonl"
            decisions = list(analyze.read_jsonl(template))
            self.assertEqual(len(decisions), 1)
            decisions[0]["judgment"] = "correct"
            template.write_text(json.dumps(decisions[0]) + "\n", encoding="utf-8")
            original = template.read_bytes()
            resolved = analyze.evaluate(data, adjudications=list(analyze.read_jsonl(template)))
            analyze.write_results(resolved, out)
            self.assertEqual(resolved["summary"]["unresolved_items"], 0)
            self.assertEqual(template.read_bytes(), original)
            self.assertEqual(resolved["agreement"], result["agreement"])

    def test_sample_report_counts_shared_documents_once_and_detects_gaps(self):
        tasks = {
            "W1": {"field": "Medicine", "fields": [{"name": "objective"}], "null_fields": []},
            "W2": {"field": "Medicine", "fields": [{"name": "exact_boolean_queries[0]"}], "null_fields": []},
        }
        data = {"alice": {"W1": record("W1", shared=True)},
                "bob": {"W1": record("W1", shared=True)}}
        report = analyze.sample_report(data, tasks)
        self.assertEqual(report["discipline_composition"][0], {
            "discipline": "Medicine", "sampled": 2, "observed": 1, "submitted_done": 1, "missing": 1})
        query = next(row for row in report["task_field_coverage"] if row["field"] == "exact_boolean_queries")
        self.assertEqual(query["unassessed_tasks"], 1)
        self.assertEqual({row["cohort"] for row in report["annotator_metrics"]}, {"shared"})
        with self.assertRaises(ValueError):
            analyze.sample_report(data, tasks, {"sampled_doc_ids": ["W1"]})

    def test_error_report_deduplicates_fields_and_preserves_disagreements(self):
        first, second = record("W1", shared=True), record("W1", shared=True)
        first["field_judgments"] = [
            {"name": "exact_boolean_queries[0]", "judgment": "partial", "error_type": "missing elements"},
            {"name": "exact_boolean_queries[1]", "judgment": "incorrect", "comment": "Wrong query"},
        ]
        second["field_judgments"] = [
            {"name": "exact_boolean_queries[0]", "judgment": "partial", "error_type": "missing elements"},
            {"name": "exact_boolean_queries[1]", "judgment": "correct"},
        ]
        result = analyze.evaluate({"alice": {"W1": first}, "bob": {"W1": second}})
        cases, counts = analyze.error_report(result)
        query = next(row for row in cases if row["field"] == "exact_boolean_queries")
        self.assertEqual(query["status"], "unresolved_flag")
        self.assertIsNone(query["judgment"])
        self.assertEqual(query["n_flagged_ratings"], 3)
        self.assertEqual(query["error_types"], ["missing elements", "unclassified"])
        category = next(row for row in counts if row["error_type"] == "missing elements")
        self.assertEqual(category["n_document_fields"], 1)
        self.assertEqual(category["n_flagged_ratings"], 2)

    def test_error_report_retains_overridden_tags_without_counting_confirmed_error(self):
        entry = record("W1", label="partial")
        entry["field_judgments"][0]["error_type"] = "wrong value"
        result = analyze.evaluate({"alice": {"W1": entry}}, adjudications=[
            {"doc_id": "W1", "kind": "field", "field": "objective", "judgment": "correct"}])
        cases, counts = analyze.error_report(result)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["status"], "resolved_nonerror_flag")
        self.assertEqual(counts[0]["error_type"], "wrong value")
        self.assertEqual(cases[0]["items"][0]["ratings"]["alice"]["judgment"], "partial")

    def test_query_table_suppresses_unassessed_recall_and_reports_error_categories(self):
        entry = record("W1")
        entry["field_judgments"] = [{"name": "exact_boolean_queries[0]", "judgment": "partial",
                                     "error_type": "wrong term"}]
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            result = analyze.evaluate({"alice": {"W1": entry}})
            analyze.write_results(result, out)
            query_line = next(line for line in (out / "results_table.tex").read_text().splitlines()
                              if line.startswith("exact boolean queries &"))
            self.assertIn("& -- & -- & -- & 0", query_line)
            self.assertIn("wrong term & 1 & 0 & 0", (out / "error_table.tex").read_text())
            cases = list(analyze.read_jsonl(out / "error_cases.jsonl"))
            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0]["status"], "resolved_error")

    def test_paper_refresh_requires_tasks_to_prevent_stale_sample_publication(self):
        with patch("sys.argv", [str(SCRIPT), "--paper-tables", "paper/tables"]):
            with self.assertRaises(SystemExit) as error:
                analyze.main()
        self.assertEqual(error.exception.code, 2)

    def test_paper_refresh_does_not_publish_stale_comment_table_without_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exports, out, paper = root / "exports", root / "reports", root / "paper"
            exports.mkdir()
            out.mkdir()
            entry = dict(record("W1"), annotator="alice")
            (exports / "annotations_alice.jsonl").write_text(json.dumps(entry) + "\n")
            tasks = root / "tasks.jsonl"
            tasks.write_text(json.dumps({"doc_id": "W1", "fields": [{"name": "objective"}], "null_fields": ["year_range"]}) + "\n")
            (out / "comment_themes_table.tex").write_text("STALE COMMENT TABLE")
            argv = [str(SCRIPT), "--dir", str(exports), "--tasks", str(tasks), "--out", str(out),
                    "--paper-tables", str(paper)]
            with patch("sys.argv", argv):
                analyze.main()
            self.assertFalse((paper / "table-validation-comment-themes.tex").exists())
            self.assertTrue((paper / "table-validation-pairwise.tex").exists())

    def test_unresolved_query_excludes_whole_document_field(self):
        first, second = record("W1", shared=True), record("W1", shared=True)
        first["field_judgments"] = [{"name": "exact_boolean_queries[0]", "judgment": "correct"},
                                    {"name": "exact_boolean_queries[1]", "judgment": "correct"}]
        second["field_judgments"] = [{"name": "exact_boolean_queries[0]", "judgment": "correct"},
                                     {"name": "exact_boolean_queries[1]", "judgment": "partial"}]
        result = analyze.evaluate({"alice": {"W1": first}, "bob": {"W1": second}})
        self.assertEqual(result["metrics"][0]["n_precision"], 0)
        self.assertEqual(result["metrics"][0]["unresolved_document_fields"], 1)

    def test_html_report_embeds_safe_complete_data_without_changing_inputs(self):
        report_script = SCRIPT.with_name("make_report.py")
        spec = importlib.util.spec_from_file_location("validation_report", report_script)
        report = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(report)
        entry = record("W1")
        unsafe = '</script><img src=x onerror="alert(1)">'
        entry["field_judgments"][0]["comment"] = unsafe
        entry["field_judgments"].append({"name": "exact_boolean_queries[0]", "judgment": "partial"})
        data = {"alice": {"W1": entry}}
        tasks = {"W1": {"doc_id": "W1", "title": unsafe,
                        "fields": [{"name": "objective"}, {"name": "exact_boolean_queries[0]"}],
                        "null_fields": ["year_range"]}}
        result = analyze.evaluate(data, tasks=tasks)
        result["sample"] = analyze.sample_report(data, tasks)
        codes = [{"comment_key": row["comment_key"], "themes": ["unclear_note"]}
                 for row in analyze.comment_inventory(result)]
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            analyze.write_results(result, out, tasks=tasks, comment_codes=codes)
            originals = {path.name: path.read_bytes() for path in out.iterdir()}
            path = report.build_report(out)
            text = path.read_text()
            embedded = text.split('<script id="report-data" type="application/json">', 1)[1].split('</script>', 1)[0]
            payload = json.loads(embedded)
            self.assertNotIn(unsafe, text)
            self.assertEqual(payload["comments"]["coded_comments"][0]["comment"], unsafe)
            self.assertEqual(payload["summary"]["observed_documents"], 1)
            self.assertEqual(len(payload["audit"]), 3)
            queries = next(row for row in payload["performance"] if row["field"] == "exact_boolean_queries")
            self.assertIsNone(queries["recall"])
            self.assertIsNone(queries["n_recall"])
            self.assertIsNone(payload.get("querySourceAudit"))
            self.assertIn('id="view-discussion"', text)
            self.assertIn('id="view-provenance"', text)

            class ReportNavigation(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.navigation = None
                    self.links = []
                    self.more_details = None

                def handle_starttag(self, tag, attributes):
                    attributes = dict(attributes)
                    if tag == "nav":
                        self.navigation = attributes.get("id")
                    if tag == "a" and self.navigation == "main-navigation":
                        self.links.append(attributes.get("href"))
                    if tag == "details" and attributes.get("id") == "more-navigation":
                        self.more_details = attributes

                def handle_endtag(self, tag):
                    if tag == "nav":
                        self.navigation = None

            navigation = ReportNavigation()
            navigation.feed(text)
            self.assertEqual(navigation.links, ["#overview", "#performance", "#errors"])
            self.assertIsNotNone(navigation.more_details)
            self.assertNotIn("open", navigation.more_details)
            for name, content in originals.items():
                self.assertEqual((out / name).read_bytes(), content)

            query_audit = {
                "field": "exact_boolean_queries", "status": "provisional", "missed": 94,
                "source_decisions": {"present": 94, "absent": 72, "unresolved": 139},
                "source_assessed": 166, "empty_query_reviews": 305,
                "extracted_query_ratings": {"correct": 60, "partial": 10, "incorrect": 2, "unresolved": 8},
                "recall": {"correct": 60, "partial": 10, "incorrect": 2, "missed": 94,
                           "denominator": 166, "estimate": 60 / 166,
                           "ci95": [0.29226018669501796, 0.4368989931695205],
                           "sensitivity_range": [0.19169329073482427, 0.39080459770114945]},
            }
            query_audit_path = out.parent / "query_source_audit.json"
            query_audit_path.write_text(json.dumps(query_audit) + "\n")
            audit_html = report.build_report(
                out, output=out / "with_query_audit.html",
                query_source_audit_path=query_audit_path,
            )
            audit_text = audit_html.read_text()
            audit_payload = json.loads(
                audit_text.split('<script id="report-data" type="application/json">', 1)[1]
                .split('</script>', 1)[0]
            )
            self.assertEqual(audit_payload["querySourceAudit"]["report"], query_audit)
            self.assertEqual(audit_payload["querySourceAudit"]["source"]["name"], query_audit_path.name)
            audit_href = audit_payload["querySourceAudit"]["source"]["href"]
            self.assertEqual((audit_html.parent / audit_href).resolve(), query_audit_path.resolve())
            self.assertIsNone(next(row for row in audit_payload["performance"]
                                   if row["field"] == "exact_boolean_queries")["recall"])
            query_recall = next(row for row in audit_payload["recallPerformance"]
                                if row["field"] == "exact_boolean_queries")
            self.assertEqual(query_recall["recall"], 60 / 166)
            self.assertEqual(query_recall["n_recall"], 166)
            self.assertEqual(query_recall["missed"], 94)
            self.assertEqual(query_recall["null_absent"], 211)
            self.assertEqual(query_recall["unresolved_document_fields"],
                             next(row for row in audit_payload["performance"]
                                  if row["field"] == "exact_boolean_queries")["unresolved_document_fields"])
            with self.assertRaises(ValueError):
                report.build_report(out, output=out / "summary.json")


if __name__ == "__main__":
    unittest.main()