import importlib.util
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "src/validation"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("adjudication_packages", SCRIPTS / "make_adjudication_packages.py")
packages = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packages)


class AdjudicationPackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def fixture(self):
        tasks, cases, sources = {}, [], []
        for number in range(1, 5):
            doc_id = f"W{number}"
            pdf = self.root / f"{doc_id}.pdf"
            pdf.write_bytes(b"fixture-pdf")
            tasks[doc_id] = {"doc_id": doc_id, "title": "</script><script>bad()</script>", "pdf": str(pdf)}
            sources.append({"doc_id": doc_id, "pdf_sha256": packages.digest(pdf),
                            "pages": ["We checked reference lists."]})
            for index in range(3 if number == 1 else 1):
                case = {"doc_id": doc_id, "kind": "null", "field": f"field_{index}",
                        "reason": "disagreement", "ratings": {}, "extraction": None}
                case["case_id"] = packages.case_id(case)
                cases.append(case)
        source_path = self.root / "sources.jsonl"
        proposal_path = self.root / "proposals.jsonl"
        packages.write_jsonl(source_path, sources)
        packages.write_jsonl(proposal_path, [{
            "doc_id": "W1", "kind": "null", "field": "field_0",
            "status": "proposed", "judgment": "present", "categories": ["presence_evidence"],
            "rationale": "Reference-list searching is stated.",
            "evidence": [{"page": 1, "quote": "We checked reference lists."}],
        }])
        tool = self.root / "tool.html"
        tool.write_text('<script type="application/json">__ADJUDICATION_DATA__</script>')
        return tasks, cases, sources, source_path, proposal_path, tool

    def build(self):
        tasks, cases, sources, source_path, proposal_path, tool = self.fixture()
        out = self.root / "packages"
        manifest = packages.build_packages(tasks, cases, [], source_path, proposal_path, out, tool)
        return out, manifest, cases

    def differing_fixture(self):
        tasks, cases, sources, source_path, proposal_path, tool = self.fixture()
        for owner, case in zip(packages.ANNOTATORS, cases[:3]):
            case["ratings"] = {name: {"judgment": "present" if name == owner else "absent"}
                               for name in packages.ANNOTATORS}
        for case in cases[3:]:
            case["kind"] = "field"
            case["case_id"] = packages.case_id(case)
            case["extraction"] = {"value": "A review aim", "evidence_span": "A review aim"}
            case["ratings"] = {name: {"judgment": label}
                               for name, label in zip(packages.ANNOTATORS, ("correct", "partial", "incorrect"))}
        return tasks, cases, sources, source_path, proposal_path, tool

    def test_differing_assignment_routes_outliers_and_splits_three_way_cases(self):
        tasks, cases, _, sources, proposals, tool = self.differing_fixture()
        out = self.root / "differing"
        manifest = packages.build_packages(tasks, cases, [], sources, proposals, out, tool,
                                           assignment_mode="differing-annotator")
        identifiers = []
        for index, owner in enumerate(packages.ANNOTATORS):
            payload = json.loads((out / f"package_{owner}/review_data.json").read_text())
            assigned = {case["case_id"] for case in payload["cases"]}
            self.assertEqual(assigned, {cases[index]["case_id"], cases[index + 3]["case_id"]})
            self.assertIn("W1", payload["documents"])
            self.assertCountEqual(manifest["case_assignment"][owner], assigned)
            identifiers.extend(assigned)
        self.assertCountEqual(identifiers, [case["case_id"] for case in cases])
        self.assertEqual(packages.assign_differing_annotators(cases),
                         packages.assign_differing_annotators(list(reversed(cases))))
        refreshed = self.root / "differing-refreshed"
        packages.build_packages(tasks, cases, [], sources, proposals, refreshed, tool,
                                previous_packages=out, assignment_mode="differing-annotator")
        for owner in packages.ANNOTATORS:
            relative = f"package_{owner}/review_data.json"
            self.assertEqual((out / relative).read_bytes(), (refreshed / relative).read_bytes())

    def test_differing_assignment_rejects_incomplete_or_agreeing_ratings(self):
        _, cases, *_ = self.differing_fixture()
        case = cases[0]
        del case["ratings"]["arno"]
        with self.assertRaisesRegex(ValueError, "complete valid ratings"):
            packages.assign_differing_annotators([case])
        case["ratings"] = {owner: {"judgment": "absent"} for owner in packages.ANNOTATORS}
        with self.assertRaisesRegex(ValueError, "not a label disagreement"):
            packages.assign_differing_annotators([case])

    def test_merge_accepts_shared_pdfs_but_not_reviews_from_the_wrong_owner(self):
        tasks, cases, _, sources, proposals, tool = self.differing_fixture()
        out = self.root / "differing"
        packages.build_packages(tasks, cases, [], sources, proposals, out, tool,
                                assignment_mode="differing-annotator")
        rows = []
        for owner in packages.ANNOTATORS:
            payload = json.loads((out / f"package_{owner}/review_data.json").read_text())
            case = next(case for case in payload["cases"] if case["doc_id"] == "W1")
            rows.append({key: case[key] for key in ("case_id", "doc_id", "kind", "field")} | {
                "version": packages.VERSION, "package_id": payload["package_id"], "reviewer": owner,
                "review_status": "reviewed", "judgment": "present", "comment": "Checked the source.",
                "page": 1, "found_text": "We checked reference lists.", "source_checked": True,
                "ts": "2026-11-19T20:00:00Z",
            })
        returned = self.root / "returned.jsonl"
        packages.write_jsonl(returned, rows)
        self.assertEqual(packages.merge_reviews(out, [returned], self.root / "merged"), (3, 3))
        rows[0]["reviewer"] = "mert_okur"
        wrong_returned = self.root / "wrong-owner.jsonl"
        packages.write_jsonl(wrong_returned, rows)
        with self.assertRaisesRegex(ValueError, "another package or reviewer"):
            packages.merge_reviews(out, [wrong_returned], self.root / "wrong-owner")

    def test_build_partitions_whole_documents_preserves_keys_and_escapes_payload(self):
        out, manifest, cases = self.build()
        all_cases, documents = [], []
        for owner in packages.ANNOTATORS:
            package = json.loads((out / f"package_{owner}/review_data.json").read_text())
            all_cases.extend(case["case_id"] for case in package["cases"])
            documents.extend(package["documents"])
            self.assertNotIn("</script><script>bad()", (out / f"package_{owner}/index.html").read_text())
            with zipfile.ZipFile(out / f"package_{owner}.zip") as archive:
                self.assertIsNone(archive.testzip())
                for doc in package["documents"].values():
                    self.assertIn(doc["pdf"], archive.namelist())
        self.assertCountEqual(all_cases, [case["case_id"] for case in cases])
        self.assertEqual(len(documents), len(set(documents)))
        self.assertEqual(manifest["ai_proposed"], 1)
        for owner in packages.ANNOTATORS:
            package = json.loads((out / f"package_{owner}/review_data.json").read_text())
            self.assertTrue(all(case["ai"]["status"] in ("proposed", "undecided") for case in package["cases"]))
            self.assertTrue(all("judgment" not in case for case in package["cases"]))

    def test_proposals_require_literal_pdf_evidence_and_cannot_pre_resolve_uncertain_cases(self):
        _, cases, sources, *_ = self.fixture()
        proposal = {"status": "proposed", "judgment": "present", "rationale": "Checked.",
                    "evidence": [{"page": 1, "quote": "Invented evidence"}]}
        with self.assertRaisesRegex(ValueError, "does not match"):
            packages.validate_proposal(cases[0], proposal, sources[0])
        proposal.update(status="undecided")
        with self.assertRaisesRegex(ValueError, "cannot have a judgment"):
            packages.validate_proposal(cases[0], proposal, sources[0])

    def test_ui_refresh_preserves_review_identities_only_for_identical_data(self):
        tasks, cases, _, sources, proposals, tool = self.fixture()
        original = self.root / "original"
        packages.build_packages(tasks, cases, [], sources, proposals, original, tool)
        tool.write_text('<h1>Refreshed UI</h1><script type="application/json">__ADJUDICATION_DATA__</script>')
        refreshed = self.root / "refreshed"
        packages.build_packages(tasks, cases, [], sources, proposals, refreshed, tool,
                                previous_packages=original)
        for owner in packages.ANNOTATORS:
            relative = f"package_{owner}/review_data.json"
            self.assertEqual((original / relative).read_bytes(), (refreshed / relative).read_bytes())
            self.assertIn("Refreshed UI", (refreshed / f"package_{owner}/index.html").read_text())
            self.assertNotIn("Refreshed UI", (original / f"package_{owner}/index.html").read_text())
        cases[0]["reason"] = "changed"
        with self.assertRaisesRegex(ValueError, "case data changed"):
            packages.build_packages(tasks, cases, [], sources, proposals, self.root / "rejected", tool,
                                    previous_packages=original)
        self.assertFalse((self.root / "rejected").exists())

    def test_humanized_comments_cannot_change_independent_labels(self):
        tasks = self.root / "tasks.jsonl"
        exports = self.root / "exports"
        exports.mkdir()
        humanized = self.root / "humanized.jsonl"
        packages.write_jsonl(tasks, [{"doc_id": "W1", "fields": [], "null_fields": ["snowballing"]}])
        for owner in packages.ANNOTATORS:
            row = {"doc_id": "W1", "annotator": owner, "shared": True, "status": "done",
                   "field_judgments": [], "null_field_judgments": [{
                       "name": "snowballing", "judgment": "present" if owner == "pierre" else "absent",
                       "comment": "Original wording"}]}
            packages.write_jsonl(exports / f"annotations_{owner}.jsonl", [row])
            if owner == "pierre":
                row["null_field_judgments"][0]["comment"] = "Natural wording"
                packages.write_jsonl(humanized, [row])
        _, cases, _ = packages.load_inputs(tasks, exports, humanized)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["ratings"]["pierre"]["comment"], "Natural wording")
        self.assertNotIn("Original wording", json.dumps(cases[0]["ratings"]["pierre"]))
        changed = json.loads(humanized.read_text())
        changed["null_field_judgments"][0]["judgment"] = "absent"
        humanized.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "only in comments"):
            packages.load_inputs(tasks, exports, humanized)

    def test_merge_requires_human_action_rejects_duplicate_wrong_owner_and_retains_queue(self):
        out, _, _ = self.build()
        package = next(
            payload for owner in packages.ANNOTATORS
            if (payload := json.loads((out / f"package_{owner}/review_data.json").read_text()))["cases"]
        )
        case = package["cases"][0]
        review = {key: case[key] for key in ("case_id", "doc_id", "kind", "field")}
        review.update(version=packages.VERSION, package_id=package["package_id"],
                      reviewer=package["reviewer"], review_status="reviewed", judgment="present",
                      comment="Checked the source", page=1, found_text="We checked reference lists.",
                      source_checked=True, ts="2026-11-19T14:00:00Z")
        returned = self.root / "returned.jsonl"
        packages.write_jsonl(returned, [review])
        self.assertEqual(packages.merge_reviews(out, [returned], self.root / "merged"), (1, 5))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            packages.merge_reviews(out, [returned, returned], self.root / "duplicate")
        for change in ({"source_checked": False}, {"reviewer": "somebody_else"},
                       {"review_status": "discussion"}, {"page": 0}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                packages.validate_review(review | change, case, package, 1)
        packages.validate_review(review | {"review_status": "discussion", "judgment": None}, case, package, 1)
        packages.validate_review(review | {"review_status": "unreviewed", "judgment": None}, case, package, 1)


if __name__ == "__main__":
    unittest.main()
