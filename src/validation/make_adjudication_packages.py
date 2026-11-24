#!/usr/bin/env python3
"""Prepare source evidence, build disjoint review packages, or merge human reviews."""

import argparse
import copy
import hashlib
import json
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import analyze


ANNOTATORS = ("arno", "mert_okur", "pierre")
VERSION = "sr4all-adjudication-1"
TOOL = Path(__file__).resolve().parents[2] / "data/validation/templates/adjudicate.html"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def case_id(row):
    return json.dumps([row["doc_id"], row["kind"], row["field"]], separators=(",", ":"))


def indexed(rows, key, description):
    result = {}
    for row in rows:
        identifier = key(row)
        if identifier in result:
            raise ValueError(f"duplicate {description}: {identifier}")
        result[identifier] = row
    return result


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def without_comments(value):
    if isinstance(value, dict):
        return {key: without_comments(item) for key, item in value.items() if key != "comment"}
    if isinstance(value, list):
        return [without_comments(item) for item in value]
    return value


def load_inputs(tasks_path, exports_path, humanized_path):
    tasks = indexed(analyze.read_jsonl(tasks_path), lambda row: row["doc_id"], "task")
    data = analyze.load_exports(exports_path)
    if set(data) != set(ANNOTATORS):
        raise ValueError(f"expected exactly {ANNOTATORS}, got {sorted(data)}")
    humanized = indexed(analyze.read_jsonl(humanized_path), lambda row: row["doc_id"], "humanized record")
    if set(humanized) != set(data["pierre"]) or any(
        without_comments(humanized[key]) != without_comments(data["pierre"][key])
        for key in humanized
    ):
        raise ValueError("Pierre's humanized export must differ only in comments")
    display = copy.deepcopy(data)
    display["pierre"] = humanized
    result = analyze.evaluate(display, tasks=tasks)
    cases = []
    for item in result["audit"]:
        if item["source"] != "unresolved":
            continue
        task = tasks[item["doc_id"]]
        extraction = next((node for node in task["fields"] if node["name"] == item["field"]), None)
        if (extraction is None) != (item["kind"] == "null"):
            raise ValueError(f"missing or unexpected extraction: {case_id(item)}")
        cases.append({
            "case_id": case_id(item), "doc_id": item["doc_id"], "kind": item["kind"],
            "field": item["field"], "reason": item["reason"], "ratings": item["ratings"],
            "extraction": extraction,
        })
    if not cases:
        raise ValueError("no unresolved items found")
    paths = [Path(tasks_path), Path(humanized_path), *sorted(Path(exports_path).glob("annotations_*.jsonl"))]
    provenance = [{"path": str(path), "sha256": digest(path)} for path in paths]
    return tasks, cases, provenance


def prepare_sources(tasks, cases, out):
    from pypdf import PdfReader

    doc_ids = sorted({case["doc_id"] for case in cases})
    for doc_id in doc_ids:
        if not Path(tasks[doc_id]["pdf"]).is_file():
            raise ValueError(f"missing PDF: {doc_id}")
    rows = []
    for doc_id in doc_ids:
        path = Path(tasks[doc_id]["pdf"])
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
        rows.append({"doc_id": doc_id, "pdf_sha256": digest(path), "pages": pages})
    write_jsonl(out, rows)
    return rows


def normalized(text):
    return " ".join(text.split())


def validate_proposal(case, proposal, source):
    if proposal.get("status") not in ("proposed", "undecided"):
        raise ValueError(f"invalid AI status: {case['case_id']}")
    if not isinstance(proposal.get("rationale"), str) or not proposal["rationale"].strip():
        raise ValueError("AI assessment needs a rationale")
    categories = proposal.get("categories", [])
    if not isinstance(categories, list) or set(categories) - set(analyze.COMMENT_THEMES):
        raise ValueError("unknown AI category")
    if proposal["status"] == "undecided":
        if proposal.get("judgment") is not None:
            raise ValueError("undecided AI assessment cannot have a judgment")
        return
    if proposal.get("judgment") not in analyze.LABELS[case["kind"]]:
        raise ValueError("invalid proposed judgment")
    evidence = proposal.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("AI proposal requires source evidence")
    for passage in evidence:
        page, quote = passage.get("page"), passage.get("quote")
        if type(page) is not int or not 1 <= page <= len(source["pages"]):
            raise ValueError("AI evidence page is out of range")
        if not isinstance(quote, str) or not normalized(quote):
            raise ValueError("AI evidence quote is empty")
        if normalized(quote) not in normalized(source["pages"][page - 1]):
            raise ValueError(f"AI quote does not match PDF text: {case['case_id']}, page {page}")


def assign_documents(cases, sources):
    grouped = defaultdict(list)
    for case in cases:
        grouped[case["doc_id"]].append(case)
    # Reading a document has a setup cost even when it contains only one disputed field.
    weights = {doc_id: len(items) + 2 * len(sources[doc_id]["pages"]) for doc_id, items in grouped.items()}
    loads = Counter({name: 0 for name in ANNOTATORS})
    assignment = {name: [] for name in ANNOTATORS}
    for doc_id in sorted(grouped, key=lambda identifier: (-weights[identifier], identifier)):
        owner = min(ANNOTATORS, key=lambda name: (loads[name], len(assignment[name]), name))
        assignment[owner].append(doc_id)
        loads[owner] += weights[doc_id]
    return {name: sorted(docs) for name, docs in assignment.items()}, weights


def assign_differing_annotators(cases):
    assignment = {name: [] for name in ANNOTATORS}
    reasons, three_way = {}, []
    for case in sorted(cases, key=lambda case: case["case_id"]):
        ratings = case["ratings"]
        if set(ratings) != set(ANNOTATORS) or any(
            not isinstance(ratings[name], dict)
            or ratings[name].get("judgment") not in analyze.LABELS[case["kind"]]
            for name in ANNOTATORS
        ):
            raise ValueError(f"differing-annotator assignment needs complete valid ratings: {case['case_id']}")
        votes = {name: ratings[name]["judgment"] for name in ANNOTATORS}
        counts = Counter(votes.values())
        if sorted(counts.values()) == [1, 2]:
            owner = next(name for name, label in votes.items() if counts[label] == 1)
            assignment[owner].append(case["case_id"])
            reasons[case["case_id"]] = "two_to_one_differing_label"
        elif len(counts) == 3:
            three_way.append(case["case_id"])
        else:
            raise ValueError(f"not a label disagreement: {case['case_id']}")
    for index, identifier in enumerate(three_way):
        assignment[ANNOTATORS[index % len(ANNOTATORS)]].append(identifier)
        reasons[identifier] = "three_way_rotation"
    return {owner: sorted(identifiers) for owner, identifiers in assignment.items()}, reasons


def build_packages(tasks, cases, provenance, sources_path, proposals_path, out, tool=TOOL,
                   previous_packages=None, assignment_mode="documents"):
    out = Path(out)
    if out.exists():
        raise ValueError(f"output already exists; choose a fresh directory: {out}")
    sources = indexed(analyze.read_jsonl(sources_path), lambda row: row["doc_id"], "source")
    expected_docs = {case["doc_id"] for case in cases}
    if set(sources) != expected_docs:
        raise ValueError("source documents do not match unresolved documents")
    for doc_id, source in sources.items():
        if not isinstance(source.get("pages"), list) or not source["pages"] or any(
            not isinstance(page, str) for page in source["pages"]
        ):
            raise ValueError(f"invalid source pages: {doc_id}")
        if digest(tasks[doc_id]["pdf"]) != source["pdf_sha256"]:
            raise ValueError(f"PDF changed since source preparation: {doc_id}")
    supplied = indexed(analyze.read_jsonl(proposals_path), case_id, "AI assessment")
    if set(supplied) - {case["case_id"] for case in cases}:
        raise ValueError("AI assessment refers to a non-disagreement item")
    cases = copy.deepcopy(cases)
    for case in cases:
        proposal = supplied.get(case["case_id"], {
            "status": "undecided", "judgment": None, "categories": [],
            "rationale": "No source-verified AI proposal. Human source review is required.",
            "evidence": [],
        })
        validate_proposal(case, proposal, sources[case["doc_id"]])
        case["ai"] = {key: value for key, value in proposal.items()
                      if key not in ("doc_id", "kind", "field")}
    document_assignment, weights = assign_documents(cases, sources)
    if assignment_mode == "documents":
        case_assignment = {
            owner: [case["case_id"] for case in cases if case["doc_id"] in doc_ids]
            for owner, doc_ids in document_assignment.items()
        }
        assignment_reasons = {case["case_id"]: "whole_document_balance" for case in cases}
        assignment_rule = "Disjoint whole documents; descending (items + 2 * PDF pages), least-loaded reviewer."
    elif assignment_mode == "differing-annotator":
        case_assignment, assignment_reasons = assign_differing_annotators(cases)
        assignment_rule = (
            "2-to-1 cases go to the differing annotator; sorted three-way cases rotate "
            "through arno, mert_okur, pierre. Cases are disjoint; PDFs may appear in multiple packages."
        )
    else:
        raise ValueError(f"unknown assignment mode: {assignment_mode}")
    case_owners = {identifier: owner for owner, identifiers in case_assignment.items()
                   for identifier in identifiers}
    selected_by_owner = {owner: [case for case in cases if case_owners[case["case_id"]] == owner]
                         for owner in ANNOTATORS}
    assignment = {owner: sorted({case["doc_id"] for case in selected})
                  for owner, selected in selected_by_owner.items()}
    document_payloads = {}
    for doc_id in sources:
        if not re.fullmatch(r"W[0-9]+", doc_id):
            raise ValueError(f"unsafe document identifier: {doc_id}")
        document_payloads[doc_id] = {
            "title": tasks[doc_id].get("title", ""),
            "pdf": f"pdfs/{doc_id}.pdf", **sources[doc_id],
        }
    previous_ids = {}
    if previous_packages is not None:
        previous_packages = Path(previous_packages)
        previous_manifest = json.loads((previous_packages / "assignment.json").read_text())
        for owner, doc_ids in assignment.items():
            previous = json.loads((previous_packages / f"package_{owner}/review_data.json").read_text())
            expected_cases = selected_by_owner[owner]
            expected_documents = {doc_id: document_payloads[doc_id] for doc_id in doc_ids}
            if (previous.get("version") != VERSION or previous.get("reviewer") != owner
                    or previous.get("cases") != expected_cases
                    or previous.get("documents") != expected_documents):
                raise ValueError(f"cannot reuse {owner}'s identity: assignment or case data changed")
            package_id = previous.get("package_id")
            if not package_id or package_id != previous_manifest["packages"][owner]["package_id"]:
                raise ValueError(f"previous package identity mismatch: {owner}")
            previous_ids[owner] = package_id
        provenance = [*provenance, {
            "path": str(previous_packages / "assignment.json"),
            "sha256": digest(previous_packages / "assignment.json"),
        }]
    provenance = [*provenance, *[
        {"path": str(path), "sha256": digest(path)}
        for path in (sources_path, proposals_path, tool, Path(__file__), Path(analyze.__file__))
    ]]
    fingerprint = hashlib.sha256(json.dumps(
        {"inputs": provenance, "assignment": assignment, "case_assignment": case_assignment,
         "assignment_mode": assignment_mode, "version": VERSION}, sort_keys=True
    ).encode()).hexdigest()
    template = Path(tool).read_text(encoding="utf-8")
    if template.count("__ADJUDICATION_DATA__") != 1:
        raise ValueError("HTML template must have exactly one data placeholder")
    manifest = {
        "version": VERSION, "run_id": fingerprint, "inputs": provenance,
        "n_cases": len(cases), "n_documents": len(expected_docs),
        "ai_proposed": sum(case["ai"]["status"] == "proposed" for case in cases),
        "assignment_mode": assignment_mode, "assignment_rule": assignment_rule,
        "case_assignment": case_assignment, "case_assignment_reasons": assignment_reasons,
        "document_weights": weights, "assignment": assignment, "packages": {},
        "note": "AI suggestions are not human decisions. Assigned human resolutions are not three-person consensus.",
    }
    out.mkdir(parents=True)
    for owner, doc_ids in assignment.items():
        package_id = previous_ids.get(owner, f"{fingerprint[:24]}-{owner}")
        selected = selected_by_owner[owner]
        documents = {}
        package = out / f"package_{owner}"
        (package / "pdfs").mkdir(parents=True)
        for doc_id in doc_ids:
            documents[doc_id] = document_payloads[doc_id]
            shutil.copyfile(tasks[doc_id]["pdf"], package / documents[doc_id]["pdf"])
        payload = {"version": VERSION, "package_id": package_id, "reviewer": owner,
                   "documents": documents, "cases": selected}
        encoded = json.dumps(payload, ensure_ascii=True).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
        (package / "index.html").write_text(
            template.replace("__ADJUDICATION_DATA__", encoded), encoding="utf-8"
        )
        write_json(package / "review_data.json", payload)
        instructions = (
            f"SR4ALL adjudication review: {owner}\n\n"
            "Unzip the entire package, then open index.html in Chrome, Edge, or Firefox.\n"
            "No server or internet connection is needed. PDFs and review data stay local.\n"
            "Use Open PDF if your browser does not display the embedded PDF.\n"
            "Proposed-resolution cases appear first; undecided cases follow.\n"
            "The GPT6.1 Medium assessment appears at the top of each card.\n"
            "Read the source for every card, including cards with a GPT6.1 Medium suggestion.\n"
            f"Assignment: {assignment_rule}\n"
            "A differing original label does not mean that label is wrong. Retain it if the source supports it.\n"
            "Select a judgment, give a rationale and a one-based PDF page, tick source checked,\n"
            "then Save & next (Save decision on the last case). For missed fields, also quote the supporting text.\n"
            "Use Previous/Next to browse without finalizing drafts; Browse cases opens filters and documents.\n"
            "Use Needs discussion if you cannot decide. No GPT6.1 Medium suggestion is accepted automatically.\n"
            "Drafts autosave in this browser; export reviews regularly for a durable backup.\n"
            "Import that JSONL export to resume elsewhere. Return your latest complete export.\n"
            "Review progress is not shared between annotators or browsers.\n"
            "This is an assigned human review, not a record of three-person consensus.\n"
        )
        (package / "INSTRUCTIONS.txt").write_text(instructions, encoding="utf-8")
        zip_path = out / f"package_{owner}.zip"
        with zipfile.ZipFile(zip_path, "x", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(package.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(package))
        manifest["packages"][owner] = {
            "package_id": package_id, "documents": len(doc_ids), "cases": len(selected),
            "estimated_work": len(selected) + 2 * sum(len(sources[doc_id]["pages"]) for doc_id in doc_ids),
            "ai_proposed": sum(case["ai"]["status"] == "proposed" for case in selected),
            "zip": zip_path.name, "sha256": digest(zip_path),
        }
    write_json(out / "assignment.json", manifest)
    write_jsonl(out / "ai_assessments.jsonl", [
        {**{key: case[key] for key in ("doc_id", "kind", "field")}, **case["ai"]} for case in cases
    ])
    return manifest


def validate_review(row, case, package, pages):
    if row.get("package_id") != package["package_id"] or row.get("reviewer") != package["reviewer"]:
        raise ValueError("review belongs to another package or reviewer")
    if row.get("version") != VERSION or row.get("case_id") != case["case_id"]:
        raise ValueError("review version or case identifier mismatch")
    if any(row.get(key) != case[key] for key in ("doc_id", "kind", "field")):
        raise ValueError("review item identity mismatch")
    status, label = row.get("review_status"), row.get("judgment")
    if status not in ("unreviewed", "reviewed", "discussion"):
        raise ValueError("invalid human review status")
    if not isinstance(row.get("comment"), str) or not isinstance(row.get("found_text"), str):
        raise ValueError("invalid human review text")
    if status != "reviewed":
        if label is not None:
            raise ValueError("unresolved review cannot carry a final judgment")
        if status == "discussion" and not str(row.get("comment") or "").strip():
            raise ValueError("discussion case needs a rationale")
        return
    if label not in analyze.LABELS[case["kind"]] or row.get("source_checked") is not True:
        raise ValueError("human decision needs a valid judgment and explicit source check")
    if not isinstance(row.get("comment"), str) or not row["comment"].strip():
        raise ValueError("human decision needs a rationale")
    page = row.get("page")
    if type(page) is not int or not 1 <= page <= pages:
        raise ValueError("human decision needs a valid one-based PDF page")
    if not isinstance(row.get("ts"), str) or not row["ts"].strip():
        raise ValueError("human decision needs a timestamp")
    if case["kind"] == "null" and label == "present" and not str(row.get("found_text") or "").strip():
        raise ValueError("missed field needs supporting text")


def merge_reviews(packages_dir, review_paths, out):
    out = Path(out)
    if out.exists():
        raise ValueError(f"output already exists: {out}")
    expected = {}
    packages = {}
    for owner in ANNOTATORS:
        path = Path(packages_dir) / f"package_{owner}" / "review_data.json"
        package = json.loads(path.read_text(encoding="utf-8"))
        packages[owner] = package
        for case in package["cases"]:
            if case["case_id"] in expected:
                raise ValueError("overlapping package cases")
            expected[case["case_id"]] = (case, package)
    reviews = indexed(
        (row for path in review_paths for row in analyze.read_jsonl(path)),
        lambda row: row["case_id"], "returned review",
    )
    if set(reviews) - set(expected):
        raise ValueError("returned review contains an unknown case")
    resolutions, discussion, audit = [], [], []
    for identifier, (case, package) in expected.items():
        row = reviews.get(identifier)
        if row is not None:
            validate_review(row, case, package, len(package["documents"][case["doc_id"]]["pages"]))
        audit.append({"case": case, "human_review": row})
        if row is not None and row["review_status"] == "reviewed":
            resolutions.append({key: row.get(key) for key in (
                "doc_id", "kind", "field", "judgment", "comment", "reviewer",
                "page", "found_text", "ts", "package_id", "source_checked",
            )} | {"resolution_method": "assigned_human_review"})
        else:
            discussion.append({"case": case, "human_review": row,
                               "reason": "not_returned" if row is None else row["review_status"]})
    out.mkdir(parents=True)
    write_jsonl(out / "human_resolutions.jsonl", resolutions)
    write_jsonl(out / "meeting_queue.jsonl", discussion)
    write_jsonl(out / "review_audit.jsonl", audit)
    write_json(out / "summary.json", {
        "total": len(expected), "human_reviewed": len(resolutions), "remaining": len(discussion),
        "note": "Assigned human decisions, not three-person consensus. No source ratings were modified.",
        "inputs": [{"path": str(path), "sha256": digest(path)} for path in review_paths],
    })
    return len(resolutions), len(discussion)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "build"):
        command = commands.add_parser(name)
        command.add_argument("--tasks", type=Path, required=True)
        command.add_argument("--exports", type=Path, required=True)
        command.add_argument("--pierre-humanized", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
        if name == "build":
            command.add_argument("--sources", type=Path, required=True)
            command.add_argument("--proposals", type=Path, required=True)
            command.add_argument("--previous-packages", type=Path,
                                 help="Reuse review identities only if all assigned case and source data are unchanged")
            command.add_argument("--assignment-mode", choices=("documents", "differing-annotator"),
                                 default="documents",
                                 help="Keep whole documents together, or assign each case to its differing annotator")
    merge = commands.add_parser("merge")
    merge.add_argument("--packages", type=Path, required=True)
    merge.add_argument("--reviews", type=Path, nargs="+", required=True)
    merge.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "merge":
        reviewed, remaining = merge_reviews(args.packages, args.reviews, args.out)
        print(f"{reviewed} human-reviewed; {remaining} remaining")
        return
    tasks, cases, provenance = load_inputs(args.tasks, args.exports, args.pierre_humanized)
    if args.command == "prepare":
        rows = prepare_sources(tasks, cases, args.out)
        print(f"Prepared {len(rows)} PDFs for {len(cases)} disagreement items")
    else:
        manifest = build_packages(tasks, cases, provenance, args.sources, args.proposals, args.out,
                                  previous_packages=args.previous_packages,
                                  assignment_mode=args.assignment_mode)
        print(json.dumps(manifest["packages"], indent=2))


if __name__ == "__main__":
    main()
