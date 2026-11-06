#!/usr/bin/env python3
"""Evaluate independent exports without modifying them; see the toolkit README."""
import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


LABELS = {"field": ("correct", "partial", "incorrect"), "null": ("absent", "present")}
JUDGMENTS = {"field": "field_judgments", "null": "null_field_judgments"}
COMMENT_THEMES = {
    "incomplete_value": "An extracted value omits elements, conditions, qualifiers, or a query component.",
    "wrong_value": "An extracted term, number, or content differs substantively from the reported value.",
    "evidence_mismatch": "The cited evidence is missing, mismatched, or insufficient to support the extraction.",
    "ocr_layout": "OCR, line numbers, page headers, line breaks, or merged/split text affect the value.",
    "query_syntax": "Query operators, grouping, quotation, field restrictions, or exact syntax differ.",
    "normalization_scope": "Field meaning, scope, normalization, or unit of extraction requires interpretation.",
    "source_inconsistency": "The source itself gives conflicting statements or counts.",
    "source_access": "Unavailable main text or supplements, mismatched/checklist-only material, language, or readability limits assessment.",
    "presence_evidence": "The comment supplies or identifies reported information for a null field.",
    "absence_statement": "The comment explicitly states that a field was not found or reported.",
    "supporting_note": "The comment supports the current extraction or explains an acceptable variant.",
    "unclear_note": "The comment is too vague, referential, or nonspecific to establish a mechanism.",
    "other": "A substantive note not captured by the other themes.",
}
RECALL_NOTE = (
    "Recall is conditional on every non-null field being reported in the PDF: "
    "correct / (correct + partial + incorrect + null-present). The exports do not "
    "independently establish presence for incorrect extractions. An unsupported "
    "evidence span does not establish field absence. Partial counts as not correct."
)


def wilson(successes, total, z=1.959963984540054):
    """Return a proportion and its two-sided 95% Wilson score interval."""
    if not 0 <= successes <= total:
        raise ValueError("require 0 <= successes <= total")
    if total == 0:
        return None, None, None
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return proportion, max(0.0, center - half), min(1.0, center + half)


def fleiss_kappa(item_label_counts, categories):
    """Fleiss' kappa and mean pairwise agreement, or None when undefined."""
    if not item_label_counts:
        return None, None
    n_raters = sum(item_label_counts[0].values())
    if any(sum(counts.values()) != n_raters for counts in item_label_counts):
        raise ValueError("all items must have the same number of ratings")
    if n_raters < 2:
        return None, None
    totals = Counter()
    agreement_sum = 0.0
    for counts in item_label_counts:
        if set(counts) - set(categories) or any(count < 0 for count in counts.values()):
            raise ValueError("invalid category counts")
        agreement_sum += (sum(count * count for count in counts.values()) - n_raters) / (
            n_raters * (n_raters - 1)
        )
        totals.update(counts)
    agreement = agreement_sum / len(item_label_counts)
    expected = sum((totals[label] / (len(item_label_counts) * n_raters)) ** 2 for label in categories)
    return (None if expected >= 1 else (agreement - expected) / (1 - expected)), agreement


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield row


def load_exports(directory):
    files = sorted(Path(directory).glob("annotations_*.jsonl"))
    if not files:
        raise ValueError(f"no annotations_*.jsonl files found in {directory}")
    data = defaultdict(dict)
    for path in files:
        for row in read_jsonl(path):
            annotator, doc_id = row.get("annotator"), row.get("doc_id")
            if not isinstance(annotator, str) or not annotator or not isinstance(doc_id, str) or not doc_id:
                raise ValueError(f"{path}: missing annotator or doc_id")
            if doc_id in data[annotator]:
                raise ValueError(f"{path}: duplicate {annotator}/{doc_id}; keep one export per document")
            data[annotator][doc_id] = row
    if not data:
        raise ValueError("exports contain no records")
    return data


def field_type(name):
    return re.sub(r"\[\d+\]$", "", name)


def agreement_stats(counts, categories):
    kappa, agreement = fleiss_kappa(counts, categories)
    unanimous = sum(sum(count > 0 for count in item.values()) == 1 for item in counts)
    return {"n_items": len(counts), "n_unanimous": unanimous,
            "fleiss_kappa": kappa, "raw_agreement": agreement}


def evaluate(data, tasks=None, adjudications=None, expected_annotators=3):
    """Resolve labels explicitly, retaining original ratings for agreement."""
    annotators = sorted(data)
    if not annotators:
        raise ValueError("no annotators")
    documents = sorted({doc_id for docs in data.values() for doc_id in docs})
    if tasks is not None and set(documents) - set(tasks):
        raise ValueError(f"export documents absent from master tasks: {sorted(set(documents) - set(tasks))}")
    judgments = defaultdict(dict)
    expected = set()
    shared_docs = set()
    incomplete = set()
    owners = {}
    for doc_id in documents:
        records = {name: data[name][doc_id] for name in annotators if doc_id in data[name]}
        flags = {row.get("shared", False) for row in records.values()}
        if len(flags) != 1 or any(not isinstance(flag, bool) for flag in flags):
            raise ValueError(f"{doc_id}: inconsistent shared flags")
        shared = True in flags
        if not shared and len(records) > 1:
            raise ValueError(f"{doc_id}: multiple annotators for a non-shared document")
        if shared:
            shared_docs.add(doc_id)
        owners[doc_id] = annotators if shared else list(records)
        task_keys = None
        if tasks is not None:
            task = tasks[doc_id]
            task_keys = {(doc_id, "field", node["name"]) for node in task["fields"]}
            task_keys.update((doc_id, "null", name) for name in task["null_fields"])
            expected.update(task_keys)
        for name, row in records.items():
            seen = set()
            if row.get("status") != "done":
                incomplete.add(doc_id)
            for kind, attribute in JUDGMENTS.items():
                for node in row.get(attribute, []):
                    field = node.get("name")
                    if not isinstance(field, str) or not field:
                        raise ValueError(f"{doc_id}/{name}: missing field name")
                    key = (doc_id, kind, field)
                    if key in seen or (task_keys is not None and key not in task_keys):
                        raise ValueError(f"{doc_id}/{name}: duplicate or unexpected field {field}")
                    seen.add(key)
                    expected.add(key)
                    label = node.get("judgment")
                    if label not in (*LABELS[kind], None, ""):
                        raise ValueError(f"{doc_id}/{name}/{field}: invalid judgment {label!r}")
                    if row.get("status") == "done" and label:
                        judgments[key][name] = node

    decisions = {}
    for decision in adjudications or []:
        key = (decision.get("doc_id"), decision.get("kind"), decision.get("field"))
        if key not in expected or key in decisions:
            raise ValueError(f"unknown or duplicate adjudication key: {key}")
        label = decision.get("judgment")
        if label in (None, ""):
            continue
        if label not in LABELS[key[1]]:
            raise ValueError(f"{key}: invalid adjudicated judgment {label!r}")
        decisions[key] = decision

    agreement_counts = {kind: [] for kind in LABELS}
    per_field_agreement = defaultdict(lambda: {kind: [] for kind in LABELS})
    disagreements = []
    audit = []
    groups = defaultdict(list)
    for key in sorted(expected):
        doc_id, kind, field = key
        ratings = judgments[key]
        counts = Counter(node["judgment"] for node in ratings.values())
        complete = set(ratings) == set(owners[doc_id])
        if not complete:
            incomplete.add(doc_id)
        if doc_id in shared_docs and complete and len(annotators) >= 2:
            agreement_counts[kind].append(counts)
            per_field_agreement[field_type(field)][kind].append(counts)
        reason = "missing_rating" if not complete else "disagreement" if len(counts) > 1 else ""
        if key in decisions:
            label, source = decisions[key]["judgment"], "adjudicated"
        elif not reason:
            label, source = next(iter(counts)), "agreed" if doc_id in shared_docs else "single"
        else:
            label, source = None, "unresolved"
        item = {"doc_id": doc_id, "kind": kind, "field": field, "judgment": label,
                "source": source, "reason": reason, "ratings": ratings,
                "adjudication": decisions.get(key)}
        audit.append(item)
        groups[(doc_id, kind, field_type(field))].append(label)
        if reason:
            disagreements.append(item)

    totals = defaultdict(Counter)
    for (doc_id, kind, field), labels in groups.items():
        counts = totals[field]
        if None in labels:
            counts["unresolved_document_fields"] += 1
            continue
        if kind == "field":
            label = max(labels, key=LABELS["field"].index)
            counts[label] += 1
        else:
            counts["null_" + labels[0]] += 1

    metrics = []
    for field, counts in sorted(totals.items()):
        n_precision = sum(counts[label] for label in LABELS["field"])
        n_recall = n_precision + counts["null_present"]
        precision, prec_lo, prec_hi = wilson(counts["correct"], n_precision)
        recall, rec_lo, rec_hi = wilson(counts["correct"], n_recall)
        metrics.append({
            "field": field, "correct": counts["correct"], "partial": counts["partial"],
            "incorrect": counts["incorrect"], "n_precision": n_precision,
            "precision": precision, "prec_lo": prec_lo, "prec_hi": prec_hi,
            "partial_share": counts["partial"] / n_precision if n_precision else None,
            "n_recall": n_recall, "recall": recall, "rec_lo": rec_lo, "rec_hi": rec_hi,
            "missed": counts["null_present"], "null_absent": counts["null_absent"],
            "unresolved_document_fields": counts["unresolved_document_fields"],
        })

    agreement = {"n_annotators": len(annotators), "shared_documents": len(shared_docs)}
    for kind, name in [("field", "extracted"), ("null", "null")]:
        agreement[name] = agreement_stats(agreement_counts[kind], LABELS[kind])
    agreement["per_field"] = {
        field: {kind: agreement_stats(counts[kind], LABELS[kind]) for kind in LABELS}
        for field, counts in sorted(per_field_agreement.items())
    }
    missing_docs = sorted(set(tasks or {}) - set(documents))
    unresolved = sum(item["judgment"] is None for item in audit)
    warnings = [RECALL_NOTE]
    if len(annotators) != expected_annotators:
        warnings.append(f"Paper protocol expects {expected_annotators} annotators; found {len(annotators)}.")
    if tasks is None:
        warnings.append("No master tasks supplied: sample and field completeness cannot be verified.")
    if missing_docs:
        warnings.append(f"{len(missing_docs)} sampled documents have no export.")
    if unresolved:
        warnings.append(f"{unresolved} unresolved items excluded; resolved-only estimates may be biased.")
    summary = {
        "estimate_status": "provisional" if (
            unresolved or incomplete or missing_docs or tasks is None or len(annotators) != expected_annotators
        ) else "adjudicated_or_unanimous",
        "annotators": annotators, "documents_by_annotator": {name: len(data[name]) for name in annotators},
        "sample_documents": len(tasks) if tasks is not None else None,
        "observed_documents": len(documents), "shared_documents": len(shared_docs),
        "missing_documents": missing_docs, "incomplete_documents": sorted(incomplete),
        "unresolved_items": unresolved, "disagreement_items": sum(item["reason"] == "disagreement" for item in audit),
        "adjudicated_items": len(decisions), "warnings": warnings,
        "scoring": {
            "unit": "document x field type; indexed queries collapse to the worst query judgment",
            "precision": "correct / (correct + partial + incorrect)",
            "recall": RECALL_NOTE,
            "agreement": "original independent item labels, complete shared items only; per kind and field",
        },
    }
    return {"summary": summary, "agreement": agreement, "metrics": metrics,
            "disagreements": disagreements, "audit": audit}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")


def write_tsv(path, rows, columns):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def display(value):
    return "--" if value is None else f"{value:.3f}"


def tex_escape(value):
    return str(value).translate(str.maketrans({"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#"}))


def sample_report(data, tasks, manifest=None):
    if manifest is not None and set(manifest["sampled_doc_ids"]) != set(tasks):
        raise ValueError("sampling manifest and master task IDs differ")
    observed = {doc_id for docs in data.values() for doc_id in docs}
    completed = {doc_id for doc_id in observed
                 if all(docs[doc_id].get("status") == "done" for docs in data.values() if doc_id in docs)}
    disciplines = sorted({task.get("field", "Unknown") for task in tasks.values()})
    composition = []
    for discipline in disciplines:
        ids = {doc_id for doc_id, task in tasks.items() if task.get("field", "Unknown") == discipline}
        composition.append({"discipline": discipline, "sampled": len(ids),
                            "observed": len(ids & observed), "submitted_done": len(ids & completed),
                            "missing": len(ids - observed)})
    nonnull, null = Counter(), Counter()
    for task in tasks.values():
        nonnull.update({field_type(node["name"]) for node in task["fields"]})
        null.update({field_type(name) for name in task["null_fields"]})
    coverage = [{"field": field, "nonnull_tasks": nonnull[field], "null_tasks": null[field],
                 "unassessed_tasks": len(tasks) - nonnull[field] - null[field]}
                for field in sorted(nonnull.keys() | null.keys())]
    annotator_metrics = []
    for annotator, docs in sorted(data.items()):
        for cohort, shared in (("shared", True), ("exclusive", False)):
            subset = {doc_id: row for doc_id, row in docs.items() if row.get("shared", False) == shared}
            if subset:
                result = evaluate({annotator: subset}, expected_annotators=1)
                annotator_metrics.extend(dict(row, annotator=annotator, cohort=cohort,
                                              submitted_documents=len(subset)) for row in result["metrics"])
    return {"discipline_composition": composition, "task_field_coverage": coverage,
            "sampling_manifest": manifest,
            "size_bin_completion_note": "Tasks omit original size-bin membership; only planned strata are available.",
            "annotator_metrics_note": "Descriptive original single-rater scores, not pooled estimates or adjusted labels.",
            "annotator_metrics": annotator_metrics}


def error_report(result):
    groups = defaultdict(list)
    for item in result["audit"]:
        groups[(item["doc_id"], item["kind"], field_type(item["field"]))].append(item)
    cases = []
    categories = defaultdict(lambda: [set(), 0])
    for (doc_id, kind, field), items in sorted(groups.items()):
        flagged = [dict(node, annotator=annotator, item_field=item["field"])
                   for item in items for annotator, node in item["ratings"].items()
                   if node["judgment"] in ("partial", "incorrect", "present")]
        labels = [item["judgment"] for item in items]
        label = None if None in labels else max(labels, key=LABELS[kind].index)
        if not flagged and label not in ("partial", "incorrect", "present"):
            continue
        status = ("unresolved_flag" if label is None else "resolved_error"
                  if label in ("partial", "incorrect", "present") else "resolved_nonerror_flag")
        type_counts = Counter((node.get("error_type") or "unclassified").strip() or "unclassified"
                              for node in flagged)
        if not type_counts:
            type_counts["unclassified"] = 0
        cases.append({"doc_id": doc_id, "kind": kind, "field": field, "status": status,
                      "judgment": label, "n_flagged_ratings": len(flagged),
                      "error_types": sorted(type_counts), "flagged_ratings": flagged,
                      "items": items})
        for error_type, count in type_counts.items():
            bucket = categories[(field, kind, status, error_type)]
            bucket[0].add(doc_id)
            bucket[1] += count
    counts = [{"field": field, "kind": kind, "status": status, "error_type": error_type,
               "n_document_fields": len(documents), "n_flagged_ratings": ratings}
              for (field, kind, status, error_type), (documents, ratings) in sorted(categories.items())]
    return cases, counts


def pairwise_report(result):
    annotators = result["summary"]["annotators"]
    complete = [item for item in result["audit"] if len(item["ratings"]) == len(annotators) >= 2]
    queries = [item for item in complete if field_type(item["field"]) == "exact_boolean_queries"]
    query_groups = defaultdict(list)
    for item in queries:
        query_groups[item["doc_id"]].append(item)
    query_documents = [
        {"doc_id": doc_id, "n_items": len(items),
         "n_disputed": sum(len({node["judgment"] for node in item["ratings"].values()}) > 1 for item in items),
         "labels": {name: dict(Counter(item["ratings"][name]["judgment"] for item in items)) for name in annotators}}
        for doc_id, items in sorted(query_groups.items(), key=lambda pair: (-len(pair[1]), pair[0]))]
    largest = query_documents[0]["doc_id"] if query_documents else None
    grouped = defaultdict(list)
    for item in result["audit"]:
        grouped[(item["doc_id"], item["kind"], field_type(item["field"]))].append(item)
    collapsed = []
    for (doc_id, kind, field), items in sorted(grouped.items()):
        if len(annotators) < 2 or any(set(item["ratings"]) != set(annotators) for item in items):
            continue
        ratings = {name: {"judgment": max((item["ratings"][name]["judgment"] for item in items), key=LABELS[kind].index)}
                   for name in annotators}
        collapsed.append({"doc_id": doc_id, "kind": kind, "field": field, "ratings": ratings})
    scopes = {"all": complete,
              "nonquery": [item for item in complete if field_type(item["field"]) != "exact_boolean_queries"],
              "queries": queries, "document_field": collapsed,
              "without_largest_query_document": [item for item in complete if item["doc_id"] != largest]}
    scopes.update({"field:" + field: [item for item in complete if field_type(item["field"]) == field]
                   for field in sorted({field_type(item["field"]) for item in complete})})
    rows, sensitivity = [], []
    for scope, selected in scopes.items():
        for kind in LABELS:
            if kind == "null" and scope not in ("all",) and not scope.startswith("field:"):
                continue
            items = [item for item in selected if item["kind"] == kind]
            counts = [Counter(node["judgment"] for node in item["ratings"].values()) for item in items]
            sensitivity.append(dict(agreement_stats(counts, LABELS[kind]), scope=scope, kind=kind))
            for first, second in combinations(annotators, 2):
                pairs = [(item["ratings"][first]["judgment"], item["ratings"][second]["judgment"]) for item in items]
                total = len(pairs)
                matches = sum(left == right for left, right in pairs)
                first_counts = Counter(left for left, right in pairs)
                second_counts = Counter(right for left, right in pairs)
                observed = matches / total if total else None
                expected = sum(first_counts[label] * second_counts[label] for label in LABELS[kind]) / total ** 2 if total else None
                documents = defaultdict(list)
                for item, (left, right) in zip(items, pairs):
                    documents[item["doc_id"]].append(left == right)
                balanced = sum(sum(values) / len(values) for values in documents.values()) / len(documents) if documents else None
                rows.append({"scope": scope, "kind": kind, "annotator_a": first, "annotator_b": second,
                             "n_items": total, "n_documents": len(documents), "n_matches": matches,
                             "raw_agreement": observed,
                             "cohen_kappa": (observed - expected) / (1 - expected) if expected is not None and expected < 1 else None,
                             "document_balanced_agreement": balanced,
                             "marginals_a": dict(first_counts), "marginals_b": dict(second_counts),
                             "confusion": [{"label_a": left, "label_b": right, "count": count}
                                           for (left, right), count in sorted(Counter(pairs).items())]})
    return {"pairs": rows, "sensitivity": sensitivity, "query_documents": query_documents,
            "largest_query_document": largest,
            "note": "Descriptive post-hoc diagnostics of original complete shared ratings; no rater selection or adjudication. "
                    "Document-balanced agreement averages within-document exact agreement. "
                    "Document-field sensitivity collapses queries to each rater's worst original label."}


def comment_inventory(result):
    groups = {}
    for item in result["audit"]:
        for name, node in sorted(item["ratings"].items()):
            comment = node.get("comment")
            if comment is None:
                continue
            if not isinstance(comment, str):
                raise ValueError(f"{item['doc_id']}/{name}/{item['field']}: comment must be text")
            if not comment.strip():
                continue
            identity = [item["kind"], field_type(item["field"]), comment]
            key = hashlib.sha256(json.dumps(identity, ensure_ascii=True).encode("utf-8")).hexdigest()[:16]
            entry = groups.setdefault(key, {"comment_key": key, "kind": item["kind"],
                                           "field_type": field_type(item["field"]), "comment": comment,
                                           "observations": []})
            if [entry["kind"], entry["field_type"], entry["comment"]] != identity:
                raise ValueError("comment identifier collision")
            entry["observations"].append({"doc_id": item["doc_id"], "kind": item["kind"], "field": item["field"],
                                          "annotator": name, "judgment": node["judgment"],
                                          "error_type": node.get("error_type"), "resolved_judgment": item["judgment"],
                                          "resolution_source": item["source"]})
    return sorted(groups.values(), key=lambda row: (row["kind"], row["field_type"], row["comment_key"]))


def coded_comment_report(result, codes):
    inventory = comment_inventory(result)
    expected = {item["comment_key"] for item in inventory}
    mapping = {}
    for row in codes:
        key, themes = row.get("comment_key"), row.get("themes")
        if key not in expected or key in mapping:
            raise ValueError(f"unknown or duplicate comment coding key: {key}")
        if not isinstance(themes, list) or not themes or any(not isinstance(theme, str) or theme not in COMMENT_THEMES for theme in themes):
            raise ValueError(f"{key}: require a nonempty list of known comment themes")
        if len(set(themes)) != len(themes):
            raise ValueError(f"{key}: duplicate themes")
        mapping[key] = themes
    if set(mapping) != expected:
        raise ValueError(f"comment coding incomplete: {len(expected - set(mapping))} uncoded contexts")
    labels = defaultdict(list)
    for item in result["audit"]:
        labels[(item["doc_id"], item["kind"], field_type(item["field"]))].append(item["judgment"])
    statuses = {}
    for key, values in labels.items():
        label = None if None in values else max(values, key=LABELS[key[1]].index)
        statuses[key] = "unresolved_flag" if label is None else "resolved_error" if label in ("partial", "incorrect", "present") else "resolved_nonerror"
    counts = defaultdict(lambda: {"documents": set(), "document_fields": set(), "observations": 0})
    field_counts = defaultdict(set)
    for entry in inventory:
        entry["themes"] = mapping[entry["comment_key"]]
        for observation in entry["observations"]:
            case = (observation["doc_id"], entry["kind"], entry["field_type"])
            status = statuses[case]
            observation["document_field_status"] = status
            for theme in entry["themes"]:
                bucket = counts[(theme, entry["kind"], status)]
                bucket["documents"].add(observation["doc_id"])
                bucket["document_fields"].add(case)
                bucket["observations"] += 1
                field_counts[(theme, entry["kind"], status, entry["field_type"])].add(case)
    observations = [row for entry in inventory for row in entry["observations"]]
    return {"summary": {"distinct_comment_contexts": len(inventory), "comment_observations": len(observations),
                        "documents_with_comments": len({row["doc_id"] for row in observations}),
                        "document_fields_with_comments": len({(row["doc_id"], row["kind"], field_type(row["field"])) for row in observations}),
                        "by_original_judgment": dict(Counter(row["judgment"] for row in observations))},
            "codebook": COMMENT_THEMES,
            "method": "Exhaustive AI-assisted textual coding of nonempty comments with field context; "
                      "non-exclusive themes, no independent human recoding, no new PDF verification or adjudication. "
                      "Empty comments and found_text evidence are not treated as comments. "
                      "Themes describe reported observations, not verified causes or corpus error rates.",
            "coded_comments": inventory,
            "counts": [{"theme": theme, "kind": kind, "status": status,
                        "n_documents": len(bucket["documents"]), "n_document_fields": len(bucket["document_fields"]),
                        "n_observations": bucket["observations"]}
                       for (theme, kind, status), bucket in sorted(counts.items())],
            "per_field": [{"theme": theme, "kind": kind, "status": status, "field": field, "n_document_fields": len(cases)}
                          for (theme, kind, status, field), cases in sorted(field_counts.items())]}


def write_discussion_report(result, outdir, tasks=None):
    documents = defaultdict(list)
    for item in result["audit"]:
        if item["judgment"] is None:
            documents[item["doc_id"]].append(item)
    ordered = sorted(documents, key=lambda doc_id: (-len(documents[doc_id]), doc_id))
    rows = []
    lines = ["# Annotation Discussion", "",
             f"{sum(map(len, documents.values()))} unresolved items across {len(documents)} documents.", "",
             "Performance estimates remain provisional until these items are resolved. "
             "Documents are ordered by pending item count, then document ID.", "",
             "Check the source PDF, discuss each item, and record one decision and its rationale "
             "in a working copy of adjudication_template.jsonl. Do not edit the independent exports "
             "or resolve a disagreement by majority vote. Agreement always uses original ratings.", "",
             "This report includes pending items only. Source context and individual assessments "
             "are reproduced verbatim as JSON; no suggested final labels are supplied.", ""]
    for doc_id in ordered:
        items = documents[doc_id]
        task = (tasks or {}).get(doc_id, {})
        rows.append({"doc_id": doc_id, "title": task.get("title", ""), "pending_items": len(items),
                     "pending_document_fields": len({(item["kind"], field_type(item["field"])) for item in items}),
                     "pending_extracted_items": sum(item["kind"] == "field" for item in items),
                     "pending_null_items": sum(item["kind"] == "null" for item in items)})
        lines.extend([f"## {doc_id}", ""])
        context = {key: value for key, value in task.items() if key not in ("fields", "null_fields")}
        if context:
            lines.extend(["Source context:", "", "```json", json.dumps(context, indent=2, ensure_ascii=True), "```", ""])
        for item in items:
            lines.extend([f"### {item['kind']}: {item['field']}", "",
                          f"Reason: {item['reason']}", "", "| Annotator | Judgment |",
                          "| --- | --- |"])
            for name in result["summary"]["annotators"]:
                rating = item["ratings"].get(name, {})
                lines.append(f"| {name.replace('|', r'\|')} | {rating.get('judgment', 'missing')} |")
            source = next((node for node in task.get("fields", []) if node["name"] == item["field"]), None)
            if source is not None:
                lines.extend(["", "Original extraction and evidence:", "", "```json",
                              json.dumps(source, indent=2, ensure_ascii=True), "```"])
            lines.extend(["", "Original independent assessments:", "", "```json",
                          json.dumps(item["ratings"], indent=2, ensure_ascii=True), "```", ""])
    write_tsv(outdir / "discussion_documents.tsv", rows,
              ["doc_id", "title", "pending_items", "pending_document_fields", "pending_extracted_items", "pending_null_items"])
    (outdir / "discussion_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_results(result, outdir, tasks=None, comment_codes=None):
    outdir.mkdir(parents=True, exist_ok=True)
    pairwise = pairwise_report(result)
    write_json(outdir / "pairwise_agreement.json", pairwise)
    write_tsv(outdir / "pairwise_agreement.tsv", pairwise["pairs"],
              ["scope", "kind", "annotator_a", "annotator_b", "n_items", "n_documents", "n_matches", "raw_agreement", "cohen_kappa", "document_balanced_agreement"])
    write_tsv(outdir / "agreement_sensitivity.tsv", pairwise["sensitivity"],
              ["scope", "kind", "n_items", "n_unanimous", "raw_agreement", "fleiss_kappa"])
    write_tsv(outdir / "query_document_concentration.tsv", pairwise["query_documents"], ["doc_id", "n_items", "n_disputed"])
    inventory = comment_inventory(result)
    write_json(outdir / "comment_inventory.json", inventory)
    pairs = list(combinations(result["summary"]["annotators"], 2))
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Descriptive pairwise agreement on original complete shared-item ratings. Extracted-value (E) and null-field (N) labels are separate. $P_o$ is exact agreement; $\kappa_C$ is unweighted Cohen's kappa. Query indices remain separate items.}",
             r"\begin{tabular}{lrrrr}", r"\toprule", r"Pair & $P_{o,E}$ & $\kappa_{C,E}$ & $P_{o,N}$ & $\kappa_{C,N}$ \\", r"\midrule"]
    for first, second in pairs:
        cells = [tex_escape(first + " / " + second)]
        for kind in ("field", "null"):
            row = next(row for row in pairwise["pairs"] if row["scope"] == "all" and row["kind"] == kind and row["annotator_a"] == first and row["annotator_b"] == second)
            cells.extend([display(row["raw_agreement"]), display(row["cohen_kappa"])])
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\label{tab:validation-pairwise}", r"\end{table}"])
    (outdir / "pairwise_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if comment_codes is not None:
        comments = coded_comment_report(result, comment_codes)
        write_json(outdir / "comment_analysis.json", comments)
        write_tsv(outdir / "comment_theme_counts.tsv", comments["counts"],
                  ["theme", "kind", "status", "n_documents", "n_document_fields", "n_observations"])
        write_tsv(outdir / "comment_themes_per_field.tsv", comments["per_field"],
                  ["theme", "kind", "status", "field", "n_document_fields"])
        lines = [r"\begin{table}[t]", r"\centering", r"\small",
                 r"\caption{AI-assisted coding of all nonempty annotation comments. Counts are unique document--fields within each theme and current resolution status: extracted errors (E error), unresolved extracted fields (E pending), resolved non-error extracted fields (E other), null-present (N missed), unresolved null fields (N pending), and null-absent (N absent). Themes overlap and describe comment content, not independently verified causes.}",
                 r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{lrrrrrr}", r"\toprule",
                 r"Theme & E error & E pending & E other & N missed & N pending & N absent \\", r"\midrule"]
        for theme in COMMENT_THEMES:
            selected = [row for row in comments["counts"] if row["theme"] == theme]
            if not selected:
                continue
            cells = [tex_escape(theme.replace("_", " "))]
            for kind in ("field", "null"):
                for status in ("resolved_error", "unresolved_flag", "resolved_nonerror"):
                    cells.append(str(sum(row["n_document_fields"] for row in selected if row["kind"] == kind and row["status"] == status)))
            lines.append(" & ".join(cells) + r" \\")
        lines.extend([r"\bottomrule", r"\end{tabular}}", r"\label{tab:validation-comment-themes}", r"\end{table}"])
        (outdir / "comment_themes_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(outdir / "summary.json", result["summary"])
    write_json(outdir / "fleiss_kappa.json", result["agreement"])
    agreement = result["agreement"]
    agreement_rows = [dict(stats, field="ALL", kind=kind)
                      for kind, stats in (("field", agreement["extracted"]), ("null", agreement["null"]))]
    agreement_rows.extend(dict(stats, field=field, kind=kind)
                          for field, groups in agreement["per_field"].items() for kind, stats in groups.items())
    write_tsv(outdir / "agreement_per_field.tsv", agreement_rows,
              ["field", "kind", "n_items", "n_unanimous", "raw_agreement", "fleiss_kappa"])
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Original independent ratings on shared documents: mean pairwise agreement ($P_o$) and nominal Fleiss' $\kappa$, separately for extracted-value correctness (E) and null-field presence/absence (N). Indexed queries are individual agreement items. Undefined estimates are shown as --.}",
             r"\small", r"\setlength{\tabcolsep}{3pt}", r"\resizebox{\linewidth}{!}{%",
             r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Field & $n_E$ & $P_{o,E}$ & $\kappa_E$ & $n_N$ & $P_{o,N}$ & $\kappa_N$ \\", r"\midrule"]
    groups = {"All fields": {"field": agreement["extracted"], "null": agreement["null"]}, **agreement["per_field"]}
    for field, kinds in groups.items():
        cells = [tex_escape(field.replace("_", " "))]
        for kind in ("field", "null"):
            stats = kinds[kind]
            cells.extend([str(stats["n_items"]), display(stats["raw_agreement"]), display(stats["fleiss_kappa"])])
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}}", r"\label{tab:validation-agreement}", r"\end{table}"])
    (outdir / "agreement_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_discussion_report(result, outdir, tasks)
    write_json(outdir / "results_per_field.json", result["metrics"])
    write_tsv(outdir / "results_per_field.tsv", result["metrics"], list(result["metrics"][0]) if result["metrics"] else ["field"])
    disagreement_rows = [dict(item, ratings=json.dumps(item["ratings"], ensure_ascii=True))
                         for item in result["disagreements"]]
    write_tsv(outdir / "disagreements.tsv", disagreement_rows,
              ["doc_id", "kind", "field", "reason", "judgment", "source", "ratings"])
    with (outdir / "final_labels.jsonl").open("w", encoding="utf-8") as stream:
        for item in result["audit"]:
            stream.write(json.dumps(item, ensure_ascii=True, allow_nan=False) + "\n")
    template = outdir / "adjudication_template.jsonl"
    if not template.exists():
        with template.open("x", encoding="utf-8") as stream:
            for item in result["disagreements"]:
                if item["judgment"] is None:
                    row = {key: item[key] for key in ("doc_id", "kind", "field")}
                    row.update(judgment=None, comment="", original_ratings=item["ratings"])
                    stream.write(json.dumps(row, ensure_ascii=True) + "\n")
    errors = []
    for item in result["audit"]:
        for annotator, node in item["ratings"].items():
            if node["judgment"] in ("partial", "incorrect", "present"):
                errors.append({"doc_id": item["doc_id"], "field": item["field"], "kind": item["kind"],
                               "annotator": annotator, **node})
    write_tsv(outdir / "error_analysis.tsv", errors,
              ["doc_id", "kind", "field", "annotator", "judgment", "error_type", "comment", "found_text", "page"])
    cases, counts = error_report(result)
    write_tsv(outdir / "error_counts.tsv", counts,
              ["field", "kind", "status", "error_type", "n_document_fields", "n_flagged_ratings"])
    with (outdir / "error_cases.jsonl").open("w", encoding="utf-8") as stream:
        for case in cases:
            stream.write(json.dumps(case, ensure_ascii=True, allow_nan=False) + "\n")
    if "sample" in result:
        sample = result["sample"]
        write_json(outdir / "sample_report.json", sample)
        write_tsv(outdir / "sample_composition.tsv", sample["discipline_composition"],
                  ["discipline", "sampled", "observed", "submitted_done", "missing"])
        write_tsv(outdir / "task_field_coverage.tsv", sample["task_field_coverage"],
                  ["field", "nonnull_tasks", "null_tasks", "unassessed_tasks"])
        rows = sample["annotator_metrics"]
        write_tsv(outdir / "annotator_metrics.tsv", rows, list(rows[0]) if rows else ["annotator"])
    status = result["summary"]["estimate_status"].replace("_", " ")
    lines = [
        "% Generated by analyze.py; requires booktabs. Recall is conditional, not independently identified.",
        r"\begin{table}[t]", r"\centering",
        "\\caption{Manual validation (" + status + "). Strict precision and conditional recall "
        "with 95\\% Wilson intervals. Each denominator counts document--field units once. "
        "Recall assumes all non-null fields are reported in the PDF; unresolved fields are excluded. "
        "Pending counts document--fields with unresolved items. Boolean-query recall and missed counts "
        "are unavailable because query absence was not assessed.}",
        r"\small", r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrlrrlrr}", r"\toprule",
        r"Field & $n_P$ & Precision [95\% CI] & Partial (\%) & $n_R$ & Recall [95\% CI] & Missed & Pending \\", r"\midrule",
    ]
    for row in result["metrics"]:
        field = row["field"].replace("_", " ")
        recall = f"{display(row['recall'])} [{display(row['rec_lo'])}, {display(row['rec_hi'])}]"
        n_recall = str(row["n_recall"])
        missed = str(row["missed"])
        if row["field"] == "exact_boolean_queries" and not row["missed"] + row["null_absent"]:
            recall, n_recall, missed = "--", "--", "--"
        partial = "--" if row["partial_share"] is None else f"{100 * row['partial_share']:.1f}"
        lines.append(f"{field} & {row['n_precision']} & {display(row['precision'])} "
                     f"[{display(row['prec_lo'])}, {display(row['prec_hi'])}] & {partial} & {n_recall} & "
                     f"{recall} & {missed} & {row['unresolved_document_fields']} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}}", r"\label{tab:validation-results}", r"\end{table}"])
    (outdir / "results_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    error_types = sorted({category for case in cases if case["kind"] == "field" for category in case["error_types"]})
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Provisional annotator-reported error categories for extracted fields. Counts are unique document--field units within each category, not individual query items or annotator votes. Categories are non-exclusive. Unresolved flags are not confirmed errors.}",
             r"\begin{tabular}{p{0.47\linewidth}rrr}", r"\toprule",
             r"Reported category & Partial & Incorrect & Unresolved \\", r"\midrule"]
    for category in error_types:
        selected = [case for case in cases if case["kind"] == "field" and category in case["error_types"]]
        counts = [sum(case["status"] == "resolved_error" and case["judgment"] == label for case in selected)
                  for label in ("partial", "incorrect")]
        counts.append(sum(case["status"] == "unresolved_flag" for case in selected))
        lines.append(tex_escape(category) + " & " + " & ".join(map(str, counts)) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\label{tab:validation-errors}", r"\end{table}"])
    (outdir / "error_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if "sample" in result:
        lines = [r"\begin{table}[t]", r"\centering",
                 r"\caption{Planned validation sample and provisional document coverage by discipline. Shared documents count once. Observed denotes a submitted document, not completed three-rater annotation or adjudication.}",
                 r"\begin{tabular}{p{0.56\linewidth}rrr}", r"\toprule",
                 r"Discipline & Sampled & Observed & Missing \\", r"\midrule"]
        for row in result["sample"]["discipline_composition"]:
            lines.append(tex_escape(row["discipline"]) + f" & {row['sampled']} & {row['observed']} & {row['missing']} " + r"\\")
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\label{tab:validation-sample}", r"\end{table}"])
        (outdir / "sample_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=root / "data/validation_sample/results")
    parser.add_argument("--tasks", type=Path, help="master tasks JSONL for sample and field validation")
    parser.add_argument("--manifest", type=Path, help="optional sampling manifest; defaults to tasks sibling when present")
    parser.add_argument("--adjudicated", type=Path, help="separate adjudicated labels; never used for kappa")
    parser.add_argument("--comment-codes", type=Path, help="exhaustive AI-assisted coding JSONL or directory containing codes_*.jsonl")
    parser.add_argument("--out", type=Path, help="defaults to <dir>/results")
    parser.add_argument("--paper-tables", type=Path, help="refresh generated validation tables in this manuscript directory")
    parser.add_argument("--expected-annotators", type=int, default=3, help="paper protocol; never fabricates missing raters")
    args = parser.parse_args()
    if args.paper_tables and args.tasks is None:
        parser.error("--paper-tables requires --tasks so sample tables cannot come from a stale run")
    tasks = None
    if args.tasks:
        tasks = {}
        for task in read_jsonl(args.tasks):
            if task["doc_id"] in tasks:
                parser.error(f"duplicate task: {task['doc_id']}")
            tasks[task["doc_id"]] = task
    try:
        data = load_exports(args.dir)
        result = evaluate(data, tasks,
                          list(read_jsonl(args.adjudicated)) if args.adjudicated else None,
                          expected_annotators=args.expected_annotators)
        manifest_path = args.manifest
        if manifest_path is None and args.tasks:
            candidate = args.tasks.parent / "sampling_manifest.json"
            if candidate.exists():
                manifest_path = candidate
        if tasks is not None:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path else None
            result["sample"] = sample_report(data, tasks, manifest)
        inputs = sorted(args.dir.glob("annotations_*.jsonl"))
        inputs.extend(path for path in (args.tasks, args.adjudicated, manifest_path) if path)
        coding_paths = []
        comment_codes = None
        if args.comment_codes:
            coding_paths = sorted(args.comment_codes.glob("codes_*.jsonl")) if args.comment_codes.is_dir() else [args.comment_codes]
            if not coding_paths:
                raise ValueError("no comment coding files found")
            comment_codes = [row for path in coding_paths for row in read_jsonl(path)]
            coded_comment_report(result, comment_codes)
            inputs.extend(coding_paths)
        result["summary"]["inputs"] = []
        for path in inputs:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            result["summary"]["inputs"].append({"path": str(path.resolve()), "sha256": digest})
        outdir = args.out or args.dir / "results"
        if any(path.resolve().parent == outdir.resolve() for path in inputs):
            parser.error("keep input files outside the report output directory")
        write_results(result, outdir, tasks=tasks, comment_codes=comment_codes)
        if args.paper_tables:
            args.paper_tables.mkdir(parents=True, exist_ok=True)
            for source, target in (("results_table.tex", "table-validation-results.tex"),
                                   ("error_table.tex", "table-validation-errors.tex"),
                                   ("sample_table.tex", "table-validation-sample.tex"),
                                   ("agreement_table.tex", "table-validation-agreement.tex"),
                                   ("pairwise_table.tex", "table-validation-pairwise.tex"),
                                   ("comment_themes_table.tex", "table-validation-comment-themes.tex")):
                if source == "comment_themes_table.tex" and comment_codes is None:
                    continue
                if (outdir / source).exists():
                    (args.paper_tables / target).write_text((outdir / source).read_text(encoding="utf-8"), encoding="utf-8")
    except ValueError as error:
        parser.error(str(error))
    summary = result["summary"]
    print(f"{summary['estimate_status']}: {summary['observed_documents']} documents, "
          f"{summary['shared_documents']} shared, {summary['unresolved_items']} unresolved items")
    for warning in summary["warnings"]:
        print("WARNING:", warning)
    for kind in ("extracted", "null"):
        print(kind, json.dumps(result["agreement"][kind]))
    print(f"Reports: {outdir}")


if __name__ == "__main__":
    main()
