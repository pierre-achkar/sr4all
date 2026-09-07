#!/usr/bin/env python3
"""
Merge annotator exports and compute the validation statistics for the paper.

Input:
  --dir   directory containing annotations_<name>.jsonl exports from the tool
Output (printed + written to <dir>/results/):
  - fleiss_kappa.json      kappa + raw agreement on the shared documents
  - disagreements.tsv      shared-doc items needing adjudication (majority tie or any disagreement)
  - results_per_field.tsv  per field: n, precision, 95% Wilson CI, recall, 95% Wilson CI
  - results_table.tex      the same as a LaTeX table skeleton

Label semantics (documented in the paper):
  precision: correct / (correct + partial + incorrect)   [partial counts as incorrect; partial share reported]
  recall:    correct extractions / (fields present in PDF)
             where "present" = non-null extractions judged correct or partial or incorrect-but-present*
                             + null fields judged "present"
             * simplification: every non-null extraction is counted as "field present in PDF" unless
               its error type is "hallucinated span". Stated explicitly in the paper.
Adjudication:
  For shared documents, the final label is the majority vote across annotators.
  Ties and all disagreements are listed in disagreements.tsv; re-run after fixing
  labels in the exports (or add an adjudicated_<name>.jsonl) to finalize.
"""
import argparse, json, math, sys
from collections import Counter, defaultdict
from pathlib import Path


def wilson(x, n, z=1.959963985):
    """Wilson score interval for x successes out of n."""
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = x / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, max(0.0, center - half), min(1.0, center + half))


def fleiss_kappa(item_label_counts, categories):
    """item_label_counts: list of Counter(label->count), all items rated by the same number of raters."""
    if not item_label_counts:
        return float("nan"), float("nan")
    n_raters = sum(item_label_counts[0].values())
    N = len(item_label_counts)
    # P_i per item
    P_items = []
    cat_totals = Counter()
    for c in item_label_counts:
        if sum(c.values()) != n_raters:
            raise ValueError("all items must have the same number of ratings")
        s = sum(v * v for v in c.values())
        P_items.append((s - n_raters) / (n_raters * (n_raters - 1)))
        cat_totals.update(c)
    P_bar = sum(P_items) / N
    total = N * n_raters
    P_e = sum((cat_totals[k] / total) ** 2 for k in categories)
    if P_e >= 1.0:
        return 1.0, P_bar
    kappa = (P_bar - P_e) / (1 - P_e)
    return kappa, P_bar


def load_exports(d):
    files = sorted(Path(d).glob("annotations_*.jsonl"))
    if not files:
        sys.exit(f"no annotations_*.jsonl files found in {d}")
    data = {}  # annotator -> {doc_id: record}
    for f in files:
        recs = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        if not recs:
            continue
        ann = recs[0]["annotator"]
        data[ann] = {r["doc_id"]: r for r in recs}
        print(f"loaded {f.name}: annotator={ann}, {len(recs)} docs")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    outdir = Path(args.dir) / "results"
    outdir.mkdir(exist_ok=True)

    data = load_exports(args.dir)
    annotators = sorted(data)

    # ---- collect judgments per (doc, kind, field) ----
    # judgments[(doc_id, kind, field)] = {annotator: label}
    judgments = defaultdict(dict)
    meta = {}  # doc_id -> shared flag
    for ann, docs in data.items():
        for doc_id, r in docs.items():
            meta[doc_id] = meta.get(doc_id, False) or r.get("shared", False)
            for fj in r.get("field_judgments", []):
                if fj.get("judgment"):
                    judgments[(doc_id, "field", fj["name"])][ann] = fj["judgment"]
                    judgments[(doc_id, "field", fj["name"])].setdefault(
                        "_error_types", {})[ann] = fj.get("error_type") or ""
            for nj in r.get("null_field_judgments", []):
                if nj.get("judgment"):
                    judgments[(doc_id, "null", nj["name"])][ann] = nj["judgment"]

    # ---- Fleiss kappa on shared items rated by ALL annotators ----
    shared_items_f, shared_items_n = [], []
    disagreements = []
    for (doc_id, kind, field), labels in judgments.items():
        raters = {a: l for a, l in labels.items() if a in annotators}
        if not meta.get(doc_id) or len(raters) != len(annotators) or len(annotators) < 2:
            continue
        c = Counter(raters.values())
        (shared_items_f if kind == "field" else shared_items_n).append(c)
        if len(c) > 1:
            disagreements.append((doc_id, kind, field,
                                  " | ".join(f"{a}:{raters[a]}" for a in annotators)))

    cats_f = ["correct", "partial", "incorrect"]
    cats_n = ["absent", "present"]
    kf, pf = fleiss_kappa(shared_items_f, cats_f) if shared_items_f else (float("nan"),) * 2
    kn, pn = fleiss_kappa(shared_items_n, cats_n) if shared_items_n else (float("nan"),) * 2
    k_all, p_all = fleiss_kappa(shared_items_f + shared_items_n, cats_f + cats_n) \
        if (shared_items_f or shared_items_n) else (float("nan"),) * 2

    kappa_out = {
        "n_annotators": len(annotators),
        "shared_items_extracted_fields": len(shared_items_f),
        "shared_items_null_fields": len(shared_items_n),
        "fleiss_kappa_extracted": round(kf, 4),
        "raw_agreement_extracted": round(pf, 4),
        "fleiss_kappa_null": round(kn, 4),
        "raw_agreement_null": round(pn, 4),
        "fleiss_kappa_overall": round(k_all, 4),
        "raw_agreement_overall": round(p_all, 4),
        "n_disagreements": len(disagreements),
    }
    (outdir / "fleiss_kappa.json").write_text(json.dumps(kappa_out, indent=2))
    print("\n== Inter-annotator agreement (shared docs) ==")
    for k, v in kappa_out.items():
        print(f"  {k}: {v}")

    with open(outdir / "disagreements.tsv", "w", encoding="utf-8") as fh:
        fh.write("doc_id\tkind\tfield\tlabels\n")
        for row in sorted(disagreements):
            fh.write("\t".join(row) + "\n")
    print(f"  -> {len(disagreements)} disagreements written to disagreements.tsv (adjudicate these)")

    # ---- final labels: majority vote (shared) or single label (own docs) ----
    final = {}   # (doc_id, kind, field) -> label ("TIE" if unresolved)
    err_type = {}  # same key -> representative error type
    for key, labels in judgments.items():
        raters = {a: l for a, l in labels.items() if a in annotators}
        if not raters:
            continue
        c = Counter(raters.values())
        top = c.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            final[key] = "TIE"
        else:
            final[key] = top[0][0]
        ets = labels.get("_error_types", {})
        err_type[key] = Counter(v for v in ets.values() if v).most_common(1)
        err_type[key] = err_type[key][0][0] if err_type[key] else ""

    ties = [k for k, v in final.items() if v == "TIE"]
    if ties:
        print(f"\nWARNING: {len(ties)} tied items excluded from estimates until adjudicated.")

    # ---- per-field precision / recall ----
    per_field = defaultdict(lambda: {"correct": 0, "partial": 0, "incorrect": 0,
                                     "halluc": 0, "null_present": 0, "null_absent": 0})
    for (doc_id, kind, field), label in final.items():
        if label == "TIE":
            continue
        s = per_field[field]
        if kind == "field":
            s[label] += 1
            if label != "correct" and err_type[(doc_id, kind, field)] == "hallucinated span":
                s["halluc"] += 1
        else:
            s["null_present" if label == "present" else "null_absent"] += 1

    rows = []
    for field in sorted(per_field):
        s = per_field[field]
        n_prec = s["correct"] + s["partial"] + s["incorrect"]
        p, plo, phi = wilson(s["correct"], n_prec)
        # recall denominator: non-null non-hallucinated + nulls judged present
        n_rec = (n_prec - s["halluc"]) + s["null_present"]
        r, rlo, rhi = wilson(s["correct"], n_rec)
        rows.append({
            "field": field, "n_precision": n_prec,
            "precision": p, "prec_lo": plo, "prec_hi": phi,
            "partial_share": (s["partial"] / n_prec) if n_prec else float("nan"),
            "n_recall": n_rec, "recall": r, "rec_lo": rlo, "rec_hi": rhi,
            "missed": s["null_present"],
        })

    with open(outdir / "results_per_field.tsv", "w", encoding="utf-8") as fh:
        fh.write("field\tn_prec\tprecision\tprec_CI95\tpartial_share\tn_rec\trecall\trec_CI95\tmissed\n")
        print("\n== Per-field estimates ==")
        print(f"{'field':32s} {'n':>4s} {'precision':>22s} {'n':>5s} {'recall':>22s}")
        for r in rows:
            pci = f"[{r['prec_lo']:.3f}, {r['prec_hi']:.3f}]"
            rci = f"[{r['rec_lo']:.3f}, {r['rec_hi']:.3f}]"
            fh.write(f"{r['field']}\t{r['n_precision']}\t{r['precision']:.4f}\t{pci}\t"
                     f"{r['partial_share']:.4f}\t{r['n_recall']}\t{r['recall']:.4f}\t{rci}\t{r['missed']}\n")
            print(f"{r['field']:32s} {r['n_precision']:4d} {r['precision']:8.3f} {pci:>14s}"
                  f" {r['n_recall']:5d} {r['recall']:8.3f} {rci:>14s}")

    # ---- LaTeX table skeleton ----
    with open(outdir / "results_table.tex", "w", encoding="utf-8") as fh:
        fh.write("% Auto-generated by analyze.py — per-field validation results\n")
        fh.write("\\begin{table}[t]\n\\centering\n")
        fh.write("\\caption{Manual validation results per methodological field. "
                 "Precision and recall with 95\\% Wilson score intervals; "
                 "$n$ denotes the number of expert judgments per field.}\n")
        fh.write("\\begin{tabular}{lrllrll}\n\\toprule\n")
        fh.write("\\textbf{Field} & $n$ & \\textbf{Precision} & 95\\% CI & $n$ & \\textbf{Recall} & 95\\% CI \\\\\n\\midrule\n")
        for r in rows:
            field_tex = r["field"].replace("_", "\\_")
            fh.write(f"{field_tex} & {r['n_precision']} & "
                     f"{r['precision']:.3f} & [{r['prec_lo']:.3f}, {r['prec_hi']:.3f}] & "
                     f"{r['n_recall']} & {r['recall']:.3f} & [{r['rec_lo']:.3f}, {r['rec_hi']:.3f}] \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n\\label{tab:validation-results}\n\\end{table}\n")
    print(f"\nAll outputs in {outdir}/")


if __name__ == "__main__":
    main()
