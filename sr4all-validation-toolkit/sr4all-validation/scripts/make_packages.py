#!/usr/bin/env python3
"""
Build per-annotator annotation packages for the SR4ALL extraction validation.

Input:
  --tasks     master tasks.jsonl (one line per sampled document, see README for schema)
  --pdf-dir   directory containing the PDFs referenced in tasks.jsonl ("pdf" field, filename part)
  --tool      path to index.html (the annotation tool)
  --annotators comma-separated annotator ids, e.g. pierre,tim,arno
  --shared    number of shared documents annotated by everyone (default 80)
  --seed      random seed for the shared/split assignment (default 42)
  --out       output directory

Output:
  <out>/package_<annotator>.zip  containing index.html, tasks.jsonl, pdfs/
  <out>/assignment.json          which doc went to whom (for the record / the paper's repo)

The shared documents are marked with "shared": true in each package's tasks.jsonl.
"""
import argparse, json, random, shutil, sys, zipfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--pdf-dir", required=True)
    ap.add_argument("--tool", required=True)
    ap.add_argument("--annotators", required=True)
    ap.add_argument("--shared", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="packages")
    args = ap.parse_args()

    annotators = [a.strip() for a in args.annotators.split(",") if a.strip()]
    tasks = [json.loads(l) for l in open(args.tasks, encoding="utf-8") if l.strip()]
    pdf_dir = Path(args.pdf_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.shared > len(tasks):
        sys.exit(f"--shared {args.shared} exceeds number of tasks {len(tasks)}")

    rng = random.Random(args.seed)
    order = list(range(len(tasks)))
    rng.shuffle(order)
    shared_idx = set(order[: args.shared])
    rest = order[args.shared:]

    # even round-robin split of the rest
    split = {a: [] for a in annotators}
    for i, idx in enumerate(rest):
        split[annotators[i % len(annotators)]].append(idx)

    assignment = {
        "seed": args.seed,
        "n_total": len(tasks),
        "n_shared": args.shared,
        "shared": sorted(tasks[i]["doc_id"] for i in shared_idx),
        "own": {a: sorted(tasks[i]["doc_id"] for i in split[a]) for a in annotators},
    }
    (out / "assignment.json").write_text(json.dumps(assignment, indent=2), encoding="utf-8")

    missing_pdfs = []
    for a in annotators:
        pkg = out / f"package_{a}"
        if pkg.exists():
            shutil.rmtree(pkg)
        (pkg / "pdfs").mkdir(parents=True)
        shutil.copy(args.tool, pkg / "index.html")
        shutil.copy(Path(args.tool).parent.parent / "ANNOTATOR_GUIDELINES.md",
                    pkg / "ANNOTATOR_GUIDELINES.md")

        doc_idxs = sorted(shared_idx) + sorted(split[a])
        with open(pkg / "tasks.jsonl", "w", encoding="utf-8") as fh:
            for idx in doc_idxs:
                t = dict(tasks[idx])
                t["shared"] = idx in shared_idx
                fname = Path(t["pdf"]).name
                t["pdf"] = f"pdfs/{fname}"
                doc_id = Path(fname).stem
                src = pdf_dir / doc_id[:2] / fname
                if not src.exists():
                    src = pdf_dir / fname
                if src.exists():
                    shutil.copy(src, pkg / "pdfs" / fname)
                else:
                    missing_pdfs.append(fname)
                fh.write(json.dumps(t, ensure_ascii=False) + "\n")

        zpath = out / f"package_{a}.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for p in pkg.rglob("*"):
                z.write(p, p.relative_to(pkg))
        print(f"{a}: {len(doc_idxs)} docs ({args.shared} shared + {len(split[a])} own) -> {zpath}")

    if missing_pdfs:
        print(f"\nWARNING: {len(set(missing_pdfs))} PDFs not found in {pdf_dir}:", file=sys.stderr)
        for f in sorted(set(missing_pdfs))[:20]:
            print("  " + f, file=sys.stderr)


if __name__ == "__main__":
    main()
