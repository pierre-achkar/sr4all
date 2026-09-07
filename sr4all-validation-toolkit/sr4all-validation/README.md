# SR4ALL Extraction Validation Toolkit

Everything needed to run the manual expert validation of the LLM-extracted
methodological fields: a browser-based annotation tool, a packaging script,
and the analysis script that produces the numbers for the Technical Validation
section (Fleiss' kappa, per-field precision/recall with Wilson 95% CIs, LaTeX table).

No server, no installation for annotators: each annotator gets a zip,
unzips it, opens `index.html` in Chrome/Edge/Firefox, selects the folder, done.

```
sr4all-validation/
├── tool/index.html          the annotation tool (self-contained)
├── scripts/
│   ├── make_packages.py     build per-annotator zips from the master task file
│   ├── analyze.py           merge exports -> kappa, precision/recall, LaTeX table
│   └── make_example.py      regenerates the example package
├── example/                 ready-to-try package with 3 dummy reviews
│   ├── index.html
│   ├── tasks.jsonl
│   └── pdfs/
└── README.md
```

## 0. Try it (2 minutes)

Open `example/index.html` in a browser, enter any annotator id, click
"Select folder…" and pick the `example/` folder, then "Start annotating".
The three dummy documents contain deliberate errors so all judgment paths
can be exercised:

- doc 1: `databases_used` is null but reported in the PDF (a *missed* field, recall case)
- doc 2: `boolean_query` is truncated (a *partial* case); `exclusion_criteria` missed
- doc 3: `n_studies_final` has a wrong value (an *incorrect* case)

Note: the tool loads PDF.js from a CDN, so the browser needs internet
access (the PDFs and annotations themselves never leave the machine).

## 1. Prepare the master task file

One JSON line per sampled document:

```json
{"doc_id": "W4285719203",
 "title": "Workplace interventions for burnout: a systematic review",
 "pdf": "pdfs/W4285719203.pdf",
 "fields": [
   {"name": "objective", "value": "...", "evidence_span": "..."},
   {"name": "boolean_query", "value": "...", "evidence_span": "..."}
 ],
 "null_fields": ["databases_used", "research_questions"]}
```

- `fields`: every non-null extraction for this document, with its verbatim evidence span
- `null_fields`: every schema field the pipeline returned null for
- `value` may be a string, number, boolean, or list — the tool renders all of them

Generate this from `sr4all_full.jsonl` restricted to the 385 sampled ids.

## 2. Build the annotator packages

```bash
python3 scripts/make_packages.py \
  --tasks tasks_385.jsonl \
  --pdf-dir /path/to/pdfs \
  --tool tool/index.html \
  --annotators pierre,tim,arno \
  --shared 80 --seed 42 \
  --out packages/
```

Produces `package_<name>.zip` per annotator (their ~102 own docs + the 80
shared docs, marked `"shared": true`) and `assignment.json` documenting the
split — keep that file, it goes in the released validation code.

## 3. Annotate

For each **extracted field**: judge `correct` / `partial` / `incorrect`
(keys 1/2/3). For partial/incorrect an error-type dropdown and comment appear.
"Locate in PDF" (key L) highlights the evidence span in the PDF.

For each **null field**: judge `not in PDF` / `present in PDF (missed)`
(keys 1/2). For "present", paste the text found and the page number.

Navigation: J/K next/previous field, N/P next/previous document.
Judgments autosave to the browser (localStorage). Click **Export annotations**
regularly — it downloads `annotations_<name>.jsonl`. That file is the actual
deliverable per annotator; localStorage is only the working copy.
To continue on another machine: load the export via "Resume from a previous
export" on the start screen.

## 4. Analyze

Put all three `annotations_*.jsonl` files in one directory:

```bash
python3 scripts/analyze.py --dir exports/
```

Outputs in `exports/results/`:

- `fleiss_kappa.json` — kappa + raw agreement on the shared docs (overall and split by extracted/null)
- `disagreements.tsv` — every shared item where annotators disagree; adjudicate these, fix the labels in the exports, re-run
- `results_per_field.tsv` — per field: n, precision, 95% Wilson CI, partial share, n, recall, 95% Wilson CI, missed count
- `results_table.tex` — LaTeX table skeleton for the paper

Metric definitions (state these in the paper):

- **precision** = correct / (correct + partial + incorrect); partial counts as
  not-correct, its share is reported separately
- **recall** = correct / (fields present in the PDF), where "present" =
  non-null extractions (except those with error type *hallucinated span*)
  + null fields judged *present*
- **final labels**: majority vote on shared docs, single label on own docs;
  ties are excluded and listed until adjudicated

## Notes

- Fleiss' kappa is computed only on items judged by *all* annotators, per the design (80 shared docs).
- The analysis has zero dependencies beyond the Python standard library.
- `assignment.json`, the annotation guidelines, the exports, and `analyze.py`
  together form the "validation code and data" referenced in the paper's
  Code Availability section.
