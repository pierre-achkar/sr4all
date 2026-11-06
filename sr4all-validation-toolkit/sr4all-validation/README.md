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

Run from the repository root using Slurm and Apptainer. The analyzer supports
two or more available annotators; it reports the actual count and warns when
it differs from the paper's three-annotator protocol. It never creates missing
ratings. No third-party Python packages are required.

```bash
srun --ntasks=1 --cpus-per-task=1 --mem=1G --time=00:05:00 \
  apptainer exec --overlay .overlay --bind /storage/remote/DevStorage \
  /opt/apptainer/vllm-ubuntu2404-uv.sif \
  /usr/bin/python3 sr4all-validation-toolkit/sr4all-validation/scripts/analyze.py \
  --dir data/validation_sample/results \
  --tasks data/validation_sample/tasks_385.jsonl \
  --out data/validation_sample/results/three_annotators_2026-11-04
```

Check live capacity with `sinfo -a -N -o "%P %N %T %G %C"` before allocating.
This is CPU-only work: do not request GPUs or pass `--nv`. Explicitly request
one task so inherited Slurm settings cannot start concurrent report writers.
For a batch run, use the same tested runtime:

```bash
bash -n jobs/evaluate_validation.sh
sbatch --test-only jobs/evaluate_validation.sh
sbatch jobs/evaluate_validation.sh
```

Batch stdout and stderr go to `logs/jobs/validation_eval-<jobid>.out` and
`logs/jobs/validation_eval-<jobid>.err`. Use `squeue` and `sacct` to inspect
job state and exit code. Input paths are fixed in the job script; additional
analyzer arguments can be passed after the script name.

Reports default to `data/validation_sample/results/results/`; `--out` selects
another directory. Keep report output separate from all inputs.

- `summary.json`: actual annotator counts, sample coverage, missing document
  IDs, incomplete document IDs, unresolved counts, scoring rules, warnings,
  and SHA-256 hashes of all input files.
- `fleiss_kappa.json`: Fleiss' kappa and raw pairwise agreement on original
  independent shared-item labels, separately for extracted and null fields
  and for each field type, plus unanimous-item counts. Only items rated by every available annotator are
  included. Undefined kappa is JSON `null`, including single-category data.
- `agreement_per_field.tsv`, `agreement_table.tex`: pooled and per-field
  agreement, with extracted and null judgments kept separate. The TSV also
  reports unanimity counts. Pairwise agreement is not three-rater unanimity:
  a 2-1 split contributes 1/3 pairwise agreement and zero unanimity.
- `pairwise_agreement.json` and `.tsv`, `pairwise_table.tex`: unweighted
  Cohen's kappa, exact agreement, marginals and confusion counts on identical
  complete shared items. TSV contains scalar summaries; JSON retains all counts.
- `agreement_sensitivity.tsv`, `query_document_concentration.tsv`: descriptive
  post-hoc diagnostics. Scopes retain all items, remove queries, retain queries,
  collapse each rater's indexed queries to their worst document-field label,
  or omit the largest query-bearing document. Pairwise JSON/TSV also report
  document-balanced exact agreement: mean within-document agreement, with each
  contributing document weighted equally. These are different estimands, not
  grounds for selecting raters or changing the primary agreement protocol.
- `comment_inventory.json`: every distinct nonempty verbatim comment together
  with kind and field context, a stable 16-hex SHA-256-prefix key, and all original
  observations. Correct/absent ratings are included; empty comments and
  `found_text` evidence are not treated as comments. Hash collisions fail.
- With `--comment-codes`, `comment_analysis.json`, `comment_theme_counts.tsv`,
  `comment_themes_per_field.tsv`, and `comment_themes_table.tex`: exhaustive
  AI-assisted textual theme coding, codebook, verbatim comments, observations,
  and unique document-field counts by current resolution status. Themes overlap
  and describe comments, not confirmed causes or population error rates.
- `results_per_field.tsv` and `.json`: strict precision, conditional recall,
  both 95% Wilson intervals, denominators, correct/partial/incorrect counts,
  partial share, missed fields, null-absent fields, and excluded units.
- `results_table.tex`: a booktabs LaTeX table labeled with the estimate status
  and the conditional-recall assumption. It is not inserted into the paper.
- `disagreements.tsv`: all conflicting or missing item ratings, original
  comments/evidence, and their current resolution status.
- `adjudication_template.jsonl`: unresolved items ready for adjudication.
  Created only when absent; never overwritten on rerun.
- `discussion_report.md`, `discussion_documents.tsv`: pending items grouped
  by document, ordered by pending item count and then document ID. The report
  includes original extraction/evidence, source metadata, and all independent
  labels, comments, error tags, found text, and pages. Already-adjudicated items
  are excluded. The TSV indexes pending item and document-field counts.
- `final_labels.jsonl`: an audit trail of every item, original ratings,
  resolution source, and any separate adjudication decision.
- `error_analysis.tsv`: partial, incorrect, and missed judgments with original
  error types, comments, text, and pages. These are per-annotator observations,
  not deduplicated consensus error counts.
- `error_cases.jsonl`: deduplicated document-field cases, with original indexed
  items and all ratings. `resolved_error`, `unresolved_flag`, and
  `resolved_nonerror_flag` distinguish current decisions from allegations that
  remain unresolved or were overridden by adjudication. Null-present cases are
  missed fields, not incorrect extracted values.
- `error_counts.tsv`: unique document-field counts within each reported category,
  field, kind, and resolution status, plus flagged-rating counts. Category counts
  are non-exclusive; do not sum them to obtain a total error rate. Categories
  come from original annotator tags, not automated interpretation of comments.
- `sample_report.json`, `sample_composition.tsv`, `task_field_coverage.tsv`:
  planned and observed discipline coverage, unassessed field counts, and sampling
  metadata. The sibling `sampling_manifest.json` is loaded automatically when
  present; use `--manifest` to override it. Its IDs must match the master tasks.
  The original task file omits size-bin membership, so completion by size bin is
  not inferred by computing new quartiles on the completed subset.
- `annotator_metrics.tsv`: descriptive original ratings by annotator, shared or
  exclusive cohort, and field. Shared ratings appear separately here on purpose;
  these rows must not be pooled into the main document-level performance table.
- `sample_table.tex`, `error_table.tex`: provisional manuscript tables. The
  performance table also reports partial shares, missed fields, and pending
  document-fields. Boolean-query recall is suppressed when null-query judgments
  are unavailable. Requires `booktabs` and `graphicx`.

To refresh all six generated tables used by Technical Validation, explicitly
opt in when running the analyzer or the batch launcher:

```bash
sbatch jobs/evaluate_validation.sh \
  --comment-codes data/validation_sample/comment_analysis_2026-11-04 \
  --out data/validation_sample/results/full_analysis_2026-11-04 \
  --paper-tables paper/tables
```

This refreshes only `table-validation-results.tex`, `table-validation-errors.tex`,
`table-validation-sample.tex`, and `table-validation-agreement.tex`.
With the full analysis it also refreshes `table-validation-pairwise.tex` and
`table-validation-comment-themes.tex`. Comment-theme tables are copied only
when this run supplied and validated comment codes, never from a stale report.
It does not update the manuscript narrative; reconcile its snapshot counts after
new exports or adjudication. Use a new output directory for a new snapshot.
The earlier two-annotator reports under `results/results/` are retained unchanged.

### Standalone HTML dashboard

Generate a navigable offline view of the complete snapshot without recalculating
scores or modifying the reports:

```bash
srun --ntasks=1 --cpus-per-task=1 --mem=1G --time=00:05:00 \
  apptainer exec --overlay .overlay --bind /storage/remote/DevStorage \
  /opt/apptainer/vllm-ubuntu2404-uv.sif \
  /usr/bin/python3 sr4all-validation-toolkit/sr4all-validation/scripts/make_report.py \
  --dir data/validation_sample/results/full_analysis_2026-11-04 \
  --tasks data/validation_sample/tasks_385.jsonl
```

Open `dashboard.html` in that snapshot directory. No web server or internet
connection is required. The three main pages are **Summary**, **Extraction
results**, and **Comments & examples**. The summary describes the findings in
plain language. Compact results distinguish fully correct, partly correct,
incorrect, missing, and unsettled entries. Comment topics link directly to the
original comments. **More details** contains reviewer agreement, reviews needing
a decision, and the evaluation methods. Full statistics remain in expandable
sections; the underlying scores and judgments are unchanged.

The HTML embeds performance, original agreement,
pairwise sensitivities, all coded comments, error cases, the full label audit,
sample coverage, and input provenance. `--tasks` adds source titles and original
extraction/evidence to review details. Tables support search, filters, sorting,
and CSV export. Source-file downloads link to the original files alongside the
HTML; the embedded evaluation remains usable if the HTML is moved separately.
Query recall stays unavailable, and disputed labels are never changed.
Use `--out /path/to/report.html` to choose another HTML destination.

### Full comment and pairwise analysis (2026-11-04)

`data/validation_sample/results/full_analysis_2026-11-04/` preserves the primary
performance/three-rater results and adds the complete comment analysis and
pairwise sensitivity reports. Coding inputs are the four disjoint
`codes_*.jsonl` files under `data/validation_sample/comment_analysis_2026-11-04/`.
`--comment-codes` accepts that directory or a single JSONL file. Each row has
`comment_key` and a nonempty `themes` list using `COMMENT_THEMES` in the analyzer.
Every inventory key must occur exactly once; missing, unknown, duplicate or
invalid assignments fail before CLI report generation. All coding input files
are included in the provenance hashes. New/changed comments require reviewed
coding updates rather than silent automatic categorization.

The 1,234 comment-bearing ratings cover 990 document-fields in 286 documents;
there are 1,096 distinct text/context entries. Every entry was read and coded
with AI assistance. A purposive AI-assisted audit checked 40 contexts, followed
by correction of four semantic-grouping notes incorrectly assigned OCR/layout.
This is not independent human recoding, and no coding agreement or new PDF
verification is claimed. Missing comments do not establish absence of problems.

The largest shared query-bearing document contributes 126 of 145 query items.
Both item-weighted and alternative descriptive summaries are retained so one
review cannot silently determine the interpretation of annotation reliability.
Source-access/assessability notes cover 203 document-fields across 44 documents,
including 183 currently resolved null-absent fields. This is a broad textual
theme (main article/supplement availability, checklist-only sources, language,
readability), not a finding that all 44 PDFs are wrong. It exposes reference
judgment limitations requiring clarification, without altering original labels.

### Three-annotator snapshot (2026-11-04)

`data/validation_sample/results/three_annotators_2026-11-04/` contains the complete
385-document sample (80 shared, 305 exclusive; 545 annotation records), with no
missing documents or requested judgments. Original agreement is available now:
extracted items: 717, kappa 0.141, pairwise agreement 78.2%; null items: 228,
kappa 0.593, pairwise agreement 82.7%. Agreement is not extraction accuracy.

Performance and error counts remain provisional: 292 items need discussion
(233 extracted, 59 null), corresponding to 177 document-fields across 70 shared
documents. No majority labels or discussion decisions have been inserted.
Start with `discussion_report.md` and `discussion_documents.tsv`; preserve the
snapshot template and put decisions in a separate working copy. These are
decisions on existing annotations, not a new annotation round.

### Shared and exclusive documents

The 80 shared documents estimate annotation reliability; they are not the only
documents eligible for extraction evaluation. With all three submissions, the
performance sample remains 385 unique documents, not 545 annotation
records. Shared documents contribute a unanimous or discussion-adjudicated label
once per field; the other 305 contribute their assigned annotator's label.
The current exports match 102 exclusive documents in `annotator_1` (Pierre) and
101 in `annotator_3` (Mert), plus all 102 in `annotator_2` (Arno).

Use the shared cohort to inspect differences in how annotators apply the
guidelines. Different exclusive cohorts can differ in document difficulty, so
their raw scores do not isolate annotator strictness. Do not statistically
"correct" an annotator's labels or fabricate ratings for exclusive documents.
Keep the main performance estimates, independent agreement, and descriptive
annotator summaries separate.

The current error analysis uses the recorded category dropdowns and comments.
Examples in the manuscript are traceable by document ID in `error_cases.jsonl`
and the master tasks; they have not been independently rechecked against PDFs.
Do not turn an unresolved partial/incorrect flag into a confirmed pipeline error.

### Scoring and limitations

- **Precision** = correct / (correct + partial + incorrect). Partial is not a
  strict success; its share is reported separately.
- **Conditional recall** = correct / (correct + partial + incorrect + missed),
  where missed means a null field judged `present`. This assumes every
  non-null field is reported in the PDF. The current annotation schema does
  not independently establish field presence for incorrect extractions, so
  unconditional recall is not identifiable from these exports alone. A
  `hallucinated span` is not proof that the field itself is absent and does
  not remove it from the denominator. Confirm field presence separately
  before presenting this conditional estimate as recall in the paper.
  The current task builder omits null-query tasks entirely: the 385 tasks contain
  80 documents with extracted Boolean queries and 305 with no query assessment.
  The query `recall` value retained in TSV/JSON is only the conditional arithmetic,
  not an identified coverage estimate. The manuscript table reports it as
  unavailable. Assess those 305 query-absence cases separately before reporting
  query recall; do not reinterpret their missing ratings as `absent`.
- **Unit**: one document-field type, not one annotator vote. For indexed
  `exact_boolean_queries`, all queries must be correct for a strict success;
  otherwise the document's field takes the worst query judgment
  (`incorrect` before `partial`). One unresolved query excludes that entire
  document-field from precision/recall. Agreement still uses the original
  individual query items; query indices remain in the adjudication audit.
- **Final labels**: a single owner's completed judgment, unanimous shared
  judgments, or an explicit adjudication. Any disagreement needs adjudication,
  including a three-rater majority. Raw exports must remain unchanged so that
  kappa always measures independent, pre-discussion judgments.
- **Incomplete data**: non-`done` records do not contribute ratings; absent
  ratings are never interpreted as `incorrect` or `absent`. Supplying the
  master tasks checks field coverage even for records marked `done` and
  identifies unsubmitted documents. Duplicate exports and unexpected labels,
  fields, documents, or adjudication keys fail explicitly rather than being
  silently merged. Keep one export per annotator-document.
- **Publication status**: estimates remain `provisional` while sample documents,
  expected annotators, or ratings are missing, or disagreements remain. Metrics
  exclude unresolved units, which can bias a resolved-only estimate; these are
  not final population estimates. Wilson intervals follow the paper's plan and
  do not correct incomplete response or nonresponse bias. The conditional
  recall caveat also applies after adjudication.

### Adjudicate without changing independent ratings

Create a working copy of `adjudication_template.jsonl` named
`data/validation_sample/adjudicated.jsonl`, outside the report directory. After
discussion, set each `judgment` to `correct`, `partial`, or `incorrect` for
`kind: "field"`, or `absent` or `present` for `kind: "null"`. Add a comment
documenting the decision. For example:

```json
{"doc_id":"W123","kind":"field","field":"objective","judgment":"correct","comment":"Agreed after checking PDF page 3."}
```

Blank judgments are left unresolved. Rerun with the decisions:

```bash
sbatch jobs/evaluate_validation.sh \
  --adjudicated data/validation_sample/adjudicated.jsonl \
  --out data/validation_sample/results/adjudicated
```

Decisions change precision/recall only, never kappa. Existing templates are
preserved, including filled-in decisions; use `disagreements.tsv` for the
current pending list or use a new `--out` directory to generate a fresh template
after receiving further exports.

### Regression checks

```bash
srun --ntasks=1 --cpus-per-task=1 --mem=1G --time=00:05:00 \
  apptainer exec --overlay .overlay --bind /storage/remote/DevStorage \
  /opt/apptainer/vllm-ubuntu2404-uv.sif \
  /usr/bin/python3 -m unittest discover -s tests -p test_analyze_validation.py -v
```

## Notes

- Fleiss' kappa supports two raters but is not generally identical to Cohen's
  kappa. Extracted and null judgments have different category spaces, so they
  are reported separately instead of inflating an overall agreement score by
  pooling those spaces.
- The analysis has zero dependencies beyond the Python standard library.
- `assignment.json`, the annotation guidelines, the exports, and `analyze.py`
  together form the "validation code and data" referenced in the paper's
  Code Availability section.
