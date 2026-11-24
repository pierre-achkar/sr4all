# SR4ALL Extraction Validation Toolkit

Everything needed to run the manual expert validation of the LLM-extracted
methodological fields: a browser-based annotation tool, a packaging script,
and the analysis script that produces the numbers for the Technical Validation
section (Fleiss' kappa, per-field precision/recall with Wilson 95% CIs, LaTeX table).

No server, no installation for annotators: each annotator gets a zip,
unzips it, opens `index.html` in Chrome/Edge/Firefox, selects the folder, done.

```
src/validation/
├── make_packages.py        build per-annotator zips from the master task file
├── analyze.py              merge exports -> kappa, precision/recall, LaTeX table
├── make_adjudication_packages.py  prepare, build, and merge human reviews
├── make_report.py          render an existing analysis snapshot as HTML
└── make_example.py         regenerate the synthetic tasks and PDFs
```

Python source files are flat under `src/validation/`. The version-controlled HTML
templates live under `data/validation/templates/`:

- `index.html`: the annotation tool.
- `adjudicate.html`: the offline adjudication template.
- `evaluation_report.html`: the report dashboard template.

Only these three templates are exempted from the data tree's Git-ignore rules;
generated packages, reports, and annotation exports stay ignored.
This README and the
[annotator guidelines](ANNOTATOR_GUIDELINES.md) live under `docs/validation/`;
the package builder includes those guidelines in each annotator package.

Run all commands from the repository root in the Slurm/Apptainer environment
described in section 4. The ready-to-try synthetic package is in
[`tests/fixtures/validation/`](../../tests/fixtures/validation/), containing
`index.html`, `tasks.jsonl`, and three dummy PDFs under `pdfs/`.
Current tasks, annotation exports, generated packages, and reports belong in
`data/validation/`, not under the source tree.

The former `sr4all-validation-toolkit/sr4all-validation/validation_sample/`
was compared file-by-file with the current data: all 1,988 files were
byte-identical at their matching relative paths. Its complete original directory
is preserved locally under `archive/validation_toolkit/validation_sample/`.
Neither copy was merged, overwritten, or regenerated. Existing package IDs,
exports, hashes, and historical provenance paths remain unchanged.

The current data directory was subsequently renamed from `data/validation_sample/`
to `data/validation/` without changing its contents. Historical provenance inside
existing artifacts retains the paths recorded when those artifacts were built;
new commands use `data/validation/`.

## Validation rounds and artifact layout

```text
data/validation/
├── annotation/
│   ├── tasks_385.jsonl
│   ├── sampling_manifest.json
│   ├── packages/                 original annotation packages and assignment
│   └── exports/                  independent annotations, plus humanized copies
├── adjudication/
│   ├── preparation/2026-11-19/    source pages and provisional assessments
│   ├── packages/2026-11-23_proposals_first/  distributed clarification packages
│   ├── previous_packages/        three older builds, retained unchanged
│   └── exports/                  reserved for returned clarification reviews
├── analysis/
│   ├── reports/                  original report snapshots, with names retained
│   └── comment_codes/2026-11-04/  coding inputs for comment analysis
├── demos/                        five-document task file and demo package
└── templates/                    version-controlled HTML templates
```

The initial round has three returned annotation exports. Its original assignment
also includes `package_martin`; that package is preserved alongside
`package_annotator_1`, `package_annotator_2`, and `package_annotator_3`, not
silently deleted or counted as a fourth returned annotation.

The distributed clarification set is the November 23 proposals-first build:
149 cases for Arno, 58 for Mert, and 85 for Pierre. Its earlier document-based
and differing-annotator builds remain in `adjudication/previous_packages/`.
Keep returned clarification reviews in `adjudication/exports/`, never among
the independent annotation exports. No returned clarification decisions were
created during this reorganization.

These are location-only moves: package contents and IDs, original ratings,
hashes, and historical provenance are unchanged. Report snapshots retain their
internal files and sibling relationships. Always choose fresh output directories
for new package builds and report snapshots.

## Offline adjudication packages

The separate `data/validation/templates/adjudicate.html` reviewer handles disagreements after the
independent annotation round. Each generated package embeds its cards and source
page text, includes the PDFs, and works offline by opening `index.html`. It uses
the browser's PDF viewer rather than the original annotation tool's CDN viewer.
An **Open PDF** link is available if embedded PDF viewing is unavailable.

The reviewer shows one disagreement at a time beside its source PDF. Use
**Previous/Next** to navigate without finalizing drafts, **Browse cases** to
filter or switch documents, and **Save & next** to record an explicit human
decision. All three annotators' comments remain visible. AI suggestions are
separate provisional drafts, never automatically accepted. Source text and
secondary evidence expand on demand; **More** contains rules, import, and reset.
On narrow screens the panes stack, with links between the case and PDF.

The queue puts all existing proposed-resolution cases first across the entire
package, then undecided cases; document and numeric field order are retained
within each group. Each card starts with its assessment, displayed as
**GPT6.1 Medium — proposed resolution** or **GPT6.1 Medium — undecided**.
This is the requested presentation label, not a change to recorded model
provenance, AI assessments, or human decisions. Internal `ai` keys and exported
review identities remain unchanged. Run the dependency-free UI regression tests
with `node --test tests/test_adjudication_ui.cjs`.

The generator recomputes unresolved items from the original exports, including
two-versus-one splits and missing ratings. It preserves `(doc_id, kind, field)`
and indexed query identities. All annotators' comments are displayed, but only
Pierre's humanized comment version is included in the cards. The generator
rejects a humanized export that changes non-comment data. Original exports
remain untouched.

Run Python commands below inside the Slurm/Apptainer environment described in
section 4. Source preparation uses the repository-pinned `pypdf` dependency;
building, merging, and the focused unit tests use the standard library.

```bash
SCRIPT=src/validation/make_adjudication_packages.py

python3 "$SCRIPT" prepare \
  --tasks data/validation/annotation/tasks_385.jsonl \
  --exports data/validation/annotation/exports \
  --pierre-humanized data/validation/annotation/exports/humanized/annotations_final_pierre.jsonl \
  --out /path/to/new/sources.jsonl

python3 "$SCRIPT" build \
  --tasks data/validation/annotation/tasks_385.jsonl \
  --exports data/validation/annotation/exports \
  --pierre-humanized data/validation/annotation/exports/humanized/annotations_final_pierre.jsonl \
  --sources /path/to/new/sources.jsonl \
  --proposals /path/to/ai_assessments.jsonl \
  --out /path/to/new/packages
```

AI assessments use `doc_id`, `kind`, `field`, `status` (`proposed` or
`undecided`), `judgment`, `categories`, `rationale`, and `evidence`. Categories
come from the existing comment codebook and are provisional grouping tags,
not confirmed causes. A proposal requires a valid judgment and at least one
`{"page": 2, "quote": "Literal source text"}` passage. Page numbers are one-based
PDF pages. The generator verifies each quote against the prepared page text,
allowing whitespace differences only, and checks PDF hashes. This verifies
quotation provenance, not whether the suggested judgment is substantively right.
Unspecified cases remain explicitly AI-undecided; they are not resolved by
majority vote. Undecided assessments must not carry a judgment.

Packages for `arno`, `mert_okur`, and `pierre` always contain disjoint cases.
The default `--assignment-mode documents` keeps whole documents together,
balancing the reproducible heuristic `disputed items + 2 * PDF pages`.
Case counts need not be equal. Every indexed query still has its own card.

Use `--assignment-mode differing-annotator` on `build` for targeted
reconsideration: a two-versus-one disagreement goes to the annotator whose
original label differs. Three-way cases, sorted by stable case ID, rotate
through `arno`, `mert_okur`, and `pierre`; three such cases give one to each.
This mode requires complete valid ratings from all three annotators and rejects
incomplete or non-disagreement cases explicitly. Different fields in the same
document may go to different people, so PDFs can appear in multiple packages,
but each case has exactly one owner. Routing is not a judgment: the differing
annotator may retain their original label or flag discussion, and the majority
is not assumed correct. This is targeted reconsideration, not a new independent
annotation round.

`assignment.json` records the assignment mode, document and case memberships,
each case's routing reason, workload, input/code hashes, and ZIP hashes.
Existing output directories are rejected rather than overwritten. A reassignment
requires new package identities; earlier exports cannot be imported directly
into packages with different ownership. Preserve those exports for provenance.

For a UI-only rebuild, add `--previous-packages /path/to/previous/packages` to
`build`, keep the same assignment mode, and choose a fresh `--out`.
This retains package identities only after
checking that every assigned case, AI assessment, source page, and PDF hash is
unchanged. Existing exports remain importable; the build records the previous
assignment's hash as provenance. Without this option, builds use new identities.
Browser autosave depends on the browser and origin: export before moving folders
or changing browsers, then import into the refreshed package.

Annotators must explicitly save every human decision, including confirmations
of AI proposals. A rationale, source-check confirmation, and valid PDF page are
required; missed fields also require a quotation. Drafts and discussion flags
have no final judgment. Autosave is browser-local, package-specific, and reports
storage failures. Export regularly; imports validate identities, reviewer,
statuses, labels, and full package coverage before replacing local work.

After collecting each annotator's latest complete JSONL export:

```bash
python3 "$SCRIPT" merge \
  --packages /path/to/packages \
  --reviews /path/to/arno.jsonl /path/to/mert.jsonl /path/to/pierre.jsonl \
  --out /path/to/new/merged
```

The merge rejects duplicate/unknown cases, mixed packages, and invalid human
decisions. It writes `human_resolutions.jsonl`, `meeting_queue.jsonl`,
`review_audit.jsonl`, and a summary. Missing, unreviewed, and discussion cases
remain in the meeting queue. These are **assigned human reviews, not
three-person consensus**. After team approval, the resolution file can be
supplied to the existing analyzer's `--adjudicated` option. AI assessments alone
are never passed to that option, and original agreement statistics are unchanged.

## 0. Try it (2 minutes)

Open `tests/fixtures/validation/index.html` in a browser, enter any annotator id,
click "Select folder…" and pick `tests/fixtures/validation/`, then "Start annotating".
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
python3 src/validation/make_packages.py \
  --tasks data/validation/annotation/tasks_385.jsonl \
  --pdf-dir /path/to/pdfs \
  --tool data/validation/templates/index.html \
  --annotators pierre,tim,arno \
  --shared 80 --seed 42 \
  --out data/validation/annotation/packages/
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
  /usr/bin/python3 src/validation/analyze.py \
  --dir data/validation/annotation/exports \
  --tasks data/validation/annotation/tasks_385.jsonl \
  --out data/validation/analysis/reports/three_annotators_2026-11-04
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

For the repository's original exports, reports default to
`data/validation/analysis/reports/results/`; `--out` selects another directory.
For a custom `--dir`, the existing `<dir>/results/` default is preserved.
Keep report output separate from all inputs.

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
  --comment-codes data/validation/analysis/comment_codes/2026-11-04 \
  --out data/validation/analysis/reports/full_analysis_2026-11-04 \
  --paper-tables paper/tables
```

This refreshes only `table-validation-results.tex`, `table-validation-errors.tex`,
`table-validation-sample.tex`, and `table-validation-agreement.tex`.
With the full analysis it also refreshes `table-validation-pairwise.tex` and
`table-validation-comment-themes.tex`. Comment-theme tables are copied only
when this run supplied and validated comment codes, never from a stale report.
It does not update the manuscript narrative; reconcile its snapshot counts after
new exports or adjudication. Use a new output directory for a new snapshot.
The earlier two-annotator reports under `analysis/reports/results/` are retained unchanged.

### Standalone HTML dashboard

Generate a navigable offline view of the complete snapshot without recalculating
scores or modifying the reports:

```bash
srun --ntasks=1 --cpus-per-task=1 --mem=1G --time=00:05:00 \
  apptainer exec --overlay .overlay --bind /storage/remote/DevStorage \
  /opt/apptainer/vllm-ubuntu2404-uv.sif \
  /usr/bin/python3 src/validation/make_report.py \
  --dir data/validation/analysis/reports/full_analysis_2026-11-04 \
  --tasks data/validation/annotation/tasks_385.jsonl
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

`data/validation/analysis/reports/full_analysis_2026-11-04/` preserves the primary
performance/three-rater results and adds the complete comment analysis and
pairwise sensitivity reports. Coding inputs are the four disjoint
`codes_*.jsonl` files under `data/validation/analysis/comment_codes/2026-11-04/`.
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

`data/validation/analysis/reports/three_annotators_2026-11-04/` contains the complete
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
`data/validation/adjudication/adjudicated.jsonl`, outside the report directory. After
discussion, set each `judgment` to `correct`, `partial`, or `incorrect` for
`kind: "field"`, or `absent` or `present` for `kind: "null"`. Add a comment
documenting the decision. For example:

```json
{"doc_id":"W123","kind":"field","field":"objective","judgment":"correct","comment":"Agreed after checking PDF page 3."}
```

Blank judgments are left unresolved. Rerun with the decisions:

```bash
sbatch jobs/evaluate_validation.sh \
  --adjudicated data/validation/adjudication/adjudicated.jsonl \
  --out data/validation/analysis/reports/adjudicated
```

Decisions change precision/recall only, never kappa. Existing templates are
preserved, including filled-in decisions; use `disagreements.tsv` for the
current pending list or use a new `--out` directory to generate a fresh template
after receiving further exports.

### Regression checks

The Python checks use the repository's base Python image and writable overlay:

```bash
srun --ntasks=1 --cpus-per-task=1 --mem=1G --time=00:05:00 \
  apptainer exec --fakeroot --overlay .overlay --bind /storage/remote/DevStorage \
  /opt/apptainer/images/base/ubuntu-noble/20250415-py3.12-cuda12.8/ubuntu-noble-20250415-py3.12-cuda12.8-v1.0.sif \
  python3 -m pytest tests/test_analyze_validation.py \
  tests/test_adjudication_packages.py tests/test_sample_validation.py
```

The test runner is pinned in the root `requirements.txt`. The dependency-free
UI checks use `node --test tests/test_adjudication_ui.cjs` in a Slurm/Apptainer
environment with Node.js available. Neither test command rebuilds existing
annotation packages or modifies real exports.

## Notes

- Fleiss' kappa supports two raters but is not generally identical to Cohen's
  kappa. Extracted and null judgments have different category spaces, so they
  are reported separately instead of inflating an overall agreement score by
  pooling those spaces.
- The analysis has zero dependencies beyond the Python standard library.
- `assignment.json`, the annotation guidelines, the exports, and `analyze.py`
  together form the "validation code and data" referenced in the paper's
  Code Availability section.
