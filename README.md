# Webis-SR4ALL-26

## Project for the Paper

**A Large-Scale, Cross-Disciplinary Corpus of Systematic Reviews**  
Pierre Achkar (Leipzig University; Fraunhofer ISI)  
Tim Gollub (Bauhaus-Universität Weimar)  
Arno Simons (TU Berlin)  
Harrisen Scells (University of Tübingen)  
Martin Potthast (Kassel University; hessian.AI; ScaDS.AI)

---

**License:** MIT

The final dataset is released on Zenodo under: https://doi.org/10.5281/zenodo.18431942

This repository reconstructs the Webis-SR4All dataset from OpenAlex metadata, PDFs, and structured extraction.

## Repository Layout

- `src/`: code for building, validating, and analyzing the corpus.
- `src/validation/`: Python code for manual validation packaging, analysis, and
  reports, with no nested directories or other file types.
- `docs/validation/`: [validation documentation](docs/validation/README.md) and
  annotator guidelines.
- `tests/`: automated tests.
- `tests/fixtures/validation/`: synthetic validation demo inputs and browser example.
- `paper/`: LaTeX manuscript, bibliography, and figures.
- `jobs/`: Slurm entry points for pipeline stages.
- `data/`: current pipeline inputs, intermediate artifacts, and outputs.
- `data/validation/annotation/`: initial annotation tasks, sampling manifest,
  original packages, and independent exports.
- `data/validation/adjudication/`: clarification preparation, distributed packages,
  previous package versions, and returned clarification exports.
- `data/validation/analysis/`: report snapshots and comment-coding inputs.
- `data/validation/demos/`: the five-document demo and its package.
- `data/validation/templates/`: the three version-controlled HTML UI templates;
  other validation data and generated outputs remain ignored by Git.

- `logs/`: current pipeline and job logs.
- `archive/data_old/`: historical datasets, preserved with their original internal layout.
- `archive/logs_old/`: historical logs, preserved with their original internal layout.
- `archive/validation_toolkit/`: preserved legacy toolkit data; all files matched
  their counterparts now in `data/validation/` when the toolkit was integrated.

The distributed clarification set is
`data/validation/adjudication/packages/2026-11-23_proposals_first/`.
Earlier versions remain under `adjudication/previous_packages/`; original
annotation exports and clarification decisions are kept separate.

The archive is local and ignored by Git. Historical artifacts are moved, not
deleted or regenerated. Scripts that explicitly use historical inputs reference
`archive/data_old/`; current pipeline data and logs remain under `data/` and
`logs/`. Run scripts from the repository root.

## Pipeline

The current corpus-building stages are:

1. [Retrieval](#1-retrieval): OpenAlex studies and benchmark DOI enrichment.
2. [OCR](#2-ocr): PDFs to Markdown.
3. [Extraction v2](#3-extraction-v2): extraction, alignment, fact-checking,
   repair, and a second alignment/fact-checking pass.
4. [Final dataset](#4-final-dataset-build): concatenate, join, normalize,
   finalize, and report completeness.

The legacy extraction implementation, benchmark utilities, search-strategy
splitter, and query experiments are described separately below. They are not
additional mandatory steps in this sequence.

**This is a stage map, not a one-command reconstruction recipe.** Inputs,
environment settings, and output ownership must be checked before running each
stage. The remaining handoff and rerun limitations are listed below.

## Before You Run

- Run from the repository root.
- Execute Python only inside a Slurm-managed Apptainer environment, including
  tests and dataset inspection. Check capacity with
  `sinfo -a -N -o "%P %N %T %G %C"` before allocating resources.
- Check [requirements.txt](requirements.txt) in the selected container; install
  dependencies there, not on the login node. GPU jobs also require their
  stage-specific model/runtime setup.
- Use CPU-only allocations for tests and CPU stages. Request GPUs and use `--nv`
  only when required. Keep the image, overlay, mounts, and model settings
  consistent with the relevant launcher.
- `.overlay/` is used for the CPU tests below. The local v2 launchers use
  `.overlay-vllm/`; fact-checking also uses `.factcheck_venv/`. These are local
  runtime resources, not interchangeable or supplied by a fresh Git checkout.
- Inspect each script's `CONFIG`, constants, and environment overrides. Some v2
  defaults contain the original absolute repository path; set the per-stage
  `SR4ALL_*` paths for another checkout.
- Load credentials from `.env` as required by the stage; do not put them in
  commands or source files.
- Preserve existing artifacts. Several stages overwrite outputs and logs, so
  choose fresh output locations before a new build.

CPU-only test command for the current cluster environment:

```bash
srun --partition=a100 --ntasks=1 --cpus-per-task=1 --mem=2G --time=00:05:00 \
  --immediate=30 \
  apptainer exec --fakeroot --overlay .overlay/ --bind /storage/remote/DevStorage \
  --env PYTHONDONTWRITEBYTECODE=1 \
  /opt/apptainer/images/base/ubuntu-noble/20250415-py3.12-cuda12.8/ubuntu-noble-20250415-py3.12-cuda12.8-v1.0.sif \
  python3 -m pytest -q tests
```

The Python commands below are entry points to run **inside the appropriate
allocated container**, not directly on the login node.

## 1) Retrieval

Goal: build the core OpenAlex SR corpus and full-text subset.

The table gives the current default paths; it does not execute the stages.
Steps 1-3 prepare inputs for step 4. Steps 5 and 6 both consume step 4's output.

| Entry point | Input | Primary output |
| --- | --- | --- |
| [1_oax_fetch_studies.py](src/retrieval/1_oax_fetch_studies.py) | OpenAlex API | `data/retrieval/oax_sr_full.json`, with checkpoint/shard files |
| [2_oax_filter.py](src/retrieval/2_oax_filter.py) | `data/retrieval/oax_sr_full.json` | `data/retrieval/filtered/oax_sr_title_english_refs.jsonl` |
| [3_oax_fetch_benchmark_studies.py](src/retrieval/3_oax_fetch_benchmark_studies.py) | `data/retrieval/benchmark_data/raw.parquet` (`doi` column), OpenAlex API | `data/retrieval/benchmark_data/oax_benchmark_data.jsonl` |
| [4_join_studies.py](src/retrieval/4_join_studies.py) | Filtered studies and benchmark OpenAlex records | `data/retrieval/merged/oax_merged_dedup.jsonl` |
| [5_download_pdfs.py](src/retrieval/5_download_pdfs.py) | Merged OpenAlex records | `data/retrieval/merged/pdfs/`; manifest at `logs/retrieval/5_pdf_download_manifest.jsonl` |
| [6_oax_slim.py](src/retrieval/6_oax_slim.py) | Merged OpenAlex records | `data/retrieval/merged/oax_slim.jsonl` |

The benchmark Parquet file is an external input, not generated by these six
scripts. Keep it and the retrieval checkpoints/manifests for provenance.
[7_pdf_inventory.py](src/retrieval/7_pdf_inventory.py) compares the slim metadata
with downloaded PDFs. Metadata availability alone does not establish that a PDF
or Markdown file exists.

## 2) OCR

The client is [run_vl_ocr_with_vllm.py](src/ocr/run_vl_ocr_with_vllm.py).
It expects a PaddleOCR-VL service at `http://127.0.0.1:8118/v1`.
[docker-compose.yml](src/ocr/docker-compose.yml) describes that GPU service, but
is not a Slurm launcher. Provision the service in the allocated GPU environment
before running the client there.

```bash
python3 src/ocr/run_vl_ocr_with_vllm.py \
  --input-dir data/retrieval/merged/pdfs \
  --output-dir data/retrieval/merged/mds
```

The explicit output directory matters: the client's default is `./output_md`,
whereas extraction v2 expects `data/retrieval/merged/mds`.
Preserve document identity in Markdown names and relative paths.

## 3) Extraction v2

[src/extraction_v2/](src/extraction_v2/) is the current methodology workflow.
[src/extraction/](src/extraction/) is the retained legacy implementation, not a
second extraction step.

Default directories below are relative to the configured repository root.

| Entry point | Input directory | Output directory |
| --- | --- | --- |
| [1_extraction.py](src/extraction_v2/1_extraction.py) | `data/retrieval/merged/mds` | `data/extraction/first_pass` |
| [2_alignment.py](src/extraction_v2/2_alignment.py) | `data/extraction/first_pass` | `data/extraction/aligned` |
| [3_fact_checking.py](src/extraction_v2/3_fact_checking.py) | `data/extraction/aligned` | `data/extraction/fact_checked` |
| [4_repair.py](src/extraction_v2/4_repair.py) | `data/extraction/fact_checked` | `data/extraction/repaired` |

Extraction discovers Markdown recursively and writes mirrored per-document JSON.
Its `doc_id` is the Markdown path relative to the input directory, without the
extension. Final merging matches the last component of this ID to the last
component of the OpenAlex `id`; check that naming contract before extraction.
Keep record IDs, evidence, and shard ownership stable across passes.

Repair is not the final verification step. Re-run alignment with
`SR4ALL_EXTRACTION_DIR` pointing to `data/extraction/repaired` and
`SR4ALL_ALIGNED_DIR` pointing to `data/extraction/repaired_aligned`. Then re-run
fact-checking with that aligned input and `SR4ALL_FACT_CHECKED_DIR` pointing to
`data/extraction/repaired_fact_checked`. Use absolute environment paths for a
different checkout. Final concatenation consumes this post-repair checked tree.

### Production launchers

These four reviewed v2 launchers are version-controlled. Their existing names,
resource requests, environments, and pass defaults are preserved. Other local
job scripts remain ignored unless explicitly reviewed and added.

| Launcher | Stage and current configuration |
| --- | --- |
| [extarction_v2.sh](jobs/extarction_v2.sh) | First-pass extraction; two disjoint workers, each using two GPUs |
| [alignmen_v2.sh](jobs/alignmen_v2.sh) | CPU alignment; currently hard-codes the **post-repair** input/output pair |
| [fact_checker_v2.sh](jobs/fact_checker_v2.sh) | GPU fact-checking via `.factcheck_venv/`; defaults to the **post-repair** pair, with environment overrides |
| [repair_v2.sh](jobs/repair_v2.sh) | GPU repair; defaults to `fact_checked` to `repaired` |

Do not submit these four files blindly in filename order: the alignment and
fact-checking launchers need first-pass paths when used before repair.
The alignment script currently assigns its paths unconditionally: environment
overrides alone do not select a first-pass alignment run.
[evaluate_validation.sh](jobs/evaluate_validation.sh) is the separately tracked
validation-report launcher.

From the repository root, create the Slurm log directory before submission;
creating it inside a running job is too late for Slurm's initial log setup.
Check each selected launcher independently, for example:

```bash
mkdir -p logs/jobs/extraction
bash -n jobs/extarction_v2.sh
sbatch --test-only jobs/extarction_v2.sh
```

`--test-only` checks scheduler acceptance without submitting a job. It does not
validate container dependencies, model caches, mounts, or dataset contents.
A fresh checkout includes the launchers, **not** their images, overlays,
fact-checking environment, cached weights, or inputs. Verify those local
resources and run a representative compute-node probe in the same environment
before a production submission.

## 4) Final Dataset Build

The current core build uses [src/final_ds/](src/final_ds/). Its order is
**0, 01, 2, 4**, followed by the completeness report; script 3 is a separate
legacy-path branch.

```bash
python3 src/final_ds/0_concat_extraction_results.py
python3 src/final_ds/01_merge_extraction_into_raw.py
python3 src/final_ds/2_normalize_year_range.py
python3 src/final_ds/4_finalize_year_range.py
```

Review the configured destinations before running: these stages open their
outputs and logs in overwrite mode. The normalizer's `overwrite=False` controls
replacement of the **year field**, not protection of an existing output file.

| Entry point | Default input | Default output |
| --- | --- | --- |
| `0_concat_extraction_results.py` | Recursive `*.json` under `data/extraction/repaired_fact_checked/` | `data/extraction/repaired_fact_checked/repaired_fact_checked_merged.jsonl` |
| `01_merge_extraction_into_raw.py` | Concatenated extraction plus `data/retrieval/merged/oax_slim.jsonl` | `data/final_ds/oax_slim_with_extraction.jsonl`, plus a five-matched-record sample |
| `2_normalize_year_range.py` | Joined dataset | `data/final_ds/oax_slim_with_extraction_normalized_year_range.jsonl` |
| `4_finalize_year_range.py` | Year-normalized dataset | `data/final_ds/sr4all_final.jsonl` |

Merge behavior:
- Extraction values are flattened into OpenAlex records; evidence and
  verification metadata are removed from the flattened fields. Retain the
  per-document source artifacts.
- Duplicate normalized extraction IDs use the last record encountered. Review
  duplicate, matched, and unmatched counts in
  `logs/final_ds/merge_extraction_into_raw.log`.
- OpenAlex records without matching extraction remain in the output.

Year normalization is heuristic and uses the bibliographic `year`; a missing
range can fall back to that year. Finalization promotes `year_range_normalized`
to `year_range` and removes the original range and normalization-rule fields.
Keep the intermediate dataset to retain that provenance.

### Completeness report

After finalization, run:

```bash
python3 src/final_ds/check_completeness.py
```

It reads `data/final_ds/sr4all_final.jsonl` and appends to
`logs/final_ds/completeness_check_all.log`. Missing input or malformed records
produce a nonzero exit status; malformed records include their source line
numbers. Importing the checker does not create or truncate logs.
Field presence treats empty wrappers, whitespace, and ghost queries as absent,
but preserves observed `false` and `0` values. Nested and flattened extraction
records use the same coverage rules. All-null percentages use valid records;
essentials use records with methodology beyond `year_range`. The search-strategy
splitter reuses this presence rule.

This is a report, not a completeness filter. Exit status zero means the checker
accepted the input records, **not** that every record contains all methodology
fields or that a published release has been reproduced.

## Optional and Legacy Branches

These tools remain available, but their default paths are not wired to the
current core build. Review their contracts before use; do not rename or copy
current data into legacy locations just to satisfy defaults.

### Search-strategy split

[3_split_search_strategy.py](src/final_ds/3_split_search_strategy.py) still reads
`data/final/sr4all_full_normalized_year_range.jsonl`, not
`data/final_ds/sr4all_final.jsonl`. Its output directory is `data/final/`, with:

- `sr4all_full_normalized_year_range_search_has_boolean.jsonl`
- `sr4all_full_normalized_year_range_search_keywords_only.jsonl`

The Boolean cohort includes records with a meaningful exact query, even if they
also have keywords. The keywords-only cohort excludes those records; records
with neither are not written to either cohort. Configure an explicit input and
fresh outputs before applying this branch to the current dataset.

### Query normalization and OpenAlex evaluation

[src/norm_queries/](src/norm_queries/) retains Boolean mapping, mapping repair,
year-range merging, and OpenAlex-query transformation tools:

- [transform_to_boolean.py](src/norm_queries/transform_to_boolean.py)
- [postprocess_boolean_mapping.py](src/norm_queries/postprocess_boolean_mapping.py)
- [merge_repaired_boolean_mapping.py](src/norm_queries/merge_repaired_boolean_mapping.py)
- [merge_year_range_normalized.py](src/norm_queries/merge_year_range_normalized.py)
- [transform_to_oax.py](src/norm_queries/transform_to_oax.py)
- [merge_oax_queries.py](src/norm_queries/merge_oax_queries.py)

Their configurations include repair subsets and `data/final/` artifacts, not
just whole-corpus inputs. For example, the repaired mapping merger writes under
`with_boolean/merged/`, while the year-range merger reads under
`with_boolean/final/`. These are not an automatic chained workflow.

[src/oax/](src/oax/) contains query repair, sanitization, count annotation,
count-bucket splitting, ID retrieval, result flattening, and evaluation.
Several defaults still reference `data/final_old/`, while others use
`data/final/with_oax/`. Reconcile these inputs, outputs, and selected cohorts
before running experiments; they are not required to produce the current
`data/final_ds/sr4all_final.jsonl`.

### Earlier benchmark utilities

[src/add_rw_data/](src/add_rw_data/) retains the earlier benchmark
preparation/acquisition scripts. In particular,
[2_check_availability_in_sr4all.py](src/add_rw_data/2_check_availability_in_sr4all.py)
expects `data/final/sr4all_merged.jsonl`, which the current core build does not
produce. The current retrieval branch instead consumes benchmark DOIs through
`src/retrieval/3_oax_fetch_benchmark_studies.py`.

## Build Review Checklist

- Confirm benchmark input provenance, PDF availability, and the OCR service
  setup before extraction.
- Confirm Markdown-derived IDs match the OpenAlex IDs used by the final join.
- Keep all extraction, alignment, fact-checking, and repair passes separate;
  final concatenation must select the intended post-repair checked tree.
- Review merge duplicate/matched/unmatched counts and the five-record sample.
- Retain both joined and normalized intermediate datasets before finalization.
- Review `logs/final_ds/completeness_check_all.log`; inspect coverage and
  malformed-record diagnostics, not only the process exit code.
- Reconcile record counts across stages. An output file's existence alone does
  not establish completeness, successful processing, or release equivalence.

## Remaining Stabilization Work

- Make launcher pass selection and local runtime paths explicit and portable;
  review other local jobs before versioning them.
- Reconcile the legacy splitter/query paths with an explicitly chosen current
  dataset contract.
- Add focused synthetic regression tests for merging, year normalization, and
  finalization, including duplicate IDs and record preservation.
- Harden remaining stages' import-time side effects, output/log overwrites,
  malformed-record handling, and missing-input exit statuses. The completeness
  checker's failure/logging guarantees do not yet apply to every other script.
