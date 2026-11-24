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

Run the pipeline in the following order:
1. `retrieval`
2. `ocr` (PaddleOCR)
3. `add_rw_data`
4. `extraction`
5. `final_ds`
6. `norm_queries`
7. `oax`

## Before You Run

- Run from repo root.
- Install dependencies from `requirements.txt`.
- Most scripts use hard-coded paths in `CONFIG` objects or module constants. Update these paths first.
- Some scripts assume absolute-style paths such as `/data/...` and `/logs/...`; adapt them to your environment.

## 1) Retrieval (`src/retrieval`)

Goal: build the core OpenAlex SR corpus and full-text subset.

Run in order:

```bash
python src/retrieval/1_oax_fetch_studies.py
python src/retrieval/2_oax_filter_refs_title_doi_pdf.py
python src/retrieval/3_pdf_download.py
python src/retrieval/4_oax_filter_downloaded.py
python src/retrieval/5_oax_slim_ft.py
```

Optional branches:

```bash
python src/retrieval/3_oax_filter_refs_title_doi_no_pdf.py
python src/retrieval/6_oax_slim_no_ft.py
python src/retrieval/07_abstarct_coverage.py
python src/retrieval/8_oax_filter_refs_title_doi.py
```

Primary outputs:
- `data/raw/oax_sr_full.json`
- `data/filtered/ft_subset/pdfs/...`
- `data/filtered/ft_subset/oax_sr_slim.json`

## 2) OCR (`src/ocr`)

Goal: convert retrieved PDFs to Markdown using PaddleOCR for downstream extraction.

Notes:
- This stage runs right after retrieval.
- Use PaddleOCR-based OCR processing over PDFs in `data/filtered/ft_subset/pdfs/...`.

## 3) Add RW Data (`src/add_rw_data`)

Goal: add external benchmark datasets and identify what is missing.

Run in order:

```bash
python src/add_rw_data/1_creat_ref_table.py
python src/add_rw_data/2_check_availability_in_sr4all.py
python src/add_rw_data/3_fetch_oax_doi.py
python src/add_rw_data/4_split_on_ft.py
python src/add_rw_data/5_pdf_download.py
python src/add_rw_data/6_slim_version.py
```

Notes:
- Step 2 expects `data/final/sr4all_merged.jsonl`.
- `sr4all_merged.jsonl` is typically produced after final split merge (`src/final_ds/merge_splits.py`) if you maintain full-text/no-full-text splits.

## 4) Extraction (`src/extraction`)

Goal: extract structured methodological fields from document text.

Processing flow:

```bash
python src/extraction/1_extraction.py
python src/extraction/2_alignment.py
python src/extraction/3_fact_checking.py
python src/extraction/4_repair.py
```

Then re-run alignment/fact-checking on repaired outputs as configured.

Important:
- Inputs are manifest/text-path driven (see `1_extraction.py` config).
- Ensure `doc_id` in extraction aligns with OpenAlex IDs used in retrieval outputs.

## 5) Final Dataset Build (`src/final_ds`)

Goal: flatten, validate, join with OpenAlex metadata, normalize year range, and split by search strategy.

Run in order:

```bash
python src/final_ds/concat_jsonl.py
python src/final_ds/1_intermediate_ds.py
python src/final_ds/2_repair_final_fields.py
python src/final_ds/3_join_final_ds.py
python src/final_ds/4_normalize_year_range.py
python src/final_ds/5_split_search_strategy.py
```

Key outputs:
- `data/final/sr4all_full.jsonl`
- `data/final/sr4all_full_normalized_year_range.jsonl`
- `data/final/sr4all_full_normalized_year_range_search_has_boolean.jsonl`
- `data/final/sr4all_full_normalized_year_range_search_keywords_only.jsonl`

For the current final dataset, run the standalone completeness report from the
repository root inside the Slurm/Apptainer environment:

```bash
python src/final_ds/check_completeness.py
```

It reads `data/final_ds/sr4all_final.jsonl` and appends to
`logs/final_ds/completeness_check_all.log`. Missing input or malformed records
produce a nonzero exit status; malformed records include their source line
numbers. Importing the checker does not create or truncate logs.
Field presence treats empty wrappers, whitespace, and ghost queries as absent,
but preserves observed `false` and `0` values. Nested and flattened extraction
records use the same coverage rules. All-null percentages use valid records;
essentials use records with methodology beyond `year_range`. The search-strategy
splitter reuses this presence rule. No existing dataset or report is regenerated.

## 6) Query Normalization (`src/norm_queries`)

Goal: normalize search strings and prepare OpenAlex-ready query mappings.

Common flow:

```bash
python src/norm_queries/transform_to_boolean.py
python src/norm_queries/postprocess_boolean_mapping.py
python src/norm_queries/merge_repaired_boolean_mapping.py
python src/norm_queries/merge_year_range_normalized.py
python src/norm_queries/transform_to_oax.py
```

Optional merge back into final split dataset:

```bash
python src/norm_queries/merge_oax_queries.py
```

## 7) OpenAlex Query Execution & Evaluation (`src/oax`)

Goal: run/repair OAX queries, annotate counts, fetch IDs, and evaluate retrieval.

Typical sequence:

```bash
python src/oax/repair_oax_errors.py
python src/oax/sanitize_oax_queries.py
python src/oax/annotate_oax_query_counts.py
python src/oax/split_oax_query_counts_buckets.py
python src/oax/fetch_oax_query_ids.py
python src/oax/flatten_oax_res.py
python src/oax/evaluate_oax_results.py
```

## Reconstruction Checklist

You have successfully reconstructed the dataset when you have all of these:

1. Core joined dataset: `data/final/sr4all_full.jsonl`
2. Year-normalized dataset: `data/final/sr4all_full_normalized_year_range.jsonl`
3. Search-strategy splits:
   - `data/final/sr4all_full_normalized_year_range_search_has_boolean.jsonl`
   - `data/final/sr4all_full_normalized_year_range_search_keywords_only.jsonl`
4. Optional merged release (if you use FT/no-FT split): `data/final/sr4all_merged.jsonl`
5. Optional OAX-enriched/eval artifacts in `data/final/with_oax/`

## Practical Tips

- Check each stage log folder before moving to the next stage:
  - `logs/retrieval/`
  - `logs/add_data/`
  - `logs/extraction/`
  - `logs/final_ds/`
  - `logs/oax/`
- Many scripts are resumable; keep output files and manifests intact between runs.
- If you run on shards, keep shard IDs consistent across extraction to final concat.
