# Webis-SR4ALL-26

This project provides a full pipeline for extracting, verifying, and normalizing systematic review methodology from OpenAlex and associated PDFs. Below are step-by-step instructions for each major stage:

---

## 1. Retrieval (src/retrieval)

**Purpose:** Fetch systematic review records from OpenAlex, filter for valid PDFs, and slim metadata for downstream processing.

**Steps:**
- Run `1_oax_fetch_studies.py` to download SR records from OpenAlex.
- Run `2_oax_filter.py` to filter records for valid titles, DOIs, and PDF links.
- Run `3_pdf_download.py` to download PDFs and log results in `pdf_download_manifest.jsonl`.
- Run `4_oax_filter_downloaded.py` to filter for successfully downloaded PDFs.
- Run `5_oax_slim.py` to slim records for extraction.

**Outputs:**
- Filtered OpenAlex records and PDFs in `data/filtered/`.

---

## 2. OCR (src/ocr)

**Purpose:** Parse downloaded PDFs into plain text for extraction.

**Steps:**
- Use scripts in `src/ocr` to convert PDFs in `data/filtered/pdfs/` to text files.
- Ensure output text files are referenced in the extraction manifest.

---

## 3. Extraction (src/extraction)

**Purpose:** Extract structured methodology from SR texts using LLMs, verify evidence, fact-check, and repair missing fields.

**Steps:**
- Run `1_extraction.py` to extract candidate information from raw texts (requires manifest).
- Run `2_alignment.py` to verify extracted evidence against source text.
- Run `3_fact_checking.py` to fact-check and null unsupported fields.
- Run `4_repair.py` to repair missing or null fields using LLMs.

**Outputs:**
- Structured extractions in JSONL format in `data/extraction_v1/`.

---

## 4. Query Normalization (src/norm_queries)

**Purpose:** Normalize extracted boolean queries and keywords for OpenAlex API compatibility.

**Steps:**
- Use scripts in `src/norm_queries` to convert extracted queries into OpenAlex-compatible query strings.
- Output normalized queries for downstream OpenAlex retrieval.

---

## 5. OpenAlex Querying (src/oax)

**Purpose:** Execute normalized queries against OpenAlex and collect retrieval statistics.

**Steps:**
- Use scripts in `src/oax` to run normalized queries and fetch results from OpenAlex.
- Analyze retrieval statistics and output for final dataset construction.

---

## Notes
- All scripts use relative paths in their CONFIG dicts; update these for new runs. 
- Intermediate and final outputs are stored in `data/filtered/`, `data/final/`, and `data/raw/`.
- For detailed schema and prompt conventions, see `src/extraction/schema.py` and `src/extraction/prompts.py`.

---
