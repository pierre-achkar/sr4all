"""
Merge two JSONL datasets and deduplicate in three stages:
1) DOI
2) OpenAlex ID
3) Normalized title

Outputs a clean merged JSONL and logs all counts.
"""
from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from typing import Callable

# Config
INPUT_FILTERED_JSONL = "./data/retrieval/filtered/oax_sr_title_english_refs.jsonl"
INPUT_BENCHMARK_JSONL = "./data/retrieval/benchmark_data/oax_benchmark_data.jsonl"

OUTPUT_DIR = "./data/retrieval/merged"
OUTPUT_JSONL = os.path.join(OUTPUT_DIR, "oax_merged_dedup.jsonl")
LOG_FILE = "./logs/retrieval/4_join_studies.log"

_DOI_PREFIX_RE = re.compile(r"^doi:\s*", re.IGNORECASE)
_DOI_URL_RE = re.compile(r"^https?://(dx\.)?doi\.org/", re.IGNORECASE)
_OA_WORK_ID_RE = re.compile(r"(W\d+)$", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^\w\s]")
_SPACE_RE = re.compile(r"\s+")


def setup_logging() -> None:
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        encoding="utf-8",
        force=True,
        filemode="w",
    )


def load_jsonl(path: str) -> list[dict]:
    records: list[dict] = []
    invalid_lines = 0
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                invalid_lines += 1
                logging.warning("Invalid JSON at %s line %d", path, i)
                continue
            if isinstance(obj, dict):
                records.append(obj)
            else:
                invalid_lines += 1
                logging.warning("Non-object JSON at %s line %d", path, i)
    if invalid_lines:
        logging.warning("Skipped %d invalid lines in %s", invalid_lines, path)
    return records


def normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    doi = str(value).strip().lower()
    doi = _DOI_PREFIX_RE.sub("", doi)
    doi = _DOI_URL_RE.sub("", doi)
    return doi or None


def extract_doi(rec: dict) -> str | None:
    doi = normalize_doi(rec.get("doi"))
    if doi:
        return doi
    ids = rec.get("ids") or {}
    if isinstance(ids, dict):
        doi = normalize_doi(ids.get("doi"))
        if doi:
            return doi
    return normalize_doi(rec.get("requested_doi"))


def normalize_openalex_id(value: str | None) -> str | None:
    if not value:
        return None
    s = str(value).strip()
    if not s:
        return None
    m = _OA_WORK_ID_RE.search(s)
    if m:
        return m.group(1).upper()
    return None


def extract_openalex_id(rec: dict) -> str | None:
    oa_id = normalize_openalex_id(rec.get("id"))
    if oa_id:
        return oa_id
    ids = rec.get("ids") or {}
    if isinstance(ids, dict):
        return normalize_openalex_id(ids.get("openalex"))
    return None


def normalize_title(value: str | None) -> str | None:
    if not value:
        return None
    s = unicodedata.normalize("NFKC", str(value)).lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _SPACE_RE.sub(" ", s).strip()
    return s or None


def extract_title_key(rec: dict) -> str | None:
    title = rec.get("display_name") or rec.get("title")
    return normalize_title(title)


def deduplicate(
    records: list[dict],
    key_fn: Callable[[dict], str | None],
    stage_name: str,
) -> tuple[list[dict], int, int]:
    seen: set[str] = set()
    output: list[dict] = []
    duplicate_count = 0
    no_key_count = 0

    for rec in records:
        key = key_fn(rec)
        if not key:
            no_key_count += 1
            output.append(rec)
            continue
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        output.append(rec)

    logging.info(
        "Dedup stage=%s | input=%d | duplicates=%d | missing_key=%d | output=%d",
        stage_name,
        len(records),
        duplicate_count,
        no_key_count,
        len(output),
    )
    return output, duplicate_count, no_key_count


def write_jsonl(path: str, records: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    setup_logging()
    logging.info("Starting merge+dedup pipeline")
    logging.info("Input filtered path: %s", INPUT_FILTERED_JSONL)
    logging.info("Input benchmark path: %s", INPUT_BENCHMARK_JSONL)

    filtered_records = load_jsonl(INPUT_FILTERED_JSONL)
    benchmark_records = load_jsonl(INPUT_BENCHMARK_JSONL)

    logging.info("Total before merge | filtered=%d", len(filtered_records))
    logging.info("Total before merge | benchmark=%d", len(benchmark_records))

    # "Concatenate 1 in 2": keep benchmark records first, then append filtered.
    merged_records = benchmark_records + filtered_records
    logging.info("Total after concatenation (before dedup)=%d", len(merged_records))

    dedup_1, dup_doi, _ = deduplicate(merged_records, extract_doi, "doi")
    dedup_2, dup_openalex, _ = deduplicate(dedup_1, extract_openalex_id, "openalex")
    dedup_3, dup_title, _ = deduplicate(dedup_2, extract_title_key, "normalized_title")

    write_jsonl(OUTPUT_JSONL, dedup_3)

    logging.info("Duplicate summary | doi=%d | openalex=%d | normalized_title=%d", dup_doi, dup_openalex, dup_title)
    logging.info("Final total clean records=%d", len(dedup_3))
    logging.info("Output written to %s", OUTPUT_JSONL)

    print(f"Input filtered: {len(filtered_records)}")
    print(f"Input benchmark: {len(benchmark_records)}")
    print(f"Merged before dedup: {len(merged_records)}")
    print(f"Duplicates removed by DOI: {dup_doi}")
    print(f"Duplicates removed by OpenAlex: {dup_openalex}")
    print(f"Duplicates removed by normalized title: {dup_title}")
    print(f"Final clean total: {len(dedup_3)}")
    print(f"Output JSONL: {OUTPUT_JSONL}")
    print(f"Log file: {LOG_FILE}")


if __name__ == "__main__":
    main()
