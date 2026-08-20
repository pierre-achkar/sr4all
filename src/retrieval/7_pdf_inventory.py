"""
Inventory downloaded PDFs against the slim OpenAlex JSONL.

- Reads slim OpenAlex JSONL records
- Resolves the expected sharded PDF path for each OpenAlex work ID
- Counts how many records exist and how many have a downloaded PDF on disk
- Logs a compact summary for downstream auditing
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, Generator, Optional


INPUT_JSONL = "./data/retrieval/merged/oax_slim.jsonl"
PDF_DIR = "./data/retrieval/merged/pdfs"
LOG_FILE = "./logs/retrieval/7_pdf_inventory.log"
MIN_PDF_BYTES = 1024


os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
    force=True,
    filemode="w",
)


def stream_jsonl(path: str) -> Generator[Dict[str, Any], None, None]:
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logging.warning("Invalid JSON at line %d in %s", line_no, path)
                continue
            if not isinstance(rec, dict):
                logging.warning("Non-object JSON at line %d in %s", line_no, path)
                continue
            yield rec


def extract_work_id(openalex_id: Any) -> Optional[str]:
    if not isinstance(openalex_id, str):
        return None
    match = re.search(r"([A-Z]\d+)$", openalex_id.strip())
    return match.group(1).upper() if match else None


def shard_path_for_work(work_id: str) -> str:
    shard = "W0"
    if len(work_id) >= 2 and work_id[1].isdigit():
        shard = f"W{work_id[1]}"
    return os.path.join(PDF_DIR, shard, f"{work_id}.pdf")


def has_downloaded_pdf(path: str) -> bool:
    try:
        return os.path.exists(path) and os.path.getsize(path) >= MIN_PDF_BYTES
    except OSError:
        return False


def count_pdfs_on_disk(pdf_dir: str) -> int:
    total = 0
    for root, _, files in os.walk(pdf_dir):
        for name in files:
            if not name.lower().endswith(".pdf"):
                continue
            path = os.path.join(root, name)
            if has_downloaded_pdf(path):
                total += 1
    return total


def main() -> None:
    total_records = 0
    records_with_valid_id = 0
    records_with_pdf = 0
    records_missing_pdf = 0
    records_missing_id = 0

    for rec in stream_jsonl(INPUT_JSONL):
        total_records += 1
        work_id = extract_work_id(rec.get("id"))
        if not work_id:
            records_missing_id += 1
            continue

        records_with_valid_id += 1
        pdf_path = shard_path_for_work(work_id)
        if has_downloaded_pdf(pdf_path):
            records_with_pdf += 1
        else:
            records_missing_pdf += 1

    pdfs_on_disk = count_pdfs_on_disk(PDF_DIR)
    pct_with_pdf = (records_with_pdf / total_records * 100.0) if total_records else 0.0

    logging.info("PDF inventory started | input=%s | pdf_dir=%s", INPUT_JSONL, PDF_DIR)
    logging.info("Total slim JSONL entries: %d", total_records)
    logging.info("Entries with valid OpenAlex work ID: %d", records_with_valid_id)
    logging.info("Entries missing valid OpenAlex work ID: %d", records_missing_id)
    logging.info("Entries with downloaded PDF: %d", records_with_pdf)
    logging.info("Entries missing downloaded PDF: %d", records_missing_pdf)
    logging.info("Valid PDFs found on disk: %d", pdfs_on_disk)
    logging.info("Downloaded PDF coverage: %.2f%%", pct_with_pdf)

    print(f"Total slim JSONL entries: {total_records}")
    print(f"Entries with downloaded PDF: {records_with_pdf}")
    print(f"Entries missing downloaded PDF: {records_missing_pdf}")
    print(f"Valid PDFs found on disk: {pdfs_on_disk}")
    print(f"Downloaded PDF coverage: {pct_with_pdf:.2f}%")
    print(f"Log written to: {LOG_FILE}")


if __name__ == "__main__":
    main()