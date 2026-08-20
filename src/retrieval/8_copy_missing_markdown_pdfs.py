"""
Copy downloaded PDFs that do not yet have parsed markdown.

- Scans downloaded PDFs in the merged retrieval PDF directory
- Checks whether a matching markdown file exists in parsed markdown storage
- Copies unparsed PDFs to a dedicated missing-PDF directory (preserving shard layout)
- Logs summary counts for parsed vs still-to-parse
"""
from __future__ import annotations

import logging
import os
import shutil


PDF_DIR = "./data/retrieval/merged/pdfs"
PARSED_MD_DIR = "/storage/remote/DevStorage/sr4all"
MISSING_PDF_DIR = "/storage/remote/DevStorage/sr4all_missing_pdfs"
LOG_FILE = "./logs/retrieval/8_copy_missing_markdown_pdfs.log"
MIN_PDF_BYTES = 1024


os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs(MISSING_PDF_DIR, exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
    force=True,
    filemode="w",
)


def is_valid_pdf(path: str) -> bool:
    try:
        return os.path.exists(path) and os.path.getsize(path) >= MIN_PDF_BYTES
    except OSError:
        return False


def iter_valid_pdfs(pdf_dir: str):
    for root, _, files in os.walk(pdf_dir):
        for name in files:
            if not name.lower().endswith(".pdf"):
                continue
            path = os.path.join(root, name)
            if is_valid_pdf(path):
                yield path


def md_exists(parsed_md_dir: str, rel_pdf_path: str) -> bool:
    rel_no_ext, _ = os.path.splitext(rel_pdf_path)
    md_path = os.path.join(parsed_md_dir, rel_no_ext + ".md")
    markdown_path = os.path.join(parsed_md_dir, rel_no_ext + ".markdown")
    return os.path.exists(md_path) or os.path.exists(markdown_path)


def main() -> None:
    total_valid_pdfs = 0
    already_parsed = 0
    still_to_parse = 0
    copied_now = 0
    already_in_missing_dir = 0

    logging.info(
        "Starting missing-markdown PDF copy | pdf_dir=%s | parsed_md_dir=%s | missing_pdf_dir=%s",
        PDF_DIR,
        PARSED_MD_DIR,
        MISSING_PDF_DIR,
    )

    for pdf_path in iter_valid_pdfs(PDF_DIR):
        total_valid_pdfs += 1
        rel_pdf_path = os.path.relpath(pdf_path, PDF_DIR)

        if md_exists(PARSED_MD_DIR, rel_pdf_path):
            already_parsed += 1
            continue

        still_to_parse += 1
        dst_path = os.path.join(MISSING_PDF_DIR, rel_pdf_path)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        if os.path.exists(dst_path):
            already_in_missing_dir += 1
            continue

        shutil.copy2(pdf_path, dst_path)
        copied_now += 1

    pct_parsed = (already_parsed / total_valid_pdfs * 100.0) if total_valid_pdfs else 0.0
    pct_missing = (still_to_parse / total_valid_pdfs * 100.0) if total_valid_pdfs else 0.0

    logging.info("Total valid downloaded PDFs: %d", total_valid_pdfs)
    logging.info("Already parsed to markdown: %d", already_parsed)
    logging.info("Still to parse: %d", still_to_parse)
    logging.info("Copied to missing folder in this run: %d", copied_now)
    logging.info("Already present in missing folder: %d", already_in_missing_dir)
    logging.info("Parsed coverage: %.2f%%", pct_parsed)
    logging.info("Missing coverage: %.2f%%", pct_missing)

    print(f"Total valid downloaded PDFs: {total_valid_pdfs}")
    print(f"Already parsed to markdown: {already_parsed}")
    print(f"Still to parse: {still_to_parse}")
    print(f"Copied to missing folder in this run: {copied_now}")
    print(f"Already present in missing folder: {already_in_missing_dir}")
    print(f"Parsed coverage: {pct_parsed:.2f}%")
    print(f"Missing coverage: {pct_missing:.2f}%")
    print(f"Log written to: {LOG_FILE}")


if __name__ == "__main__":
    main()