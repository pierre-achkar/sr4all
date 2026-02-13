"""
Split OpenAlex JSONL records into with-PDF and no-PDF subsets,
after dropping records without a non-empty referenced_works list.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

# =========================
# Config
# =========================
Config = {
    "input_jsonl": "./data/rw_ds/raw/unmatched_refs_id_doi_openalex.jsonl",
    "output_with_pdf": "./data/rw_ds/filtered/unmatched_refs_id_doi_openalex_with_refs_has_pdf.jsonl",
    "output_no_pdf": "./data/rw_ds/filtered/unmatched_refs_id_doi_openalex_with_refs_no_pdf.jsonl",
    "log_file": "./logs/add_data/split_on_ft.log",
}


# =========================
# Helpers
# =========================


def has_references(rec: dict) -> bool:
    """Check if the actual list of references exists and is not empty."""
    refs = rec.get("referenced_works")
    return isinstance(refs, list) and len(refs) > 0


def has_pdf(rec: dict) -> bool:
    def _ok(url):
        return isinstance(url, str) and url.strip() != ""

    pl = rec.get("primary_location") or {}
    if _ok(pl.get("pdf_url")):
        return True

    boa = rec.get("best_oa_location") or {}
    if _ok(boa.get("pdf_url")):
        return True

    for loc in rec.get("locations") or []:
        if _ok(loc.get("pdf_url")):
            return True
    return False


def _get_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("split_on_ft")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.propagate = False
    return logger


# =========================
# Main
# =========================


def main() -> None:
    input_path = Path(Config["input_jsonl"])
    out_with_pdf = Path(Config["output_with_pdf"])
    out_no_pdf = Path(Config["output_no_pdf"])
    log_path = Path(Config["log_file"])

    logger = _get_logger(log_path)
    out_with_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_no_pdf.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_lines": 0,
        "parsed_records": 0,
        "drop_no_refs": 0,
        "write_with_pdf": 0,
        "write_no_pdf": 0,
        "json_errors": 0,
        "blank_lines": 0,
    }

    logger.info("Starting split_on_ft")
    logger.info("Input: %s", input_path)
    logger.info("Output (with pdf): %s", out_with_pdf)
    logger.info("Output (no pdf): %s", out_no_pdf)

    with input_path.open("r", encoding="utf-8") as fin, out_with_pdf.open(
        "w", encoding="utf-8"
    ) as fout_pdf, out_no_pdf.open("w", encoding="utf-8") as fout_nopdf:

        for line in fin:
            stats["total_lines"] += 1
            s = line.strip()
            if not s:
                stats["blank_lines"] += 1
                continue

            try:
                rec = json.loads(s)
            except json.JSONDecodeError:
                stats["json_errors"] += 1
                continue

            stats["parsed_records"] += 1

            if not has_references(rec):
                stats["drop_no_refs"] += 1
                continue

            if has_pdf(rec):
                fout_pdf.write(json.dumps(rec) + "\n")
                stats["write_with_pdf"] += 1
            else:
                fout_nopdf.write(json.dumps(rec) + "\n")
                stats["write_no_pdf"] += 1

    logger.info("Done. Stats: %s", stats)


if __name__ == "__main__":
    main()
