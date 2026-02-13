"""
Merge year_range_normalized into a target JSONL by id mapping.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Optional

# ========================
# Config
# ========================
CONFIG = {
    "year_range_jsonl": Path(
        "./data/final/sr4all_full_normalized_year_range_search_keywords_only.jsonl"
    ),
    "input_jsonl": Path(
        "./data/final/with_boolean/final/sr4all_full_normalized_keywords_only_mapping_merged_2.jsonl"
    ),
    "output_jsonl": Path(
        "./data/final/with_boolean/final/sr4all_full_normalized_keywords_only_mapping_merged_2_with_year_range.jsonl"
    ),
    "log_file": Path("./logs/final_ds/merge_year_range_normalized.log"),
}

# ========================
# Logging
# ========================
CONFIG["log_file"].parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(CONFIG["log_file"]),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    filemode="w",
)
logger = logging.getLogger("merge_year_range_normalized")


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def get_record_id(rec: Dict) -> Optional[str]:
    return rec.get("id") or rec.get("doc_id") or rec.get("rec_id")


def load_year_range_mapping(path: Path) -> Dict[str, Optional[str]]:
    mapping: Dict[str, Optional[str]] = {}
    total = 0
    kept = 0
    for rec in iter_jsonl(path):
        total += 1
        rec_id = get_record_id(rec)
        if not rec_id:
            continue
        if "year_range_normalized" not in rec:
            continue
        mapping[str(rec_id)] = rec.get("year_range_normalized")
        kept += 1
    logger.info("year_range_records_total=%d year_range_records_kept=%d", total, kept)
    return mapping


def main() -> None:
    year_range_path = CONFIG["year_range_jsonl"]
    input_path = CONFIG["input_jsonl"]
    output_path = CONFIG["output_jsonl"]

    if not year_range_path.exists():
        logger.error("Year-range input not found: %s", year_range_path)
        return
    if not input_path.exists():
        logger.error("Target input not found: %s", input_path)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading year_range_normalized mapping...")
    mapping = load_year_range_mapping(year_range_path)
    logger.info("Mapping size=%d", len(mapping))

    total = 0
    matched = 0
    missing = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for rec in iter_jsonl(input_path):
            total += 1
            rec_id = get_record_id(rec)
            if rec_id and str(rec_id) in mapping:
                rec["year_range_normalized"] = mapping[str(rec_id)]
                matched += 1
            else:
                missing += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(
        "Done. total=%d matched=%d missing=%d output=%s",
        total,
        matched,
        missing,
        output_path,
    )


if __name__ == "__main__":
    main()
