"""
Split final dataset into 3 cohorts by search strategy:
- boolean_only
- keywords_only
- both

Input:  sr4all_full_normalized_year_range.jsonl
Output: three JSONL files + logging stats
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CONFIG = {
    "input_file": Path("/home/fhg/pie65738/projects/sr4all/data/final/sr4all_full_normalized_year_range.jsonl"),
    "output_dir": Path("/home/fhg/pie65738/projects/sr4all/data/final"),
    "log_file": Path("/home/fhg/pie65738/projects/sr4all/logs/final_ds/split_search_strategy.log"),
}

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
CONFIG["log_file"].parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(CONFIG["log_file"], mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("SplitSearchStrategy")

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def is_filled(field_data: Any) -> bool:
    """
    Checks if a field has valid content.
    Matches semantics from check_completeness.py.
    """
    if field_data is None:
        return False

    # Case 1: Evidence object {"value": ...}
    if isinstance(field_data, dict):
        val = field_data.get("value")
        if val is None:
            return False
        if isinstance(val, list) and len(val) == 0:
            return False
        return True

    # Case 2: List of objects (exact_boolean_queries)
    if isinstance(field_data, list):
        if not field_data:
            return False
        first_item = field_data[0]
        if isinstance(first_item, dict):
            if first_item.get("boolean_query_string") is None:
                return False
        return True

    return False


def write_jsonl_line(fp, record: Dict[str, Any]):
    fp.write(json.dumps(record, ensure_ascii=False) + "\n")

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    input_path = CONFIG["input_file"]
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)

    out_both = CONFIG["output_dir"] / "sr4all_full_normalized_year_range_search_both.jsonl"
    out_boolean_only = CONFIG["output_dir"] / "sr4all_full_normalized_year_range_search_boolean_only.jsonl"
    out_keywords_only = CONFIG["output_dir"] / "sr4all_full_normalized_year_range_search_keywords_only.jsonl"

    counts = {
        "total": 0,
        "both": 0,
        "boolean_only": 0,
        "keywords_only": 0,
        "neither": 0,
        "missing_extraction": 0,
    }

    logger.info(f"Reading: {input_path}")
    logger.info(f"Writing outputs to: {CONFIG['output_dir']}")

    with open(input_path, "r", encoding="utf-8") as fin, \
        open(out_both, "w", encoding="utf-8") as fb, \
        open(out_boolean_only, "w", encoding="utf-8") as fbo, \
        open(out_keywords_only, "w", encoding="utf-8") as fko:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            counts["total"] += 1

            data = rec.get("extraction")
            if not isinstance(data, dict):
                # Some final datasets are already flattened (no "extraction" wrapper)
                data = rec

            if not data:
                counts["missing_extraction"] += 1
                continue

            has_boolean = is_filled(data.get("exact_boolean_queries"))
            has_keywords = is_filled(data.get("keywords_used"))

            if has_boolean and has_keywords:
                counts["both"] += 1
                write_jsonl_line(fb, rec)
            elif has_boolean and not has_keywords:
                counts["boolean_only"] += 1
                write_jsonl_line(fbo, rec)
            elif has_keywords and not has_boolean:
                counts["keywords_only"] += 1
                write_jsonl_line(fko, rec)
            else:
                counts["neither"] += 1

    logger.info("Split complete.")
    logger.info(
        "Totals | total=%d both=%d boolean_only=%d keywords_only=%d neither=%d missing_extraction=%d",
        counts["total"],
        counts["both"],
        counts["boolean_only"],
        counts["keywords_only"],
        counts["neither"],
        counts["missing_extraction"],
    )

    logger.info(f"Output: {out_both}")
    logger.info(f"Output: {out_boolean_only}")
    logger.info(f"Output: {out_keywords_only}")


if __name__ == "__main__":
    main()
