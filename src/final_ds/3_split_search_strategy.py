"""
Split final dataset into 2 cohorts by search strategy:
- has_boolean (has any exact_boolean_queries, regardless of keywords)
- keywords_only (no exact_boolean_queries but has keywords)

Input:  sr4all_full_normalized_year_range.jsonl
Output: two JSONL files + logging stats
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

if __package__:
    from .check_completeness import is_filled
else:
    from check_completeness import is_filled

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CONFIG = {
    "input_file": Path("./data/final/sr4all_full_normalized_year_range.jsonl"),
    "output_dir": Path("./data/final"),
    "log_file": Path("./logs/final_ds/split_search_strategy.log"),
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

    out_has_boolean = (
        CONFIG["output_dir"]
        / "sr4all_full_normalized_year_range_search_has_boolean.jsonl"
    )
    out_keywords_only = (
        CONFIG["output_dir"]
        / "sr4all_full_normalized_year_range_search_keywords_only.jsonl"
    )

    counts = {
        "total": 0,
        "has_boolean": 0,
        "keywords_only": 0,
        "has_boolean_and_keywords": 0,
        "neither": 0,
        "missing_extraction": 0,
    }

    logger.info(f"Reading: {input_path}")
    logger.info(f"Writing outputs to: {CONFIG['output_dir']}")

    with open(input_path, "r", encoding="utf-8") as fin, open(
        out_has_boolean, "w", encoding="utf-8"
    ) as fbo, open(out_keywords_only, "w", encoding="utf-8") as fko:

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

            if has_boolean:
                counts["has_boolean"] += 1
                if has_keywords:
                    counts["has_boolean_and_keywords"] += 1
                write_jsonl_line(fbo, rec)
            elif has_keywords:
                counts["keywords_only"] += 1
                write_jsonl_line(fko, rec)
            else:
                counts["neither"] += 1

    logger.info("Split complete.")
    logger.info(
        "Totals | total=%d has_boolean=%d has_boolean_and_keywords=%d keywords_only=%d neither=%d missing_extraction=%d",
        counts["total"],
        counts["has_boolean"],
        counts["has_boolean_and_keywords"],
        counts["keywords_only"],
        counts["neither"],
        counts["missing_extraction"],
    )

    logger.info(f"Output: {out_has_boolean}")
    logger.info(f"Output: {out_keywords_only}")


if __name__ == "__main__":
    main()
