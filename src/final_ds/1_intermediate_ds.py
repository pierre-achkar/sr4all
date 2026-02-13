"""
Final Dataset Flattener & Filter.

Input:  fact_checked_repaired_corpus_0.jsonl
Output: sr4all_final_v1.jsonl

Tasks:
1. FLATTEN: Hoist nested 'extraction' fields to top level.
2. FILTER: Drop documents with no extraction or all extracted fields null/empty.
"""

import json
from pathlib import Path
import logging
import re

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
INPUT_FILE = Path(
    "./data/sr4all/extraction_v1/repaired_fact_checked/repaired_fact_checked_corpus_all.jsonl"
)
OUTPUT_FILE = Path(
    "./data/sr4all/extraction_v1/intermediate/sr4all_intermediate_all.jsonl"
)
LOGGING_FILE = Path("./logs/final_ds/intermediate_dataset_flattener_all.log")

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOGGING_FILE, mode="w"), logging.StreamHandler()],
)


# -----------------------------------------------------------------------------
# HELPER: VALIDATION LOGIC
# -----------------------------------------------------------------------------
def is_filled(field_data):
    """
    Checks if a field has valid content (not null, not empty list, not ghost object).
    """
    if field_data is None:
        return False

    # Case 1: Evidence Object {"value": ...}
    if isinstance(field_data, dict):
        val = field_data.get("value")
        if val is None:
            return False
        if isinstance(val, list) and len(val) == 0:
            return False
        return True

    # Case 2: List of Objects (Boolean Queries)
    if isinstance(field_data, list):
        if not field_data:
            return False  # Empty list

        # Check for ghost object [{"boolean_query_string": null}]
        if isinstance(field_data[0], dict):
            if field_data[0].get("boolean_query_string") is None:
                return False
        return True

    return False


def has_any_filled(extraction: dict) -> bool:
    """
    Returns True if ANY extracted field is non-null/non-empty.
    """
    if not extraction:
        return False
    return any(is_filled(v) for v in extraction.values())


def _strip_verbatim_sources(extraction: dict) -> dict:
    """
    Returns a copy of extraction with only the main values (no verbatim_source).
    Evidence objects become their `value`, and boolean query items drop verbatim_source.
    """
    if not extraction:
        return {}

    cleaned = {}
    for key, value in extraction.items():
        # Evidence object -> keep only value
        if isinstance(value, dict) and "value" in value:
            cleaned[key] = value.get("value")
            continue

        # Boolean queries list -> remove verbatim_source per item
        if key == "exact_boolean_queries" and isinstance(value, list):
            cleaned[key] = [
                {
                    "boolean_query_string": item.get("boolean_query_string"),
                    "database_source": item.get("database_source"),
                }
                for item in value
                if isinstance(item, dict)
            ]
            continue

        cleaned[key] = value

    return cleaned


# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
def main():
    if not INPUT_FILE.exists():
        logging.error(f"Error: Input file not found at {INPUT_FILE}")
        return

    logging.info(f"Reading from: {INPUT_FILE.name}")
    logging.info(f"Writing to:   {OUTPUT_FILE.name}")
    total_read = 0
    total_saved = 0
    total_dropped_no_extraction = 0
    total_dropped_all_null = 0

    with open(INPUT_FILE, "r") as fin, open(OUTPUT_FILE, "w") as fout:
        for line in fin:
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    logging.error("Skipping error line: record is not a dict")
                    continue
                total_read += 1

                extraction = record.get("extraction", {})
                if extraction is None:
                    extraction = {}
                if not isinstance(extraction, dict):
                    logging.error("Skipping error line: extraction is not a dict")
                    continue

                # --- FILTER STEP ---
                if not extraction:
                    total_dropped_no_extraction += 1
                    continue
                if not has_any_filled(extraction):
                    total_dropped_all_null += 1
                    continue

                # --- FLATTEN STEP ---
                # 1. Keep only doc_id + extracted values (no verbatim_source)
                final_record = {
                    "file_path": record.get("file_path"),
                    "doc_id": record.get("doc_id"),
                }

                # 2. Hoist cleaned extracted fields
                if extraction:
                    final_record.update(_strip_verbatim_sources(extraction))

                # 3. Save (Clean, no extra stats)
                fout.write(json.dumps(final_record) + "\n")
                total_saved += 1

            except Exception as e:
                logging.error(f"Skipping error line: {e}")

    logging.info("-" * 40)
    logging.info(f"PROCESSING COMPLETE")
    logging.info(f"Total Read:     {total_read}")
    logging.info(f"Filtered Out:   {total_read - total_saved}")
    logging.info(f"Dropped (no extraction): {total_dropped_no_extraction}")
    logging.info(f"Dropped (all fields null/empty): {total_dropped_all_null}")
    logging.info(f"Final Dataset:  {total_saved} documents")
    logging.info(f"Saved to:       {OUTPUT_FILE}")
    logging.info("-" * 40)


if __name__ == "__main__":
    main()
