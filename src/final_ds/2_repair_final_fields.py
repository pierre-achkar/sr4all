"""
Check/repair final dataset fields against schema.py (without verbatim_source).

- Ensures every record has all fields from ReviewExtraction.
- If a field is missing, adds it with null.
- Writes a log file with counts per field.
- Writes a corrected copy to output_file (input is left unchanged).
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict

# Ensure we can import src
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from extraction.schema import ReviewExtraction

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CONFIG = {
    "input_file": Path(
        "./data/sr4all/extraction_v1/intermediate/sr4all_intermediate_all.jsonl"
    ),
    "output_file": Path(
        "./data/sr4all/extraction_v1/intermediate/sr4all_intermediate_all_corrected.jsonl"
    ),
    "log_file": Path("./logs/final_ds/repair_fields_all.log"),
}

# Setup Logging
CONFIG["log_file"].parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(CONFIG["log_file"], mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("FinalFieldsCheck")


def main():
    input_path = CONFIG["input_file"]
    output_path = CONFIG["output_file"]
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    # All required fields from schema (top-level only)
    required_fields = list(ReviewExtraction.model_fields.keys())

    # Stats
    total_records = 0
    field_add_counts: Dict[str, int] = {k: 0 for k in required_fields}

    with open(input_path, "r", encoding="utf-8") as fin, open(
        output_path, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSON line.")
                continue

            total_records += 1

            # Ensure required fields exist and order by schema
            ordered_record: Dict = {}
            for field in required_fields:
                if field not in record:
                    ordered_record[field] = None
                    field_add_counts[field] += 1
                else:
                    ordered_record[field] = record[field]

            # Preserve any extra fields after schema fields (if present)
            for k, v in record.items():
                if k not in ordered_record:
                    ordered_record[k] = v

            fout.write(json.dumps(ordered_record) + "\n")

    # Log summary
    logger.info("-" * 60)
    logger.info(f"Processed records: {total_records}")
    logger.info("Fields added (missing -> null):")
    for field, count in field_add_counts.items():
        if count > 0:
            logger.info(f"  {field}: {count}")
    logger.info("-" * 60)


if __name__ == "__main__":
    main()
