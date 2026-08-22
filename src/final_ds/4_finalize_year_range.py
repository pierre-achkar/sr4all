"""Promote normalized year ranges into the final dataset field."""

import json
import logging
from pathlib import Path


INPUT_JSONL = Path(
    "./data/final_ds/oax_slim_with_extraction_normalized_year_range.jsonl"
)
OUTPUT_JSONL = Path("./data/final_ds/sr4all_final.jsonl")
LOG_FILE = Path("./logs/final_ds/finalize_year_range.log")

OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger("FinalizeYearRange")


def main() -> None:
    if not INPUT_JSONL.exists():
        logger.error("Input file not found: %s", INPUT_JSONL)
        return

    total = 0
    invalid = 0
    missing_normalized = 0

    with INPUT_JSONL.open("r", encoding="utf-8") as input_file, OUTPUT_JSONL.open(
        "w", encoding="utf-8"
    ) as output_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                invalid += 1
                logger.warning("Skipping invalid JSON on line %d: %s", line_number, error)
                continue

            total += 1
            normalized = record.pop("year_range_normalized", None)
            record.pop("year_range", None)
            record.pop("year_range_normalization_rule", None)
            record["year_range"] = normalized
            if normalized is None:
                missing_normalized += 1

            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("FINAL YEAR-RANGE CLEANUP COMPLETE")
    logger.info("Records written:          %d", total)
    logger.info("Missing normalized value: %d", missing_normalized)
    logger.info("Invalid records skipped:  %d", invalid)
    logger.info("Output:                   %s", OUTPUT_JSONL)


if __name__ == "__main__":
    main()