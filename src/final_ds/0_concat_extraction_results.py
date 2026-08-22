"""
Concatenate JSON files from subdirectories into a single JSONL file.

Usage: edit CONFIG below and run this script.
"""

import json
import logging
from pathlib import Path
from typing import Iterator, List

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CONFIG = {
    "input_dir": Path("./data/extraction/repaired_fact_checked"),
    "output_file": Path(
        "./data/extraction/repaired_fact_checked/repaired_fact_checked_merged.jsonl"
    ),
    "log_file": Path("./logs/final_ds/concat_extraction_results.log"),
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
logger = logging.getLogger("ConcatJSONL")


def _find_json_files(input_dir: Path, output: Path) -> List[Path]:
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return []
    if not input_dir.is_dir():
        logger.error(f"Input path is not a directory: {input_dir}")
        return []

    return sorted(
        path
        for path in input_dir.rglob("*.json")
        if path.is_file() and path != output
    )


def _iter_json_records(path: Path) -> Iterator[object]:
    with path.open("r", encoding="utf-8") as in_f:
        data = json.load(in_f)

    if isinstance(data, list):
        yield from data
    else:
        yield data


def _validate_inputs(paths: List[Path]) -> List[Path]:
    valid = []
    for p in paths:
        valid.append(p)
    return valid


def concat_json(inputs: List[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    total_entries = 0
    logger.info(f"Merging {len(inputs)} JSON files from {CONFIG['input_dir']}")
    with output.open("w", encoding="utf-8") as out_f:
        for path in inputs:
            for record in _iter_json_records(path):
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_entries += 1

    logger.info(f"Done. Merged {total_entries} entries into {output}")


def main() -> None:
    inputs = _validate_inputs(
        _find_json_files(CONFIG["input_dir"], CONFIG["output_file"])
    )
    if not inputs:
        logger.error("No JSON input files found. Exiting.")
        return

    concat_json(inputs, CONFIG["output_file"])


if __name__ == "__main__":
    main()
