"""
Concatenate multiple JSONL files into a single JSONL file.

Usage: edit CONFIG below and run this script.
"""

import logging
from pathlib import Path
from typing import List

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CONFIG = {
    "input_files": [
        Path(
            "./data/sr4all/extraction_v1/repaired_fact_checked/repaired_fact_checked_corpus_0.jsonl"
        ),
        Path(
            "./data/sr4all/extraction_v1/repaired_fact_checked/repaired_fact_checked_corpus_1.jsonl"
        ),
        Path(
            "./data/sr4all/extraction_v1/repaired_fact_checked/repaired_fact_checked_corpus_2.jsonl"
        ),
    ],
    "output_file": Path(
        "./data/sr4all/extraction_v1/repaired_fact_checked/repaired_fact_checked_corpus_all.jsonl"
    ),
    "log_file": Path("./logs/utils/concat_jsonl.log"),
}

# Setup Logging
CONFIG["log_file"].parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(CONFIG["log_file"]),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("ConcatJSONL")


def _validate_inputs(paths: List[Path]) -> List[Path]:
    valid = []
    for p in paths:
        if not p.exists():
            logger.warning(f"Missing input file: {p}")
            continue
        if p.is_dir():
            logger.warning(f"Skipping directory input: {p}")
            continue
        valid.append(p)
    return valid


def concat_jsonl(inputs: List[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    total_lines = 0
    with open(output, "w", encoding="utf-8") as out_f:
        for path in inputs:
            logger.info(f"Appending: {path}")
            with open(path, "r", encoding="utf-8") as in_f:
                file_lines = 0
                for line in in_f:
                    if not line.strip():
                        continue
                    out_f.write(line)
                    total_lines += 1
                    file_lines += 1

                logger.info(f"Lines in {path}: {file_lines}")

    logger.info(f"Done. Wrote {total_lines} lines to {output}")


def main() -> None:
    inputs = _validate_inputs(CONFIG["input_files"])
    if not inputs:
        logger.error("No valid input files found. Exiting.")
        return

    concat_jsonl(inputs, CONFIG["output_file"])


if __name__ == "__main__":
    main()
