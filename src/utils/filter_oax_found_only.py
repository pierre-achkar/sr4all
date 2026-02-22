#!/usr/bin/env python3
"""
Filter OpenAlex DOI JSONL output to keep only found records.

This script removes placeholder lines written as:
{"requested_doi": "...", "status": "not_found"}
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep only found OpenAlex records from a JSONL file."
    )
    parser.add_argument(
        "--input",
        default="data/benchmark_data/oax_doi_results.jsonl",
        help="Path to input JSONL file.",
    )
    parser.add_argument(
        "--output",
        default="data/benchmark_data/oax_doi_results.found_only.jsonl",
        help="Path to output JSONL file.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500000,
        help="Log progress every N lines.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("filter_oax_found_only")

    total = 0
    kept = 0
    filtered_out = 0
    invalid_json = 0

    with input_path.open("r", encoding="utf-8") as in_f, output_path.open(
        "w", encoding="utf-8"
    ) as out_f:
        for line in in_f:
            total += 1
            s = line.strip()
            if not s:
                filtered_out += 1
                continue

            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                invalid_json += 1
                filtered_out += 1
                continue

            if obj.get("status") == "not_found":
                filtered_out += 1
                continue

            out_f.write(line)
            kept += 1

            if args.progress_every > 0 and total % args.progress_every == 0:
                logger.info(
                    "Progress total=%d kept=%d filtered_out=%d invalid_json=%d",
                    total,
                    kept,
                    filtered_out,
                    invalid_json,
                )

    logger.info("Input file: %s", input_path)
    logger.info("Output file: %s", output_path)
    logger.info("Total lines: %d", total)
    logger.info("Kept lines: %d", kept)
    logger.info("Filtered out lines: %d", filtered_out)
    logger.info("Invalid JSON lines: %d", invalid_json)


if __name__ == "__main__":
    main()
