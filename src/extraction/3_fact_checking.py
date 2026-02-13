"""
Job C: Fact-Checking and Hallucination Mitigation
- Reads aligned candidates from Job B
- Uses a FactChecker module (based on a smaller, efficient model) to verify each candidate
- If a candidate fails fact-checking, it is "nuked" (value and source set to null) in the JSON structure
- Saves the fact-checked corpus to a new JSONL file for downstream use (e.g., training, analysis)
"""

import sys
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

# Ensure we can import src
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from extraction.fact_checker import FactChecker

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CONFIG = {
    "input_file": Path(
        "/data/sr4all/extraction_v1/repaired_aligned/aligned_repaired_candidates_0.jsonl"
    ),
    "output_file": Path(
        "/data/sr4all/extraction_v1/repaired_fact_checked/repaired_fact_checked_corpus_0.jsonl"
    ),
    "log_file": Path("/logs/extraction/repaired_factcheck_0.log"),
    # Batch size for the FactChecker (Chunking)
    "batch_size": 128,
    # Save progress to disk every N documents
    "save_interval": 50,
}

# Setup Logging
CONFIG["output_file"].parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(CONFIG["log_file"]), logging.StreamHandler()],
)
logger = logging.getLogger("JobE")


def main():
    if not CONFIG["input_file"].exists():
        logger.error(f"Input file not found: {CONFIG['input_file']}")
        return

    # 1. Initialize Model
    # IMPORTANT: Ensure Job A (Qwen) is NOT running on the same GPU.
    try:
        checker = FactChecker(batch_size=CONFIG["batch_size"])
    except Exception as e:
        logger.critical(f"Failed to load FactChecker: {e}")
        return

    # 2. Load Data
    logger.info(f"Loading data from {CONFIG['input_file']}...")
    records = []
    with open(CONFIG["input_file"], "r") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except:
                pass

    # Check Resume Status
    completed_ids = set()
    if CONFIG["output_file"].exists():
        with open(CONFIG["output_file"], "r") as f:
            for line in f:
                try:
                    completed_ids.add(json.loads(line).get("doc_id"))
                except:
                    pass

    to_process = [r for r in records if r.get("doc_id") not in completed_ids]

    if not to_process:
        logger.info("All documents fact-checked. Exiting.")
        return

    logger.info(f"Checking {len(to_process)} remaining documents...")

    # 3. Processing Loop
    buffer = []

    for record in tqdm(to_process):
        data = record.get("extraction", {})

        # Skip empty records
        if not data:
            buffer.append(record)
            continue

        # A. Collect verifyable pairs for this doc
        # We recursively find every node that has BOTH 'verbatim_source' and 'value'
        pairs_to_check = []  # List of (source, value)
        field_map = []  # Metadata to map results back to JSON: (path_list, None)

        def recurse_collect(item, path):
            if isinstance(item, dict):
                # Is this an Evidence Node? (Job B ensures we only have valid sources here)
                if "verbatim_source" in item and "value" in item:
                    val = item["value"]
                    src = item["verbatim_source"]

                    # Only verify if we have data (non-null)
                    if val is not None and src is not None:
                        pairs_to_check.append((src, val))
                        field_map.append((path, None))

                # Recurse deeper
                for k, v in item.items():
                    if k != "verbatim_source":  # Don't recurse into strings
                        recurse_collect(v, path + [k])

            elif isinstance(item, list):
                for i, sub in enumerate(item):
                    recurse_collect(sub, path + [i])

        recurse_collect(data, [])

        # B. Run Inference (If there is anything to check)
        if pairs_to_check:
            # This calls the module we just tested
            results = checker.verify_batch(pairs_to_check)

            # C. Apply Results (The "Fact Check Nuke")
            hallucination_count = 0

            for (path, _), res in zip(field_map, results):
                if res["status"] != "PASS":
                    # HALLUCINATION DETECTED by MiniCheck
                    hallucination_count += 1

                    # Navigate to the item in the dict and Nuke it
                    target = data
                    # Traverse to parent
                    for p in path[:-1]:
                        target = target[p]

                    # Nuke the specific field
                    last_key = path[-1]
                    if isinstance(target, dict) and last_key in target:
                        target[last_key]["value"] = None
                        target[last_key]["verbatim_source"] = None

            # Record stats
            record["fact_check_stats"] = {
                "checked": len(pairs_to_check),
                "failed": hallucination_count,
            }

        # Save result (whether we modified it or not)
        buffer.append(record)

        # Incremental Save
        if len(buffer) >= CONFIG["save_interval"]:
            _save_chunk(buffer, CONFIG["output_file"])
            buffer = []

    # Final Save
    if buffer:
        _save_chunk(buffer, CONFIG["output_file"])

    logger.info("Fact-Checking Complete.")


def _save_chunk(data, path):
    with open(path, "a") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")


if __name__ == "__main__":
    main()
