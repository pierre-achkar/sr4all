"""
Job B: Alignment and Verification of Extracted Candidates
- Reads raw extractions from Job A
- Loads the original text files based on file paths
- Uses a custom AlignmentVerifier to check if the extracted information matches the text
- Saves the aligned and verified candidates to a new JSONL file for Job C to process
"""

import sys
import json
import logging
import time
import multiprocessing
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

# Ensure we can import src
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from extraction.verifier import AlignmentVerifier

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CONFIG = {
    "input_file": Path(
        "/data/sr4all/extraction_v1/repaired/repaired_raw_candidates_0.jsonl"
    ),
    "output_file": Path(
        "/data/sr4all/extraction_v1/repaired_aligned/aligned_repaired_candidates_0.jsonl"
    ),
    "log_file": Path("/logs/extraction/repaired_alignment_0.log"),
    # Verification Settings
    "threshold": 70,  # Low threshold for noisy OCR
    "min_quote_len": 5,  # Ignore quotes shorter than this (exact match only)
    # Execution
    "processes": max(1, multiprocessing.cpu_count() - 2),  # Leave 2 cores for system
    "chunk_size": 100,
}

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(CONFIG["log_file"]), logging.StreamHandler()],
)
logger = logging.getLogger("JobB")


# -----------------------------------------------------------------------------
# WORKER FUNCTION (Must be top-level for Multiprocessing)
# -----------------------------------------------------------------------------
def process_single_record(record: Dict) -> Dict:
    """
    Worker function to verify a single document.
    """
    doc_id = record.get("doc_id")
    file_path_str = record.get("file_path")
    data = record.get("extraction")

    # Skip if extraction failed previously
    if not data or record.get("error"):
        record["verification"] = {"status": "SKIPPED", "reason": "No extraction data"}
        return record

    try:
        # 1. Load Text
        path = Path(file_path_str)
        if not path.exists():
            record["verification"] = {"status": "ERROR", "reason": "File not found"}
            return record

        text = path.read_text(encoding="utf-8", errors="replace")

        # 2. Verify
        # Initialize verifier inside worker to be safe with fork/spawn contexts
        verifier = AlignmentVerifier(
            threshold=CONFIG["threshold"], min_len=CONFIG["min_quote_len"]
        )

        # This returns the VerificationResult object (is_valid, score, errors, cleaned_data)
        result = verifier.verify(data, text)

        # 3. Attach Results
        record["verification"] = {
            "status": "PASS" if result.is_valid else "FAIL",
            "score": result.score,
            "errors": result.errors,
            "threshold_used": CONFIG["threshold"],
        }

        # IMPORTANT: Replace the raw extraction with the CLEANED version
        # (Where invalid fields are set to null)
        # We keep the original in 'raw_extraction' if you ever want to debug, or just overwrite it.
        # Here we overwrite 'extraction' to ensure downstream steps only use valid data.
        record["extraction"] = result.cleaned_data

        return record

    except Exception as e:
        record["verification"] = {"status": "CRITICAL_ERROR", "reason": str(e)}
        return record


# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
def main():
    input_path = CONFIG["input_file"]
    output_path = CONFIG["output_file"]

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    # 1. Load Input Data
    logger.info(f"Loading extracted candidates from {input_path}...")
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except:
                pass

    # 2. Check Resume Status
    completed_ids = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    completed_ids.add(d.get("doc_id"))
                except:
                    pass
        logger.info(f"Resuming: Found {len(completed_ids)} already verified.")

    to_process = [r for r in records if r["doc_id"] not in completed_ids]

    if not to_process:
        logger.info("All records verified. Exiting.")
        return

    logger.info(
        f"Starting verification on {len(to_process)} docs using {CONFIG['processes']} cores..."
    )

    # 3. Multiprocessing Loop
    start_time = time.perf_counter()
    buffer = []

    # Use imap_unordered for better memory efficiency with large lists
    with multiprocessing.Pool(processes=CONFIG["processes"]) as pool:
        iterator = pool.imap_unordered(process_single_record, to_process, chunksize=10)

        for result in tqdm(iterator, total=len(to_process)):
            buffer.append(result)

            # Incremental Save
            if len(buffer) >= CONFIG["chunk_size"]:
                _save_chunk(buffer, output_path)
                buffer = []

    # Final Save
    if buffer:
        _save_chunk(buffer, output_path)

    duration = time.perf_counter() - start_time
    logger.info(
        f"Alignment Complete. Verified {len(to_process)} docs in {duration:.2f}s."
    )


def _save_chunk(data: List[Dict], filepath: Path):
    with open(filepath, "a", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
