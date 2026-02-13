"""
Job A: Extraction of Candidate Information from Raw Texts
- Reads raw text files based on a manifest (Parquet)
- Uses a Qwen3-32B model on H100s to extract structured information
- Saves raw extractions to JSONL for Job B to process
"""

import sys
import json
import logging
import time
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any

# Ensure we can import src
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from extraction.inference_engine_batch import QwenInference

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CONFIG = {
    # Manifest
    "input_parquet": Path("/data/sr4all/unprocessed_part2.parquet"),
    # Output
    "output_dir": Path("/data/sr4all/extraction_v1"),
    # Model
    "model_path": "Qwen/Qwen3-32B",
    "tensor_parallel": 2,
    # Performance Settings
    "batch_size": 50,  # Number of docs to process in one GPU call
}

# Setup Logging
CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(CONFIG["output_dir"] / "job_a_extraction_2.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("JobA")


# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
def main():
    output_file = CONFIG["output_dir"] / "raw_candidates_2.jsonl"

    # 1. Load Manifest
    logger.info(f"Loading manifest from {CONFIG['input_parquet']}...")
    if not CONFIG["input_parquet"].exists():
        logger.error("Input Parquet not found!")
        return

    df = pd.read_parquet(CONFIG["input_parquet"])

    # Sort by token count to minimize padding (faster batching)
    if "token_count" in df.columns:
        logger.info("Sorting by token count for efficiency...")
        df = df.sort_values("token_count", ascending=True)

    all_records = df.to_dict(orient="records")
    total_docs = len(all_records)
    logger.info(f"Manifest contains {total_docs} documents.")

    # 2. Check Resume Status
    completed_ids = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    completed_ids.add(str(data.get("doc_id") or data.get("id")))
                except:
                    pass
        logger.info(f"Resuming: Found {len(completed_ids)} already processed.")

    # Filter Work
    to_process = [r for r in all_records if str(r["doc_id"]) not in completed_ids]

    if not to_process:
        logger.info("All documents processed. Exiting.")
        return

    # 3. Initialize Engine
    logger.info("Initializing H100 Engine (Batch Mode)...")
    try:
        engine = QwenInference(
            CONFIG["model_path"], tensor_parallel=CONFIG["tensor_parallel"]
        )
    except Exception as e:
        logger.critical(f"Failed to load engine: {e}")
        return

    # 4. Batch Processing Loop
    batch_size = CONFIG["batch_size"]
    total_remaining = len(to_process)

    logger.info(
        f"Starting extraction on {total_remaining} docs in batches of {batch_size}..."
    )
    start_time = time.perf_counter()

    # We iterate through the list in chunks
    for i in tqdm(range(0, total_remaining, batch_size), desc="Processing Batches"):
        batch_records = to_process[i : i + batch_size]

        # A. IO Phase: Load texts for this batch
        valid_records = []  # Records that successfully loaded
        valid_texts = []  # Texts for the LLM
        error_results = []  # Failed loads (save immediately)

        for record in batch_records:
            doc_id = str(record.get("doc_id"))
            file_path_str = record.get("file_path")

            try:
                path = Path(file_path_str)
                if not path.exists():
                    error_results.append(
                        _create_error(doc_id, file_path_str, "FILE_NOT_FOUND")
                    )
                    continue

                text = path.read_text(encoding="utf-8", errors="replace")
                if not text.strip():
                    error_results.append(
                        _create_error(doc_id, file_path_str, "EMPTY_TEXT")
                    )
                    continue

                valid_records.append(record)
                valid_texts.append(text)

            except Exception as e:
                error_results.append(
                    _create_error(doc_id, file_path_str, f"READ_ERROR: {e}")
                )

        # B. Compute Phase: Run Batch Inference
        if valid_texts:
            llm_results = engine.generate_batch(valid_texts)

            # C. Merge Phase: Combine records with results
            success_results = []
            for record, result in zip(valid_records, llm_results):
                doc_id = str(record.get("doc_id"))

                entry = {
                    "doc_id": doc_id,
                    "file_path": record.get("file_path"),
                    "extraction": result["parsed"],  # Dict or None
                    "raw_output": result["raw"],  # String
                    "timestamp": time.time(),
                }

                if result["error"]:
                    entry["error"] = result["error"]
                    logger.warning(f"Doc {doc_id} failed generation: {result['error']}")

                success_results.append(entry)

            # Combine successes and IO errors
            batch_output = success_results + error_results
        else:
            # Whole batch failed IO
            batch_output = error_results

        # D. Save Phase
        if batch_output:
            _save_chunk(batch_output, output_file)

    duration = time.perf_counter() - start_time
    logger.info(
        f"Extraction Complete. Processed {total_remaining} docs in {duration:.2f}s."
    )


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def _create_error(doc_id: str, file_path: str, msg: str) -> Dict:
    return {
        "doc_id": doc_id,
        "file_path": file_path,
        "extraction": None,
        "error": msg,
        "timestamp": time.time(),
    }


def _save_chunk(data: List[Dict], filepath: Path):
    """Appends data to JSONL."""
    with open(filepath, "a", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
