"""
Job A: Batch Extraction (GPU).

Input:  Clean Corpus Parquet (Manifest with file_paths).
Output: raw_candidates.jsonl
"""

import sys
import json
import logging
import time
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from typing import List, Dict

# Ensure we can import src
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from extraction.inference_engine import QwenInference

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CONFIG = {
    # The Parquet is your "Manifest" - it tells us WHAT to process
    "input_parquet": Path("/home/fhg/pie65738/projects/sr4all/data/sr4all/clean_corpus.parquet"),
    
    # We append all results to this one file
    "output_dir": Path("/home/fhg/pie65738/projects/sr4all/data/sr4all/extraction_v1"),
    
    # Model
    "model_path": "Qwen/Qwen3-32B", 
    "tensor_parallel": 2,
    
    # Save to disk every N docs to prevent data loss on crash
    "chunk_size": 50,  
}

# Setup Logging
CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(CONFIG["output_dir"] / "job_a_extraction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("JobA")

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
def main():
    output_file = CONFIG["output_dir"] / "raw_candidates.jsonl"
    
    # 1. Load Manifest (Parquet)
    logger.info(f"Loading manifest from {CONFIG['input_parquet']}...")
    if not CONFIG['input_parquet'].exists():
        logger.error("Input Parquet not found!")
        return

    df = pd.read_parquet(CONFIG['input_parquet'])
    
    # Optional: Sort by token count for efficient vLLM processing (shorter first usually warms up faster)
    if "token_count" in df.columns:
        df = df.sort_values("token_count", ascending=True)
        
    all_records = df.to_dict(orient="records")
    logger.info(f"Manifest contains {len(all_records)} documents.")

    # 2. Check Resume Status (The JSONL Advantage)
    # We just read the existing JSONL to see which IDs are done.
    completed_ids = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # Support both "doc_id" and "id" depending on your schema
                    completed_ids.add(data.get("doc_id") or data.get("id"))
                except: pass
        logger.info(f"Resuming: Found {len(completed_ids)} already processed.")
    
    # Filter remaining work
    to_process = [r for r in all_records if str(r["doc_id"]) not in completed_ids]
    
    if not to_process:
        logger.info("All documents processed. Exiting.")
        return

    # 3. Initialize Engine
    logger.info("Initializing H100 Engine...")
    engine = QwenInference(CONFIG["model_path"], tensor_parallel=CONFIG["tensor_parallel"])

    # 4. Processing Loop
    buffer = []
    start_time = time.perf_counter()
    
    logger.info(f"Starting extraction on {len(to_process)} remaining documents...")
    
    for i, record in enumerate(tqdm(to_process)):
        doc_id = str(record.get("doc_id"))
        file_path_str = record.get("file_path")
        
        # A. Load Text from Disk
        try:
            file_path = Path(file_path_str)
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                buffer.append({"doc_id": doc_id, "error": "FILE_NOT_FOUND"})
                continue
                
            text = file_path.read_text(encoding="utf-8", errors="replace")
            
            # Sanity check on empty files
            if not text.strip():
                buffer.append({"doc_id": doc_id, "error": "EMPTY_TEXT"})
                continue

        except Exception as e:
            logger.error(f"Read error on {doc_id}: {e}")
            buffer.append({"doc_id": doc_id, "error": f"READ_ERROR: {str(e)}"})
            continue
            
        # B. Generate
        try:
            # Returns (parsed_dict, raw_str)
            result_json, raw_text = engine.generate(text)
            
            # Construct Result Record
            output_entry = {
                "doc_id": doc_id,
                "file_path": file_path_str,
                "extraction": result_json,
                # Only save raw output if extraction failed (to save space), or save always if you prefer debuggability
                #"raw_output": raw_text if not result_json else None, 
                "raw_output": raw_text,
                "timestamp": time.time()
            }
            
            if not result_json:
                logger.error(f"Failed to parse JSON for {doc_id}")
                output_entry["error"] = "JSON_PARSE_FAIL"
                
            buffer.append(output_entry)

        except Exception as e:
            logger.error(f"Inference error on {doc_id}: {e}")
            buffer.append({"doc_id": doc_id, "error": str(e)})

        # C. Incremental Save (The Safety Net)
        if len(buffer) >= CONFIG["chunk_size"]:
            _save_chunk(buffer, output_file)
            buffer = []

    # Final Save
    if buffer:
        _save_chunk(buffer, output_file)
        
    duration = time.perf_counter() - start_time
    logger.info(f"Job A Complete. Processed {len(to_process)} docs in {duration:.2f}s.")

def _save_chunk(data: List[Dict], filepath: Path):
    """Appends data to JSONL."""
    with open(filepath, "a", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    main()