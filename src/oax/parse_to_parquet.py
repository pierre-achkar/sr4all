"""
OpenAlex Snapshot Parser (Works -> Parquet)

Description:
Reads raw OpenAlex Works snapshot files (.gz), extracts critical fields 
(ID, Title, Year, Abstract), reconstructs the abstract text from the 
inverted index, and saves to Parquet format.

Hardware Note:
This is CPU-bound (decompression + JSON parsing). It uses ProcessPoolExecutor 
to saturate available cores.
"""

import gzip
import json
import logging
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

# ==========================================
# CONFIGURATION
# ==========================================
# Dynamic paths based on where you run the script
BASE_DIR = Path.cwd()

CONFIG = {
    # Input: recursively searches for .gz files inside this folder
    "input_dir": BASE_DIR / "data" / "raw_openalex" / "works",
    
    # Output: Flat folder for Parquet files
    "output_dir": BASE_DIR / "data" / "parquet_works",
    
    # Number of CPU processes to use (Leave 8-16 for your cluster)
    "max_workers": int(os.environ.get("SLURM_CPUS_PER_TASK", 16)),
    
    # Batch size not strictly needed as we process 1 file -> 1 file
    # but kept for reference if logic changes
    "batch_size": 50_000,
}

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("OAXParser")


def reconstruct_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
    """
    Reconstructs abstract text from OpenAlex inverted index.
    Input: {'The': [0], 'cat': [1]} -> Output: "The cat"
    """
    if not inverted_index:
        return ""
    
    # Flatten the index: [ (0, "The"), (1, "cat") ]
    tokens = []
    for word, positions in inverted_index.items():
        for pos in positions:
            tokens.append((pos, word))
    
    # Sort by position to restore sentence order
    tokens.sort(key=lambda x: x[0])
    
    # Join with spaces
    return " ".join([t[1] for t in tokens])


def process_single_file(file_path: Path, output_dir: Path) -> str:
    """
    Worker function to process a single .gz file.
    """
    # CRITICAL FIX: Include parent folder name in output filename 
    # to avoid overwriting files (e.g. updated_date=2024..._part_001.parquet)
    unique_name = f"{file_path.parent.name}_{file_path.name.replace('.gz', '')}.parquet"
    output_filename = output_dir / unique_name
    
    if output_filename.exists():
        return f"SKIP (Exists): {unique_name}"

    # Buffers
    ids = []
    titles = []
    years = []
    abstracts = []
    
    count = 0
    
    try:
        # Stream the GZ file line by line
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # --- Extraction Logic ---
                doc_id = doc.get("id")
                
                # Basic validity check: Must have ID
                if not doc_id: 
                    continue

                # Reconstruct Abstract
                inv_index = doc.get("abstract_inverted_index")
                abstract_text = reconstruct_abstract(inv_index)

                # Append to Lists
                ids.append(doc_id)
                titles.append(doc.get("title") or "")
                years.append(doc.get("publication_year"))
                abstracts.append(abstract_text)
                
                count += 1

        # --- Write to Parquet ---
        if count > 0:
            # Create a PyArrow Table
            table = pa.Table.from_pydict({
                "id": ids,
                "title": titles,
                "publication_year": years,
                "abstract": abstracts
            })
            
            # Write with compression (ZSTD is fast and small)
            pq.write_table(table, output_filename, compression='zstd')
            return f"DONE: {unique_name} ({count} records)"
        else:
            return f"EMPTY: {unique_name}"

    except Exception as e:
        return f"ERROR: {unique_name} - {str(e)}"


def main():
    start_time = time.time()
    
    input_dir = CONFIG["input_dir"]
    output_dir = CONFIG["output_dir"]
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Scanning for .gz files in: {input_dir}")
    
    # Find all .gz files recursively 
    files = list(input_dir.rglob("*.gz"))
    
    if not files:
        logger.error(f"No .gz files found. Is the path correct?")
        return

    logger.info(f"Found {len(files)} files. Starting processing pool...")

    # --- Parallel Execution ---
    processed_count = 0
    with ProcessPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
        # Submit all jobs
        future_to_file = {
            executor.submit(process_single_file, f, output_dir): f 
            for f in files
        }
        
        # Monitor progress
        for future in as_completed(future_to_file):
            result_msg = future.result()
            processed_count += 1
            
            # Log every 20 files to keep output clean but visible
            if processed_count % 20 == 0:
                logger.info(f"[{processed_count}/{len(files)}] Last: {result_msg}")

    elapsed = time.time() - start_time
    logger.info(f"Finished. Processed {len(files)} files in {elapsed / 60:.2f} minutes.")


if __name__ == "__main__":
    main()