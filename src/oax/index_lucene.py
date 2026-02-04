"""
OpenAlex Parquet -> Pyserini (Lucene) Indexer
Status: Production Ready
Fixes: Java 21 Config, EmptyDocumentException, Schema Alignment
"""
import logging
import time
import json
import os
import sys
from pathlib import Path
import pyarrow.parquet as pq

# ==========================================
# 1. JAVA CONFIGURATION (CRITICAL)
# Must run before importing pyserini
# ==========================================
project_root = "/home/fhg/pie65738/projects/sr4all" 
jdk_path = os.path.join(project_root, "jdk-21.0.10") 

# Validation
if not os.path.exists(os.path.join(jdk_path, "bin", "javac")):
    print(f"FATAL: Java not found at {jdk_path}/bin/javac")
    print("Please check the folder name/path.")
    sys.exit(1)

os.environ["JAVA_HOME"] = jdk_path
os.environ["PATH"] = f"{jdk_path}/bin:{os.environ['PATH']}"
# ==========================================

from pyserini.index.lucene import LuceneIndexer

# CONFIGURATION
BASE_DIR = Path.cwd()
INPUT_DIR = BASE_DIR / "data/raw_openalex" / "parquet_works"
INDEX_DIR = BASE_DIR / "data/raw_openalex" / "indices" / "lucene-works"
THREADS = 16 
BATCH_SIZE = 50_000

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("Indexer")

def doc_generator(input_dir):
    """
    Reads Parquet, sanitizes data, and yields Lucene-ready dictionaries.
    """
    files = sorted(list(input_dir.glob("*.parquet")))
    if not files:
        logger.error(f"No .parquet files found in {input_dir}")
        return

    logger.info(f"Found {len(files)} parquet files. Starting ingestion...")

    for file_idx, file_path in enumerate(files):
        try:
            table = pq.read_table(file_path)
            pydict = table.to_pydict()
            
            # Extract columns safely (handle missing columns in schema)
            ids = pydict.get('id', [])
            titles = pydict.get('title', [])
            abstracts = pydict.get('abstract', [])
            years = pydict.get('publication_year', [])
            
            num_rows = len(ids)
            
            for i in range(num_rows):
                doc_id = ids[i]
                if not doc_id: 
                    continue

                # --- SANITIZATION (The Fix for 7.5M Errors) ---
                # Convert None to "" to prevent Java NullPointer/EmptyExceptions
                title = titles[i] if (i < len(titles) and titles[i] is not None) else ""
                abstract = abstracts[i] if (i < len(abstracts) and abstracts[i] is not None) else ""
                year = years[i] if (i < len(years) and years[i] is not None) else 0
                
                # Clean strings
                title = str(title).strip()
                abstract = str(abstract).strip()

                # Construct the TIAB (Title + Abstract) field
                tiab_text = f"{title} {abstract}".strip()
                
                # --- FILTERING ---
                # If a doc has NO text, do not index it. 
                # This prevents "EmptyDocumentException" logs.
                if not tiab_text:
                    continue

                # --- SCHEMA DEFINITION ---
                # We store a lightweight JSON in 'raw' for retrieval
                raw_payload = {
                    "id": str(doc_id),
                    "title": title,
                    "year": int(year)
                }

                yield {
                    "id": str(doc_id),
                    
                    # Searchable Fields
                    "title": title,
                    "abstract": abstract,
                    "tiab": tiab_text,       # PRIMARY SEARCH FIELD
                    "contents": tiab_text,   # Default Fallback Field
                    
                    # Filter Field (Must be Int)
                    "publication_year": int(year),
                    
                    # Stored Data (for display/debugging)
                    "raw": json.dumps(raw_payload, ensure_ascii=False)
                }

        except Exception as e:
            logger.error(f"Failed to process file {file_path.name}: {e}")

def main():
    start_time = time.time()
    
    # 1. Clean Slate Warning
    if INDEX_DIR.exists():
        logger.warning(f"Index folder exists at {INDEX_DIR}.")
        logger.warning("If this run fails or is a restart, ensure you deleted the old folder first!")

    # 2. Initialize Indexer
    try:
        # threads=THREADS enables parallel indexing in Java
        indexer = LuceneIndexer(str(INDEX_DIR), threads=THREADS)
    except Exception as e:
        logger.error(f"Failed to start Lucene Indexer. Check Java Config. Error: {e}")
        return
    
    logger.info(f"Writing index to: {INDEX_DIR}")
    
    count = 0
    valid_docs = 0
    
    # 3. Stream & Index
    for doc in doc_generator(INPUT_DIR):
        indexer.add_doc_dict(doc)
        valid_docs += 1
        
        # Periodic Status Update
        if valid_docs % 100_000 == 0:
            logger.info(f"Indexed {valid_docs:,} active documents...")

    # 4. Finalize
    logger.info("Committing and optimizing index (Waiting for Java)...")
    indexer.close()
    
    duration = (time.time() - start_time) / 60
    logger.info(f"DONE. Indexed {valid_docs:,} works in {duration:.2f} minutes.")

if __name__ == "__main__":
    main()