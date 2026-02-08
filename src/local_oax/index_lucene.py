"""
This script performs the indexing of OpenAlex works data into a Lucene index using a hybrid Python-Java approach. 
It reads Parquet files containing works metadata, extracts relevant fields, 
and constructs Lucene documents with both stored and indexed fields. 
The script is designed for production use, with robust error handling, logging, 
and performance optimizations to handle large datasets efficiently.
"""
import logging
import time
import os
import shutil
import json
import sys
from pathlib import Path
import pyarrow.parquet as pq

# ==========================================
# 1. JAVA CONFIGURATION
# ==========================================
PROJECT_ROOT = "/home/fhg/pie65738/projects/sr4all" 
JDK_PATH = os.path.join(PROJECT_ROOT, "jdk-21.0.10") 

if not os.path.exists(os.path.join(JDK_PATH, "bin", "javac")):
    print(f"FATAL: Java not found at {JDK_PATH}")
    sys.exit(1)

os.environ["JAVA_HOME"] = JDK_PATH
os.environ["PATH"] = f"{JDK_PATH}/bin:{os.environ['PATH']}"

# ==========================================
# 2. HYBRID IMPORT
# ==========================================
from pyserini.index.lucene import LuceneIndexer
from jnius import autoclass

# CONFIGURATION
BASE_DIR = Path.cwd()
INPUT_DIR = BASE_DIR / "data/raw_openalex" / "parquet_works"
INDEX_DIR = BASE_DIR / "data/raw_openalex" / "indices" / "lucene-works" # <--- REAL PRODUCTION PATH
THREADS = 16 
MEMORY_BUFFER_MB = 16384 # 16GB Heap for massive indexing

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("ProductionIndexer")

# ---------------------------------------------------------
# JAVA CLASS LOADERS
# ---------------------------------------------------------
JPaths = autoclass('java.nio.file.Paths')
JFSDirectory = autoclass('org.apache.lucene.store.FSDirectory')
JEnglishAnalyzer = autoclass('org.apache.lucene.analysis.en.EnglishAnalyzer')
JIndexWriterConfig = autoclass('org.apache.lucene.index.IndexWriterConfig')
JOpenMode = autoclass('org.apache.lucene.index.IndexWriterConfig$OpenMode')

JIndexWriter = autoclass('org.apache.lucene.index.IndexWriter')
JDocument = autoclass('org.apache.lucene.document.Document')

# Field Types
JString = autoclass('java.lang.String')
JStringField = autoclass('org.apache.lucene.document.StringField')
JTextField = autoclass('org.apache.lucene.document.TextField')
JIntPoint = autoclass('org.apache.lucene.document.IntPoint')
JStoredField = autoclass('org.apache.lucene.document.StoredField')
JFieldStore = autoclass('org.apache.lucene.document.Field$Store')
JBinaryDocValuesField = autoclass('org.apache.lucene.document.BinaryDocValuesField') 
JBytesRef = autoclass('org.apache.lucene.util.BytesRef')

def doc_generator(input_dir):
    all_files = sorted(list(input_dir.glob("*.parquet")))
    if not all_files:
        logger.error(f"No .parquet files found in {input_dir}")
        return

    # PRODUCTION: Process ALL files
    logger.info(f"PRODUCTION MODE: Starting ingestion of {len(all_files)} files...")

    for file_path in all_files:
        try:
            table = pq.read_table(file_path)
            pydict = table.to_pydict()
            
            ids = pydict.get('id', [])
            titles = pydict.get('title', [])
            abstracts = pydict.get('abstract', [])
            years = pydict.get('publication_year', [])
            
            # Vectorized length check
            num_rows = len(ids)
            
            for i in range(num_rows):
                doc_id = ids[i]
                if not doc_id: continue

                title = str(titles[i] or "").strip()
                abstract = str(abstracts[i] or "").strip()
                year = years[i] if years[i] is not None else 0
                tiab_text = f"{title} {abstract}".strip()
                
                if not tiab_text: continue

                doc_dict = {
                    "id": str(doc_id),
                    "contents": tiab_text,
                    "publication_year": int(year)
                }

                yield doc_dict

        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")

def main():
    start_time = time.time()
    
    # Safety Check: If index exists, warn user.
    if INDEX_DIR.exists():
        logger.warning(f"Index folder exists at {INDEX_DIR}.")
        logger.warning("Assuming you have CLEARED this folder if you want a fresh start.")
        # We do NOT auto-delete in production to avoid accidents, 
        # but the Java OpenMode.CREATE will overwrite segments.
        
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Initializing Production IndexWriter at {INDEX_DIR}")

    try:
        path = JPaths.get(str(INDEX_DIR))
        directory = JFSDirectory.open(path)
        analyzer = JEnglishAnalyzer() 
        config = JIndexWriterConfig(analyzer)
        config.setOpenMode(JOpenMode.CREATE) 
        config.setRAMBufferSizeMB(float(MEMORY_BUFFER_MB)) 
        
        writer = JIndexWriter(directory, config)
    except Exception as e:
        logger.error(f"Failed to create IndexWriter: {e}")
        return

    valid_docs = 0
    try:
        for doc_dict in doc_generator(INPUT_DIR):
            doc = JDocument()
            
            doc_id = doc_dict["id"]
            text = doc_dict["contents"]
            year = doc_dict["publication_year"]
            
            # 1. ID (Stored String)
            doc.add(JStringField("id", doc_id, JFieldStore.YES))
            
            # 2. ID (Binary DocValues)
            java_id = JString(doc_id)
            doc.add(JBinaryDocValuesField("id", JBytesRef(java_id)))
            
            # 3. Raw JSON (Stored)
            doc.add(JStoredField("raw", json.dumps(doc_dict)))
            
            # 4. Contents (Text + Positions)
            doc.add(JTextField("contents", text, JFieldStore.YES))
            
            # 5. Year (IntPoint + Stored)
            doc.add(JIntPoint("publication_year", year))
            doc.add(JStoredField("publication_year", year))
            
            writer.addDocument(doc)
            valid_docs += 1
            
            if valid_docs % 100_000 == 0:
                logger.info(f"Indexed {valid_docs:,} documents...")
                
    except Exception as e:
        logger.error(f"Indexing crashed: {e}")
    
    logger.info("Committing and closing...")
    writer.commit()
    writer.close()
    directory.close()
    
    duration = (time.time() - start_time) / 60
    logger.info(f"DONE. Indexed {valid_docs:,} docs in {duration:.2f} minutes.")

if __name__ == "__main__":
    main()