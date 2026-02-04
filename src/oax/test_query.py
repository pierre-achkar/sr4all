"""
OpenAlex Boolean Export Engine (High-Performance)
"""
import os
import gc
from pathlib import Path

# --- 1. JAVA CONFIG ---
PROJECT_ROOT = "/home/fhg/pie65738/projects/sr4all" 
JDK_PATH = os.path.join(PROJECT_ROOT, "jdk-21.0.10") 
os.environ["JAVA_HOME"] = JDK_PATH
os.environ["PATH"] = f"{JDK_PATH}/bin:{os.environ['PATH']}"

# --- 2. IMPORTS ---
from pyserini.search.lucene import LuceneSearcher

# --- CONFIG ---
INDEX_PATH = "data/raw_openalex/indices/lucene-works"
OUTPUT_FILE = "data/filtered/boolean_results.txt"

# Search Logic
# Grouping everything under tiab:() ensures no term leaks to default fields
QUERY_LOGIC = '"B12 Vitamin" AND "Bone Mineral Density" AND "Risk of Fractures"'
START_YEAR = 2025
END_YEAR = 1900 
MAX_INT = 2147483647 # Lucene's absolute limit (Uncapped)

def main():
    print(f"Loading Index: {INDEX_PATH}")
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"Index not found at {INDEX_PATH}")
        
    searcher = LuceneSearcher(INDEX_PATH)
    
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target_years = range(START_YEAR, END_YEAR - 1, -1)
    total_found = 0

    print(f"Executing Boolean Export...")
    print(f"Logic: tiab:({QUERY_LOGIC})")
    print("-" * 50)

    with open(output_path, "w", encoding="utf-8") as f:
        for year in target_years:
            # Construct strict boolean query
            # We use the index-level field name 'publication_year'
            query = f"tiab:({QUERY_LOGIC}) AND publication_year:{year}"
            
            try:
                # k=MAX_INT turns this into a full index dump for the query
                # searcher.search is efficient if we don't request the document body
                hits = searcher.search(query, k=MAX_INT)
                
                count = len(hits)
                if count > 0:
                    # Extract IDs only - this is fast
                    # docid is already the string OpenAlex ID from your indexer
                    ids = [hit.docid for hit in hits]
                    f.write("\n".join(ids) + "\n")
                    
                    total_found += count
                    print(f"[{year}] Found {count:6,} docs")
                
                # Cleanup per year to keep RAM stable
                del hits
                gc.collect()
                
            except Exception as e:
                print(f"Error in year {year}: {str(e)}")

    print("-" * 50)
    print(f"DONE. Total IDs Extracted: {total_found:,}")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()