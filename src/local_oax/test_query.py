"""
OpenAlex Retrieval Script
Function: Runs a Boolean Query + Year Filter -> Exports ALL matching IDs to a file.
"""
import os
import sys

# ==========================================
# 1. JAVA CONFIGURATION (MUST BE FIRST)
# ==========================================
PROJECT_ROOT = "/home/fhg/pie65738/projects/sr4all" 
JDK_PATH = os.path.join(PROJECT_ROOT, "jdk-21.0.10") 

if not os.path.exists(os.path.join(JDK_PATH, "bin", "javac")):
    print(f"FATAL: Java not found at {JDK_PATH}")
    sys.exit(1)

os.environ["JAVA_HOME"] = JDK_PATH
os.environ["PATH"] = f"{JDK_PATH}/bin:{os.environ['PATH']}"

# ==========================================
# 2. IMPORTS
# ==========================================
from pyserini.search.lucene import LuceneSearcher
from pyserini.pyclass import autoclass

# CONFIGURATION
# Point this to your TEST index first, then switch to the real one later
INDEX_PATH = "data/raw_openalex/indices/lucene-works-test" 
OUTPUT_FILE = "retrieved_ids.txt"

# QUERY PARAMETERS
# Note: 'contents' is the default field.
QUERY_TEXT = '("machine learning")'
YEAR_START = 2000
YEAR_END = 2025

def main():
    if not os.path.exists(INDEX_PATH):
        print(f"FATAL: Index path does not exist: {INDEX_PATH}")
        return

    print(f"[INIT] Loading Index from {INDEX_PATH}...")
    try:
        searcher = LuceneSearcher(INDEX_PATH)
    except Exception as e:
        print(f"FATAL: Failed to load index. Error: {e}")
        return

    # ---------------------------------------------------------
    # A. SETUP JAVA QUERY PARSER
    # ---------------------------------------------------------
    # We use the Java parser directly to ensure IntPoint range queries work perfectly.
    JQueryParser = autoclass('org.apache.lucene.queryparser.classic.QueryParser')
    JAnalyzer = autoclass('org.apache.lucene.analysis.en.EnglishAnalyzer')
    
    # Parse queries against the 'contents' field by default
    parser = JQueryParser("contents", JAnalyzer())
    
    # ---------------------------------------------------------
    # B. CONSTRUCT QUERY
    # ---------------------------------------------------------
    # Syntax: (TEXT_QUERY) AND publication_year:[START TO END]
    # IMPORTANT: Lucene Range Syntax relies on the IntPoint field we just built.
    full_query_str = f"({QUERY_TEXT}) AND publication_year:[{YEAR_START} TO {YEAR_END}]"
    
    print("-" * 50)
    print(f"[QUERY] {full_query_str}")
    print("-" * 50)

    try:
        lucene_query = parser.parse(full_query_str)
        
        # ---------------------------------------------------------
        # C. COUNT TOTAL HITS (Uncapped)
        # ---------------------------------------------------------
        # We ask Lucene "How many documents match?" (O(1) lookup)
        total_hits = searcher.object.searcher.count(lucene_query)
        
        print(f"[RESULT] Found {total_hits:,} matching documents.")
        
        if total_hits == 0:
            print("[WARN] No documents found. Verify your year range matches the data in the test file.")
            return

        # ---------------------------------------------------------
        # D. RETRIEVE & SAVE
        # ---------------------------------------------------------
        print(f"[IO] Fetching {total_hits:,} IDs and saving to {OUTPUT_FILE}...")
        
        # Fetch ALL results (k=total_hits)
        hits = searcher.search(lucene_query, k=total_hits)
        
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for i, hit in enumerate(hits):
                # hit.docid matches the StringField "id" we indexed
                f.write(f"{hit.docid}\n")
                
                if (i + 1) % 10_000 == 0:
                    print(f"  -> Written {i + 1:,} IDs...")

        print("-" * 50)
        print(f"[SUCCESS] Done. Saved to {OUTPUT_FILE}")

    except Exception as e:
        print(f"\n[ERROR] Search Failed: {e}")

if __name__ == "__main__":
    main()