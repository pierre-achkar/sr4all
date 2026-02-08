import os
import sys

# ==========================================
# 1. JAVA CONFIGURATION (MUST BE FIRST)
# ==========================================
PROJECT_ROOT = "/home/fhg/pie65738/projects/sr4all" 
JDK_PATH = os.path.join(PROJECT_ROOT, "jdk-21.0.10") 

# Verify path exists to avoid confusing errors later
if not os.path.exists(os.path.join(JDK_PATH, "bin", "javac")):
    print(f"FATAL: Java not found at {JDK_PATH}/bin/javac")
    sys.exit(1)

# Set Environment Variables
os.environ["JAVA_HOME"] = JDK_PATH
os.environ["PATH"] = f"{JDK_PATH}/bin:{os.environ['PATH']}"

# ==========================================
# 2. NOW IMPORT PYSERINI
# ==========================================
import json
from pyserini.search.lucene import LuceneSearcher

INDEX_PATH = "data/raw_openalex/indices/lucene-works-test"

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

    # 1. Total Count Verification
    print(f"[INFO] Total Documents in Index: {searcher.num_docs:,}")

    if searcher.num_docs == 0:
        print("STOP: Index is empty.")
        return

    # 2. Schema Inspection (The "Truth")
    print("-" * 50)
    print(" INSPECTING DOCUMENT #0 ")
    print("-" * 50)
    
    try:
        # Fetch the first internal Lucene document (ID 0)
        doc = searcher.doc(0)
        
        # We try to parse the raw content
        raw_content = doc.raw()
        if raw_content:
            try:
                json_content = json.loads(raw_content)
                print(f"Fields Found in JSON: {list(json_content.keys())}")
                
                # Check for Text Fields
                if 'contents' in json_content:
                    print(f" - Field 'contents' found. (Length: {len(json_content['contents'])})")
                elif 'tiab' in json_content:
                    print(f" - Field 'tiab' found. (Length: {len(json_content['tiab'])})")
                
                # Check Year Type
                year_val = json_content.get('publication_year')
                print(f" - Year Value: {year_val} (Type: {type(year_val)})")
            except json.JSONDecodeError:
                print(f"Raw content is not JSON: {raw_content[:100]}...")
        else:
            print("WARNING: Doc #0 has no stored raw content (did you use -storeRaw?)")

    except Exception as e:
        print(f"Error inspecting doc: {e}")

    # 3. Simple Test Query
    print("-" * 50)
    test_q = "education"
    print(f"Test Query: '{test_q}'")
    hits = searcher.search(test_q, k=5)
    print(f"Found {len(hits)} hits for '{test_q}'")
    for h in hits:
        print(f" - {h.docid} ({h.score:.2f})")

if __name__ == "__main__":
    main()