import os
# --- JAVA SETUP ---
project_root = "/home/fhg/pie65738/projects/sr4all" 
jdk_path = os.path.join(project_root, "jdk-21.0.10") 
os.environ["JAVA_HOME"] = jdk_path
os.environ["PATH"] = f"{jdk_path}/bin:{os.environ['PATH']}"
# ------------------

from pyserini.search.lucene import LuceneSearcher

searcher = LuceneSearcher("data/raw_openalex/indices/lucene-works")

# 1. Test a simple term to verify the 'tiab' field exists
# (Should return >0 hits)
print("Testing simple term 'tiab:kidney'...")
hits_simple = searcher.search("tiab:kidney", k=10)
print(f"-> Found {len(hits_simple)} matches for 'kidney'.")

# 2. Test your complex query WITHOUT the year
# (Should return a small, realistic number, e.g., 50-200)
complex_query = "(tiab:oncocytoma OR tiab:oncocytic) AND (tiab:biopsy OR tiab:surveillance) AND (tiab:kidney OR tiab:renal)"
print(f"\nTesting complex query: {complex_query}")
hits_complex = searcher.search(complex_query, k=1000)
print(f"-> Found {len(hits_complex)} matches.")

if len(hits_simple) == 0:
    print("\n[FATAL] 'tiab:kidney' returned 0. The 'tiab' field does not exist in your index.")
    print("Action: Delete the index folder and re-run the Indexer script.")
elif len(hits_complex) > 10000:
     print("\n[FATAL] Query returned too many results. Syntax is still ignoring logic.")
else:
     print("\n[SUCCESS] Index is healthy and query logic is working.")