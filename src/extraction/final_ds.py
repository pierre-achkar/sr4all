"""
Final Dataset Flattener & Filter.

Input:  fact_checked_repaired_corpus_0.jsonl
Output: sr4all_final_v1.jsonl

Tasks:
1. FLATTEN: Hoist nested 'extraction' fields to top level.
2. FILTER: Keep ONLY documents that meet the "Fully Complete" standard:
   - Must have Objective.
   - Must have Search Strategy (Boolean Queries OR Keywords).
   - Must have Criteria (Inclusion OR Exclusion).
"""

import json
from pathlib import Path

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
INPUT_FILE = Path("/home/fhg/pie65738/projects/sr4all/data/sr4all/extraction_v1_old/fact_checked_repaired_corpus_0.jsonl")
OUTPUT_FILE = Path("/home/fhg/pie65738/projects/sr4all/data/sr4all/extraction_v1_old/sr4all_final_v1.jsonl")

# -----------------------------------------------------------------------------
# HELPER: VALIDATION LOGIC
# -----------------------------------------------------------------------------
def is_filled(field_data):
    """
    Checks if a field has valid content (not null, not empty list, not ghost object).
    """
    if field_data is None: 
        return False
    
    # Case 1: Evidence Object {"value": ...}
    if isinstance(field_data, dict):
        val = field_data.get("value")
        if val is None: 
            return False
        if isinstance(val, list) and len(val) == 0:
            return False
        return True
        
    # Case 2: List of Objects (Boolean Queries)
    if isinstance(field_data, list):
        if not field_data: 
            return False # Empty list
        
        # Check for ghost object [{"boolean_query_string": null}]
        if isinstance(field_data[0], dict):
            if field_data[0].get("boolean_query_string") is None:
                return False
        return True

    return False

def check_completeness(data):
    """
    Returns True if doc meets the methodological completeness standard.
    """
    if not data: return False

    # 1. Check Fields
    obj_ok  = is_filled(data.get("objective"))
    bool_ok = is_filled(data.get("exact_boolean_queries"))
    key_ok  = is_filled(data.get("keywords_used"))
    inc_ok  = is_filled(data.get("inclusion_criteria"))
    exc_ok  = is_filled(data.get("exclusion_criteria"))

    # 2. Logic Groups
    has_objective = obj_ok
    has_search    = bool_ok or key_ok
    has_criteria  = inc_ok or exc_ok

    # 3. Final Verdict
    return has_objective and has_search and has_criteria

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
def main():
    if not INPUT_FILE.exists():
        print(f"Error: Input file not found at {INPUT_FILE}")
        return

    print(f"Reading from: {INPUT_FILE.name}")
    print(f"Writing to:   {OUTPUT_FILE.name}")

    total_read = 0
    total_saved = 0
    
    with open(INPUT_FILE, "r") as fin, open(OUTPUT_FILE, "w") as fout:
        for line in fin:
            try:
                record = json.loads(line)
                total_read += 1
                
                extraction = record.get("extraction", {})
                
                # --- FILTER STEP ---
                if not check_completeness(extraction):
                    continue # Skip incomplete docs

                # --- FLATTEN STEP ---
                # 1. Start with metadata
                final_record = {
                    "doc_id": record.get("doc_id"),
                    "file_path": record.get("file_path")
                }
                
                # 2. Hoist extracted fields
                if extraction:
                    final_record.update(extraction)
                
                # 3. Save (Clean, no extra stats)
                fout.write(json.dumps(final_record) + "\n")
                total_saved += 1

            except Exception as e:
                print(f"Skipping error line: {e}")

    print("-" * 40)
    print(f"PROCESSING COMPLETE")
    print(f"Total Read:     {total_read}")
    print(f"Filtered Out:   {total_read - total_saved}")
    print(f"Final Dataset:  {total_saved} documents")
    print(f"Saved to:       {OUTPUT_FILE}")
    print("-" * 40)

if __name__ == "__main__":
    main()