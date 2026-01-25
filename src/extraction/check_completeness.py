"""
Completeness Checker.

Scans the corpus to see how many documents are "methodologically complete".

Definition of Valid Field:
- Not None.
- Not an empty list [].
- Not a "ghost" object (e.g., [{"boolean_query_string": null}]).

Definition of Complete Document:
1. Has Objective.
2. Has Search Strategy (Boolean Queries OR Keywords).
3. Has Criteria (Inclusion OR Exclusion).
"""

import json
from pathlib import Path
from collections import Counter

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
INPUT_FILE = Path("/home/fhg/pie65738/projects/sr4all/data/sr4all/extraction_v1_old/fact_checked_repaired_corpus_0.jsonl")

def is_filled(field_data):
    """
    Checks if a field has valid content.
    Returns True if data exists, False if it is effectively empty/null.
    """
    if field_data is None: 
        return False
    
    # Case 1: Evidence Object {"value": ...} (Standard fields)
    if isinstance(field_data, dict):
        val = field_data.get("value")
        if val is None: 
            return False
        if isinstance(val, list) and len(val) == 0:
            return False
        return True
        
    # Case 2: List of Objects (Exact Boolean Queries)
    if isinstance(field_data, list):
        if not field_data: 
            return False # Empty list []
        
        # Check for ghost object: [{"boolean_query_string": null, ...}]
        first_item = field_data[0]
        if isinstance(first_item, dict):
            # It is ONLY valid if the query string is NOT null
            if first_item.get("boolean_query_string") is None:
                return False
        
        return True

    return False

def main():
    if not INPUT_FILE.exists():
        print(f"Error: Input file not found at {INPUT_FILE}")
        return

    print(f"Scanning: {INPUT_FILE.name}...")
    
    total_docs = 0
    stats = Counter()
    
    # Logic Group Counters
    has_objective = 0
    has_search = 0
    has_criteria = 0
    fully_complete = 0

    with open(INPUT_FILE, "r") as f:
        for line in f:
            try:
                rec = json.loads(line)
                data = rec.get("extraction", {})
                
                # If extraction is null, skip
                if not data: continue

                total_docs += 1
                
                # 1. Check Individual Fields using helper
                obj_ok = is_filled(data.get("objective"))
                bool_ok = is_filled(data.get("exact_boolean_queries"))
                key_ok = is_filled(data.get("keywords_used"))
                inc_ok = is_filled(data.get("inclusion_criteria"))
                exc_ok = is_filled(data.get("exclusion_criteria"))

                # 2. Update Stats for individual fields
                if obj_ok: stats["objective"] += 1
                if bool_ok: stats["exact_boolean_queries"] += 1
                if key_ok: stats["keywords_used"] += 1
                if inc_ok: stats["inclusion_criteria"] += 1
                if exc_ok: stats["exclusion_criteria"] += 1

                # 3. Check Logic Groups
                
                # Group A: Objective
                if obj_ok: 
                    has_objective += 1

                # Group B: Search Strategy (Boolean OR Keywords)
                search_group_ok = bool_ok or key_ok
                if search_group_ok:
                    has_search += 1
                
                # Group C: Criteria (Inclusion OR Exclusion)
                criteria_group_ok = inc_ok or exc_ok
                if criteria_group_ok:
                    has_criteria += 1

                # 4. Full Completeness (A + B + C)
                if obj_ok and search_group_ok and criteria_group_ok:
                    fully_complete += 1

            except Exception as e:
                pass

    # --- REPORT ---
    print("\n" + "="*60)
    print(f"COMPLETENESS REPORT (N={total_docs})")
    print("="*60)
    
    print(f"\n{'INDIVIDUAL FIELD':<35} | {'COUNT':<10} | {'%':<6}")
    print("-" * 60)
    
    for k, v in sorted(stats.items()):
        pct = (v / total_docs) * 100
        print(f"{k:<35} | {v:<10} | {pct:.1f}%")
    
    print("-" * 60)
    print(f"\n{'LOGIC GROUP':<35} | {'COUNT':<10} | {'%':<6}")
    print("-" * 60)
    
    print(f"1. Objective (Value present)        | {has_objective:<10} | {(has_objective/total_docs)*100:.1f}%")
    print(f"2. Search (Queries OR Keywords)     | {has_search:<10} | {(has_search/total_docs)*100:.1f}%")
    print(f"3. Criteria (Inclusion OR Exclusion)| {has_criteria:<10} | {(has_criteria/total_docs)*100:.1f}%")
    
    print("="*60)
    print(f"✅ FULLY COMPLETE DOCS (1+2+3)      | {fully_complete:<10} | {(fully_complete/total_docs)*100:.1f}%")
    print("="*60)

if __name__ == "__main__":
    main()