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
import logging
import re

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
INPUT_FILE = Path("/home/fhg/pie65738/projects/sr4all/data/sr4all/extraction_v1/repaired_fact_checked/repaired_fact_checked_corpus_all.jsonl")
OUTPUT_FILE = Path("/home/fhg/pie65738/projects/sr4all/data/sr4all/extraction_v1/intermediate/sr4all_intermediate_all.jsonl")
LOGGING_FILE = Path("/home/fhg/pie65738/projects/sr4all/logs/final_ds/intermediate_dataset_flattener_all.log")

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOGGING_FILE, mode="w"),
        logging.StreamHandler()
    ]
)

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

_PLACEHOLDER_ONLY_RE = re.compile(r"^(?:#?\d+|AND|OR|NOT|\(|\)|\s)+$", re.IGNORECASE)

def is_placeholder_only(query: str) -> bool:
    if not query or not isinstance(query, str):
        return False
    return _PLACEHOLDER_ONLY_RE.fullmatch(query.strip()) is not None

def has_only_placeholder_queries(field_data) -> bool:
    if not isinstance(field_data, list) or not field_data:
        return False
    queries = []
    for item in field_data:
        if isinstance(item, dict):
            q = item.get("boolean_query_string")
            if q is not None:
                queries.append(q)
    if not queries:
        return False
    return all(is_placeholder_only(q) for q in queries)

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

def _strip_verbatim_sources(extraction: dict) -> dict:
    """
    Returns a copy of extraction with only the main values (no verbatim_source).
    Evidence objects become their `value`, and boolean query items drop verbatim_source.
    """
    if not extraction:
        return {}

    cleaned = {}
    for key, value in extraction.items():
        # Evidence object -> keep only value
        if isinstance(value, dict) and "value" in value:
            cleaned[key] = value.get("value")
            continue

        # Boolean queries list -> remove verbatim_source per item
        if key == "exact_boolean_queries" and isinstance(value, list):
            cleaned[key] = [
                {
                    "boolean_query_string": item.get("boolean_query_string"),
                    "database_source": item.get("database_source"),
                }
                for item in value
                if isinstance(item, dict)
            ]
            continue

        cleaned[key] = value

    return cleaned

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
def main():
    if not INPUT_FILE.exists():
        logging.error(f"Error: Input file not found at {INPUT_FILE}")
        return

    logging.info(f"Reading from: {INPUT_FILE.name}")
    logging.info(f"Writing to:   {OUTPUT_FILE.name}")
    total_read = 0
    total_saved = 0
    total_placeholder_only = 0
    total_placeholder_only_dropped = 0
    total_placeholder_only_kept_keywords = 0
    
    with open(INPUT_FILE, "r") as fin, open(OUTPUT_FILE, "w") as fout:
        for line in fin:
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    logging.error("Skipping error line: record is not a dict")
                    continue
                total_read += 1
                
                extraction = record.get("extraction", {})
                if extraction is None:
                    extraction = {}
                if not isinstance(extraction, dict):
                    logging.error("Skipping error line: extraction is not a dict")
                    continue
                exact_queries = extraction.get("exact_boolean_queries")
                
                # --- FILTER STEP ---
                placeholder_only = has_only_placeholder_queries(exact_queries)
                if placeholder_only:
                    total_placeholder_only += 1
                    if is_filled(extraction.get("keywords_used")):
                        total_placeholder_only_kept_keywords += 1
                    else:
                        total_placeholder_only_dropped += 1
                        continue
                if not check_completeness(extraction):
                    continue # Skip incomplete docs

                # --- FLATTEN STEP ---
                # 1. Keep only doc_id + extracted values (no verbatim_source)
                final_record = {
                    "file_path": record.get("file_path"),
                    "doc_id": record.get("doc_id")
                }
                
                # 2. Hoist cleaned extracted fields
                if extraction:
                    final_record.update(_strip_verbatim_sources(extraction))
                
                # 3. Save (Clean, no extra stats)
                fout.write(json.dumps(final_record) + "\n")
                total_saved += 1

            except Exception as e:
                logging.error(f"Skipping error line: {e}")

    logging.info("-" * 40)
    logging.info(f"PROCESSING COMPLETE")
    logging.info(f"Total Read:     {total_read}")
    logging.info(f"Filtered Out:   {total_read - total_saved}")
    logging.info(f"Placeholder-only total: {total_placeholder_only}")
    logging.info(f"Placeholder-only dropped (no keywords): {total_placeholder_only_dropped}")
    logging.info(f"Placeholder-only kept (has keywords): {total_placeholder_only_kept_keywords}")
    logging.info(f"Final Dataset:  {total_saved} documents")
    logging.info(f"Saved to:       {OUTPUT_FILE}")
    logging.info("-" * 40)

if __name__ == "__main__":
    main()