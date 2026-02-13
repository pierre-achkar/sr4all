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
import logging
import re

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
INPUT_FILE = Path(
    "./data/sr4all/extraction_v1/repaired_fact_checked/repaired_fact_checked_corpus_all.jsonl"
)

# setup logging to a file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            Path("./logs/final_ds/completeness_check_all.log"), mode="w"
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("CompletenessChecker")


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
            return False  # Empty list []

        # Check for ghost object: [{"boolean_query_string": null, ...}]
        first_item = field_data[0]
        if isinstance(first_item, dict):
            # It is ONLY valid if the query string is NOT null
            if first_item.get("boolean_query_string") is None:
                return False

        return True

    return False


_PLACEHOLDER_ONLY_RE = re.compile(r"^(?:#?\d+|AND|OR|NOT|\(|\)|\s)+$", re.IGNORECASE)


def is_placeholder_only(query: str) -> bool:
    if not query or not isinstance(query, str):
        return False
    return _PLACEHOLDER_ONLY_RE.fullmatch(query.strip()) is not None


def main():
    if not INPUT_FILE.exists():
        logger.error(f"Input file not found at {INPUT_FILE}")
        return

    logger.info(f"Scanning: {INPUT_FILE.name}...")

    total_records = 0
    total_docs = 0
    total_docs_with_null_extraction = 0
    total_docs_all_null_fields = 0
    total_docs_all_fields_filled = 0
    stats = Counter()

    # Logic Group Counters
    has_objective = 0
    has_search = 0
    has_criteria = 0
    fully_complete = 0

    # Essentials (Objective + Strategy + Eligibility)
    essentials_complete = 0
    has_strategy = 0
    has_eligibility = 0

    # Search Strategy Breakdown
    search_bool_only = 0
    search_keywords_only = 0
    search_both = 0
    search_bool_any = 0
    search_keywords_any = 0
    search_none = 0

    # Placeholder-only query stats
    placeholder_only_queries = 0
    placeholder_only_docs = 0

    fields_to_check = [
        "objective",
        "research_questions",
        "n_studies_initial",
        "n_studies_final",
        "year_range",
        "snowballing",
        "keywords_used",
        "exact_boolean_queries",
        "databases_used",
        "inclusion_criteria",
        "exclusion_criteria",
    ]

    with open(INPUT_FILE, "r") as f:
        for line in f:
            try:
                rec = json.loads(line)
                total_records += 1
                data = rec.get("extraction", {})

                # If extraction is null, skip
                if not data:
                    total_docs_with_null_extraction += 1
                    continue

                total_docs += 1

                # 1. Check Individual Fields using helper
                obj_ok = is_filled(data.get("objective"))
                rq_ok = is_filled(data.get("research_questions"))
                n_init_ok = is_filled(data.get("n_studies_initial"))
                n_final_ok = is_filled(data.get("n_studies_final"))
                year_ok = is_filled(data.get("year_range"))
                snow_ok = is_filled(data.get("snowballing"))
                bool_ok = is_filled(data.get("exact_boolean_queries"))
                key_ok = is_filled(data.get("keywords_used"))
                inc_ok = is_filled(data.get("inclusion_criteria"))
                exc_ok = is_filled(data.get("exclusion_criteria"))

                # All-null / all-filled checks across all fields
                per_field_filled = [is_filled(data.get(k)) for k in fields_to_check]
                if not any(per_field_filled):
                    total_docs_all_null_fields += 1
                if all(per_field_filled):
                    total_docs_all_fields_filled += 1

                # Placeholder-only checks inside boolean queries
                placeholder_in_doc = False
                for q in data.get("exact_boolean_queries") or []:
                    q_str = (q or {}).get("boolean_query_string")
                    if is_placeholder_only(q_str):
                        placeholder_only_queries += 1
                        placeholder_in_doc = True
                if placeholder_in_doc:
                    placeholder_only_docs += 1

                # 2. Update Stats for individual fields
                if obj_ok:
                    stats["objective"] += 1
                if rq_ok:
                    stats["research_questions"] += 1
                if n_init_ok:
                    stats["n_studies_initial"] += 1
                if n_final_ok:
                    stats["n_studies_final"] += 1
                if year_ok:
                    stats["year_range"] += 1
                if snow_ok:
                    stats["snowballing"] += 1
                if bool_ok:
                    stats["exact_boolean_queries"] += 1
                if key_ok:
                    stats["keywords_used"] += 1
                if inc_ok:
                    stats["inclusion_criteria"] += 1
                if exc_ok:
                    stats["exclusion_criteria"] += 1

                # 3. Check Logic Groups

                # Group A: Objective
                if obj_ok:
                    has_objective += 1

                # Group B: Search Strategy (Boolean OR Keywords)
                search_group_ok = bool_ok or key_ok
                if search_group_ok:
                    has_search += 1

                if bool_ok and key_ok:
                    search_both += 1
                elif bool_ok and not key_ok:
                    search_bool_only += 1
                elif key_ok and not bool_ok:
                    search_keywords_only += 1
                else:
                    search_none += 1

                if bool_ok:
                    search_bool_any += 1
                if key_ok:
                    search_keywords_any += 1

                # Group C: Criteria (Inclusion OR Exclusion)
                criteria_group_ok = inc_ok or exc_ok
                if criteria_group_ok:
                    has_criteria += 1

                # Essentials: Objective + Strategy + Eligibility
                strategy_ok = search_group_ok
                eligibility_ok = criteria_group_ok

                if strategy_ok:
                    has_strategy += 1
                if eligibility_ok:
                    has_eligibility += 1

                if obj_ok and strategy_ok and eligibility_ok:
                    essentials_complete += 1

                # 4. Full Completeness (A + B + C)
                if obj_ok and search_group_ok and criteria_group_ok:
                    fully_complete += 1

            except Exception as e:
                pass

    # --- REPORT ---
    logger.info("\n" + "=" * 60)
    logger.info(
        f"COMPLETENESS REPORT (records={total_records}, docs_with_extraction={total_docs})"
    )
    logger.info("=" * 60)

    logger.info(
        f"Docs with ALL fields null/empty      | {total_docs_all_null_fields:<10} | {(total_docs_all_null_fields/max(total_docs,1))*100:.1f}%"
    )
    logger.info(
        f"Docs with ALL fields filled         | {total_docs_all_fields_filled:<10} | {(total_docs_all_fields_filled/max(total_docs,1))*100:.1f}%"
    )

    logger.info(f"\n{'PER-FIELD COMPLETENESS':<35} | {'COUNT':<10} | {'%':<6}")
    logger.info("-" * 60)

    for k, v in sorted(stats.items()):
        pct = (v / max(total_docs, 1)) * 100
        logger.info(f"{k:<35} | {v:<10} | {pct:.1f}%")

    logger.info("=" * 60)
    logger.info("ESSENTIALS COMPLETENESS (Objective + Strategy + Eligibility)")
    logger.info("-" * 60)
    logger.info(
        f"Objective                            | {has_objective:<10} | {(has_objective/max(total_docs,1))*100:.1f}%"
    )
    logger.info(
        f"Strategy (Queries OR Keywords)       | {has_strategy:<10} | {(has_strategy/max(total_docs,1))*100:.1f}%"
    )
    logger.info(
        f"Eligibility (Inclusion OR Exclusion) | {has_eligibility:<10} | {(has_eligibility/max(total_docs,1))*100:.1f}%"
    )
    logger.info(
        f"Essentials complete (all 3)          | {essentials_complete:<10} | {(essentials_complete/max(total_docs,1))*100:.1f}%"
    )

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
