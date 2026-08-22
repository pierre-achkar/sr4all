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
    "./data/final_ds/sr4all_final.jsonl"
)
LOG_FILE = Path("./logs/final_ds/completeness_check_all.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# setup logging to a file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_FILE, mode="w"
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

    # Evidence object {"value": ...} (standard extraction fields).
    if isinstance(field_data, dict):
        if "value" not in field_data:
            return bool(field_data)
        val = field_data.get("value")
        if val is None:
            return False
        if isinstance(val, list) and len(val) == 0:
            return False
        return True

    # Lists are filled when they contain at least one meaningful item.
    if isinstance(field_data, list):
        if not field_data:
            return False  # Empty list []

        return True

    if isinstance(field_data, str):
        return bool(field_data.strip())

    return True


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
    field_counts = Counter()
    field_names = set()
    invalid_records = 0

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

    extraction_fields = [
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
    extraction_presence_fields = [
        field for field in extraction_fields if field != "year_range"
    ]

    with open(INPUT_FILE, "r") as f:
        for line in f:
            try:
                rec = json.loads(line)
                total_records += 1
                field_names.update(rec)
                for field_name, field_value in rec.items():
                    field_filled = is_filled(field_value)
                    if field_name == "exact_boolean_queries" and isinstance(
                        field_value, list
                    ):
                        field_filled = any(
                            isinstance(item, dict)
                            and item.get("boolean_query_string") is not None
                            for item in field_value
                        )
                    if field_filled:
                        field_counts[field_name] += 1

                data = rec.get("extraction")
                if not isinstance(data, dict):
                    data = {key: rec.get(key) for key in extraction_fields}

                # If extraction is null, skip
                if not any(
                    is_filled(data.get(key)) for key in extraction_presence_fields
                ):
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
                bool_ok = is_filled(data.get("exact_boolean_queries")) and not (
                    isinstance(data.get("exact_boolean_queries"), list)
                    and all(
                        isinstance(item, dict)
                        and item.get("boolean_query_string") is None
                        for item in data["exact_boolean_queries"]
                    )
                )
                key_ok = is_filled(data.get("keywords_used"))
                inc_ok = is_filled(data.get("inclusion_criteria"))
                exc_ok = is_filled(data.get("exclusion_criteria"))

                # All-null / all-filled checks across all fields
                per_field_filled = [is_filled(data.get(k)) for k in extraction_fields]
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

                # Check logic groups.

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

                # Full completeness (A + B + C).
                if obj_ok and search_group_ok and criteria_group_ok:
                    fully_complete += 1

            except Exception as e:
                invalid_records += 1
                logger.warning("Skipping invalid record: %s", e)

    # --- REPORT ---
    logger.info("\n" + "=" * 60)
    logger.info(
        f"COMPLETENESS REPORT (records={total_records}, docs_with_extraction={total_docs}, invalid={invalid_records})"
    )
    logger.info("=" * 60)

    logger.info(
        f"Docs with ALL fields null/empty      | {total_docs_all_null_fields:<10} | {(total_docs_all_null_fields/max(total_docs,1))*100:.1f}%"
    )
    logger.info(
        f"Docs with ALL fields filled         | {total_docs_all_fields_filled:<10} | {(total_docs_all_fields_filled/max(total_docs,1))*100:.1f}%"
    )

    pipeline_field_names = set(extraction_fields)
    raw_field_names = field_names - pipeline_field_names

    for section_name, section_fields in (
        ("RAW OPENALEX FIELDS", raw_field_names),
        ("PIPELINE-EXTRACTED FIELDS", pipeline_field_names & field_names),
    ):
        logger.info(f"\n{section_name:<35} | {'COUNT':<10} | {'%':<6}")
        logger.info("-" * 60)
        for field_name in sorted(section_fields):
            count = field_counts[field_name]
            pct = (count / max(total_records, 1)) * 100
            logger.info(f"{field_name:<35} | {count:<10} | {pct:.1f}%")

    logger.info(
        "Percentages above use all records as the denominator; essentials use documents with extraction."
    )

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
