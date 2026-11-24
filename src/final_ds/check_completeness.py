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
logger = logging.getLogger("CompletenessChecker")


def is_filled(field_data: object) -> bool:
    """
    Checks if a field has valid content.
    Returns True if data exists, False if it is effectively empty/null.
    """
    if field_data is None:
        return False

    if isinstance(field_data, dict):
        if "value" in field_data:
            return is_filled(field_data["value"])
        if "boolean_query_string" in field_data or "database_source" in field_data:
            return is_filled(field_data.get("boolean_query_string"))
        return any(is_filled(value) for value in field_data.values())

    if isinstance(field_data, list):
        return any(is_filled(item) for item in field_data)

    if isinstance(field_data, str):
        return bool(field_data.strip())

    return True


_PLACEHOLDER_ONLY_RE = re.compile(r"^(?:#?\d+|AND|OR|NOT|\(|\)|\s)+$", re.IGNORECASE)


def is_placeholder_only(query: str) -> bool:
    if not query or not isinstance(query, str):
        return False
    return _PLACEHOLDER_ONLY_RE.fullmatch(query.strip()) is not None


def main() -> int:
    if not INPUT_FILE.exists():
        logger.error(f"Input file not found at {INPUT_FILE}")
        return 1

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

    field_names.update(extraction_fields)
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as error:
                invalid_records += 1
                logger.warning("Skipping invalid JSON at line %d: %s", line_number, error)
                continue

            if not isinstance(rec, dict):
                invalid_records += 1
                logger.warning("Skipping invalid record at line %d: expected an object", line_number)
                continue
            data = rec.get("extraction")
            if data is not None and not isinstance(data, dict):
                invalid_records += 1
                logger.warning("Skipping invalid record at line %d: extraction must be an object or null", line_number)
                continue
            if data is None:
                data = {key: rec.get(key) for key in extraction_fields}
            queries = data.get("exact_boolean_queries")
            if queries is not None and (
                not isinstance(queries, list)
                or any(
                    query is not None and (
                        not isinstance(query, dict)
                        or not isinstance(query.get("boolean_query_string"), (str, type(None)))
                    )
                    for query in queries
                )
            ):
                invalid_records += 1
                logger.warning("Skipping invalid record at line %d: expected a list of query objects or nulls", line_number)
                continue

            total_records += 1
            record_fields = {key: value for key, value in rec.items() if key != "extraction"}
            record_fields.update({key: data[key] for key in extraction_fields if key in data})
            field_names.update(record_fields)
            for field_name, field_value in record_fields.items():
                if is_filled(field_value):
                    field_counts[field_name] += 1

            filled = {key: is_filled(data.get(key)) for key in extraction_fields}
            if not any(filled.values()):
                total_docs_all_null_fields += 1
            if not any(filled[key] for key in extraction_presence_fields):
                total_docs_with_null_extraction += 1
                continue

            total_docs += 1
            if all(filled.values()):
                total_docs_all_fields_filled += 1

            obj_ok = filled["objective"]
            bool_ok = filled["exact_boolean_queries"]
            key_ok = filled["keywords_used"]
            inc_ok = filled["inclusion_criteria"]
            exc_ok = filled["exclusion_criteria"]

            placeholder_in_doc = False
            for query in queries or []:
                if query is not None and is_placeholder_only(query.get("boolean_query_string")):
                    placeholder_only_queries += 1
                    placeholder_in_doc = True
            if placeholder_in_doc:
                placeholder_only_docs += 1

            if obj_ok:
                has_objective += 1

            search_group_ok = bool_ok or key_ok
            if search_group_ok:
                has_search += 1
            if bool_ok and key_ok:
                search_both += 1
            elif bool_ok:
                search_bool_only += 1
            elif key_ok:
                search_keywords_only += 1
            else:
                search_none += 1
            if bool_ok:
                search_bool_any += 1
            if key_ok:
                search_keywords_any += 1

            criteria_group_ok = inc_ok or exc_ok
            if criteria_group_ok:
                has_criteria += 1
            strategy_ok = search_group_ok
            eligibility_ok = criteria_group_ok
            if strategy_ok:
                has_strategy += 1
            if eligibility_ok:
                has_eligibility += 1
            if obj_ok and strategy_ok and eligibility_ok:
                essentials_complete += 1
            if obj_ok and search_group_ok and criteria_group_ok:
                fully_complete += 1

    # --- REPORT ---
    logger.info("\n" + "=" * 60)
    logger.info(
        f"COMPLETENESS REPORT (records={total_records}, docs_with_extraction={total_docs}, invalid={invalid_records})"
    )
    logger.info("=" * 60)

    logger.info(
        f"Docs with ALL fields null/empty      | {total_docs_all_null_fields:<10} | {(total_docs_all_null_fields/max(total_records,1))*100:.1f}%"
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
        "Field coverage and all-null percentages use valid records; all-filled and essentials use documents with extraction (excluding year-only records)."
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
    if invalid_records:
        logger.error("Completeness check failed: %d invalid record(s)", invalid_records)
        return 1
    return 0


if __name__ == "__main__":
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    raise SystemExit(main())
