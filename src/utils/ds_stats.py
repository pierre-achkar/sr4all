import json
import logging
from typing import Any

PATH = "./data/final/sr4all_full.jsonl"
LOG_FILE = "./logs/final_ds/ds_stats.log"

# --- Setup logging ---
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
    force=True,
    filemode="w",
)

# read the jsonl file
with open(PATH, "r") as f:
    data = [json.loads(line) for line in f]

logging.info(f"Total number of samples: {len(data)}")

# Keys in the data:
# {'exclusion_criteria', 'exact_boolean_queries', 'doi', 'keywords_used_used', 'subfield',
# 'topics', 'keywords_used', 'referenced_works', 'research_questions', 'year_range_normalized',
# 'inclusion_criteria', 'title', 'references_abstract_coverage', 'type', 'pdf_url', 'id',
# 'cited_by_count', 'language', 'n_studies_final', 'field', 'year_range_normalization_rule',
# 'year', 'year_range', 'databases_used', 'objective', 'source', 'referenced_works_count',
# 'snowballing', 'abstract', 'n_studies_initial', 'authors'}

# drop year_range_normalization_rule
for sample in data:
    if "year_range_normalization_rule" in sample:
        del sample["year_range_normalization_rule"]

# drop references_abstract_coverage
for sample in data:
    if "references_abstract_coverage" in sample:
        del sample["references_abstract_coverage"]


def is_filled(field_data: Any) -> bool:
    """Checks if a field has valid content."""
    if field_data is None:
        return False

    # Case 1: Dict values
    if isinstance(field_data, dict):
        # Flat boolean-query object
        if isinstance(field_data.get("boolean_query_string"), str):
            return bool(field_data["boolean_query_string"].strip())

        # Evidence object {"value": ...}
        val = field_data.get("value")
        if val is not None:
            if isinstance(val, list):
                if len(val) == 0:
                    return False
                if all(isinstance(v, str) for v in val):
                    return any(v.strip() for v in val)
            if isinstance(val, str):
                return bool(val.strip())
            return True

        # Generic dict fallback: any non-empty value
        return any(is_filled(v) for v in field_data.values())

    # Case 2: List values
    if isinstance(field_data, list):
        if not field_data:
            return False
        # List of dicts (e.g., exact_boolean_queries)
        if all(isinstance(item, dict) for item in field_data):
            # Require at least one non-empty boolean_query_string
            for item in field_data:
                bqs = item.get("boolean_query_string")
                if isinstance(bqs, str) and bqs.strip():
                    return True
            return False
        # List of strings (keywords)
        if all(isinstance(item, str) for item in field_data):
            return any(item.strip() for item in field_data)
        return True

    if isinstance(field_data, str):
        return bool(field_data.strip())

    return bool(field_data)


def has_exact_boolean_queries(sample: dict) -> bool:
    """
    True if sample has at least one non-empty boolean query.
    Counts at sample level (one sample -> one count), regardless of how many query strings exist.
    """
    queries = sample.get("exact_boolean_queries")
    if not isinstance(queries, list) or not queries:
        return False

    for item in queries:
        if not isinstance(item, dict):
            continue
        bqs = item.get("boolean_query_string")
        if isinstance(bqs, str) and bqs.strip():
            return True
        if isinstance(bqs, list) and any(isinstance(x, str) and x.strip() for x in bqs):
            return True
    return False


def has_database_info(sample: dict) -> bool:
    """
    True if sample has database info either in `databases_used` or in
    `exact_boolean_queries[].database_source`.
    """
    if is_filled(sample.get("databases_used")):
        return True

    queries = sample.get("exact_boolean_queries")
    if not isinstance(queries, list):
        return False

    for item in queries:
        if not isinstance(item, dict):
            continue
        db = item.get("database_source")
        if isinstance(db, str) and db.strip():
            return True
        if isinstance(db, list) and any(isinstance(x, str) and x.strip() for x in db):
            return True
    return False


# count how many has title filled
title_count = sum(1 for sample in data if "title" in sample and sample["title"])
logging.info(f"Number of samples with 'title' field filled: {title_count}")

# count how many has abstract filled
abstract_count = sum(
    1 for sample in data if "abstract" in sample and sample["abstract"]
)
logging.info(f"Number of samples with 'abstract' field filled: {abstract_count}")

# count the number of samples that have the field "objective" even if it is empty
objective_count = sum(1 for sample in data if "objective" in sample)
logging.info(f"Number of samples with full text: {objective_count}")

# count all samples that have title, abstract, and objective filled (not empty)
title_abstract_objective_count = sum(
    1
    for sample in data
    if "title" in sample
    and sample["title"]
    and "abstract" in sample
    and sample["abstract"]
    and "objective" in sample
    and sample["objective"]
)
logging.info(
    f"Number of samples with title, abstract, and full-text filled: {title_abstract_objective_count}"
)

# count the number of samples that have the field "objective" filled (not empty)
objective_filled_count = sum(
    1 for sample in data if "objective" in sample and sample["objective"]
)
logging.info(
    f"Number of samples with 'objective' field filled: {objective_filled_count}"
)

# count how many have research_questions filled
research_questions_count = sum(
    1
    for sample in data
    if "research_questions" in sample and sample["research_questions"]
)
logging.info(
    f"Number of samples with 'research_questions' field filled: {research_questions_count}"
)

# count how many have keywords_used filled
keywords_used_count = sum(
    1 for sample in data if is_filled(sample.get("keywords_used"))
)
logging.info(
    f"Number of samples with 'keywords_used' field filled: {keywords_used_count}"
)

# count how many have exact_boolean_queries filled (at least one with non-empty 'boolean_query_string')
exact_boolean_queries_count = sum(
    1 for sample in data if has_exact_boolean_queries(sample)
)
logging.info(
    f"Number of samples with 'exact_boolean_queries' field filled: {exact_boolean_queries_count}"
)

# count how many have inclusion_criteria or exclusion_criteria filled
inclusion_criteria_count = sum(
    1
    for sample in data
    if "inclusion_criteria" in sample and sample["inclusion_criteria"]
)
exclusion_criteria_count = sum(
    1
    for sample in data
    if "exclusion_criteria" in sample and sample["exclusion_criteria"]
)
logging.info(
    f"Number of samples with 'inclusion_criteria' field filled: {inclusion_criteria_count}"
)
logging.info(
    f"Number of samples with 'exclusion_criteria' field filled: {exclusion_criteria_count}"
)
logging.info(
    f"Number of samples with either 'inclusion_criteria' or 'exclusion_criteria' field filled: {inclusion_criteria_count + exclusion_criteria_count}"
)

# count how many have year_range_normalized filled
year_range_normalized_count = sum(
    1
    for sample in data
    if "year_range_normalized" in sample and sample["year_range_normalized"]
)
logging.info(
    f"Number of samples with 'year_range_normalized' field filled: {year_range_normalized_count}"
)

# count how many have all of above fields filled (title, abstract, objective, research_questions, keywords_used, exact_boolean_queries, inclusion_criteria, exclusion_criteria, year_range_normalized)
all_fields_count = sum(
    1
    for sample in data
    if (
        "title" in sample
        and sample["title"]
        and "abstract" in sample
        and sample["abstract"]
        and "objective" in sample
        and sample["objective"]
        and "research_questions" in sample
        and sample["research_questions"]
        and is_filled(sample.get("keywords_used"))
        and has_exact_boolean_queries(sample)
        and (
            ("inclusion_criteria" in sample and sample["inclusion_criteria"])
            or ("exclusion_criteria" in sample and sample["exclusion_criteria"])
        )
        and "year_range_normalized" in sample
        and sample["year_range_normalized"]
    )
)
logging.info(f"Number of samples with all key fields filled: {all_fields_count}")

# how many has databases_used filled
databases_used_count = sum(1 for sample in data if has_database_info(sample))
logging.info(
    f"Number of samples with 'databases_used' information filled: {databases_used_count}"
)

# count how many have n_studies_initial filled
n_studies_initial_count = sum(
    1
    for sample in data
    if "n_studies_initial" in sample and sample["n_studies_initial"] is not None
)
logging.info(
    f"Number of samples with 'n_studies_initial' field filled: {n_studies_initial_count}"
)

# count how many have n_studies_final filled
n_studies_final_count = sum(
    1
    for sample in data
    if "n_studies_final" in sample and sample["n_studies_final"] is not None
)
logging.info(
    f"Number of samples with 'n_studies_final' field filled: {n_studies_final_count}"
)

# how many has referenced_works_count filled
referenced_works_count = sum(
    1
    for sample in data
    if "referenced_works_count" in sample
    and sample["referenced_works_count"] is not None
)
logging.info(
    f"Number of samples with 'referenced_works_count' field filled: {referenced_works_count}"
)

# how many has database info, n_studies_initial, n_studies_final, referenced_works_count all filled
all_stats_count = sum(
    1
    for sample in data
    if has_database_info(sample)
    and "n_studies_initial" in sample
    and sample["n_studies_initial"] is not None
    and "n_studies_final" in sample
    and sample["n_studies_final"] is not None
    and "referenced_works_count" in sample
    and sample["referenced_works_count"] is not None
)
logging.info(f"Number of samples with all stats fields filled: {all_stats_count}")

# how many have all key fields and all stats fields filled
all_fields_and_stats_count = sum(
    1
    for sample in data
    if (
        "title" in sample
        and sample["title"]
        and "abstract" in sample
        and sample["abstract"]
        and "objective" in sample
        and sample["objective"]
        and "research_questions" in sample
        and sample["research_questions"]
        and is_filled(sample.get("keywords_used"))
        and has_exact_boolean_queries(sample)
        and (
            ("inclusion_criteria" in sample and sample["inclusion_criteria"])
            or ("exclusion_criteria" in sample and sample["exclusion_criteria"])
        )
        and "year_range_normalized" in sample
        and sample["year_range_normalized"]
        # databases_used OR at least one exact_boolean_queries[].database_source filled
        and has_database_info(sample)
        and "n_studies_initial" in sample
        and sample["n_studies_initial"] is not None
        and "n_studies_final" in sample
        and sample["n_studies_final"] is not None
        and "referenced_works_count" in sample
        and sample["referenced_works_count"] is not None
    )
)
logging.info(
    f"Number of samples with all key fields and all stats fields filled: {all_fields_and_stats_count}"
)
