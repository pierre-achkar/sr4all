"""
Repair Impact Analyzer.

Compares 'fact_checked_corpus.jsonl' (Baseline) vs 'repaired_raw_candidates.jsonl' (Result)
to calculate the Recovery Rate (Recall Gain).
"""

import json
from pathlib import Path
from collections import defaultdict
import logging

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
# 1. Baseline: The output of Job E (Clean but sparse)
BASELINE_FILE = Path(
    "/data/sr4all/extraction_v1/raw_fact_checked/raw_fact_checked_corpus_all.jsonl"
)

# 2. Repaired: The output of Job C (Inference results)
REPAIRED_FILE = Path(
    "/data/sr4all/extraction_v1/repaired_fact_checked/repaired_fact_checked_corpus_all.jsonl"
)

LOG_FILE = Path("/logs/extraction/repair_impact_analysis_all.log")

# setup logging to a file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger("RepairImpactAnalyzer")


def load_corpus(path: Path):
    """Loads corpus into a dict {doc_id: extraction_data}"""
    data = {}
    if not path.exists():
        print(f"File not found: {path}")
        return {}

    with open(path, "r") as f:
        for line in f:
            try:
                rec = json.loads(line)
                doc_id = rec.get("doc_id")
                if not doc_id:
                    continue

                # --- FIX: Handle explicit Nulls safely ---
                ext = rec.get("extraction")
                if ext is None:
                    ext = {}

                data[doc_id] = ext
            except:
                pass
    return data


def is_present(field_data):
    """Checks if a field has a valid non-null value."""
    if field_data is None:
        return False

    # Check Evidence Object {"value": ...}
    if isinstance(field_data, dict):
        return field_data.get("value") is not None

    # Check List
    if isinstance(field_data, list):
        if not field_data:
            return False
        # Special check for boolean queries
        if isinstance(field_data[0], dict):
            # It's only "present" if the query string is not null
            return field_data[0].get("boolean_query_string") is not None
        return True  # Generic list is present

    return False


def main():
    logger.info(f"Loading Baseline: {BASELINE_FILE.name}...")
    baseline = load_corpus(BASELINE_FILE)
    logger.info(f"Loaded {len(baseline)} baseline docs.")

    logger.info(f"Loading Repaired: {REPAIRED_FILE.name}...")
    repaired = load_corpus(REPAIRED_FILE)
    logger.info(f"Loaded {len(repaired)} repaired docs.")

    # Overlapping docs only
    common_ids = set(baseline.keys()) & set(repaired.keys())
    logger.info(f"Comparing {len(common_ids)} common documents...\n")
    stats = defaultdict(lambda: {"missing_before": 0, "recovered": 0, "regressed": 0})
    total_recovered = 0
    total_missing_before = 0

    for doc_id in common_ids:
        base_data = baseline[doc_id]
        rep_data = repaired[doc_id]

        # Use keys from baseline schema (or simple union)
        # We assume the schema is consistent, but let's be safe
        all_keys = set(base_data.keys()) | set(rep_data.keys())

        for key in all_keys:
            # Skip metadata keys if they sneaked in
            if key in ["repair_attempted", "fact_check_stats"]:
                continue

            was_present = is_present(base_data.get(key))
            is_present_now = is_present(rep_data.get(key))

            if not was_present:
                stats[key]["missing_before"] += 1
                total_missing_before += 1
                if is_present_now:
                    stats[key]["recovered"] += 1
                    total_recovered += 1

            elif was_present and not is_present_now:
                stats[key]["regressed"] += 1

    # --- REPORT ---
    logger.info(
        f"{'FIELD':<25} | {'MISSING (Base)':<15} | {'RECOVERED':<10} | {'GAIN %':<10}"
    )
    logger.info("-" * 70)

    for key, metrics in sorted(stats.items()):
        missing = metrics["missing_before"]
        recov = metrics["recovered"]
        gain = (recov / missing * 100) if missing > 0 else 0.0

        logger.info(f"{key:<25} | {missing:<15} | {recov:<10} | {gain:.1f}%")

    logger.info("-" * 70)
    grand_gain = (
        (total_recovered / total_missing_before * 100)
        if total_missing_before > 0
        else 0.0
    )
    logger.info(
        f"{'TOTAL':<25} | {total_missing_before:<15} | {total_recovered:<10} | {grand_gain:.1f}%"
    )


if __name__ == "__main__":
    main()
