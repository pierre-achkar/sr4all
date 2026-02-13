"""
Extract records with null boolean_queries for re-inference.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Set

# ========================
# Config
# ========================
CONFIG = {
    "mapping_jsonl": Path(
        "./data/final/with_boolean/merged/sr4all_full_normalized_boolean_mapping_merged_1.jsonl"
    ),
    "source_jsonl": Path(
        "./data/final/sr4all_full_normalized_year_range_search_has_boolean.jsonl"
    ),
    "output_jsonl": Path(
        "./data/final/with_boolean/null_subsets/sr4all_full_normalized_boolean_null_repair_subset_2.jsonl"
    ),
    "log_file": Path("./logs/oax/extract_null_boolean_subset.log"),
}

# ========================
# Logging
# ========================
CONFIG["log_file"].parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(CONFIG["log_file"]),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    filemode="w",
)
logger = logging.getLogger("oax_extract_null_boolean_subset")


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def collect_null_ids(mapping_path: Path) -> Set[str]:
    ids: Set[str] = set()
    total = 0
    nulls = 0
    for rec in iter_jsonl(mapping_path):
        total += 1
        rec_id = rec.get("id")
        if rec.get("boolean_queries") is None and isinstance(rec_id, str) and rec_id:
            ids.add(rec_id)
            nulls += 1
    logger.info("mapping_total=%d mapping_nulls=%d", total, nulls)
    return ids


def main() -> None:
    mapping_path = CONFIG["mapping_jsonl"]
    source_path = CONFIG["source_jsonl"]
    output_path = CONFIG["output_jsonl"]

    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping not found: {mapping_path}")
    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    ids = collect_null_ids(mapping_path)
    logger.info("null_ids=%d", len(ids))

    total = 0
    matched = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for rec in iter_jsonl(source_path):
            total += 1
            rec_id = rec.get("id")
            if rec_id in ids:
                matched += 1
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = f"Subset extract complete | source_total={total} matched={matched} output={output_path}"
    print(summary)
    logger.info(summary)


if __name__ == "__main__":
    main()
