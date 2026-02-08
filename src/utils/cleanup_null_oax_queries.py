"""
Clean up oax_boolean_queries by removing redundant nulls while preserving single-null cases.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# ========================
# Config
# ========================
CONFIG = {
    "input_jsonl": Path(
        "/data/final/with_oax/sr4all_full_normalized_year_range_search_keywords_only_oax_mapping_repaired_v2.jsonl"
    ),
    "output_jsonl": Path(
        "/data/final/with_oax/sr4all_full_normalized_year_range_search_keywords_only_oax_mapping_repaired_v2_clean.jsonl"
    ),
    "log_file": Path(
        "/logs/oax/cleanup_oax_queries_keywords_only.log"
    ),
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
logger = logging.getLogger("oax_cleanup")


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


def normalize_queries(items: List[Optional[str]]) -> List[Optional[str]]:
    if not isinstance(items, list) or len(items) == 0:
        return items

    non_null = [q for q in items if isinstance(q, str) and q.strip()]
    if non_null:
        return non_null

    # All null/empty -> keep exactly one null
    return [None]


def main() -> None:
    input_path = CONFIG["input_jsonl"]
    output_path = CONFIG["output_jsonl"]

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    changed = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for rec in iter_jsonl(input_path):
            total += 1
            items = rec.get("oax_boolean_queries")
            if isinstance(items, list):
                cleaned = normalize_queries(items)
                if cleaned != items:
                    rec["oax_boolean_queries"] = cleaned
                    changed += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info("Cleanup complete | total=%d changed=%d output=%s", total, changed, output_path)
    print(f"Cleanup complete | total={total} changed={changed} output={output_path}")


if __name__ == "__main__":
    main()
