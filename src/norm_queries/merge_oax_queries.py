"""
Merge OAX mapping outputs back into the final split datasets by id.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Optional

# ========================
# Config
# ========================
CONFIG = {
    "input_jsonl": Path(
        "./data/final/sr4all_full_normalized_year_range_search_keywords_only.jsonl"
    ),
    "mapping_jsonl": Path(
        "./data/final/with_oax/sr4all_full_normalized_year_range_search_keywords_only_oax_mapping_repaired_v2.jsonl"
    ),
    "output_jsonl": Path(
        "./data/final/with_oax/sr4all_full_normalized_year_range_search_keywords_only_with_oax.jsonl"
    ),
    "log_file": Path("./logs/oax/merge_oax_queries_keywords_only.log"),
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
logger = logging.getLogger("oax_merge")


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


def get_record_id(rec: Dict) -> Optional[str]:
    return rec.get("id") or rec.get("doc_id") or rec.get("rec_id")


def main() -> None:
    input_path = CONFIG["input_jsonl"]
    mapping_path = CONFIG["mapping_jsonl"]
    output_path = CONFIG["output_jsonl"]

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return
    if not mapping_path.exists():
        logger.error("Mapping file not found: %s", mapping_path)
        return

    mapping: Dict[str, Dict] = {}
    for rec in iter_jsonl(mapping_path):
        rec_id = get_record_id(rec)
        if rec_id:
            mapping[rec_id] = rec

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    matched = 0
    missing = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for rec in iter_jsonl(input_path):
            total += 1
            rec_id = get_record_id(rec)
            mapped = mapping.get(rec_id) if rec_id else None

            if mapped:
                rec["oax_boolean_queries"] = mapped.get("oax_boolean_queries")
                rec["oax_transform_error"] = mapped.get("oax_transform_error")
                matched += 1
            else:
                missing += 1

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(
        "Merge complete | total=%d matched=%d missing=%d output=%s",
        total,
        matched,
        missing,
        output_path,
    )
    print(
        f"Merge complete | total={total} matched={matched} missing={missing} output={output_path}"
    )


if __name__ == "__main__":
    main()
