"""
Merge repaired boolean mappings back into the fixed mapping file.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Iterable

# ========================
# Config
# ========================
CONFIG = {
    "base_mapping_jsonl": Path(
        "./data/final/with_boolean/merged/sr4all_full_normalized_keywords_only_mapping_merged_1.jsonl"
    ),
    "repaired_mapping_jsonl": Path(
        "./data/final/with_boolean/repaired_fixed/sr4all_full_normalized_keywords_only_repaired_fixed_mapping_2.jsonl"
    ),
    "output_jsonl": Path(
        "./data/final/with_boolean/merged/sr4all_full_normalized_keywords_only_mapping_merged_2.jsonl"
    ),
    "log_file": Path("./logs/oax/merge_repaired_keywords_only_mapping.log"),
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
logger = logging.getLogger("oax_merge_repaired_boolean_mapping")


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


def load_repaired(path: Path) -> Dict[str, Dict]:
    repaired = {}
    total = 0
    usable = 0
    for rec in iter_jsonl(path):
        total += 1
        rec_id = rec.get("id")
        if not isinstance(rec_id, str) or not rec_id:
            continue
        if rec.get("boolean_queries") is None:
            continue
        repaired[rec_id] = rec
        usable += 1
    logger.info("repaired_total=%d repaired_usable=%d", total, usable)
    return repaired


def main() -> None:
    base_path = CONFIG["base_mapping_jsonl"]
    repaired_path = CONFIG["repaired_mapping_jsonl"]
    output_path = CONFIG["output_jsonl"]

    if not base_path.exists():
        raise FileNotFoundError(f"Base mapping not found: {base_path}")
    if not repaired_path.exists():
        raise FileNotFoundError(f"Repaired mapping not found: {repaired_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    repaired = load_repaired(repaired_path)

    total = 0
    replaced = 0
    kept = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for rec in iter_jsonl(base_path):
            total += 1
            rec_id = rec.get("id")
            if rec_id in repaired:
                new_rec = repaired[rec_id]
                rec["boolean_queries"] = new_rec.get("boolean_queries")
                rec["boolean_error"] = new_rec.get("boolean_error")
                rec["keywords_only"] = new_rec.get(
                    "keywords_only", rec.get("keywords_only")
                )
                replaced += 1
            else:
                kept += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = (
        f"Merge complete | base_total={total} replaced={replaced} kept={kept} "
        f"output={output_path}"
    )
    print(summary)
    logger.info(summary)


if __name__ == "__main__":
    main()
