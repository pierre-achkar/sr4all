"""Check OAX transform output coverage and normalized-query completeness."""

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

# ========================
# Config
# ========================
CONFIG = {
    "input_jsonl": Path(
        "/home/fhg/pie65738/projects/sr4all/data/final/sr4all_full_normalized_year_range_search_boolean_only.jsonl"
    ),
    "output_jsonl": Path(
        "/home/fhg/pie65738/projects/sr4all/data/final/with_oax/sr4all_full_normalized_year_range_search_boolean_only_oax_mapping_repaired.jsonl"
    ),
    "log_file": Path(
        "/home/fhg/pie65738/projects/sr4all/logs/oax/check_oax_output_boolean_only.log"
    ),
    "missing_ids_out": Path(
        "/home/fhg/pie65738/projects/sr4all/logs/oax/missing_oax_ids_boolean_only.txt"
    ),
    "error_ids_out": Path(
        "/home/fhg/pie65738/projects/sr4all/logs/oax/oax_error_ids_boolean_only.txt"
    ),
    "error_ids_by_type_out": Path(
        "/home/fhg/pie65738/projects/sr4all/logs/oax/oax_error_ids_by_type_boolean_only.json"
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
logger = logging.getLogger("oax_check")


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


def has_oax_error(rec: Dict) -> bool:
    return bool(rec.get("oax_transform_error"))


def has_any_normalized_query(rec: Dict) -> bool:
    items = rec.get("oax_boolean_queries")
    if not isinstance(items, list) or len(items) == 0:
        return False
    for item in items:
        if isinstance(item, str) and item.strip():
            return True
    return False


def main() -> None:
    input_path = CONFIG["input_jsonl"]
    output_path = CONFIG["output_jsonl"]

    if not input_path.exists():
        msg = f"Input not found: {input_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)
    if not output_path.exists():
        msg = f"Output not found: {output_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    input_ids: Set[str] = set()
    for rec in iter_jsonl(input_path):
        rec_id = get_record_id(rec)
        if rec_id:
            input_ids.add(rec_id)

    output_ids: Set[str] = set()
    error_ids: Set[str] = set()
    error_type_counts: Dict[str, int] = {}
    error_ids_by_type: Dict[str, Set[str]] = {}
    no_normalized_ids: Set[str] = set()
    no_normalized_with_expected: Set[str] = set()
    no_normalized_with_zero_expected: Set[str] = set()
    for rec in iter_jsonl(output_path):
        rec_id = get_record_id(rec)
        if rec_id:
            output_ids.add(rec_id)
        if has_oax_error(rec):
            if rec_id:
                error_ids.add(rec_id)
            err = rec.get("oax_transform_error")
            if isinstance(err, str) and err:
                error_type_counts[err] = error_type_counts.get(err, 0) + 1
                if rec_id:
                    error_ids_by_type.setdefault(err, set()).add(rec_id)
        if not has_any_normalized_query(rec):
            if rec_id:
                no_normalized_ids.add(rec_id)
                expected_len = rec.get("oax_expected_len")
                if isinstance(expected_len, int) and expected_len > 0:
                    no_normalized_with_expected.add(rec_id)
                else:
                    no_normalized_with_zero_expected.add(rec_id)

    missing_ids = sorted(input_ids - output_ids)

    CONFIG["missing_ids_out"].parent.mkdir(parents=True, exist_ok=True)
    CONFIG["error_ids_out"].parent.mkdir(parents=True, exist_ok=True)
    CONFIG["error_ids_by_type_out"].parent.mkdir(parents=True, exist_ok=True)

    with CONFIG["missing_ids_out"].open("w", encoding="utf-8") as f:
        for rec_id in missing_ids:
            f.write(f"{rec_id}\n")

    with CONFIG["error_ids_out"].open("w", encoding="utf-8") as f:
        for rec_id in sorted(error_ids):
            f.write(f"{rec_id}\n")

    with CONFIG["error_ids_by_type_out"].open("w", encoding="utf-8") as f:
        payload = {k: sorted(v) for k, v in sorted(error_ids_by_type.items())}
        json.dump(payload, f, ensure_ascii=False, indent=2)

    summary_lines = [
        "=== OAX Output Check ===",
        f"Input records:  {len(input_ids)}",
        f"Output records: {len(output_ids)}",
        f"Missing records: {len(missing_ids)}",
        f"Error records: {len(error_ids)}",
        f"No normalized queries: {len(no_normalized_ids)}",
        f"  - With expected_len > 0: {len(no_normalized_with_expected)}",
        f"  - With expected_len == 0: {len(no_normalized_with_zero_expected)}",
        f"Missing IDs saved to: {CONFIG['missing_ids_out']}",
        f"Error IDs saved to:   {CONFIG['error_ids_out']}",
        f"Error IDs by type:    {CONFIG['error_ids_by_type_out']}",
    ]
    if error_type_counts:
        summary_lines.append("Error types (count):")
        for err_type in sorted(error_type_counts.keys()):
            summary_lines.append(f"  - {err_type}: {error_type_counts[err_type]}")
    for line in summary_lines:
        print(line)
    logger.info("\n".join(summary_lines))


if __name__ == "__main__":
    main()
