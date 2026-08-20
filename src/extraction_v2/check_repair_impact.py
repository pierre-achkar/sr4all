"""Measure the effect of the v2 repair stage.

Compares mirrored per-document trees:

    data/extraction/fact_checked/W1/id.json
    data/extraction/repaired_fact_checked/W1/id.json

The analyzer is read-only. It writes an impact log and a field-coverage summary
log, and does not create a report artifact beside either dataset.
"""

import argparse
import json
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

PROJECT = Path(os.environ.get("SR4ALL_PROJECT", "/home/fhg/pie65738/projects/sr4all"))


def _path(env_name: str, default: Path) -> Path:
    return Path(os.environ.get(env_name, str(default)))


BASELINE_ROOT = _path(
    "SR4ALL_FACT_CHECKED_DIR", PROJECT / "data/extraction/fact_checked"
)
REPAIRED_ROOT = _path(
    "SR4ALL_REPAIRED_FACT_CHECKED_DIR",
    PROJECT / "data/extraction/repaired_fact_checked",
)
LOG_FILE = _path(
    "SR4ALL_REPAIR_IMPACT_LOG",
    PROJECT / "logs/extraction/repaired_fact_checked/repair_impact.log",
)
SUMMARY_LOG_FILE = _path(
    "SR4ALL_REPAIR_FIELD_SUMMARY_LOG",
    PROJECT / "logs/extraction/field_fill_summary.log",
)

FIELD_ORDER = (
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
)


def setup_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("RepairImpact")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def write_field_summary(
    log_file: Path,
    stats: Dict[str, Counter],
    fields: Iterable[str],
    document_count: int,
) -> None:
    """Write document-level field coverage before and after repair."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as handle:
        handle.write(f"Documents compared: {document_count}\n")
        handle.write(
            "FIELD | BASELINE_FILLED | BASELINE_PCT | REPAIRED_FILLED | "
            "REPAIRED_PCT | ABS_CHANGE | PCT_POINT_CHANGE\n"
        )
        handle.write("-" * 120 + "\n")
        for field in fields:
            metric = stats[field]
            baseline_filled = metric["filled_before"]
            repaired_filled = metric["filled_after"]
            baseline_pct = (
                100.0 * baseline_filled / document_count
                if document_count else 0.0
            )
            repaired_pct = (
                100.0 * repaired_filled / document_count
                if document_count else 0.0
            )
            handle.write(
                f"{field} | {baseline_filled} | {baseline_pct:.1f}% | "
                f"{repaired_filled} | {repaired_pct:.1f}% | "
                f"{repaired_filled - baseline_filled} | "
                f"{repaired_pct - baseline_pct:.1f} percentage points\n"
            )


def find_documents(root: Path) -> Dict[str, Path]:
    """Return mirrored document files keyed by relative POSIX path."""
    if not root.is_dir():
        return {}
    return {
        path.relative_to(root).as_posix(): path
        for path in root.glob("*/**/*.json")
        if path.is_file() and not path.stem.startswith("_")
    }


def load_record(path: Path) -> Tuple[Any, str]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, f"READ_ERROR: {error}"
    if not isinstance(record, dict):
        return None, "INVALID_RECORD: root is not an object"
    return record, ""


def is_present(field_data: Any) -> bool:
    """Match the repair detector's notion of a usable field."""
    if isinstance(field_data, dict):
        value = field_data.get("value")
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, list):
            return bool(value)
        return True

    if isinstance(field_data, list):
        return any(
            isinstance(item, dict)
            and bool(str(item.get("boolean_query_string") or "").strip())
            for item in field_data
        )

    return False


def extraction_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    extraction = record.get("extraction")
    return extraction if isinstance(extraction, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--repaired", type=Path, default=REPAIRED_ROOT)
    parser.add_argument("--log-file", type=Path, default=LOG_FILE)
    parser.add_argument(
        "--summary-log-file", type=Path, default=SUMMARY_LOG_FILE
    )
    args = parser.parse_args()

    logger = setup_logging(args.log_file)
    baseline_files = find_documents(args.baseline)
    repaired_files = find_documents(args.repaired)

    logger.info("Repair impact analysis started")
    logger.info("Baseline root: %s", args.baseline)
    logger.info("Repaired root: %s", args.repaired)
    logger.info("Baseline files discovered: %d", len(baseline_files))
    logger.info("Repaired files discovered: %d", len(repaired_files))

    common = sorted(set(baseline_files) & set(repaired_files))
    missing_repaired = sorted(set(baseline_files) - set(repaired_files))
    extra_repaired = sorted(set(repaired_files) - set(baseline_files))
    logger.info("Common documents: %d", len(common))
    logger.info("Missing repaired outputs: %d", len(missing_repaired))
    logger.info("Unexpected repaired outputs: %d", len(extra_repaired))

    stats = defaultdict(Counter)
    totals = Counter()
    read_errors = 0

    for relative_path in common:
        baseline, baseline_error = load_record(baseline_files[relative_path])
        repaired, repaired_error = load_record(repaired_files[relative_path])
        if baseline_error or repaired_error:
            read_errors += 1
            logger.warning(
                "%s | baseline=%s | repaired=%s",
                relative_path,
                baseline_error or "OK",
                repaired_error or "OK",
            )
            continue

        baseline_fields = extraction_fields(baseline)
        repaired_fields = extraction_fields(repaired)
        fields = list(FIELD_ORDER)
        fields.extend(
            key for key in sorted(set(baseline_fields) | set(repaired_fields))
            if key not in FIELD_ORDER
        )

        totals["documents_compared"] += 1
        patched_keys = set(repaired.get("repair_patched_keys") or [])
        if repaired.get("repair_attempted"):
            totals["repair_attempted"] += 1
        if patched_keys:
            totals["documents_with_patches"] += 1

        for field in fields:
            before = is_present(baseline_fields.get(field))
            after = is_present(repaired_fields.get(field))
            if before:
                stats[field]["filled_before"] += 1
            if after:
                stats[field]["filled_after"] += 1
            if not before:
                stats[field]["missing_before"] += 1
                totals["missing_before"] += 1
                if after:
                    stats[field]["recovered"] += 1
                    totals["recovered"] += 1
            elif not after:
                stats[field]["regressed"] += 1
                totals["regressed"] += 1

            if field in patched_keys:
                stats[field]["marked_patched"] += 1

    logger.info("Documents with read errors: %d", read_errors)
    logger.info("Repair attempts: %d", totals["repair_attempted"])
    logger.info("Documents with patches: %d", totals["documents_with_patches"])
    logger.info("")
    logger.info(
        "%s | %s | %s | %s | %s | %s",
        "FIELD",
        "MISSING_BEFORE",
        "RECOVERED",
        "GAIN_PCT",
        "REGRESSED",
        "MARKED_PATCHED",
    )
    logger.info("-" * 100)

    ordered_fields: Iterable[str] = (
        list(FIELD_ORDER)
        + [field for field in sorted(stats) if field not in FIELD_ORDER]
    )
    for field in ordered_fields:
        metric = stats[field]
        missing = metric["missing_before"]
        recovered = metric["recovered"]
        gain = 100.0 * recovered / missing if missing else 0.0
        logger.info(
            "%s | %d | %d | %.1f%% | %d | %d",
            field,
            missing,
            recovered,
            gain,
            metric["regressed"],
            metric["marked_patched"],
        )

    total_missing = totals["missing_before"]
    total_recovered = totals["recovered"]
    total_gain = 100.0 * total_recovered / total_missing if total_missing else 0.0
    logger.info("-" * 100)
    logger.info(
        "TOTAL | %d | %d | %.1f%% | %d | -",
        total_missing,
        total_recovered,
        total_gain,
        totals["regressed"],
    )
    write_field_summary(
        args.summary_log_file,
        stats,
        ordered_fields,
        totals["documents_compared"],
    )
    logger.info("Field fill summary written to: %s", args.summary_log_file)
    logger.info("Repair impact analysis complete")


if __name__ == "__main__":
    main()
