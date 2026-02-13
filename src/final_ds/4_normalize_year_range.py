"""Heuristically normalize year_range values using study year as cap."""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

CONFIG = {
    "input_jsonl": Path("./sr4all/data/final/sr4all_full.jsonl"),
    "output_jsonl": Path("./sr4all/data/final/sr4all_full_normalized_year_range.jsonl"),
    "log_file": Path("./sr4all/logs/final_ds/normalize_year_range.log"),
    "overwrite": False,  # if True, overwrite year_range.value (or year_range if string)
}

CONFIG["log_file"].parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(CONFIG["log_file"]),
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("normalize_year_range")

_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
_YEAR_RANGE_RE = re.compile(r"^(?:19|20)\d{2}-(?:19|20)\d{2}$")
_YEAR_ONLY_RE = re.compile(r"^(?:19|20)\d{2}$")
_PRESENT_RE = re.compile(
    r"\b(present|current|to date|to-date|now|today)\b", re.IGNORECASE
)
_LAST_RE = re.compile(
    r"\b(last|past|previous|recent)\s+(\d{1,3})\s+years?\b", re.IGNORECASE
)
_SINCE_RE = re.compile(r"\b(since|from)\b", re.IGNORECASE)
_DECADE_RE = re.compile(
    r"\b(last|past|previous|recent)\s+(\d{1,2})\s+decades?\b", re.IGNORECASE
)
_DECADE_WORD_RE = re.compile(r"\b(last|past|previous|recent)\s+decade\b", re.IGNORECASE)
_FEW_RE = re.compile(r"\b(last|past|recent)\s+few\s+years?\b", re.IGNORECASE)
_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
_LAST_WORD_RE = re.compile(
    r"\b(last|past|previous|recent)\s+(one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\s+years?\b",
    re.IGNORECASE,
)


def _extract_years(text: str) -> list[int]:
    return [int(y) for y in _YEAR_RE.findall(text)]


def _is_valid_range(value: str) -> bool:
    return bool(_YEAR_RANGE_RE.match(value) or _YEAR_ONLY_RE.match(value))


def normalize_year_range(
    raw: Optional[str], study_year: Optional[int]
) -> Tuple[Optional[str], Optional[str]]:
    if not raw:
        if study_year:
            return str(study_year), "fallback_study_year"
        return None, None

    text = str(raw).strip()
    years = _extract_years(text)

    # Case 1: two explicit years (sorted)
    if len(years) >= 2:
        y1, y2 = sorted(years[:2])
        candidate = f"{y1}-{y2}"
        return (
            (candidate, "two_years")
            if _is_valid_range(candidate)
            else (None, "invalid_two_years")
        )

    # Case 2: year to present/current
    if len(years) == 1 and _PRESENT_RE.search(text):
        if study_year:
            candidate = f"{years[0]}-{study_year}"
            return (
                (candidate, "year_to_present")
                if _is_valid_range(candidate)
                else (None, "invalid_year_to_present")
            )
        return None, "year_to_present_missing_study_year"

    # Case 3: since/from YEAR (implicit end)
    if len(years) == 1 and _SINCE_RE.search(text):
        if study_year:
            candidate = f"{years[0]}-{study_year}"
            return (
                (candidate, "since_year")
                if _is_valid_range(candidate)
                else (None, "invalid_since_year")
            )
        return None, "since_year_missing_study_year"

    # Case 4: last/past N years (numeric)
    m = _LAST_RE.search(text)
    if m and study_year:
        n = int(m.group(2))
        start = max(study_year - n + 1, 0)
        candidate = f"{start}-{study_year}"
        return (
            (candidate, "last_n_years")
            if _is_valid_range(candidate)
            else (None, "invalid_last_n_years")
        )

    # Case 5: last/past N years (word numbers)
    m = _LAST_WORD_RE.search(text)
    if m and study_year:
        n = _NUMBER_WORDS.get(m.group(2).lower())
        if n:
            start = max(study_year - n + 1, 0)
            candidate = f"{start}-{study_year}"
            return (
                (candidate, "last_word_years")
                if _is_valid_range(candidate)
                else (None, "invalid_last_word_years")
            )

    # Case 6: last/past N decades
    m = _DECADE_RE.search(text)
    if m and study_year:
        n = int(m.group(2))
        years_span = n * 10
        start = max(study_year - years_span + 1, 0)
        candidate = f"{start}-{study_year}"
        return (
            (candidate, "last_n_decades")
            if _is_valid_range(candidate)
            else (None, "invalid_last_n_decades")
        )

    # Case 7: last/past decade (single)
    if _DECADE_WORD_RE.search(text) and study_year:
        start = max(study_year - 10 + 1, 0)
        candidate = f"{start}-{study_year}"
        return (
            (candidate, "last_decade")
            if _is_valid_range(candidate)
            else (None, "invalid_last_decade")
        )

    # Case 8: last/past few years
    if _FEW_RE.search(text) and study_year:
        start = max(study_year - 5 + 1, 0)
        candidate = f"{start}-{study_year}"
        return (
            (candidate, "last_few_years")
            if _is_valid_range(candidate)
            else (None, "invalid_last_few_years")
        )

    if study_year:
        return str(study_year), "fallback_study_year"
    return None, "no_rule"


def _get_year_range_value(rec: Dict) -> Optional[str]:
    yr = rec.get("year_range")
    if isinstance(yr, dict):
        return yr.get("value")
    if isinstance(yr, str):
        return yr
    return None


def _set_year_range_value(rec: Dict, normalized: str):
    yr = rec.get("year_range")
    if isinstance(yr, dict):
        yr["value"] = normalized
        rec["year_range"] = yr
    else:
        rec["year_range"] = normalized


def main():
    input_path = CONFIG["input_jsonl"]
    output_path = CONFIG["output_jsonl"]

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    normalized_count = 0
    rule_counts: Dict[str, int] = {}

    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1
            raw = _get_year_range_value(rec)
            study_year = rec.get("year")
            normalized, rule = normalize_year_range(raw, study_year)

            if rule:
                rule_counts[rule] = rule_counts.get(rule, 0) + 1

            if normalized:
                normalized_count += 1
                if CONFIG["overwrite"]:
                    _set_year_range_value(rec, normalized)
                else:
                    rec["year_range_normalized"] = normalized
                    rec["year_range_normalization_rule"] = rule
            else:
                if not CONFIG["overwrite"]:
                    rec["year_range_normalized"] = None
                    rec["year_range_normalization_rule"] = rule

            fout.write(json.dumps(rec) + "\n")

    logger.info(
        "Done. Total=%d, normalized=%d, output=%s",
        total,
        normalized_count,
        output_path,
    )
    logger.info("Rule counts: %s", json.dumps(rule_counts, sort_keys=True))


if __name__ == "__main__":
    main()
