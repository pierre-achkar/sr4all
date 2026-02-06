"""Debug OpenAlex query failures with HTTP error details + local heuristics."""

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlencode, urlparse, urlunparse

import requests
from tqdm import tqdm

# ========================
# Config (adjust paths)
# ========================
CONFIG = {
    "input_jsonl": Path(
        "/home/fhg/pie65738/projects/sr4all/data/final/with_oax/sr4all_full_normalized_boolean_with_year_range_oax_with_counts.jsonl"
    ),
    "output_jsonl": Path(
        "/home/fhg/pie65738/projects/sr4all/data/final/with_oax/debug_errors/debug_oax_query_failures.jsonl"
    ),
    "summary_out": Path(
        "/home/fhg/pie65738/projects/sr4all/data/final/with_oax/debug_errors/debug_oax_query_failures_summary.json"
    ),
    "log_file": Path(
        "/home/fhg/pie65738/projects/sr4all/logs/oax/debug_oax_query_failures.log"
    ),
    "mailto": "piero.achkar.17@gmail.com",
    "api_key_env": "OPENALEX_API_KEY_2",
    "timeout_seconds": 60,
    "max_records": None,  # set to int for sampling
    "probe_zero_count": True,  # also probe count==0 queries
    "probe_only_errors": False,  # if True, only probe queries with oax_query_errors
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
logger = logging.getLogger("oax_debug")


@dataclass
class ProbeResult:
    status: int
    error: Optional[str]
    message: Optional[str]


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


def _count_jsonl_records(path: Path) -> int:
    total = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total += 1
    return total


def get_record_id(rec: Dict) -> Optional[str]:
    return rec.get("id") or rec.get("doc_id") or rec.get("rec_id")


def _prepare_oax_url(url: str) -> str:
    if not isinstance(url, str) or not url.strip():
        return ""
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    query["per-page"] = ["1"]
    mailto = CONFIG.get("mailto")
    if mailto:
        query["mailto"] = [mailto]
    api_key = os.getenv(CONFIG.get("api_key_env", "OPENALEX_API_KEY"), "")
    if api_key:
        query["api_key"] = [api_key]
    new_query = urlencode(query, doseq=True)
    return urlunparse(parsed._replace(query=new_query))


def _extract_search_string(url: str) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    raw = query.get("search", [""])[0]
    return unquote(raw)


def _heuristic_flags(search: str) -> List[str]:
    flags: List[str] = []
    if not search:
        return ["missing_search"]
    if search.count("\"") % 2 != 0:
        flags.append("unbalanced_quotes")
    if search.count("(") != search.count(")"):
        flags.append("unbalanced_parens")
    if re.search(r"^\s*(AND|OR|NOT)\b", search, flags=re.IGNORECASE):
        flags.append("leading_operator")
    if re.search(r"\b(AND|OR|NOT)\s*$", search, flags=re.IGNORECASE):
        flags.append("trailing_operator")
    if re.search(r"\b(AND|OR|NOT)\s+(AND|OR|NOT)\b", search, flags=re.IGNORECASE):
        flags.append("double_operator")
    if re.search(r"\(\s*\)", search):
        flags.append("empty_parens")
    if len(search) > 1000:
        flags.append("long_query")
    return flags


def _probe(url: str) -> ProbeResult:
    prepared = _prepare_oax_url(url)
    if not prepared:
        return ProbeResult(status=0, error="empty_url", message=None)

    try:
        resp = requests.get(prepared, timeout=CONFIG["timeout_seconds"])
    except Exception as exc:
        return ProbeResult(status=0, error="request_exception", message=str(exc))

    if resp.status_code == 200:
        return ProbeResult(status=200, error=None, message=None)

    try:
        payload = resp.json()
    except Exception:
        payload = None

    err = None
    msg = None
    if isinstance(payload, dict):
        err = payload.get("error") or payload.get("message")
        msg = payload.get("message") or payload.get("detail")
    if not err:
        err = f"http_{resp.status_code}"
    if not msg:
        msg = resp.text[:500]

    return ProbeResult(status=resp.status_code, error=err, message=msg)


def _should_probe(
    url: str,
    counts: List[int],
    errors: List[Optional[str]],
    idx: int,
) -> bool:
    if not isinstance(url, str) or not url.strip():
        return True

    if CONFIG["probe_only_errors"]:
        return bool(errors and idx < len(errors) and errors[idx])

    if errors and idx < len(errors) and errors[idx]:
        return True

    if CONFIG["probe_zero_count"]:
        if counts and idx < len(counts) and counts[idx] == 0:
            return True

    return False


def main() -> None:
    input_path = CONFIG["input_jsonl"]
    output_path = CONFIG["output_jsonl"]
    summary_path = CONFIG["summary_out"]

    if not input_path.exists():
        logger.error("Input not found: %s", input_path)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_records = 0
    probed = 0
    http_errors = 0
    empty_url = 0
    flag_counts: Dict[str, int] = {}
    status_counts: Dict[str, int] = {}

    total_expected = None
    if CONFIG["max_records"]:
        total_expected = CONFIG["max_records"]
    else:
        total_expected = _count_jsonl_records(input_path)

    with output_path.open("w", encoding="utf-8") as f_out:
        for rec in tqdm(iter_jsonl(input_path), total=total_expected, desc="Records"):
            total_records += 1
            if CONFIG["max_records"] and total_records > CONFIG["max_records"]:
                break

            rec_id = get_record_id(rec)
            urls = rec.get("oax_query") or []
            counts = rec.get("oax_query_counts") or []
            errors = rec.get("oax_query_errors") or []

            if not isinstance(urls, list):
                urls = []

            for idx, url in enumerate(urls):
                if not _should_probe(url, counts, errors, idx):
                    continue

                search = _extract_search_string(url)
                flags = _heuristic_flags(search)
                for flag in flags:
                    flag_counts[flag] = flag_counts.get(flag, 0) + 1

                result = _probe(url)
                probed += 1

                if result.error:
                    http_errors += 1
                if result.error == "empty_url":
                    empty_url += 1

                status_key = str(result.status)
                status_counts[status_key] = status_counts.get(status_key, 0) + 1

                f_out.write(
                    json.dumps(
                        {
                            "id": rec_id,
                            "query_index": idx,
                            "oax_query": url,
                            "search": search,
                            "heuristic_flags": flags,
                            "http_status": result.status,
                            "http_error": result.error,
                            "http_message": result.message,
                            "oax_query_count": counts[idx] if idx < len(counts) else None,
                            "oax_query_error": errors[idx] if idx < len(errors) else None,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    summary = {
        "total_records": total_records,
        "probed_queries": probed,
        "http_errors": http_errors,
        "empty_url": empty_url,
        "status_counts": status_counts,
        "heuristic_flags": flag_counts,
    }

    with summary_path.open("w", encoding="utf-8") as f_sum:
        json.dump(summary, f_sum, ensure_ascii=False, indent=2)

    logger.info("Summary: %s", summary)
    print(f"Wrote debug output to: {output_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
