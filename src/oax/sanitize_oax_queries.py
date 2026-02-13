"""
Sanitize oax_boolean_queries: ensure search= prefix, ASCII double quotes, and uppercase operators.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# ========================
# Config
# ========================
CONFIG = {
    "input_jsonl": Path(
        "./data/final_old/with_oax/sr4all_full_normalized_year_range_search_keywords_only_oax_mapping_repaired_v2.jsonl"
    ),
    "output_jsonl": Path(
        "./data/final_old/with_oax/sr4all_full_normalized_year_range_search_keywords_only_oax_mapping_repaired_v2_sanitized.jsonl"
    ),
    "log_file": Path("./logs/oax/sanitize_oax_queries_keywords_only.log"),
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
logger = logging.getLogger("oax_sanitize")

SMART_QUOTES = {
    "“": '"',
    "”": '"',
    "„": '"',
    "‟": '"',
    "’": '"',
    "‘": '"',
    "‚": '"',
    "‹": '"',
    "›": '"',
}


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


def normalize_quotes(text: str) -> str:
    for src, dst in SMART_QUOTES.items():
        text = text.replace(src, dst)
    return text


def ensure_search_prefix(query: str) -> str:
    q = query.strip()
    if not q:
        return q
    lower = q.lower()
    if lower.startswith("search="):
        return q
    if q.startswith("("):
        return f"search={q}"
    return f"search=({q})"


def uppercase_operators(query: str) -> str:
    return re.sub(
        r"\b(and|or|not)\b", lambda m: m.group(1).upper(), query, flags=re.IGNORECASE
    )


def quote_phrases(query: str) -> str:
    # Split on operators and parentheses, keep delimiters
    parts = re.split(r"(\bAND\b|\bOR\b|\bNOT\b|\(|\))", query, flags=re.IGNORECASE)
    out: List[str] = []
    for part in parts:
        if part is None or part == "":
            continue
        token = part.strip()
        if token.upper() in {"AND", "OR", "NOT", "(", ")"}:
            out.append(token.upper())
            continue
        if token.startswith("search="):
            out.append(token)
            continue
        # If token contains spaces and no quotes, wrap in quotes
        if " " in token and '"' not in token:
            token = f'"{token}"'
        out.append(token)
    return " ".join(out).replace("( ", "(").replace(" )", ")")


def sanitize_query(query: str) -> str:
    q = normalize_quotes(query)
    q = ensure_search_prefix(q)
    q = uppercase_operators(q)
    q = quote_phrases(q)
    return q


def sanitize_list(items: List[Optional[str]]) -> List[Optional[str]]:
    out: List[Optional[str]] = []
    for item in items:
        if item is None:
            out.append(None)
            continue
        if not isinstance(item, str):
            out.append(item)
            continue
        cleaned = sanitize_query(item)
        out.append(cleaned)
    return out


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
                sanitized = sanitize_list(items)
                if sanitized != items:
                    rec["oax_boolean_queries"] = sanitized
                    changed += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(
        "Sanitize complete | total=%d changed=%d output=%s", total, changed, output_path
    )
    print(f"Sanitize complete | total={total} changed={changed} output={output_path}")


if __name__ == "__main__":
    main()
