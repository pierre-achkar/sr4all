"""Count raw output lengths from OAX trace JSONL."""

import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from tqdm import tqdm
from transformers import AutoTokenizer

# Ensure we can import from src
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ========================
# Config
# ========================
CONFIG = {
    "input_jsonl": Path(
        "/home/fhg/pie65738/projects/sr4all/data/final/with_oax/sr4all_full_normalized_year_range_search_both_oax_trace.jsonl"
    ),
    "log_file": Path(
        "/home/fhg/pie65738/projects/sr4all/logs/oax/raw_output_length_count_both.log"
    ),
    "model_path": "Qwen/Qwen3-32B",
    "sample_size": 0,  # 0 = process all, otherwise limit to N records
}

# ========================
# Logging
# ========================
CONFIG["log_file"].parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(CONFIG["log_file"]),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("oax_raw_len")


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
    return rec.get("rec_id") or rec.get("id") or rec.get("doc_id")


def percentile(values: List[int], p: float) -> Optional[int]:
    if not values:
        return None
    if p <= 0:
        return min(values)
    if p >= 1:
        return max(values)
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1))
    return ordered[idx]


def main():
    input_path = CONFIG["input_jsonl"]
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return

    logger.info("Loading tokenizer for %s...", CONFIG["model_path"])
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_path"], trust_remote_code=True)

    token_counts: List[int] = []
    char_counts: List[int] = []
    processed = 0
    skipped = 0

    with tqdm(desc="Counting raw output length", unit="rec") as pbar:
        for record in iter_jsonl(input_path):
            if CONFIG["sample_size"] and CONFIG["sample_size"] > 0:
                if processed >= CONFIG["sample_size"]:
                    break

            rec_id = get_record_id(record)
            raw = record.get("raw") or record.get("raw_output")
            if not raw:
                skipped += 1
                pbar.update(1)
                processed += 1
                continue

            token_len = len(tokenizer.encode(raw))
            char_len = len(raw)
            token_counts.append(token_len)
            char_counts.append(char_len)

            logger.info(
                "rec_id=%s raw_tokens=%d raw_chars=%d",
                rec_id,
                token_len,
                char_len,
            )

            pbar.update(1)
            processed += 1

    if token_counts:
        summary = {
            "total_records": len(token_counts),
            "skipped_empty": skipped,
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "mean_tokens": sum(token_counts) / len(token_counts),
            "p50_tokens": percentile(token_counts, 0.50),
            "p95_tokens": percentile(token_counts, 0.95),
            "p99_tokens": percentile(token_counts, 0.99),
            "min_chars": min(char_counts),
            "max_chars": max(char_counts),
            "mean_chars": sum(char_counts) / len(char_counts),
            "p50_chars": percentile(char_counts, 0.50),
            "p95_chars": percentile(char_counts, 0.95),
            "p99_chars": percentile(char_counts, 0.99),
        }
    else:
        summary = {
            "total_records": 0,
            "skipped_empty": skipped,
            "min_tokens": None,
            "max_tokens": None,
            "mean_tokens": None,
            "p50_tokens": None,
            "p95_tokens": None,
            "p99_tokens": None,
            "min_chars": None,
            "max_chars": None,
            "mean_chars": None,
            "p50_chars": None,
            "p95_chars": None,
            "p99_chars": None,
        }

    logger.info("Summary: %s", summary)


if __name__ == "__main__":
    main()
