"""
Count prompt token lengths for OpenAlex query normalization inputs.
"""

import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm
from transformers import AutoTokenizer

# Ensure we can import from src
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from oax.io_llm import LLMInput, LLMQueryItem
from oax.prompts import TransformerToOAXPrompts

# ========================
# Config
# ========================
CONFIG = {
    "input_jsonl": Path(
        "/home/fhg/pie65738/projects/sr4all/data/final/sr4all_full_normalized_year_range_search_keywords_only.jsonl"
    ),
    "log_file": Path(
        "/home/fhg/pie65738/projects/sr4all/logs/oax/prompt_length_count_keywords_only.log"
    ),
    "model_path": "Qwen/Qwen3-32B",
    "enable_thinking": False,
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
logger = logging.getLogger("oax_prompt_len")


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


def build_llm_input(queries: List[Dict], keywords: List[str]) -> Tuple[LLMInput, int]:
    llm_items: List[LLMQueryItem] = []
    for q in queries:
        q_str = (q or {}).get("boolean_query_string")
        db_src = (q or {}).get("database_source")
        if not q_str:
            llm_items.append(LLMQueryItem(boolean_query_string="", database_source=db_src))
        else:
            llm_items.append(LLMQueryItem(boolean_query_string=q_str, database_source=db_src))

    if len(queries) == 0 and len(keywords) > 0:
        llm_input = LLMInput(queries=[], keywords=keywords)
        expected_len = 1
    else:
        llm_input = LLMInput(
            queries=llm_items,
            keywords=keywords if len(keywords) > 0 else None,
        )
        expected_len = len(queries)

    return llm_input, expected_len


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

    counts: List[int] = []
    processed = 0
    skipped = 0

    with tqdm(desc="Counting prompt tokens", unit="rec") as pbar:
        for record in iter_jsonl(input_path):
            if CONFIG["sample_size"] and CONFIG["sample_size"] > 0:
                if processed >= CONFIG["sample_size"]:
                    break

            rec_id = get_record_id(record)
            if not rec_id:
                pbar.update(1)
                processed += 1
                continue

            queries = record.get("exact_boolean_queries") or []
            keywords = record.get("keywords_used") or []
            if not isinstance(queries, list):
                queries = []
            if not isinstance(keywords, list):
                keywords = []

            if len(queries) == 0 and len(keywords) == 0:
                skipped += 1
                pbar.update(1)
                processed += 1
                continue

            llm_input, expected_len = build_llm_input(queries, keywords)
            system_prompt, user_prompt = TransformerToOAXPrompts.render(llm_input)

            full_prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=CONFIG["enable_thinking"],
            )

            token_count = len(tokenizer.encode(full_prompt))
            counts.append(token_count)

            logger.info(
                "rec_id=%s prompt_tokens=%d queries=%d keywords=%d expected_len=%d",
                rec_id,
                token_count,
                len(queries),
                len(keywords),
                expected_len,
            )

            pbar.update(1)
            processed += 1

    if counts:
        summary = {
            "total_records": len(counts),
            "skipped_empty": skipped,
            "min_tokens": min(counts),
            "max_tokens": max(counts),
            "mean_tokens": sum(counts) / len(counts),
            "p50_tokens": percentile(counts, 0.50),
            "p95_tokens": percentile(counts, 0.95),
            "p99_tokens": percentile(counts, 0.99),
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
        }

    logger.info("Summary: %s", summary)


if __name__ == "__main__":
    main()
