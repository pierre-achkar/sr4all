"""Normalize extracted Boolean queries into OpenAlex /works query fragments using Qwen (vLLM)."""

import json
import logging
import sys
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

# Ensure we can import from src
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from oax.inference_engine import QwenInference
from oax.io_llm import LLMInput, LLMQueryItem

# ========================
# Config
# ========================
CONFIG = {
    "input_jsonl": Path(
        "/home/fhg/pie65738/projects/sr4all/data/final/sr4all_full_normalized_year_range_search_boolean_only.jsonl"
    ),
    "output_jsonl": Path(
        "/home/fhg/pie65738/projects/sr4all/data/final/with_oax/sr4all_full_normalized_year_range_search_boolean_only_oax.jsonl"
    ),
    "debug_output_jsonl": Path(
        "/home/fhg/pie65738/projects/sr4all/data/final/with_oax/sr4all_full_normalized_year_range_search_boolean_only_oax_debug.jsonl"
    ),
    "log_file": Path(
        "/home/fhg/pie65738/projects/sr4all/logs/oax/transform_to_openalex_search_boolean_only.log"
    ),
    "model_path": "Qwen/Qwen3-32B",
    "tensor_parallel": 2,
    "batch_size": 25,
    "save_every": 10,
    "skip_done": True,
    "sample_size": 0,  # 0 = process all, otherwise limit to N records
    "retry_on_error": False,
    "mode": "repair",  # "normal" | "repair"
    "structured_outputs": True,
    "enable_thinking": False,
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
logger = logging.getLogger("oax_transformer")


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
    return rec.get("id") or rec.get("doc_id")


def has_oax_error(rec: Dict) -> bool:
    if rec.get("oax_transform_error"):
        return True
    items = rec.get("oax_boolean_queries")
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict) and item.get("error"):
                return True
    return False


def normalize_outputs(outputs, prompts_meta: List[Tuple[str, int]]):
    results: List[Dict] = []
    for output, (rec_id, expected_len) in zip(outputs, prompts_meta):
        raw = output.get("raw")
        parsed = output.get("parsed")
        err = output.get("error")
        thinking = None
        if isinstance(raw, str):
            match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL | re.IGNORECASE)
            if match:
                thinking = match.group(1).strip()

        if err:
            results.append(
                {
                    "rec_id": rec_id,
                    "expected_len": expected_len,
                    "oax_list": None,
                    "error": err,
                    "raw": raw,
                    "thinking": thinking,
                    "parsed": parsed,
                }
            )
            continue

        payload = parsed or {}
        oax_list = payload.get("oax_boolean_queries")

        if not isinstance(oax_list, list):
            results.append(
                {
                    "rec_id": rec_id,
                    "expected_len": expected_len,
                    "oax_list": None,
                    "error": "MISSING_OR_INVALID_LIST",
                    "raw": raw,
                    "thinking": thinking,
                    "parsed": parsed,
                }
            )
            continue

        if expected_len != len(oax_list):
            results.append(
                {
                    "rec_id": rec_id,
                    "expected_len": expected_len,
                    "oax_list": oax_list,
                    "error": "LENGTH_MISMATCH",
                    "raw": raw,
                    "thinking": thinking,
                    "parsed": parsed,
                }
            )
            continue

        results.append(
            {
                "rec_id": rec_id,
                "expected_len": expected_len,
                "oax_list": oax_list,
                "error": None,
                "raw": raw,
                "thinking": thinking,
                "parsed": parsed,
            }
        )
    return results


def main():
    input_path = CONFIG["input_jsonl"]
    base_output_path = CONFIG["output_jsonl"]
    output_path = base_output_path
    debug_output_path = CONFIG["debug_output_jsonl"]

    if CONFIG["mode"] == "repair":
        output_path = output_path.parent / "repaired" / output_path.name
        debug_output_path = debug_output_path.parent / "repaired" / debug_output_path.name

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    debug_output_path.parent.mkdir(parents=True, exist_ok=True)

    completed_ids = set()
    error_ids = None
    base_records: Dict[str, Dict] = {}
    if base_output_path.exists():
        if CONFIG["mode"] == "repair":
            error_ids = set()
            total_base = 0
            copied_clean = 0

            if output_path.exists():
                output_path.unlink()
            if debug_output_path.exists():
                debug_output_path.unlink()

            with output_path.open("a", encoding="utf-8") as out_f:
                for rec in iter_jsonl(base_output_path):
                    total_base += 1
                    rec_id = get_record_id(rec)
                    if not rec_id:
                        continue
                    if has_oax_error(rec):
                        error_ids.add(rec_id)
                        continue
                    out_f.write(json.dumps(rec) + "\n")
                    copied_clean += 1
            logger.info(
                f"Repair mode: found {len(error_ids)} records with errors (base={total_base})."
            )
            logger.info(
                "Repair mode: will reprocess %d error records and copy %d clean records.",
                len(error_ids),
                copied_clean,
            )
        elif CONFIG["skip_done"]:
            for rec in iter_jsonl(base_output_path):
                rec_id = get_record_id(rec)
                if rec_id:
                    completed_ids.add(rec_id)
            logger.info(f"Resuming: found {len(completed_ids)} already processed.")
    elif CONFIG["mode"] == "repair":
        logger.error("Repair mode requires an existing output file to locate errors.")
        return

    logger.info("Initializing OAX inference engine...")
    engine = QwenInference(
        CONFIG["model_path"],
        tensor_parallel=CONFIG["tensor_parallel"],
        structured_outputs=CONFIG["structured_outputs"],
        enable_thinking=CONFIG["enable_thinking"],
    )

    total_records = 0
    if CONFIG["mode"] == "repair":
        total_records = len(error_ids or [])
    else:
        for rec in iter_jsonl(input_path):
            total_records += 1
    if CONFIG["sample_size"] and CONFIG["sample_size"] > 0:
        total_records = min(total_records, CONFIG["sample_size"])

    buffer: List[Dict] = []
    debug_buffer: List[Dict] = []
    batch_records: List[Dict] = []
    batch_count = 0

    def build_llm_input(queries: List[Dict], keywords: List[str]):
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

        input_meta = {
            "expected_len": expected_len,
            "queries": [
                {
                    "boolean_query_string": (q or {}).get("boolean_query_string"),
                    "database_source": (q or {}).get("database_source"),
                }
                for q in queries
            ],
            "keywords": keywords,
        }

        return llm_input, expected_len, input_meta

    def finalize_record(record: Dict, queries: List[Dict], oax_list, err, expected_len: int) -> Dict:
        record["oax_boolean_queries"] = []
        record["oax_expected_len"] = expected_len
        record["oax_transform_error"] = err

        if oax_list is None:
            oax_list = [None] * max(len(queries), 1)

        if len(oax_list) != expected_len:
            if len(oax_list) < expected_len:
                oax_list = oax_list + [None] * (expected_len - len(oax_list))
            else:
                oax_list = oax_list[:expected_len]
            err = err or "LENGTH_MISMATCH"
            record["oax_transform_error"] = err

        if len(queries) == 0:
            record["oax_boolean_queries"] = [
                {
                    "boolean_query_string": None,
                    "database_source": None,
                    "oax_boolean_query": oax,
                    **({"error": err} if err else {}),
                }
                for oax in oax_list
            ]
        else:
            for q, oax in zip(queries, oax_list):
                item = {
                    "boolean_query_string": (q or {}).get("boolean_query_string"),
                    "database_source": (q or {}).get("database_source"),
                    "oax_boolean_query": oax,
                }
                if err:
                    item["error"] = err
                record["oax_boolean_queries"].append(item)

        if oax_list is None and len(queries) > 0:
            record["oax_boolean_queries"] = [
                {
                    "boolean_query_string": (q or {}).get("boolean_query_string"),
                    "database_source": (q or {}).get("database_source"),
                    "oax_boolean_query": None,
                    "error": err or "NO_OUTPUT",
                }
                for q in queries
            ]
            record["oax_transform_error"] = err or "NO_OUTPUT"

        return record

    def flush_buffers():
        nonlocal buffer, debug_buffer
        if buffer:
            with output_path.open("a", encoding="utf-8") as f:
                for rec in buffer:
                    f.write(json.dumps(rec) + "\n")
            buffer = []
        if debug_buffer:
            with debug_output_path.open("a", encoding="utf-8") as f:
                for rec in debug_buffer:
                    f.write(json.dumps(rec) + "\n")
            debug_buffer = []

    def process_batch(batch: List[Dict]):
        nonlocal batch_count
        if not batch:
            return 0

        batch_count += 1
        logger.info("Processing batch %d (n=%d)", batch_count, len(batch))

        batch_inputs: List[LLMInput] = []
        batch_meta: List[Tuple[str, int]] = []
        batch_ctx: List[Dict] = []

        for rec in batch:
            queries = rec.get("_queries", [])
            keywords = rec.get("_keywords", [])
            llm_input, expected_len, input_meta = build_llm_input(queries, keywords)
            batch_inputs.append(llm_input)
            batch_meta.append((rec["_rec_id"], expected_len))
            batch_ctx.append(
                {
                    "rec": rec,
                    "queries": queries,
                    "expected_len": expected_len,
                    "input_meta": input_meta,
                }
            )

        outputs = engine.generate_batch(batch_inputs)
        normalized = normalize_outputs(outputs, batch_meta)

        for result, ctx in zip(normalized, batch_ctx):
            rec_id = result["rec_id"]
            oax_list = result["oax_list"]
            err = result["error"]
            input_meta = ctx["input_meta"]

            debug_buffer.append(
                {
                    "rec_id": rec_id,
                    "expected_len": result["expected_len"],
                    "input": input_meta,
                    "raw": result["raw"],
                    "parsed": result["parsed"],
                    "error": err,
                        "thinking": result.get("thinking"),
                }
            )

            if CONFIG["retry_on_error"] and err:
                retry_input = build_llm_input(ctx["queries"], input_meta.get("keywords", []))[0]
                retry_outputs = engine.generate_batch([retry_input])
                retry_meta = [(rec_id, result["expected_len"])]
                retry_result = normalize_outputs(retry_outputs, retry_meta)[0]
                oax_list = retry_result["oax_list"]
                err = retry_result["error"]
                debug_buffer.append(
                    {
                        "rec_id": rec_id,
                        "expected_len": retry_result["expected_len"],
                        "input": input_meta,
                        "raw": retry_result["raw"],
                        "parsed": retry_result["parsed"],
                        "error": retry_result["error"],
                            "thinking": retry_result.get("thinking"),
                        "retry": True,
                    }
                )

            record = finalize_record(
                ctx["rec"],
                ctx["queries"],
                oax_list,
                err,
                ctx["expected_len"],
            )
            buffer.append(record)

        if len(buffer) >= CONFIG["save_every"] or len(debug_buffer) >= CONFIG["save_every"]:
            flush_buffers()

        return len(batch)

    processed_count = 0
    with tqdm(total=total_records, desc="Normalizing", unit="rec") as pbar:
        for record in iter_jsonl(input_path):
            if CONFIG["sample_size"] and CONFIG["sample_size"] > 0:
                if processed_count >= CONFIG["sample_size"]:
                    break

            rec_id = get_record_id(record)
            if not rec_id:
                pbar.update(1)
                processed_count += 1
                continue

            if CONFIG["mode"] == "repair":
                if error_ids is None:
                    error_ids = set()
                if rec_id not in error_ids:
                    continue
            elif CONFIG["skip_done"] and rec_id in completed_ids:
                pbar.update(1)
                processed_count += 1
                continue

            queries = record.get("exact_boolean_queries") or []
            keywords = record.get("keywords_used") or []
            if not isinstance(queries, list):
                queries = []
            if not isinstance(keywords, list):
                keywords = []

            if len(queries) == 0 and len(keywords) == 0:
                record["oax_boolean_queries"] = []
                record["oax_expected_len"] = 0
                record["oax_transform_error"] = None
                buffer.append(record)
                pbar.update(1)
                processed_count += 1
                if len(buffer) >= CONFIG["save_every"]:
                    flush_buffers()
                continue

            batch_records.append(
                {
                    "_rec_id": rec_id,
                    "_queries": queries,
                    "_keywords": keywords,
                    **record,
                }
            )

            if len(batch_records) >= CONFIG["batch_size"]:
                processed_count += process_batch(batch_records)
                pbar.update(len(batch_records))
                batch_records = []

        if batch_records:
            processed_count += process_batch(batch_records)
            pbar.update(len(batch_records))
            batch_records = []

        flush_buffers()

    logger.info(f"Done. Output saved to {output_path}")


if __name__ == "__main__":
    main()
