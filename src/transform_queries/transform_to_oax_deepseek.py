"""Normalize extracted Boolean queries into OpenAlex /works query fragments using DeepSeek (Azure API)."""

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv
from pydantic import ValidationError
from tqdm import tqdm

# Ensure we can import from src
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from oax.deepseek_v3 import DeepSeekV3
from oax.io_llm import LLMInput, LLMQueryItem, LLMOutput
from oax.prompts import TransformerToOAXPrompts

# ========================
# Config
# ========================
CONFIG = {
    "input_jsonl": Path(
        "/home/fhg/pie65738/projects/sr4all/data/final/sr4all_full_normalized_year_range_search_boolean_only.jsonl"
    ),
    "mapping_output_jsonl": Path(
        "/home/fhg/pie65738/projects/sr4all/data/final/with_oax/sr4all_full_normalized_year_range_search_boolean_only_oax_mapping_deepseek.jsonl"
    ),
    "trace_output_jsonl": Path(
        "/home/fhg/pie65738/projects/sr4all/data/final/with_oax/sr4all_full_normalized_year_range_search_boolean_only_oax_trace_deepseek.jsonl"
    ),
    "log_file": Path(
        "/home/fhg/pie65738/projects/sr4all/logs/oax/transform_to_openalex_search_boolean_only_deepseek.log"
    ),
    "batch_size": 50,
    "save_every": 10,
    "skip_done": True,
    "sample_size": 50,  # 0 = process all, otherwise limit to N records

    # DeepSeek API settings (Azure)
    "temperature": 0.1,
    "top_p": 0.95,
    "max_tokens": 14000,
    "timeout": 800,
    "retries": 3,
    "backoff": 5.0,
    "request_delay": 0.0,  # seconds between requests
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
logger = logging.getLogger("oax_transformer_deepseek")

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_ANALYSIS_RE = re.compile(r"<analysis>.*?</analysis>", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    if not text:
        return text
    text = _THINK_RE.sub("", text)
    text = _ANALYSIS_RE.sub("", text)
    return text.strip()


def _extract_json_candidate(text: str) -> str:
    if not text:
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def iter_jsonl(path: Path) -> Iterable[Dict]:
    """Iterate over JSONL file, yielding one record at a time."""
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


def normalize_outputs(outputs, prompts_meta: List[Tuple[str, int]]):
    results: List[Dict] = []
    output_count = len(outputs)
    meta_count = len(prompts_meta)

    if output_count != meta_count:
        logger.error(
            "Batch output count mismatch: outputs=%d meta=%d",
            output_count,
            meta_count,
        )

    min_count = min(output_count, meta_count)

    for output, (rec_id, expected_len) in zip(outputs[:min_count], prompts_meta[:min_count]):
        raw = output.get("raw")
        parsed = output.get("parsed")
        err = output.get("error")

        if err:
            results.append(
                {
                    "rec_id": rec_id,
                    "expected_len": expected_len,
                    "oax_list": None,
                    "error": err,
                    "raw": raw,
                    "parsed": parsed,
                }
            )
            continue

        payload = parsed or {}
        oax_list = payload.get("oax_boolean_queries")
        edits = payload.get("edits")

        if not isinstance(oax_list, list):
            results.append(
                {
                    "rec_id": rec_id,
                    "expected_len": expected_len,
                    "oax_list": None,
                    "error": "MISSING_OR_INVALID_LIST",
                    "raw": raw,
                    "parsed": parsed,
                    "edits": edits,
                }
            )
            continue

        if not isinstance(edits, list):
            results.append(
                {
                    "rec_id": rec_id,
                    "expected_len": expected_len,
                    "oax_list": oax_list,
                    "error": "MISSING_OR_INVALID_EDITS",
                    "raw": raw,
                    "parsed": parsed,
                    "edits": edits,
                }
            )
            continue

        if expected_len != len(edits):
            results.append(
                {
                    "rec_id": rec_id,
                    "expected_len": expected_len,
                    "oax_list": oax_list,
                    "error": "EDITS_LENGTH_MISMATCH",
                    "raw": raw,
                    "parsed": parsed,
                    "edits": edits,
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
                    "parsed": parsed,
                    "edits": edits,
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
                "parsed": parsed,
                "edits": edits,
            }
        )

    if meta_count > output_count:
        for rec_id, expected_len in prompts_meta[output_count:]:
            results.append(
                {
                    "rec_id": rec_id,
                    "expected_len": expected_len,
                    "oax_list": None,
                    "error": "MISSING_OUTPUT",
                    "raw": None,
                    "parsed": None,
                    "edits": None,
                }
            )
    return results


def _build_default_edits(expected_len: int, status: str = "ok", reason: Optional[str] = None) -> List[Dict]:
    return [
        {
            "input_index": i + 1,
            "status": status,
            "reason_if_null": reason,
            "removed_metadata": [],
            "proximity_rewrites": [],
            "wildcard_expansions": [],
        }
        for i in range(max(expected_len, 1))
    ]


def _extract_queries_from_text(text: str) -> List[str]:
    if not text:
        return []

    # Try to locate a JSON list after the oax_boolean_queries key
    match = re.search(r"oax_boolean_queries\s*:\s*\[", text)
    if match:
        start = match.end() - 1
        end = text.find("]", start)
        if end != -1:
            candidate = text[start:end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

    # Fallback: extract search=... fragments
    fragments = re.findall(r"search\s*=\s*[^\n\r]+", text, flags=re.IGNORECASE)
    cleaned = []
    for frag in fragments:
        frag = frag.strip().rstrip(",").strip()
        cleaned.append(frag)
    if cleaned:
        return cleaned

    # Fallback: any bracketed list
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return []


def _run_deepseek(client: DeepSeekV3, llm_input: LLMInput, expected_len: int) -> Dict[str, Optional[str]]:
    system_prompt, user_prompt = TransformerToOAXPrompts.render(llm_input)
    client.system_prompt = f"{system_prompt}\nDo not include <think> or <analysis> tags. Return only JSON."

    try:
        logger.info("Sent request")
        raw = client.get_response(
            user_prompt,
            retries=CONFIG["retries"],
            backoff=CONFIG["backoff"],
        )
    except Exception as e:
        return {"parsed": None, "raw": None, "error": str(e)}

    cleaned_text = _strip_thinking(raw)
    json_text = _extract_json_candidate(cleaned_text)

    try:
        obj = json.loads(json_text)
        if isinstance(obj, dict):
            if isinstance(obj.get("oax_boolean_queries"), list) and not isinstance(obj.get("edits"), list):
                obj["edits"] = _build_default_edits(expected_len)
        parsed = LLMOutput.model_validate(obj).model_dump(by_alias=True)
        logger.info("OK")
        return {"parsed": parsed, "raw": raw, "error": None}
    except ValidationError as e:
        extracted = _extract_queries_from_text(cleaned_text)
        if extracted:
            if len(extracted) < expected_len:
                extracted = extracted + [None] * (expected_len - len(extracted))
            elif len(extracted) > expected_len:
                extracted = extracted[:expected_len]
            parsed = {
                "oax_boolean_queries": extracted,
                "edits": _build_default_edits(expected_len),
            }
            logger.info("OK")
            return {"parsed": parsed, "raw": raw, "error": None}
        return {"parsed": None, "raw": raw, "error": f"SCHEMA_VALIDATION_ERROR: {str(e)}"}
    except json.JSONDecodeError as e:
        extracted = _extract_queries_from_text(cleaned_text)
        if extracted:
            if len(extracted) < expected_len:
                extracted = extracted + [None] * (expected_len - len(extracted))
            elif len(extracted) > expected_len:
                extracted = extracted[:expected_len]
            parsed = {
                "oax_boolean_queries": extracted,
                "edits": _build_default_edits(expected_len),
            }
            logger.info("OK")
            return {"parsed": parsed, "raw": raw, "error": None}
        return {"parsed": None, "raw": raw, "error": f"JSON_PARSE_ERROR: {str(e)}"}
    except Exception as e:
        return {"parsed": None, "raw": raw, "error": f"UNKNOWN_ERROR: {str(e)}"}


def main():
    input_path = CONFIG["input_jsonl"]
    mapping_output_path = CONFIG["mapping_output_jsonl"]
    trace_output_path = CONFIG["trace_output_jsonl"]

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    mapping_output_path.parent.mkdir(parents=True, exist_ok=True)
    trace_output_path.parent.mkdir(parents=True, exist_ok=True)

    dotenv_path = SRC_DIR / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path, override=True)

    # Skip if already done (resume)
    completed_ids = set()
    if CONFIG["skip_done"] and mapping_output_path.exists():
        for rec in iter_jsonl(mapping_output_path):
            rec_id = get_record_id(rec)
            if rec_id:
                completed_ids.add(rec_id)
    client = DeepSeekV3(
        system_prompt=TransformerToOAXPrompts.SYSTEM,
        temperature=CONFIG["temperature"],
        top_p=CONFIG["top_p"],
        max_tokens=CONFIG["max_tokens"],
        timeout=CONFIG["timeout"],
    )

    mapping_buffer: List[Dict] = []
    trace_buffer: List[Dict] = []
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

    def flush_buffers():
        nonlocal mapping_buffer, trace_buffer
        if mapping_buffer:
            with mapping_output_path.open("a", encoding="utf-8") as f:
                for rec in mapping_buffer:
                    f.write(json.dumps(rec) + "\n")
            mapping_buffer = []
        if trace_buffer:
            with trace_output_path.open("a", encoding="utf-8") as f:
                for rec in trace_buffer:
                    f.write(json.dumps(rec) + "\n")
            trace_buffer = []

    def process_batch(batch: List[Dict]):
        nonlocal batch_count
        if not batch:
            return 0

        batch_count += 1

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

        outputs = []
        for llm_input, expected_len in tqdm(
            zip(batch_inputs, [m[1] for m in batch_meta]),
            desc="DeepSeek requests",
            unit="req",
            leave=False,
            total=len(batch_inputs),
        ):
            outputs.append(_run_deepseek(client, llm_input, expected_len))
            if CONFIG["request_delay"] and CONFIG["request_delay"] > 0:
                time.sleep(CONFIG["request_delay"])

        normalized = normalize_outputs(outputs, batch_meta)
        if len(normalized) != len(batch_ctx):
            logger.error(
                "Normalized length mismatch: normalized=%d batch_ctx=%d",
                len(normalized),
                len(batch_ctx),
            )

        for i, ctx in enumerate(batch_ctx):
            result = normalized[i]
            rec_id = result["rec_id"]
            oax_list = result["oax_list"]
            err = result["error"]
            input_meta = ctx["input_meta"]
            has_query_text = any((q or {}).get("boolean_query_string") for q in ctx["queries"])

            trace_buffer.append(
                {
                    "rec_id": rec_id,
                    "expected_len": result["expected_len"],
                    "input": input_meta,
                    "raw": result["raw"],
                    "parsed": result["parsed"],
                    "error": err,
                }
            )

            if oax_list is None:
                oax_list = [None] * max(ctx["expected_len"], 1)
            elif len(oax_list) != ctx["expected_len"]:
                if len(oax_list) < ctx["expected_len"]:
                    oax_list = oax_list + [None] * (ctx["expected_len"] - len(oax_list))
                else:
                    oax_list = oax_list[: ctx["expected_len"]]

            mapping_buffer.append(
                {
                    "id": rec_id,
                    "oax_boolean_queries": oax_list,
                    "oax_expected_len": ctx["expected_len"],
                    "keywords_only": (not has_query_text) and len(input_meta.get("keywords", [])) > 0,
                    "oax_transform_error": err,
                    "oax_edits": result.get("edits"),
                }
            )

        if len(mapping_buffer) >= CONFIG["save_every"] or len(trace_buffer) >= CONFIG["save_every"]:
            flush_buffers()

        return len(batch)

    processed_count = 0
    try:
        with tqdm(desc="Normalizing", unit="rec") as pbar:
            for record in iter_jsonl(input_path):
                if CONFIG["sample_size"] and CONFIG["sample_size"] > 0:
                    if processed_count >= CONFIG["sample_size"]:
                        break

                rec_id = get_record_id(record)
                if not rec_id:
                    pbar.update(1)
                    processed_count += 1
                    continue

                if CONFIG["skip_done"] and rec_id in completed_ids:
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
                    trace_buffer.append(
                        {
                            "rec_id": rec_id,
                            "expected_len": 0,
                            "input": {
                                "expected_len": 0,
                                "queries": [],
                                "keywords": [],
                            },
                            "raw": None,
                            "parsed": None,
                            "error": None,
                            "skipped": True,
                        }
                    )
                    mapping_buffer.append(
                        {
                            "id": rec_id,
                            "oax_boolean_queries": [],
                            "oax_expected_len": 0,
                            "keywords_only": False,
                            "oax_transform_error": None,
                            "oax_edits": [],
                        }
                    )
                    pbar.update(1)
                    processed_count += 1
                    if len(mapping_buffer) >= CONFIG["save_every"] or len(trace_buffer) >= CONFIG["save_every"]:
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
                    n = process_batch(batch_records)
                    processed_count += n
                    pbar.update(n)
                    batch_records = []

            if batch_records:
                n = process_batch(batch_records)
                processed_count += n
                pbar.update(n)
                batch_records = []
    finally:
        flush_buffers()

    return


if __name__ == "__main__":
    main()