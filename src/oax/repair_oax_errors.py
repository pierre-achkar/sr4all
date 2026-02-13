"""
Repair OAX transform errors by re-sending failed records to the LLM and patching mapping output.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

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
        "./data/final_old/sr4all_full_normalized_year_range_search_boolean_only.jsonl"
    ),
    "mapping_output_jsonl": Path(
        "./data/final_old/with_oax/sr4all_full_normalized_year_range_search_boolean_only_oax_mapping.jsonl"
    ),
    "repaired_output_jsonl": Path(
        "./data/final_old/with_oax/sr4all_full_normalized_year_range_search_boolean_only_oax_mapping_repaired.jsonl"
    ),
    "error_ids_by_type_json": Path(
        "./logs/oax/oax_error_ids_by_type_boolean_only.json"
    ),
    "log_file": Path("./logs/oax/repair_oax_errors_boolean_only.log"),
    "repaired_ids_out": Path("./logs/oax/repair_oax_repaired_ids_boolean_only.txt"),
    "still_error_ids_out": Path(
        "./logs/oax/repair_oax_still_error_ids_boolean_only.txt"
    ),
    "model_path": "Qwen/Qwen3-32B",
    "tensor_parallel": 2,
    "batch_size": 50,
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
logger = logging.getLogger("oax_repair")


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


def load_error_ids(path: Path) -> Set[str]:
    if not path.exists():
        msg = f"Error ID file not found: {path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    ids: Set[str] = set()
    ignore_types = {"LENGTH_MISMATCH", "EDITS_LENGTH_MISMATCH"}
    if isinstance(data, dict):
        for err_type, value in data.items():
            if err_type in ignore_types:
                continue
            if isinstance(value, list):
                ids.update(str(v) for v in value if v)
    return ids


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

    for output, (rec_id, expected_len) in zip(
        outputs[:min_count], prompts_meta[:min_count]
    ):
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


def build_llm_input(queries: List[Dict], keywords: List[str]):
    llm_items: List[LLMQueryItem] = []
    for q in queries:
        q_str = (q or {}).get("boolean_query_string")
        db_src = (q or {}).get("database_source")
        if not q_str:
            llm_items.append(
                LLMQueryItem(boolean_query_string="", database_source=db_src)
            )
        else:
            llm_items.append(
                LLMQueryItem(boolean_query_string=q_str, database_source=db_src)
            )

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


def build_mapping_entry(
    rec_id: str,
    expected_len: int,
    oax_list: Optional[List],
    err: Optional[str],
    edits: Optional[List],
    has_query_text: bool,
    keywords: List[str],
):
    if oax_list is None:
        oax_list = [None] * max(expected_len, 1)
    elif len(oax_list) != expected_len:
        if len(oax_list) < expected_len:
            oax_list = oax_list + [None] * (expected_len - len(oax_list))
        else:
            oax_list = oax_list[:expected_len]

    return {
        "id": rec_id,
        "oax_boolean_queries": oax_list,
        "oax_expected_len": expected_len,
        "keywords_only": (not has_query_text) and len(keywords) > 0,
        "oax_transform_error": err,
        "oax_edits": edits,
    }


def main() -> None:
    input_path = CONFIG["input_jsonl"]
    mapping_path = CONFIG["mapping_output_jsonl"]
    repaired_path = CONFIG["repaired_output_jsonl"]
    error_ids_path = CONFIG["error_ids_by_type_json"]

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return
    if not mapping_path.exists():
        logger.error("Mapping file not found: %s", mapping_path)
        return

    error_ids = load_error_ids(error_ids_path)
    if not error_ids:
        logger.info("No error IDs found. Nothing to repair.")
        return

    logger.info("Found %d error IDs to repair.", len(error_ids))

    # Collect input records to repair
    repair_records: List[Dict] = []
    for rec in iter_jsonl(input_path):
        rec_id = get_record_id(rec)
        if rec_id and rec_id in error_ids:
            repair_records.append(rec)

    if not repair_records:
        logger.info("No matching input records for error IDs. Exiting.")
        return

    logger.info("Preparing %d records for reprocessing.", len(repair_records))

    engine = QwenInference(
        CONFIG["model_path"],
        tensor_parallel=CONFIG["tensor_parallel"],
        structured_outputs=CONFIG["structured_outputs"],
        enable_thinking=CONFIG["enable_thinking"],
    )

    repaired_map: Dict[str, Dict] = {}
    repaired_ok: Set[str] = set()
    still_error: Set[str] = set()
    batch_size = CONFIG["batch_size"]

    for i in tqdm(
        range(0, len(repair_records), batch_size), desc="Repairing", unit="rec"
    ):
        batch = repair_records[i : i + batch_size]
        batch_inputs: List[LLMInput] = []
        batch_meta: List[Tuple[str, int]] = []
        batch_ctx: List[Dict] = []

        for rec in batch:
            rec_id = get_record_id(rec)
            if not rec_id:
                continue
            queries = rec.get("exact_boolean_queries") or []
            keywords = rec.get("keywords_used") or []
            if not isinstance(queries, list):
                queries = []
            if not isinstance(keywords, list):
                keywords = []

            llm_input, expected_len, _ = build_llm_input(queries, keywords)
            batch_inputs.append(llm_input)
            batch_meta.append((rec_id, expected_len))
            batch_ctx.append(
                {
                    "rec_id": rec_id,
                    "queries": queries,
                    "keywords": keywords,
                    "expected_len": expected_len,
                }
            )

        if not batch_inputs:
            continue

        try:
            outputs = engine.generate_batch(batch_inputs)
        except Exception as e:
            err_label = f"INFERENCE_EXCEPTION:{type(e).__name__}"
            for ctx in batch_ctx:
                rec_id = ctx["rec_id"]
                has_query_text = any(
                    (q or {}).get("boolean_query_string") for q in ctx["queries"]
                )
                repaired_map[rec_id] = build_mapping_entry(
                    rec_id=rec_id,
                    expected_len=ctx["expected_len"],
                    oax_list=None,
                    err=err_label,
                    edits=None,
                    has_query_text=has_query_text,
                    keywords=ctx["keywords"],
                )
                still_error.add(rec_id)
            continue

        normalized = normalize_outputs(outputs, batch_meta)

        for result, ctx in zip(normalized, batch_ctx):
            rec_id = result["rec_id"]
            oax_list = result["oax_list"]
            err = result["error"]
            edits = result.get("edits")
            has_query_text = any(
                (q or {}).get("boolean_query_string") for q in ctx["queries"]
            )

            repaired_map[rec_id] = build_mapping_entry(
                rec_id=rec_id,
                expected_len=result["expected_len"],
                oax_list=oax_list,
                err=err,
                edits=edits,
                has_query_text=has_query_text,
                keywords=ctx["keywords"],
            )
            if err:
                still_error.add(rec_id)
            else:
                repaired_ok.add(rec_id)

    # Write repaired output by patching the original mapping file
    repaired_path.parent.mkdir(parents=True, exist_ok=True)
    CONFIG["repaired_ids_out"].parent.mkdir(parents=True, exist_ok=True)
    CONFIG["still_error_ids_out"].parent.mkdir(parents=True, exist_ok=True)
    replaced = 0
    with repaired_path.open("w", encoding="utf-8") as fout:
        for rec in iter_jsonl(mapping_path):
            rec_id = get_record_id(rec)
            if rec_id and rec_id in repaired_map:
                fout.write(json.dumps(repaired_map[rec_id], ensure_ascii=False) + "\n")
                replaced += 1
            else:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with CONFIG["repaired_ids_out"].open("w", encoding="utf-8") as f:
        for rec_id in sorted(repaired_ok):
            f.write(f"{rec_id}\n")

    with CONFIG["still_error_ids_out"].open("w", encoding="utf-8") as f:
        for rec_id in sorted(still_error):
            f.write(f"{rec_id}\n")

    logger.info(
        "Repaired %d records. Output: %s | ok=%d still_error=%d",
        replaced,
        repaired_path,
        len(repaired_ok),
        len(still_error),
    )


if __name__ == "__main__":
    main()
