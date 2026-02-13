"""
Normalize extracted Boolean queries into Lucene-compatible query strings using Qwen (vLLM).
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

# Ensure we can import from src
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# --- UPDATED IMPORTS ---
from transform_queries import schemas  # The new Pydantic models
from transform_queries.prompts import TransformerToSimplePrompts  # The prompt class
from transform_queries.inference_engine import QwenInference

# ========================
# Config
# ========================
CONFIG = {
    # Update these paths to your actual file locations
    "input_jsonl": Path(
        "./data/final/with_boolean/null_subsets/sr4all_full_normalized_keywords_only_null_repair_subset_2.jsonl"
    ),
    "mapping_output_jsonl": Path(
        "./data/final/with_boolean/repaired/sr4all_full_normalized_keywords_only_repaired_mapping_2.jsonl"
    ),
    "trace_output_jsonl": Path(
        "./data/final/with_boolean/traces/sr4all_full_normalized_keywords_only_repaired_trace_2.jsonl"
    ),
    "log_file": Path("./logs/oax/transform_to_boolean_keywords_only_repaired_2.log"),
    "model_path": "Qwen/Qwen3-32B",
    "tensor_parallel": 2,
    "batch_size": 200,
    "save_every": 10,
    "skip_done": True,
    "sample_size": 0,  # 0 = process all
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
logger = logging.getLogger("lucene_transformer")


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


def normalize_outputs(outputs, prompts_meta: List[Dict]) -> List[Dict]:
    """
    Matches LLM outputs back to the input records.
    The new schema returns a list of result objects with IDs.
    """
    results: List[Dict] = []

    # Safety check for batch size mismatch
    min_count = min(len(outputs), len(prompts_meta))

    for output, meta in zip(outputs[:min_count], prompts_meta[:min_count]):
        rec_id = meta["rec_id"]
        expected_len = meta["expected_len"]

        raw = output.get("raw")
        parsed = output.get("parsed")
        err = output.get("error")

        if err:
            results.append(
                {
                    "rec_id": rec_id,
                    "expected_len": expected_len,
                    "boolean_results": None,
                    "error": err,
                    "raw": raw,
                    "parsed": parsed,
                }
            )
            continue

        # Parse the 'results' list from the LLM output
        llm_results = parsed.get("results") if parsed else None

        if not isinstance(llm_results, list):
            results.append(
                {
                    "rec_id": rec_id,
                    "expected_len": expected_len,
                    "lucene_results": None,
                    "error": "INVALID_JSON_STRUCTURE",
                    "raw": raw,
                    "parsed": parsed,
                }
            )
            continue

        # Map back to ordered list based on IDs (q_0, q_1, etc.)
        # We initialize a list of Nones
        ordered_queries = [None] * expected_len
        processing_errors = []

        for res in llm_results:
            q_id = res.get("id")
            q_bool = res.get("boolean_query")
            q_status = res.get("status")
            q_err = res.get("error_reason")

            # Parse ID "q_0" -> index 0
            try:
                idx = None
                if isinstance(q_id, str) and "_" in q_id:
                    idx = int(q_id.split("_")[1])
                elif isinstance(q_id, str) and q_id.isdigit():
                    # Keyword-only mode often returns "1" as the sole ID
                    idx = int(q_id) - 1
                if idx is None:
                    raise ValueError("Unrecognized ID format")
                if 0 <= idx < expected_len:
                    if q_status == "skipped" or q_bool is None:
                        ordered_queries[idx] = (
                            None  # Explicitly None for skipped/invalid
                        )
                        if q_err:
                            processing_errors.append(f"Idx {idx}: {q_err}")
                    else:
                        ordered_queries[idx] = q_bool
            except (IndexError, ValueError, AttributeError):
                processing_errors.append(f"Bad ID: {q_id}")

        results.append(
            {
                "rec_id": rec_id,
                "expected_len": expected_len,
                "boolean_results": ordered_queries,
                "error": "; ".join(processing_errors) if processing_errors else None,
                "raw": raw,
                "parsed": parsed,
            }
        )

    return results


def main():
    input_path = CONFIG["input_jsonl"]
    mapping_output_path = CONFIG["mapping_output_jsonl"]
    trace_output_path = CONFIG["trace_output_jsonl"]

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    mapping_output_path.parent.mkdir(parents=True, exist_ok=True)
    trace_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume logic
    completed_ids = set()
    if CONFIG["skip_done"] and mapping_output_path.exists():
        for rec in iter_jsonl(mapping_output_path):
            rec_id = get_record_id(rec)
            if rec_id:
                completed_ids.add(rec_id)
        logger.info(f"Resuming: found {len(completed_ids)} already processed.")

    logger.info("Initializing Inference Engine...")
    engine = QwenInference(
        CONFIG["model_path"],
        response_model=schemas.TransformationOutput,
        tensor_parallel=CONFIG["tensor_parallel"],
        structured_outputs=CONFIG["structured_outputs"],
        enable_thinking=CONFIG["enable_thinking"],
    )

    # Buffer setup
    mapping_buffer: List[Dict] = []
    trace_buffer: List[Dict] = []
    batch_records: List[Dict] = []
    batch_count = 0

    def build_llm_input(
        queries: List[Dict], keywords: List[str]
    ) -> Tuple[schemas.TransformationInput, int]:
        """
        Constructs the Pydantic input object.
        Generates stable IDs (q_0, q_1) for mapping back later.
        """
        query_items = []
        has_nonempty_query = False

        # 1. Handle explicit Boolean Queries
        for idx, q in enumerate(queries):
            raw_text = (q or {}).get("boolean_query_string")
            if not isinstance(raw_text, str):
                if raw_text is not None:
                    logger.warning(
                        "Non-string boolean_query_string at idx %d: %r", idx, raw_text
                    )
                raw_text = ""
            if raw_text.strip():
                has_nonempty_query = True
            # We treat empty strings as valid entries to maintain index alignment
            query_items.append(schemas.RawQueryItem(id=f"q_{idx}", raw_string=raw_text))

        # If all queries are empty, treat as no-queries
        if not has_nonempty_query:
            query_items = []

        # 2. Construct the Input Object
        # Case A: Only Keywords (Keyword Mode)
        if not query_items and keywords:
            llm_input = schemas.TransformationInput(queries=[], keywords=keywords)
            expected_len = 1  # Keywords produce exactly 1 combined query

        # Case B: Queries exist (Query Mode)
        else:
            llm_input = schemas.TransformationInput(
                queries=query_items,
                keywords=keywords if keywords else None,  # Optional context
            )
            expected_len = len(query_items)

        return llm_input, expected_len

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
        logger.info("Processing batch %d (n=%d)", batch_count, len(batch))

        prompts_to_generate = []  # List of (sys, user) tuples
        batch_meta = []  # Metadata to track requests

        # 1. Build Prompts
        for rec in batch:
            queries = rec.get("_queries", [])
            keywords = rec.get("_keywords", [])
            rec_id = rec["_rec_id"]

            # Create Pydantic Object
            llm_input_obj, expected_len = build_llm_input(queries, keywords)

            # RENDER PROMPT (Explicitly calling our new class)
            sys_prompt, user_prompt = TransformerToSimplePrompts.render(llm_input_obj)

            prompts_to_generate.append((sys_prompt, user_prompt))

            batch_meta.append(
                {
                    "rec_id": rec_id,
                    "expected_len": expected_len,
                    "queries": queries,
                    "keywords": keywords,
                    "input_obj": llm_input_obj.model_dump(),  # Save for trace
                }
            )

        # 2. Inference
        try:
            # Assumes engine.generate_batch accepts list of (sys, user)
            # If your engine expects objects, revert to passing llm_input_obj
            outputs = engine.generate_batch(prompts_to_generate)
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            # Handle catastrophic batch failure if needed
            return len(batch)

        # 3. Parse & Normalize
        normalized_results = normalize_outputs(outputs, batch_meta)

        # 4. Store Results
        for i, res in enumerate(normalized_results):
            rec_id = res["rec_id"]
            meta = batch_meta[i]

            # Trace Log (Full Debug Info)
            trace_buffer.append(
                {
                    "rec_id": rec_id,
                    "input_prompt": meta["input_obj"],
                    "raw_output": res["raw"],
                    "parsed_output": res["parsed"],
                    "error": res["error"],
                }
            )

            # Mapping Log (Clean Result)
            mapping_buffer.append(
                {
                    "id": rec_id,
                    "boolean_queries": res[
                        "boolean_results"
                    ],  # List of strings or None
                    "boolean_error": res["error"],
                    "keywords_only": (not meta["queries"]) and bool(meta["keywords"]),
                }
            )

        if len(mapping_buffer) >= CONFIG["save_every"]:
            flush_buffers()

        return len(batch)

    # --- MAIN LOOP ---
    processed_count = 0
    total_records = None
    if CONFIG["sample_size"] > 0:
        total_records = CONFIG["sample_size"]
    else:
        try:
            with input_path.open("r", encoding="utf-8") as f:
                total_records = sum(1 for _ in f)
        except Exception as e:
            logger.warning("Could not count total records for tqdm: %s", e)
    try:
        with tqdm(total=total_records, desc="Normalizing", unit="rec") as pbar:
            for record in iter_jsonl(input_path):
                # Sampling Check
                if (
                    CONFIG["sample_size"] > 0
                    and processed_count >= CONFIG["sample_size"]
                ):
                    break

                rec_id = get_record_id(record)
                if not rec_id:
                    continue

                # Skip Done
                if CONFIG["skip_done"] and rec_id in completed_ids:
                    continue

                # Prepare Data
                queries = record.get("exact_boolean_queries") or []
                keywords = record.get("keywords_used") or []
                if not isinstance(queries, list):
                    queries = []
                if not isinstance(keywords, list):
                    keywords = []

                # Skip Empty (Edge Case)
                if not queries and not keywords:
                    # Log empty record
                    mapping_buffer.append(
                        {
                            "id": rec_id,
                            "boolean_queries": [],
                            "boolean_error": "NO_INPUT_DATA",
                        }
                    )
                    continue

                batch_records.append(
                    {
                        "_rec_id": rec_id,
                        "_queries": queries,
                        "_keywords": keywords,
                    }
                )

                if len(batch_records) >= CONFIG["batch_size"]:
                    n = process_batch(batch_records)
                    processed_count += n
                    pbar.update(n)
                    batch_records = []

            # Process Remainder
            if batch_records:
                n = process_batch(batch_records)
                processed_count += n
                pbar.update(n)

    finally:
        flush_buffers()

    logger.info("Done. Saved to %s", mapping_output_path)


if __name__ == "__main__":
    main()
