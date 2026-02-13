"""
Job D: Repairing Failed Extractions
- Reads the fact-checked corpus from Job C (which contains nulls where data failed)
- Uses a Qwen3-32B model to attempt to repair missing fields based on the original text and the context of the missing information
- Handles various failure modes (simple nulls, evidence nodes with null values, and "ghost objects" in boolean queries)
- Saves the repaired corpus to a new JSONL file for Job B to process again (alignment and verification)
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

# Ensure imports
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from extraction.inference_engine_batch import QwenInference
from extraction.repair_prompt import REPAIR_SYSTEM_PROMPT, get_repair_user_prompt

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CONFIG = {
    # Input: The output from Job E (which contains nulls where data failed)
    "input_file": Path(
        "/data/sr4all/extraction_v1/raw_fact_checked/raw_fact_checked_corpus_2.jsonl"
    ),
    # Output: This becomes the input for Job B (Alignment)
    "output_file": Path(
        "/data/sr4all/extraction_v1/repaired/repaired_raw_candidates_2.jsonl"
    ),
    "log_file": Path("/logs/extraction/repair_job_2.log"),
    "model_path": "Qwen/Qwen3-32B",
    # Repair Settings
    "batch_size": 20,
    "temperature": 0.1,  # Lower temp for more precise extraction
}

# Setup Logging
CONFIG["output_file"].parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("RepairJob")


# -----------------------------------------------------------------------------
# HELPER: DETECTION LOGIC
# -----------------------------------------------------------------------------
def detect_missing_keys(data: Dict) -> List[str]:
    """
    Identifies fields that need repair.
    Handles simple nulls, empty lists, and 'ghost objects' in boolean queries.
    """
    missing = []
    if not data:
        return missing

    for k, v in data.items():
        # Case A: Simple Null
        if v is None:
            missing.append(k)

        # Case B: Evidence Node Null (e.g. {"value": null})
        elif isinstance(v, dict) and "value" in v and v["value"] is None:
            missing.append(k)

        # Case C: Complex List (exact_boolean_queries)
        elif k == "exact_boolean_queries":
            if v is None or not isinstance(v, list):
                missing.append(k)
            elif len(v) == 0:
                missing.append(k)
            else:
                # Check for "ghost object" (e.g. [{"boolean_query_string": null}])
                first_item = v[0]
                if (
                    isinstance(first_item, dict)
                    and first_item.get("boolean_query_string") is None
                ):
                    missing.append(k)

    return missing


# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
def main():
    if not CONFIG["input_file"].exists():
        logger.error(f"Input file not found: {CONFIG['input_file']}")
        return

    # 1. SCAN & FILTER
    logger.info("Scanning corpus for repair candidates...")
    to_repair = []
    pass_through_count = 0

    # Check Resume Status
    completed_ids = set()
    if CONFIG["output_file"].exists():
        with open(CONFIG["output_file"], "r") as f:
            for line in f:
                try:
                    completed_ids.add(json.loads(line).get("doc_id"))
                except:
                    pass

    with open(CONFIG["input_file"], "r") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("doc_id") in completed_ids:
                    continue

                data = rec.get("extraction", {})
                missing_keys = detect_missing_keys(data)

                if missing_keys:
                    rec["missing_keys"] = missing_keys
                    to_repair.append(rec)
                else:
                    # Document is perfect, just pass it through
                    _save_record(rec, CONFIG["output_file"])
                    pass_through_count += 1
            except:
                pass

    logger.info(f"Docs already clean: {pass_through_count}")
    logger.info(f"Docs needing repair: {len(to_repair)}")

    if not to_repair:
        logger.info("Nothing to repair. Exiting.")
        return

    # 2. INITIALIZE ENGINE
    logger.info(f"Initializing Qwen (Temp={CONFIG['temperature']})...")
    engine = QwenInference(CONFIG["model_path"])
    engine.sampling_params.temperature = CONFIG["temperature"]

    # 3. REPAIR LOOP
    batch_size = CONFIG["batch_size"]

    for i in tqdm(range(0, len(to_repair), batch_size), desc="Repairing"):
        batch = to_repair[i : i + batch_size]

        # --- A. Prepare Prompts ---
        prompts = []
        valid_batch_indices = []

        for idx, rec in enumerate(batch):
            try:
                file_path = Path(rec["file_path"])
                if not file_path.exists():
                    # Cannot repair without text, pass through original
                    _save_record(rec, CONFIG["output_file"])
                    continue

                text = file_path.read_text(encoding="utf-8", errors="replace")

                # Dynamic Prompt Construction
                user_content = get_repair_user_prompt(text, rec["missing_keys"])
                msgs = [
                    {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ]

                # Apply Chat Template manually (bypassing generate_batch default)
                full_prompt = engine.tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )

                prompts.append(full_prompt)
                valid_batch_indices.append(idx)

            except Exception as e:
                logger.error(f"Error loading {rec['doc_id']}: {e}")
                _save_record(rec, CONFIG["output_file"])

        if not prompts:
            continue

        # --- B. Run Inference ---
        # Direct LLM call to use our custom prompts
        outputs = engine.llm.generate(prompts, engine.sampling_params, use_tqdm=False)

        # --- C. Patch & Save ---
        for j, output in enumerate(outputs):
            rec_idx = valid_batch_indices[j]
            record = batch[rec_idx]

            try:
                generated_json = json.loads(output.outputs[0].text)

                # Patch only the requested missing fields
                for key in record["missing_keys"]:
                    new_item = generated_json.get(key)

                    # Only overwrite if we actually got a value (not null)
                    # We rely on Job B/E to verify correctness later
                    has_data = False
                    if isinstance(new_item, dict) and new_item.get("value") is not None:
                        has_data = True
                    elif isinstance(new_item, list) and len(new_item) > 0:
                        if isinstance(new_item[0], dict) and new_item[0].get(
                            "boolean_query_string"
                        ):
                            has_data = True

                    if has_data:
                        record["extraction"][key] = new_item

                # Add metadata about the repair attempt
                record["repair_attempted"] = True

            except json.JSONDecodeError:
                logger.warning(f"Repair JSON Parse Fail for {record['doc_id']}")
                # Save original without changes

            _save_record(record, CONFIG["output_file"])

    logger.info("Repair Job Complete.")


def _save_record(rec, path):
    # Remove temporary processing keys before saving
    if "missing_keys" in rec:
        del rec["missing_keys"]

    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
