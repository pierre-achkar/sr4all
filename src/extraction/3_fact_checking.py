"""
Job C: Fact-Checking and Hallucination Mitigation
- Reads aligned candidates from Job B
- Uses a FactChecker module (based on a smaller, efficient model) to verify each candidate
- If a candidate fails fact-checking, it is "nuked" (value and source set to null) in the JSON structure
- Saves the fact-checked corpus to a new JSONL file for downstream use (e.g., training, analysis)
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

# Ensure we can import src
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from extraction.fact_checker import FactChecker

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CONFIG = {
    "input_file": Path(
        "/data/sr4all/extraction_v1/raw_aligned/aligned_raw_candidates_2.jsonl"
    ),
    "output_file": Path(
        "/data/sr4all/extraction_v1/raw_fact_checked/raw_fact_checked_corpus_2.jsonl"
    ),
    "log_file": Path("/logs/extraction/raw_factcheck_2.log"),
    # Batch size for the FactChecker (Chunking)
    "batch_size": 128,
    # Number of records to collect before one MiniCheck pass.
    "record_batch_size": 50,
    # Save progress to disk every N documents
    "save_interval": 50,
}

# Setup Logging
CONFIG["output_file"].parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(CONFIG["log_file"]), logging.StreamHandler()],
)
logger = logging.getLogger("JobE")


def main():
    if not CONFIG["input_file"].exists():
        logger.error(f"Input file not found: {CONFIG['input_file']}")
        return

    # 1. Initialize Model
    # IMPORTANT: Ensure Job A (Qwen) is NOT running on the same GPU.
    try:
        checker = FactChecker(batch_size=CONFIG["batch_size"])
    except Exception as e:
        logger.critical(f"Failed to load FactChecker: {e}")
        return

    # 2. Load Data
    logger.info(f"Loading data from {CONFIG['input_file']}...")
    records = []
    with open(CONFIG["input_file"], "r") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except:
                pass

    # Check Resume Status
    completed_ids = set()
    if CONFIG["output_file"].exists():
        with open(CONFIG["output_file"], "r") as f:
            for line in f:
                try:
                    completed_ids.add(json.loads(line).get("doc_id"))
                except:
                    pass

    to_process = [r for r in records if r.get("doc_id") not in completed_ids]

    if not to_process:
        logger.info("All documents fact-checked. Exiting.")
        return

    logger.info(f"Checking {len(to_process)} remaining documents...")

    # 3. Processing Loop
    buffer = []

    for i in tqdm(
        range(0, len(to_process), CONFIG["record_batch_size"]), desc="Fact-checking"
    ):
        record_batch = to_process[i : i + CONFIG["record_batch_size"]]
        _fact_check_record_batch(record_batch, checker)
        buffer.extend(record_batch)

        # Incremental Save
        if len(buffer) >= CONFIG["save_interval"]:
            _save_chunk(buffer, CONFIG["output_file"])
            buffer = []

    # Final Save
    if buffer:
        _save_chunk(buffer, CONFIG["output_file"])

    logger.info("Fact-Checking Complete.")


def _fact_check_record_batch(records: List[Dict[str, Any]], checker: FactChecker):
    pairs_to_check = []
    field_map = []

    for record_idx, record in enumerate(records):
        data = record.get("extraction", {})
        if not data:
            continue
        _collect_pairs(data, [], record_idx, pairs_to_check, field_map)

    if not pairs_to_check:
        return

    results = checker.verify_batch(pairs_to_check)
    record_stats = {
        idx: {"checked": 0, "failed": 0}
        for idx, record in enumerate(records)
        if record.get("extraction")
    }
    failed_list_items = {}
    passed_list_items = {}

    for metadata, result in zip(field_map, results):
        record_idx = metadata["record_idx"]
        path = metadata["path"]
        path_key = (record_idx, tuple(path))

        record_stats[record_idx]["checked"] += 1

        if metadata["kind"] == "list_item":
            if result["status"] == "PASS":
                passed_list_items.setdefault(path_key, set()).add(metadata["index"])
            else:
                failed_list_items.setdefault(path_key, set()).add(metadata["index"])
                record_stats[record_idx]["failed"] += 1
            continue

        if result["status"] != "PASS":
            record_stats[record_idx]["failed"] += 1
            _null_evidence_node(records[record_idx]["extraction"], path)

    for record_idx, path_tuple in failed_list_items:
        _prune_failed_list_items(
            records[record_idx]["extraction"],
            list(path_tuple),
            passed_list_items.get((record_idx, path_tuple), set()),
        )

    for record_idx, stats in record_stats.items():
        if stats["checked"] > 0:
            records[record_idx]["fact_check_stats"] = stats


def _collect_pairs(
    item: Any,
    path: List[Any],
    record_idx: int,
    pairs_to_check: List,
    field_map: List[Dict[str, Any]],
):
    if isinstance(item, dict):
        if "verbatim_source" in item and "value" in item:
            val = item["value"]
            src = item["verbatim_source"]

            if val is not None and src is not None:
                if isinstance(val, list):
                    for item_idx, item_value in enumerate(val):
                        pairs_to_check.append((src, item_value))
                        field_map.append(
                            {
                                "record_idx": record_idx,
                                "path": path,
                                "kind": "list_item",
                                "index": item_idx,
                            }
                        )
                else:
                    pairs_to_check.append((src, val))
                    field_map.append(
                        {"record_idx": record_idx, "path": path, "kind": "field"}
                    )

        for k, v in item.items():
            if k != "verbatim_source":
                _collect_pairs(v, path + [k], record_idx, pairs_to_check, field_map)

    elif isinstance(item, list):
        for i, sub in enumerate(item):
            _collect_pairs(sub, path + [i], record_idx, pairs_to_check, field_map)


def _null_evidence_node(data: Dict[str, Any], path: List[Any]):
    target = data
    for p in path[:-1]:
        target = target[p]

    last_key = path[-1]
    if isinstance(target, dict) and last_key in target:
        target[last_key]["value"] = None
        target[last_key]["verbatim_source"] = None


def _prune_failed_list_items(data: Dict[str, Any], path: List[Any], passed_indices: set):
    target = data
    for p in path[:-1]:
        target = target[p]

    last_key = path[-1]
    if not isinstance(target, dict) or last_key not in target:
        return

    node = target[last_key]
    if not isinstance(node, dict) or not isinstance(node.get("value"), list):
        return

    node["value"] = [
        item for item_idx, item in enumerate(node["value"]) if item_idx in passed_indices
    ]

    if not node["value"]:
        node["value"] = None
        node["verbatim_source"] = None


def _save_chunk(data, path):
    with open(path, "a") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")


if __name__ == "__main__":
    main()
