"""
Job A: extraction of candidate information from Markdown source files.
- Recursively reads Markdown files from an input directory
- Runs Qwen3.6-27B with schema-constrained decoding
- Writes one extraction JSON next to the mirrored source-relative path

For example, input_dir/W1/example.md becomes output_dir/W1/example.json.
Workers select deterministic modulo shards from the same lexical source order,
allowing independent tensor-parallel workers to share one output tree safely.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from extraction_v2.inference_engine_batch import QwenInference

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CONFIG = {
    "input_dir": Path(os.environ.get(
        "SR4ALL_MDS_DIR", "/home/fhg/pie65738/projects/sr4all/data/retrieval/merged/mds"
    )),
    "output_dir": Path(os.environ.get(
        "SR4ALL_EXTRACTION_DIR", "/home/fhg/pie65738/projects/sr4all/data/extraction/first_pass"
    )),
    "log_dir": Path(os.environ.get(
        "SR4ALL_EXTRACTION_LOG_DIR", "/home/fhg/pie65738/projects/sr4all/logs/extraction"
    )),
    # None means the full corpus. Set SR4ALL_MAX_DOCUMENTS for a bounded run.
    "max_documents": (
        int(os.environ["SR4ALL_MAX_DOCUMENTS"])
        if os.environ.get("SR4ALL_MAX_DOCUMENTS") else None
    ),
    "shard_index": int(os.environ.get("SR4ALL_SHARD_INDEX", "0")),
    "shard_count": int(os.environ.get("SR4ALL_SHARD_COUNT", "1")),
    "worker_name": os.environ.get("SR4ALL_EXTRACTION_WORKER", "main"),
    "model_path": os.environ.get("SR4ALL_EXTRACTION_MODEL", "Qwen/Qwen3.6-27B"),
    "tensor_parallel": int(os.environ.get("SR4ALL_EXTRACTION_TP", "2")),
    "batch_size": int(os.environ.get("SR4ALL_EXTRACTION_BATCH_SIZE", "50")),
    # Keep the longest-doc probe only for bounded smoke tests.
    "smoke_test_longest": int(os.environ.get("SR4ALL_SMOKE_TEST_LONGEST", "0")),
}

CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
CONFIG["log_dir"].mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            CONFIG["log_dir"] / f"job_a_extraction_{CONFIG['worker_name']}.log"
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("JobA")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    out_dir = CONFIG["output_dir"]

    if CONFIG["shard_count"] < 1 or not 0 <= CONFIG["shard_index"] < CONFIG["shard_count"]:
        logger.error(
            "Invalid shard settings: shard_index=%s, shard_count=%s",
            CONFIG["shard_index"], CONFIG["shard_count"],
        )
        return

    # 1. Discover one deterministic shard from the full corpus.
    input_dir = CONFIG["input_dir"]
    logger.info(f"Discovering Markdown files under {input_dir}...")
    if not input_dir.is_dir():
        logger.error("Input Markdown directory not found!")
        return
    all_records = _discover_records(
        input_dir,
        CONFIG["max_documents"],
        CONFIG["shard_index"],
        CONFIG["shard_count"],
    )
    logger.info(
        "Selected %d documents (shard %d/%d, max_documents=%s).",
        len(all_records),
        CONFIG["shard_index"] + 1,
        CONFIG["shard_count"],
        CONFIG["max_documents"] if CONFIG["max_documents"] is not None else "all",
    )
    if not all_records:
        logger.warning("No Markdown files found. Exiting.")
        return

    # 2. Resume only successful per-document extractions.
    completed_ids = _load_completed(out_dir)
    logger.info(f"Resuming: {len(completed_ids)} already extracted.")
    to_process = [r for r in all_records if r["doc_id"] not in completed_ids]
    if not to_process:
        logger.info("All documents processed. Exiting.")
        return

    # 3. Engine
    logger.info("Initializing engine...")
    try:
        engine = QwenInference(
            CONFIG["model_path"], tensor_parallel=CONFIG["tensor_parallel"]
        )
    except Exception as e:
        logger.critical(f"Failed to load engine: {e}")
        return

    fingerprint = engine.fingerprint()
    _write_run_manifest(CONFIG["log_dir"], fingerprint, len(to_process))
    logger.info(f"Run fingerprint: {fingerprint}")

    # 4. Batch loop.
    batches = _plan_batches(
        to_process, CONFIG["batch_size"], CONFIG["smoke_test_longest"]
    )
    total = len(to_process)
    logger.info(f"Extracting {total} docs in {len(batches)} batches...")
    start = time.perf_counter()

    for batch_records in tqdm(batches, desc="Processing Batches"):
        valid_records, valid_texts, io_errors = _load_texts(batch_records, engine)

        batch_output = list(io_errors)
        if valid_texts:
            llm_results = engine.generate_batch(valid_texts)
            for record, result in zip(valid_records, llm_results):
                doc_id = str(record.get("doc_id"))
                entry = {
                    "doc_id": doc_id,
                    "file_path": record.get("file_path"),
                    "extraction": result["parsed"],
                    "raw_output": result["raw"],
                    "token_metadata": result.get("token_metadata"),
                    "prompt_hash": fingerprint["prompt_hash"],
                    "schema_hash": fingerprint["schema_hash"],
                    "timestamp": time.time(),
                }
                if record.get("input_truncated"):
                    entry["input_truncated"] = True
                if result["error"]:
                    entry["error"] = result["error"]
                    logger.warning(f"Doc {doc_id} failed: {result['error']}")
                batch_output.append(entry)

        for entry in batch_output:
            _save_output(entry, out_dir)

    duration = time.perf_counter() - start
    logger.info(f"Complete. {total} docs in {duration:.2f}s.")


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def _discover_records(
    input_dir: Path,
    max_documents: int | None,
    shard_index: int,
    shard_count: int,
) -> List[Dict[str, str]]:
    paths = sorted(
        input_dir.rglob("*.md"), key=lambda path: path.relative_to(input_dir).as_posix()
    )
    if max_documents is not None:
        paths = paths[:max_documents]
    records = []
    for position, path in enumerate(paths):
        if position % shard_count != shard_index:
            continue
        relative_path = path.relative_to(input_dir)
        records.append(
            {
                "doc_id": relative_path.with_suffix("").as_posix(),
                "file_path": str(path),
                "relative_path": relative_path.as_posix(),
            }
        )
    return records


def _load_completed(output_dir: Path) -> set:
    """Doc ids with a usable extraction. Errors are deliberately not included."""
    done = set()
    for output_file in output_dir.rglob("*.json"):
        if output_file.name == "run_manifest.json":
            continue
        try:
            data = json.loads(output_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning(f"Unreadable output {output_file}; it will be retried.")
            continue
        if data.get("extraction") is not None and not data.get("error"):
            done.add(str(data.get("doc_id")))
    return done


def _plan_batches(records: List[Dict], batch_size: int, n_smoke: int) -> List[List[Dict]]:
    """Longest n_smoke documents as batch 0, then the rest in order."""
    if n_smoke <= 0 or len(records) <= n_smoke:
        return [records[i : i + batch_size] for i in range(0, len(records), batch_size)]
    smoke = records[-n_smoke:]  # records are sorted cheapest-first
    rest = records[:-n_smoke]
    return [smoke] + [rest[i : i + batch_size] for i in range(0, len(rest), batch_size)]


def _load_texts(batch_records: List[Dict], engine: QwenInference):
    valid_records, valid_texts, errors = [], [], []
    for record in batch_records:
        doc_id = str(record.get("doc_id"))
        file_path_str = record.get("file_path")
        try:
            path = Path(file_path_str)
            if not path.exists():
                errors.append(_create_error(doc_id, file_path_str, "FILE_NOT_FOUND"))
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if not text.strip():
                errors.append(_create_error(doc_id, file_path_str, "EMPTY_TEXT"))
                continue
            text, was_truncated = engine.truncate_document(text)
            if was_truncated:
                record["input_truncated"] = True
                logger.info(f"Doc {doc_id} truncated to fit the model context window.")
            valid_records.append(record)
            valid_texts.append(text)
        except Exception as e:  # noqa: BLE001
            errors.append(_create_error(doc_id, file_path_str, f"READ_ERROR: {e}"))
    return valid_records, valid_texts, errors


def _create_error(doc_id: str, file_path: Any, msg: str) -> Dict:
    return {
        "doc_id": doc_id,
        "file_path": file_path,
        "extraction": None,
        "raw_output": None,
        "error": msg,
        "timestamp": time.time(),
    }


def _save_output(entry: Dict, output_dir: Path) -> None:
    relative_path = Path(entry["doc_id"] + ".json")
    output_file = output_dir / relative_path
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_suffix(".json.tmp")
    temporary_file.write_text(
        json.dumps(entry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_file.replace(output_file)


def _write_run_manifest(out_dir: Path, fingerprint: Dict, n_planned: int) -> None:
    payload = dict(fingerprint)
    payload.update(
        {
            "batch_size": CONFIG["batch_size"],
            "documents_planned": n_planned,
            "shard_index": CONFIG["shard_index"],
            "shard_count": CONFIG["shard_count"],
            "max_documents": CONFIG["max_documents"],
            "started": time.time(),
        }
    )
    (out_dir / f"run_manifest_{CONFIG['worker_name']}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()