"""
Job D: targeted repair of missing extraction fields.

Reads fact-checked per-document JSONs, re-asks the model for fields that are still
missing, and writes mirrored per-document JSONs.

    data/extraction/fact_checked/W1/abc.json -> data/extraction/repaired/W1/abc.json

Points that differ from a naive repair pass:
  * Fields are asked for INDIVIDUALLY via a subset schema, so constrained decoding
    cannot emit the nine fields we already have.
  * Rejections are fed back as negative evidence. With greedy decoding, a plain
    re-ask usually reproduces the rejected value; naming it rules it out.
  * Verification annotations are STRIPPED from the extraction before writing, so the
    output can be re-validated by the alignment stage (schema is extra="forbid").
    The prior verdicts are preserved under `previous_verdicts`.
  * Generation errors are NOT written, so a transient failure stays retryable.
    Terminal conditions (source missing, document too long) are written.
  * repair_round is capped, so a field that keeps failing is not re-asked forever.

    python src/extraction_v2/4_repair.py --dry-run
    python src/extraction_v2/4_repair.py --limit 100
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from extraction_v2.repair_prompt import (
    REPAIR_SYSTEM_PROMPT, get_repair_user_prompt, subset_schema,
)

if TYPE_CHECKING:
    from extraction_v2.inference_engine_batch import QwenInference

PROJECT = Path(os.environ.get("SR4ALL_PROJECT", "/home/fhg/pie65738/projects/sr4all"))


def _path(env_name: str, default: Path) -> Path:
    return Path(os.environ.get(env_name, str(default)))


CONFIG = {
    # Must match the fact-checking stage's output_root.
    "input_root": _path("SR4ALL_FACT_CHECKED_DIR", PROJECT / "data/extraction/fact_checked"),
    "output_root": _path("SR4ALL_REPAIRED_DIR", PROJECT / "data/extraction/repaired"),
    "mds_root": _path("SR4ALL_MDS_DIR", PROJECT / "data/retrieval/merged/mds"),
    "log_dir": _path("SR4ALL_REPAIR_LOG_DIR", PROJECT / "logs/extraction"),
    "model_path": os.environ.get("SR4ALL_REPAIR_MODEL", "Qwen/Qwen3.6-27B"),
    "tensor_parallel": int(os.environ.get("SR4ALL_REPAIR_TP", "2")),
    "batch_size": int(os.environ.get("SR4ALL_REPAIR_BATCH_SIZE", "20")),
    "max_repair_rounds": int(os.environ.get("SR4ALL_MAX_REPAIR_ROUNDS", "1")),
    "shard_index": int(os.environ.get("SR4ALL_SHARD_INDEX", "0")),
    "shard_count": int(os.environ.get("SR4ALL_SHARD_COUNT", "1")),
    "worker_name": os.environ.get("SR4ALL_REPAIR_WORKER", "main"),
}

# Annotations written by the alignment and fact-check stages. The schema forbids
# extra keys, so these must not survive into a record that goes back through
# alignment.
ANNOTATION_KEYS = (
    "verification", "verification_score",
    "fact_check", "fact_check_probability", "fact_check_probabilities",
)

logger = logging.getLogger("JobD")


def _progress(iterable, **kwargs):
    try:
        from tqdm import tqdm

        return tqdm(iterable, **kwargs)
    except Exception:  # noqa: BLE001
        return iterable


# -----------------------------------------------------------------------------
# DETECTION
# -----------------------------------------------------------------------------
def detect_missing_keys(extraction: Dict[str, Any]) -> List[str]:
    if not isinstance(extraction, dict):
        return []
    missing: List[str] = []
    for key, node in extraction.items():
        if key == "exact_boolean_queries":
            if not isinstance(node, list) or not any(
                isinstance(i, dict) and str(i.get("boolean_query_string") or "").strip()
                for i in node
            ):
                missing.append(key)
            continue
        if not isinstance(node, dict) or "value" not in node:
            continue
        value = node.get("value")
        if value is None:
            missing.append(key)
        elif isinstance(value, str) and not value.strip():
            missing.append(key)
        elif isinstance(value, list) and not value:
            missing.append(key)
    return missing


def collect_rejections(record: Dict[str, Any], fields: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Why is each field null? Three causes, and only the first is a pure recall play:
      never extracted  -> no annotation, no rejection entry
      alignment        -> verification in NOT_FOUND / MISSING_SOURCE / SPAN_TOO_SHORT
      fact-check       -> fact_check == FAIL
    The rejected value comes from extraction_raw, which the alignment stage retains.
    """
    current = record.get("extraction") or {}
    raw = record.get("extraction_raw") or {}
    out: Dict[str, Dict[str, Any]] = {}
    for f in fields:
        node = current.get(f)
        if not isinstance(node, dict):
            continue
        verdict, fc = node.get("verification"), node.get("fact_check")
        reason = None
        if fc == "FAIL":
            prob = node.get("fact_check_probability")
            reason = "fact-check: value not supported by its own span"
            if prob is not None:
                reason += f" (support {prob})"
        elif verdict in ("NOT_FOUND", "MISSING_SOURCE", "SPAN_TOO_SHORT"):
            reason = {
                "NOT_FOUND": "alignment: quoted span does not occur in the document",
                "MISSING_SOURCE": "alignment: value given with no supporting span",
                "SPAN_TOO_SHORT": "alignment: span too short to ground the value",
            }[verdict]
        if reason is None:
            continue
        prior = raw.get(f) if isinstance(raw.get(f), dict) else {}
        out[f] = {
            "value": prior.get("value"),
            "verbatim_source": prior.get("verbatim_source"),
            "reason": reason,
        }
    return out


def has_usable_data(field: str, node: Any) -> bool:
    if field == "exact_boolean_queries":
        return isinstance(node, list) and any(
            isinstance(i, dict) and str(i.get("boolean_query_string") or "").strip()
            for i in node
        )
    if not isinstance(node, dict):
        return False
    value = node.get("value")
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return len(value) > 0
    return True


# -----------------------------------------------------------------------------
# RECORD SHAPING
# -----------------------------------------------------------------------------
def strip_annotations(extraction: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split extraction into (clean extraction, per-field prior verdicts)."""
    clean: Dict[str, Any] = {}
    verdicts: Dict[str, Any] = {}
    for key, node in (extraction or {}).items():
        if isinstance(node, dict):
            kept = {k: v for k, v in node.items() if k not in ANNOTATION_KEYS}
            note = {k: v for k, v in node.items() if k in ANNOTATION_KEYS}
            clean[key] = kept
            if note:
                verdicts[key] = note
        elif isinstance(node, list):
            items, notes = [], []
            for item in node:
                if isinstance(item, dict):
                    items.append({k: v for k, v in item.items() if k not in ANNOTATION_KEYS})
                    notes.append({k: v for k, v in item.items() if k in ANNOTATION_KEYS})
                else:
                    items.append(item)
                    notes.append({})
            clean[key] = items
            if any(notes):
                verdicts[key] = notes
        else:
            clean[key] = node
    return clean, verdicts


def resolve_source_path(record: Dict[str, Any], src_json: Path) -> Optional[Path]:
    for candidate in (record.get("source_path"), record.get("file_path")):
        if candidate and Path(candidate).exists():
            return Path(candidate)
    shard = record.get("shard") or src_json.parent.name
    guess = CONFIG["mds_root"] / shard / f"{src_json.stem}.md"
    return guess if guess.exists() else None


def finalize(record: Dict[str, Any], patched: List[str], missing: List[str],
             error: Optional[str] = None, **extra) -> Dict[str, Any]:
    clean, verdicts = strip_annotations(record.get("extraction") or {})
    out = dict(record)
    out["extraction"] = clean
    if verdicts:
        out["previous_verdicts"] = verdicts
    out["repair_round"] = int(record.get("repair_round") or 0) + 1
    out["repair_missing_keys"] = missing
    out["repair_patched_keys"] = patched
    out["repair_attempted"] = bool(extra.pop("attempted", bool(missing) and error is None))
    if error:
        out["repair_error"] = error
    out.update(extra)
    return out


# -----------------------------------------------------------------------------
# SUBSET GENERATION
# -----------------------------------------------------------------------------
def generate_subset(engine: "QwenInference", prompts: List[str],
                    schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Constrained generation against a SUBSET schema, so decoding cannot emit the
    fields we already have. The engine's own sampling params are pinned to the full
    ReviewExtraction grammar, hence the local build here; sampling values are copied
    from the engine so the two stages stay identical apart from the schema.
    """
    from vllm import SamplingParams

    try:
        from vllm.sampling_params import StructuredOutputsParams

        extra = {"structured_outputs": StructuredOutputsParams(json=schema)}
    except ImportError:
        extra = {"guided_json": schema}

    params = SamplingParams(
        temperature=engine.config["temperature"],
        top_p=engine.config["top_p"],
        max_tokens=engine.config["max_tokens"],
        presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0,
        **extra,
    )

    try:
        outputs = engine.llm.generate(prompts, params, use_tqdm=False)
    except Exception as e:  # noqa: BLE001
        logger.critical(f"Repair generation failed for {len(prompts)} prompts: {e}")
        return [{"parsed": None, "error": f"GENERATION_ERROR: {e}"} for _ in prompts]

    results = []
    for output in outputs:
        completion = output.outputs[0]
        entry = {
            "parsed": None, "error": None,
            "token_metadata": {
                "input_tokens": len(output.prompt_token_ids),
                "output_tokens": len(completion.token_ids),
                "finish_reason": completion.finish_reason,
            },
        }
        # Grammar-constrained, so this should always parse; a failure means the
        # completion was truncated at max_tokens.
        try:
            entry["parsed"] = json.loads(completion.text)
        except json.JSONDecodeError as e:
            entry["error"] = f"JSON_PARSE_ERROR: {e}"
        results.append(entry)
    return results


# -----------------------------------------------------------------------------
# BATCH
# -----------------------------------------------------------------------------
def run_batch(engine: "QwenInference",
              items: List[Tuple[Path, Path, Dict[str, Any]]],
              summary: Counter, prompt_hash: str) -> None:
    # Group by requested field set: one subset schema per grammar, so documents with
    # the same missing fields share a constrained-decoding group.
    groups: Dict[Tuple[str, ...], List[Tuple[Path, Dict, List[str], str]]] = {}

    for src, dst, record in items:
        extraction = record.get("extraction") or {}
        missing = detect_missing_keys(extraction)

        if not missing:
            write_json(dst, finalize(record, [], [], attempted=False))
            summary["passthrough_clean"] += 1
            continue

        if int(record.get("repair_round") or 0) >= CONFIG["max_repair_rounds"]:
            write_json(dst, finalize(record, [], missing, attempted=False,
                                     repair_skipped="MAX_ROUNDS_REACHED"))
            summary["max_rounds_reached"] += 1
            continue

        md = resolve_source_path(record, src)
        if md is None:
            write_json(dst, finalize(record, [], missing, error="SOURCE_MARKDOWN_NOT_FOUND"))
            summary["source_missing"] += 1
            continue
        try:
            text = md.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            write_json(dst, finalize(record, [], missing, error=f"SOURCE_READ_ERROR: {e}"))
            summary["source_read_error"] += 1
            continue
        if not text.strip():
            write_json(dst, finalize(record, [], missing, error="EMPTY_SOURCE_TEXT"))
            summary["empty_source"] += 1
            continue

        rejections = collect_rejections(record, missing)
        prompt_body = get_repair_user_prompt(text, missing, rejections)
        if prompt_body is None:
            write_json(dst, finalize(record, [], missing, attempted=False,
                                     repair_skipped="NO_REPAIRABLE_FIELDS"))
            summary["no_repairable_fields"] += 1
            continue

        prompt = engine.build_prompt(REPAIR_SYSTEM_PROMPT, prompt_body)
        # Truncating would cut the tail, which is where Methods usually sits, so an
        # over-long document is skipped rather than silently blinded.
        n_tokens = len(engine.tokenizer(prompt)["input_ids"])
        budget = CONFIG["max_model_len"] - CONFIG["max_tokens"]
        if n_tokens > budget:
            write_json(dst, finalize(record, [], missing,
                                     error=f"TOO_LONG: {n_tokens} > {budget}"))
            summary["too_long"] += 1
            continue

        key = tuple(sorted(missing))
        groups.setdefault(key, []).append((dst, record, missing, prompt))

    for fields, group in groups.items():
        try:
            schema = subset_schema(list(fields))
        except ValueError:
            for dst, record, missing, _p in group:
                write_json(dst, finalize(record, [], missing, attempted=False,
                                         repair_skipped="NO_REPAIRABLE_FIELDS"))
                summary["no_repairable_fields"] += 1
            continue

        prompts = [p for _d, _r, _m, p in group]
        results = generate_subset(engine, prompts, schema)

        for (dst, record, missing, _p), result in zip(group, results):
            if result.get("error"):
                # Not written: absence of the output file is what makes it retryable.
                logger.warning(f"{record.get('doc_id')}: {result['error']}; deferred")
                summary["deferred_generation_error"] += 1
                continue

            parsed = result.get("parsed") or {}
            patched = []
            for key in missing:
                candidate = parsed.get(key)
                if has_usable_data(key, candidate):
                    record.setdefault("extraction", {})[key] = candidate
                    patched.append(key)

            out = finalize(record, patched, missing,
                           repair_token_metadata=result.get("token_metadata"),
                           repair_prompt_hash=prompt_hash)
            write_json(dst, out)
            if patched:
                summary["repaired_documents"] += 1
                summary["repaired_fields"] += len(patched)
                for k in patched:
                    summary[f"field_repaired:{k}"] += 1
            else:
                summary["no_new_data"] += 1


# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------
def find_inputs(input_root: Path) -> List[Path]:
    return sorted(
        p for d in sorted(input_root.iterdir()) if d.is_dir()
        for p in sorted(d.glob("*.json")) if not p.stem.startswith("_")
    )


def output_path(src: Path, input_root: Path, output_root: Path) -> Path:
    return output_root / src.relative_to(input_root)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def setup_logging(log_dir: Path, worker_name: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_dir / f"repair_{worker_name}.log"),
            logging.StreamHandler(),
        ],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    input_root, output_root = CONFIG["input_root"], CONFIG["output_root"]
    setup_logging(CONFIG["log_dir"], CONFIG["worker_name"])

    if CONFIG["shard_count"] < 1 or not 0 <= CONFIG["shard_index"] < CONFIG["shard_count"]:
        logger.error(
            "Invalid shard settings: shard_index=%s, shard_count=%s",
            CONFIG["shard_index"],
            CONFIG["shard_count"],
        )
        return

    if not input_root.exists():
        logger.error(f"Input root not found: {input_root}")
        return

    jobs: List[Tuple[Path, Path]] = []
    skipped = 0
    inputs = find_inputs(input_root)
    shard_inputs = [
        src for position, src in enumerate(inputs)
        if position % CONFIG["shard_count"] == CONFIG["shard_index"]
    ]
    logger.info(
        "Selected %d of %d inputs (shard %d/%d).",
        len(shard_inputs),
        len(inputs),
        CONFIG["shard_index"] + 1,
        CONFIG["shard_count"],
    )
    for src in shard_inputs:
        dst = output_path(src, input_root, output_root)
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue
        jobs.append((src, dst))
    if args.limit:
        jobs = jobs[: args.limit]
    logger.info(f"{len(jobs)} to repair, {skipped} already done, under {input_root}")

    if args.dry_run:
        for src, dst in jobs[:20]:
            logger.info(f"  {src} -> {dst}")
        logger.info(f"Dry run: {len(jobs)} documents. Model not loaded.")
        return
    if not jobs:
        logger.info("Nothing to do.")
        return

    from extraction_v2.inference_engine_batch import QwenInference
    from extraction_v2 import repair_prompt as _rp

    engine = QwenInference(
        model_path=CONFIG["model_path"], tensor_parallel=CONFIG["tensor_parallel"]
    )
    CONFIG["max_model_len"] = engine.config["max_model_len"]
    CONFIG["max_tokens"] = engine.config["max_tokens"]
    prompt_hash = hashlib.sha256(
        (REPAIR_SYSTEM_PROMPT + _rp.BASE_INSTRUCTIONS
         + json.dumps(_rp.FIELD_RULES, sort_keys=True)).encode()
    ).hexdigest()[:12]
    logger.info(f"Repair prompt hash: {prompt_hash}")

    summary = Counter({"documents_total": len(jobs), "skipped_existing": skipped})
    started = time.perf_counter()

    for i in _progress(range(0, len(jobs), CONFIG["batch_size"]), desc="Repairing"):
        chunk = jobs[i : i + CONFIG["batch_size"]]
        records: List[Tuple[Path, Path, Dict[str, Any]]] = []
        for src, dst in chunk:
            try:
                records.append((src, dst, json.loads(src.read_text(encoding="utf-8"))))
            except (OSError, json.JSONDecodeError) as e:
                logger.error(f"Unreadable {src}: {e}")
                summary["unreadable"] += 1
        run_batch(engine, records, summary, prompt_hash)

    elapsed = time.perf_counter() - started
    summary["elapsed_seconds"] = round(elapsed, 2)
    payload = dict(summary)
    payload["repair_prompt_hash"] = prompt_hash
    payload["max_repair_rounds"] = CONFIG["max_repair_rounds"]
    (CONFIG["log_dir"] / f"repair_summary_{CONFIG['worker_name']}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    logger.info(f"Repair complete in {elapsed:.1f}s")
    logger.info("Summary:\n" + json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()