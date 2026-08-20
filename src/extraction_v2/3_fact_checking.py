"""
Job C: fact checking and hallucination mitigation.

Reads aligned per-document JSONs, checks every extracted claim against its own
verbatim span with MiniCheck, and writes fact-checked JSONs in the same layout.

    data/aligned/W1/abc.json  ->  data/fact_checked/W1/abc.json

Behaviour worth knowing:
  * Values are verbalized into sentences before checking (see fact_checker.py).
  * The support threshold depends on the alignment verdict for that span. A NOISY
    span is garbled, so entailment is harder through no fault of the extraction;
    it gets a lower bar than a VERIFIED span.
    * A document with ANY errored claim is not written, so it stays retryable.
    * Fact-check failures annotate the existing evidence; they do not delete it.
        Later stages can use the probabilities without losing potentially useful spans.

    python src/extraction_v2/3_fact_checking.py --dry-run
    python src/extraction_v2/3_fact_checking.py --limit 100
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

PROJECT = Path(os.environ.get("SR4ALL_PROJECT", "/home/fhg/pie65738/projects/sr4all"))

CONFIG = {
    # Must match the alignment stage's output_root.
    "input_root": Path(os.environ.get(
        "SR4ALL_ALIGNED_DIR", PROJECT / "data/extraction/aligned"
    )),
    "output_root": Path(os.environ.get(
        "SR4ALL_FACT_CHECKED_DIR", PROJECT / "data/extraction/fact_checked"
    )),
    "log_dir": Path(os.environ.get(
        "SR4ALL_FACT_CHECK_LOG_DIR", PROJECT / "logs/extraction"
    )),
    "chunk_size": 128,
    "doc_batch_size": 50,
    # Support-probability floors, by the alignment verdict on the span.
    "threshold_verified": 0.35,
    "threshold_noisy": 0.20,
    "threshold_unknown": 0.30,
    "shard_index": int(os.environ.get("SR4ALL_SHARD_INDEX", "0")),
    "shard_count": int(os.environ.get("SR4ALL_SHARD_COUNT", "1")),
    "worker_name": os.environ.get("SR4ALL_FACT_CHECK_WORKER", "main"),
}

logger = logging.getLogger("JobC")

LIST_VALUE_FIELDS = {
    "research_questions", "keywords_used", "databases_used",
    "inclusion_criteria", "exclusion_criteria",
}


# -----------------------------------------------------------------------------
# CLAIM COLLECTION
# -----------------------------------------------------------------------------
def collect_claims(extraction: Dict[str, Any], doc_idx: int) -> List["Claim"]:
    from extraction_v2.fact_checker import Claim, verbalize

    claims: List[Claim] = []
    for field, node in (extraction or {}).items():

        if field == "exact_boolean_queries":
            for i, item in enumerate(node or []):
                span = item.get("verbatim_source")
                if not span:
                    continue
                for db in item.get("database_source") or []:
                    sentence = verbalize("database_source", db)
                    if sentence:
                        claims.append(Claim(
                            doc_idx=doc_idx, path=(field, i), field="database_source",
                            kind="query_database", span=span, sentence=sentence,
                            span_quality=item.get("verification", "UNKNOWN"),
                        ))
            continue

        if not isinstance(node, dict):
            continue
        span, value = node.get("verbatim_source"), node.get("value")
        quality = node.get("verification", "UNKNOWN")
        if not span or value is None:
            continue
        # snowballing == False is an absence, not a claim needing support.
        if isinstance(value, bool) and value is False:
            continue

        if field in LIST_VALUE_FIELDS and isinstance(value, list):
            for i, item in enumerate(value):
                sentence = verbalize(field, item)
                if sentence:
                    claims.append(Claim(
                        doc_idx=doc_idx, path=(field,), field=field, kind="list_item",
                        span=span, sentence=sentence, index=i, span_quality=quality,
                    ))
        else:
            sentence = verbalize(field, value)
            if sentence:
                claims.append(Claim(
                    doc_idx=doc_idx, path=(field,), field=field, kind="field",
                    span=span, sentence=sentence, span_quality=quality,
                ))
    return claims


def threshold_for(span_quality: str) -> float:
    return {
        "VERIFIED": CONFIG["threshold_verified"],
        "NOISY": CONFIG["threshold_noisy"],
    }.get(span_quality, CONFIG["threshold_unknown"])


# -----------------------------------------------------------------------------
# APPLYING RESULTS
# -----------------------------------------------------------------------------
def apply_results(records: List[Dict], claims: List, results: List[Dict]) -> List[bool]:
    """
    Writes verdicts into each record's extraction. Returns a per-record flag:
    True if the record had at least one ERROR claim and must not be written.
    """
    errored = [False] * len(records)
    stats = [Counter() for _ in records]
    # (doc_idx, field) -> {index: (passed, probability)}
    list_votes: Dict[Tuple[int, str], Dict[int, Tuple[bool, float]]] = {}
    query_prune: Dict[int, set] = {}

    for claim, res in zip(claims, results):
        d = claim.doc_idx
        if res["status"] == "ERROR":
            errored[d] = True
            stats[d]["errors"] += 1
            continue

        prob = res["support_probability"]
        passed = prob >= threshold_for(claim.span_quality)
        stats[d]["checked"] += 1
        stats[d]["passed" if passed else "failed"] += 1

        extraction = records[d].get("extraction") or {}

        if claim.kind == "list_item":
            list_votes.setdefault((d, claim.field), {})[claim.index] = (passed, prob)
        elif claim.kind == "query_database":
            node = extraction.get("exact_boolean_queries")[claim.path[1]]
            probs = node.setdefault("fact_check_probabilities", {})
            probs[claim.sentence.split(" in ")[-1].rstrip(".")] = round(prob, 4)
            if passed:
                node["fact_check"] = "PASS"
            else:
                query_prune.setdefault(d, set()).add(claim.path[1])
        else:
            node = extraction.get(claim.field)
            node["fact_check_probability"] = round(prob, 4)
            if passed:
                node["fact_check"] = "PASS"
            else:
                node["value"] = None
                node["verbatim_source"] = None

    # Keep only list items supported above the span-quality threshold.
    for (d, field), votes in list_votes.items():
        node = (records[d].get("extraction") or {}).get(field)
        if not isinstance(node, dict) or not isinstance(node.get("value"), list):
            continue
        kept, probs = [], []
        for i, item in enumerate(node["value"]):
            passed, prob = votes.get(i, (True, None))
            if passed:
                kept.append(item)
                probs.append(None if prob is None else round(prob, 4))
        node["fact_check_probabilities"] = probs
        node["fact_check"] = "PASS" if kept else None
        node["value"] = kept or None
        if node["value"] is None:
            node["verbatim_source"] = None

    # Query strings are copied artifacts. Drop an item when its checked database
    # attribution is unsupported.
    for d, bad_indices in query_prune.items():
        queries = (records[d].get("extraction") or {}).get("exact_boolean_queries") or []
        records[d]["extraction"]["exact_boolean_queries"] = [
            query for i, query in enumerate(queries) if i not in bad_indices
        ]

    for d, record in enumerate(records):
        if stats[d]:
            record["fact_check_stats"] = dict(stats[d])
    return errored


# -----------------------------------------------------------------------------
# RUN
# -----------------------------------------------------------------------------
def find_inputs(input_root: Path) -> List[Path]:
    """Shard subdirectories only; skips _alignment_summary.json and friends."""
    return sorted(
        p for d in sorted(input_root.iterdir()) if d.is_dir()
        for p in sorted(d.glob("*.json")) if not p.stem.startswith("_")
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    in_root, out_root = CONFIG["input_root"], CONFIG["output_root"]
    out_root.mkdir(parents=True, exist_ok=True)
    CONFIG["log_dir"].mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(
                CONFIG["log_dir"] / f"fact_checking_{CONFIG['worker_name']}.log"
            ),
            logging.StreamHandler(),
        ],
    )

    if CONFIG["shard_count"] < 1 or not 0 <= CONFIG["shard_index"] < CONFIG["shard_count"]:
        logger.error(
            "Invalid shard settings: shard_index=%s, shard_count=%s",
            CONFIG["shard_index"],
            CONFIG["shard_count"],
        )
        return

    if not in_root.exists():
        logger.error(f"Input root not found: {in_root}")
        return

    jobs: List[Tuple[Path, Path]] = []
    skipped = 0
    inputs = find_inputs(in_root)
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
        dst = out_root / src.relative_to(in_root)
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue
        jobs.append((src, dst))
    logger.info(f"{len(jobs)} to check, {skipped} already done, under {in_root}")
    if args.limit:
        jobs = jobs[: args.limit]

    if args.dry_run:
        for src, dst in jobs[:20]:
            logger.info(f"  {src} -> {dst}")
        logger.info(f"Dry run: {len(jobs)} documents. Model not loaded.")
        return
    if not jobs:
        logger.info("Nothing to do.")
        return

    # IMPORTANT: the extraction job (Qwen) must not hold the same GPU.
    from extraction_v2.fact_checker import FactChecker

    try:
        checker = FactChecker(chunk_size=CONFIG["chunk_size"])
    except Exception as e:  # noqa: BLE001
        logger.critical(f"Failed to load FactChecker: {e}")
        return

    summary = Counter()
    per_field = Counter()
    start = time.perf_counter()

    for i in _progress(range(0, len(jobs), CONFIG["doc_batch_size"]), len(jobs)):
        batch = jobs[i : i + CONFIG["doc_batch_size"]]
        records, keep = [], []
        for src, dst in batch:
            try:
                records.append(json.loads(src.read_text(encoding="utf-8")))
                keep.append((src, dst))
            except (OSError, json.JSONDecodeError) as e:
                logger.error(f"Skipping unreadable {src}: {e}")
                summary["unreadable"] += 1

        claims: List[Any] = []
        for d, rec in enumerate(records):
            if rec.get("extraction") and not rec.get("error"):
                claims.extend(collect_claims(rec["extraction"], d))

        results = checker.score_claims(claims) if claims else []
        errored = apply_results(records, claims, results) if claims else [False] * len(records)

        for claim, res in zip(claims, results):
            if res["status"] == "OK":
                per_field[
                    f"{claim.field}:"
                    f"{'pass' if res['support_probability'] >= threshold_for(claim.span_quality) else 'fail'}"
                ] += 1

        for d, (src, dst) in enumerate(keep):
            if errored[d]:
                # Leave it absent so the next run retries it.
                logger.warning(f"{records[d].get('doc_id')}: errored claims, not written")
                summary["deferred_errors"] += 1
                continue
            records[d]["fact_check_config"] = {
                **checker.config(),
                "threshold_verified": CONFIG["threshold_verified"],
                "threshold_noisy": CONFIG["threshold_noisy"],
                "checked_at": time.time(),
            }
            _write(dst, records[d])
            summary["written"] += 1

    logger.info(f"Complete in {time.perf_counter() - start:.1f}s")
    payload = {
        "documents": dict(summary),
        "claims_by_field": dict(sorted(per_field.items())),
        "thresholds": {
            "VERIFIED": CONFIG["threshold_verified"],
            "NOISY": CONFIG["threshold_noisy"],
            "UNKNOWN": CONFIG["threshold_unknown"],
        },
        "checker": checker.config(),
    }
    (CONFIG["log_dir"] / f"fact_check_summary_{CONFIG['worker_name']}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    logger.info("Summary:\n" + json.dumps(payload, indent=2))


def _progress(iterator, total):
    try:
        from tqdm import tqdm

        return tqdm(iterator, total=total, desc="Fact-checking")
    except ImportError:
        return iterator


def _write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


if __name__ == "__main__":
    main()