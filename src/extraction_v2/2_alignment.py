"""
Job B: alignment and verification of extracted candidates.

Reads one JSON per markdown from the extraction stage, verifies every span
against the source text, writes one JSON per document mirroring the shard layout.

    data/extraction/W3/abc.json   ->   data/aligned/W3/abc.json

Modes:
    python src/extraction/2_alignment.py                        # run alignment
    python src/extraction/2_alignment.py --dry-run              # selection only
    python src/extraction/2_alignment.py --sample-labels 100    # emit spans to label
    python src/extraction/2_alignment.py --calibrate labelled.json   # threshold curve

Verification is three-way per span. Only NOT_FOUND / MISSING_SOURCE /
SPAN_TOO_SHORT null a value; NOISY keeps it and annotates, because a located but
OCR-garbled span is not evidence of a bad extraction. Nulling those is what made
the pilot too strict.
"""

import argparse
import json
import logging
import multiprocessing
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from extraction_v2.schema import ReviewExtraction
from extraction_v2.verifier import (
    DEFAULT_BANDS, MIN_SPAN_CHARS, NOISY, AlignmentVerifier,
)

PROJECT = Path(os.environ.get("SR4ALL_PROJECT", "/home/fhg/pie65738/projects/sr4all"))

def _dir(env: str, default: Path) -> Path:
    """Env overrides keep the paths testable and portable across machines."""
    return Path(os.environ.get(env, default))

CONFIG = {
    "input_root": _dir("SR4ALL_EXTRACTION_DIR", PROJECT / "data/extraction/first_pass"),
    "output_root": _dir("SR4ALL_ALIGNED_DIR", PROJECT / "data/extraction/aligned"),
    "log_dir": _dir("SR4ALL_ALIGNMENT_LOG_DIR", PROJECT / "logs/extraction"),
    # Fallback if a record's source_path no longer resolves.
    "mds_root": _dir("SR4ALL_MDS_DIR", PROJECT / "data/retrieval/merged/mds"),
    "bands": DEFAULT_BANDS,
    "min_span_chars": MIN_SPAN_CHARS,
    "null_noisy": False,
    "processes": max(1, multiprocessing.cpu_count() - 2),
}

logger = logging.getLogger("JobB")
_VERIFIER: Optional[AlignmentVerifier] = None


def _init_worker():
    """One verifier per process, not one per document."""
    global _VERIFIER
    _VERIFIER = AlignmentVerifier(
        bands=CONFIG["bands"],
        min_span_chars=CONFIG["min_span_chars"],
        null_noisy=CONFIG["null_noisy"],
    )


# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------
def find_inputs(input_root: Path) -> List[Path]:
    """All extraction JSONs under the shard directories, deterministic order."""
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    return sorted(
        p for d in sorted(input_root.iterdir()) if d.is_dir()
        for p in sorted(d.glob("*.json"))
    )


def output_path(src_json: Path, input_root: Path, output_root: Path) -> Path:
    return output_root / src_json.parent.relative_to(input_root) / src_json.name


def _resolve_source(record: Dict, src_json: Path) -> Optional[Path]:
    """The markdown the extraction came from."""
    sp = record.get("file_path") or record.get("source_path")
    if sp and Path(sp).exists():
        return Path(sp)
    shard = record.get("shard") or src_json.parent.name
    guess = CONFIG["mds_root"] / shard / f"{src_json.stem}.md"
    return guess if guess.exists() else None


# -----------------------------------------------------------------------------
# WORKER
# -----------------------------------------------------------------------------
def process_one(job: Dict[str, str]) -> Dict[str, Any]:
    """Read one extraction JSON, verify, write one aligned JSON. Returns a summary."""
    src_json, dst_json = Path(job["src"]), Path(job["dst"])
    try:
        record = json.loads(src_json.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return _finish(dst_json, {"doc_id": src_json.stem}, "READ_ERROR", str(e)[:300])

    doc_id = record.get("doc_id") or f"{src_json.parent.name}/{src_json.stem}"
    data = record.get("extraction")

    if not data or record.get("error"):
        return _finish(dst_json, record, "SKIPPED", "no extraction data")

    try:
        data = ReviewExtraction.model_validate(data).model_dump(mode="json")
    except Exception as e:  # noqa: BLE001
        return _finish(dst_json, record, "SCHEMA_FAIL", str(e)[:400])

    md = _resolve_source(record, src_json)
    if md is None:
        return _finish(dst_json, record, "ERROR", "source markdown not found")

    try:
        text = md.read_text(encoding="utf-8", errors="replace")
        result = (_VERIFIER or AlignmentVerifier()).verify(data, text, doc_id=doc_id)
    except Exception as e:  # noqa: BLE001
        return _finish(dst_json, record, "CRITICAL_ERROR", str(e)[:400])

    out = dict(record)
    out["extraction_raw"] = record.get("extraction")   # keep, so re-tuning is free
    out["extraction"] = result.cleaned_data
    out["spans"] = [c.row(doc_id) for c in result.checks]
    out["verification"] = {
        "status": "PASS" if result.is_valid else "PARTIAL",
        "verified_rate": round(result.score, 3),
        "counts": result.counts,
        "nulled_fields": result.nulled_fields,
        "noisy_fields": [c.path for c in result.checks if c.decision == NOISY],
        "errors": result.errors,
        "bands_used": [list(b) for b in CONFIG["bands"]],
        "min_span_chars": CONFIG["min_span_chars"],
        "null_noisy": CONFIG["null_noisy"],
        "timestamp": time.time(),
    }
    _write(dst_json, out)
    return {
        "doc_id": doc_id, "shard": out.get("shard") or dst_json.parent.name,
        **out["verification"],
    }


def _finish(dst_json: Path, record: Dict, status: str, reason: str) -> Dict[str, Any]:
    out = dict(record)
    out["spans"] = []
    out["verification"] = {"status": status, "reason": reason, "timestamp": time.time()}
    _write(dst_json, out)
    return {
        "doc_id": record.get("doc_id"), "shard": dst_json.parent.name,
        "status": status, "reason": reason,
    }


def _write(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)   # atomic: a killed job never leaves a half-written JSON


# -----------------------------------------------------------------------------
# RUN
# -----------------------------------------------------------------------------
def run_alignment(args) -> None:
    in_root, out_root = CONFIG["input_root"], CONFIG["output_root"]
    inputs = find_inputs(in_root)
    logger.info(f"Found {len(inputs)} extraction files under {in_root}")
    logger.info(f"  shards: {dict(sorted(Counter(p.parent.name for p in inputs).items()))}")

    jobs = []
    skipped = 0
    for src in inputs:
        dst = output_path(src, in_root, out_root)
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue
        jobs.append({"src": str(src), "dst": str(dst)})
    if skipped:
        logger.info(f"Skipping {skipped} already aligned (use --overwrite to redo).")
    if args.limit:
        jobs = jobs[: args.limit]

    if args.dry_run:
        for j in jobs[:20]:
            logger.info(f"  {j['src']} -> {j['dst']}")
        logger.info(f"Dry run: {len(jobs)} documents would be aligned.")
        return
    if not jobs:
        logger.info("Nothing to do.")
        return

    logger.info(
        f"Aligning {len(jobs)} documents on {CONFIG['processes']} processes "
        f"(bands={CONFIG['bands']}, null_noisy={CONFIG['null_noisy']})"
    )
    start = time.perf_counter()
    summaries: List[Dict] = []
    with multiprocessing.Pool(CONFIG["processes"], initializer=_init_worker) as pool:
        for s in _progress(pool.imap_unordered(process_one, jobs, chunksize=8), len(jobs)):
            summaries.append(s)
    logger.info(f"Complete in {time.perf_counter() - start:.1f}s")
    _summarize(summaries, CONFIG["log_dir"])


def _progress(iterator, total):
    try:
        from tqdm import tqdm

        return tqdm(iterator, total=total)
    except ImportError:
        return iterator


def _summarize(summaries: List[Dict], log_dir: Path) -> None:
    decisions, nulled, noisy = Counter(), Counter(), Counter()
    rates = []
    for s in summaries:
        for k, n in (s.get("counts") or {}).items():
            decisions[k] += n
        for f in s.get("nulled_fields") or []:
            nulled[f] += 1
        for f in s.get("noisy_fields") or []:
            noisy[f] += 1
        if s.get("verified_rate") is not None:
            rates.append(s["verified_rate"])

    summary = {
        "documents": len(summaries),
        "status": dict(Counter(s.get("status") for s in summaries)),
        "span_decisions": dict(decisions),
        "mean_verified_rate": round(sum(rates) / len(rates), 3) if rates else None,
        "fully_verified_docs": sum(1 for r in rates if r == 1.0),
        "nulled_by_field": dict(nulled.most_common()),
        "noisy_by_field": dict(noisy.most_common()),
        "bands_used": [list(b) for b in CONFIG["bands"]],
        "min_span_chars": CONFIG["min_span_chars"],
        "null_noisy": CONFIG["null_noisy"],
    }
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "alignment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logger.info("Summary:\n" + json.dumps(summary, indent=2))


# -----------------------------------------------------------------------------
# CALIBRATION (offline, run once to choose the bands)
# -----------------------------------------------------------------------------
BAND_EDGES = [(200, "long >=200"), (80, "mid 80-199"), (0, "short <80")]


def _band_of(n: int) -> str:
    for edge, name in BAND_EDGES:
        if n >= edge:
            return name
    return "short <80"


def collect_spans(out_root: Path) -> List[Dict]:
    """Every span row from the aligned JSONs."""
    rows = []
    for d in sorted(p for p in out_root.iterdir() if p.is_dir()):
        for p in sorted(d.glob("*.json")):
            try:
                rows.extend(json.loads(p.read_text(encoding="utf-8")).get("spans") or [])
            except json.JSONDecodeError:
                logger.warning(f"Unparseable {p}; skipping.")
    return rows


def sample_labels(out_root: Path, n: int, seed: int, dest: Path) -> None:
    """
    Stratify across score deciles so the labelled set covers the decision region
    rather than the easy tails. Label each row's "label" as "good" or "bad":
    good = the span genuinely occurs in the document, however garbled.
    """
    rows = [r for r in collect_spans(out_root)
            if r.get("quote_len", 0) > 0 and r.get("score") is not None]
    by_decile = defaultdict(list)
    for r in rows:
        by_decile[min(int(r["score"] // 10), 9)].append(r)
    rng = random.Random(seed)
    per = max(1, n // max(len(by_decile), 1))
    picked: List[Dict] = []
    for d in sorted(by_decile):
        pool = by_decile[d]
        picked += rng.sample(pool, min(per, len(pool)))
    rng.shuffle(picked)
    picked = picked[:n]
    dest.write_text(json.dumps(picked, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        f"Wrote {len(picked)} spans to {dest} "
        f"(deciles present: {sorted(by_decile)}). Fill in each \"label\"."
    )


def calibrate(labels_file: Path) -> None:
    rows = json.loads(labels_file.read_text(encoding="utf-8"))
    labelled = [r for r in rows if r.get("label") in ("good", "bad")]
    if not labelled:
        logger.error("No rows labelled 'good'/'bad'.")
        return
    print(f"{len(labelled)} labelled spans "
          f"({sum(r['label'] == 'good' for r in labelled)} good)\n")

    by_band = defaultdict(list)
    for r in labelled:
        by_band[_band_of(r["quote_len"])].append(r)

    for _edge, band in BAND_EDGES:
        rows_b = by_band.get(band, [])
        if not rows_b:
            continue
        good = [r["score"] for r in rows_b if r["label"] == "good"]
        bad = [r["score"] for r in rows_b if r["label"] == "bad"]
        print(f"--- {band}  n={len(rows_b)}  good={len(good)}  bad={len(bad)}")
        if good:
            print(f"    good  min={min(good):.1f}  p05={_pct(good, 5):.1f}  median={_pct(good, 50):.1f}")
        if bad:
            print(f"    bad   max={max(bad):.1f}  p95={_pct(bad, 95):.1f}  median={_pct(bad, 50):.1f}")
        print(f"    {'thresh':>7} {'kept':>5} {'prec':>6} {'rec':>6} {'lost_good':>10}")
        best = None
        for t in range(50, 100, 2):
            kept = [r for r in rows_b if r["score"] >= t]
            tp = sum(1 for r in kept if r["label"] == "good")
            prec = tp / len(kept) if kept else 1.0
            rec = tp / len(good) if good else 1.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            if best is None or f1 > best[1]:
                best = (t, f1)
            print(f"    {t:>7} {len(kept):>5} {prec:>6.3f} {rec:>6.3f} {len(good) - tp:>10}")
        print(f"    -> best F1 at {best[0]} (F1={best[1]:.3f})\n")


def _pct(xs: List[float], p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sample-labels", type=int, metavar="N",
                    help="emit N spans from the aligned outputs for hand labelling")
    ap.add_argument("--labels-out", type=Path, default=Path("to_label.json"))
    ap.add_argument("--calibrate", type=Path, metavar="LABELLED_JSON",
                    help="print the precision/recall curve from a labelled file")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--processes", type=int, default=None)
    args = ap.parse_args()
    if args.processes:
        CONFIG["processes"] = args.processes

    CONFIG["output_root"].mkdir(parents=True, exist_ok=True)
    CONFIG["log_dir"].mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(CONFIG["log_dir"] / "alignment.log"),
            logging.StreamHandler(),
        ],
    )

    if args.calibrate:
        calibrate(args.calibrate)
    elif args.sample_labels:
        sample_labels(CONFIG["output_root"], args.sample_labels, args.seed, args.labels_out)
    else:
        run_alignment(args)


if __name__ == "__main__":
    main()