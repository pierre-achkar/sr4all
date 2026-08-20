#!/usr/bin/env python3
"""
Snapshot summary for PDF download progress (no CLI args).

This script is safe to run while 5_download_pdfs.py is still running.
It reads:
1) input JSONL (optional, for exact total/has-pdf counts)
2) manifest JSONL (latest status per record)
3) pdf output folder (valid files on disk)
"""

from __future__ import annotations

import json
import os
import re
import time
import csv
from collections import Counter
from typing import Any, Dict, Generator, Optional

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


DEFAULT_INPUT_JSON = "./data/retrieval/merged/oax_merged_dedup.jsonl"
DEFAULT_OUTPUT_DIR = "./data/retrieval/merged/pdfs"
DEFAULT_MANIFEST = "./logs/retrieval/5_pdf_download_manifest.jsonl"
DEFAULT_MIN_PDF_BYTES = 1024
DEFAULT_SHORT_PDF_LOG = "./logs/retrieval/5a_short_pdfs.jsonl"
DEFAULT_REPORT_JSON = "./logs/retrieval/5a_full_report.json"
DEFAULT_NOT_DOWNLOADED_CSV = "./data/retrieval/merged/5a_not_downloaded_with_links.csv"

# Fixed behavior 
MIN_PAGES_REQUIRED = 2
MAX_PDF_SCAN = 0  # 0 = scan all PDFs on disk
SKIP_INPUT_SCAN = True  # False = compute exact total/has-link from input JSONL
DELETE_SHORT_PDFS = True
DELETE_NONFUNCTIONAL_PDFS = True


def stream_jsonl(path: str) -> Generator[Dict[str, Any], None, None]:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                if i <= 5:
                    print(f"WARN: invalid JSONL at line {i} in {path}")
                continue
            if isinstance(rec, dict):
                yield rec


def _ok_url(url: Any) -> bool:
    return isinstance(url, str) and bool(url.strip())


def _dedupe_keep_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen = set()
    for item in items:
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def collect_pdf_urls(rec: Dict[str, Any]) -> list[str]:
    urls: list[str] = []

    def _push(url: Any) -> None:
        if _ok_url(url):
            urls.append(url.strip())

    boa = rec.get("best_oa_location") or {}
    _push(boa.get("pdf_url"))

    oa = rec.get("open_access") or {}
    _push(oa.get("oa_url"))

    repo_oa: list[str] = []
    oa_non_repo: list[str] = []
    other: list[str] = []
    for loc in rec.get("locations") or []:
        url = loc.get("pdf_url")
        if not _ok_url(url):
            continue
        src = loc.get("source") or {}
        src_type = (src.get("type") or "").lower()
        is_oa = bool(loc.get("is_oa"))
        if is_oa and src_type == "repository":
            repo_oa.append(url.strip())
        elif is_oa:
            oa_non_repo.append(url.strip())
        else:
            other.append(url.strip())

    urls.extend(repo_oa)
    urls.extend(oa_non_repo)

    pl = rec.get("primary_location") or {}
    _push(pl.get("pdf_url"))

    urls.extend(other)
    return _dedupe_keep_order(urls)


def choose_pdf_url(rec: Dict[str, Any]) -> Optional[str]:
    urls = collect_pdf_urls(rec)
    return urls[0] if urls else None


def extract_work_id(openalex_id: str) -> Optional[str]:
    if not isinstance(openalex_id, str):
        return None
    m = re.search(r"([A-Z]\d+)$", openalex_id.strip())
    return m.group(1).upper() if m else None


def shard_path_for_work(output_dir: str, work_id: str) -> str:
    shard = "W0"
    if len(work_id) >= 2 and work_id[1].isdigit():
        shard = f"W{work_id[1]}"
    return os.path.join(output_dir, shard, f"{work_id}.pdf")


def has_valid_pdf(path: str, min_pdf_bytes: int) -> bool:
    try:
        return os.path.exists(path) and os.path.getsize(path) >= min_pdf_bytes
    except OSError:
        return False


def count_valid_pdfs_on_disk(output_dir: str, min_pdf_bytes: int) -> int:
    total = 0
    for root, _, files in os.walk(output_dir):
        for name in files:
            if not name.lower().endswith(".pdf"):
                continue
            path = os.path.join(root, name)
            try:
                if os.path.getsize(path) >= min_pdf_bytes:
                    total += 1
            except OSError:
                continue
    return total


def append_jsonl(path: str, rec: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def extract_work_id_from_path(path: str) -> Optional[str]:
    base = os.path.basename(path)
    if not base.lower().endswith(".pdf"):
        return None
    work_id = base[:-4]
    m = re.match(r"^[A-Z]\d+$", work_id)
    return work_id if m else None


def safe_delete(path: str) -> tuple[bool, Optional[str]]:
    try:
        os.remove(path)
        return True, None
    except Exception as exc:
        return False, str(exc)


def scan_short_pdfs_on_disk(
    *,
    output_dir: str,
    min_pdf_bytes: int,
    min_pages: int,
    short_pdf_log: str,
    max_pdf_scan: int,
    delete_short_pdfs: bool,
    delete_nonfunctional_pdfs: bool,
) -> Dict[str, int]:
    stats = Counter()
    if min_pages <= 0:
        return stats
    if PdfReader is None:
        raise RuntimeError(
            "pypdf is required for page filtering. Install with: pip install pypdf"
        )

    if os.path.exists(short_pdf_log):
        os.remove(short_pdf_log)

    scanned = 0
    for root, _, files in os.walk(output_dir):
        for name in files:
            if not name.lower().endswith(".pdf"):
                continue
            path = os.path.join(root, name)
            work_id = extract_work_id_from_path(path)
            try:
                size = os.path.getsize(path)
            except OSError:
                stats["io_error"] += 1
                continue

            if size < min_pdf_bytes:
                stats["nonfunctional_too_small_bytes"] += 1
                deleted = False
                delete_error = None
                if delete_nonfunctional_pdfs:
                    deleted, delete_error = safe_delete(path)
                    if deleted:
                        stats["deleted_nonfunctional_pdf"] += 1
                    else:
                        stats["delete_error"] += 1
                append_jsonl(
                    short_pdf_log,
                    {
                        "kind": "nonfunctional_pdf",
                        "reason": "too_small_bytes",
                        "path": path,
                        "work_id": work_id,
                        "bytes": size,
                        "min_pdf_bytes": min_pdf_bytes,
                        "deleted": deleted,
                        "delete_error": delete_error,
                        "timestamp": time.time(),
                    },
                )
                continue

            if max_pdf_scan > 0 and scanned >= max_pdf_scan:
                stats["scan_limited"] += 1
                return stats

            scanned += 1
            stats["scanned"] += 1

            try:
                with open(path, "rb") as f:
                    head = f.read(5)
            except Exception as exc:
                stats["nonfunctional_read_head_error"] += 1
                deleted = False
                delete_error = None
                if delete_nonfunctional_pdfs:
                    deleted, delete_error = safe_delete(path)
                    if deleted:
                        stats["deleted_nonfunctional_pdf"] += 1
                    else:
                        stats["delete_error"] += 1
                append_jsonl(
                    short_pdf_log,
                    {
                        "kind": "nonfunctional_pdf",
                        "reason": "read_head_error",
                        "path": path,
                        "work_id": work_id,
                        "bytes": size,
                        "error": str(exc),
                        "deleted": deleted,
                        "delete_error": delete_error,
                        "timestamp": time.time(),
                    },
                )
                continue

            if not head.startswith(b"%PDF"):
                stats["nonfunctional_bad_pdf_header"] += 1
                deleted = False
                delete_error = None
                if delete_nonfunctional_pdfs:
                    deleted, delete_error = safe_delete(path)
                    if deleted:
                        stats["deleted_nonfunctional_pdf"] += 1
                    else:
                        stats["delete_error"] += 1
                append_jsonl(
                    short_pdf_log,
                    {
                        "kind": "nonfunctional_pdf",
                        "reason": "bad_pdf_header",
                        "path": path,
                        "work_id": work_id,
                        "bytes": size,
                        "header_bytes": head.decode("latin1", errors="replace"),
                        "deleted": deleted,
                        "delete_error": delete_error,
                        "timestamp": time.time(),
                    },
                )
                continue

            try:
                reader = PdfReader(path)
                pages = len(reader.pages)
            except Exception as exc:
                stats["nonfunctional_reader_error"] += 1
                deleted = False
                delete_error = None
                if delete_nonfunctional_pdfs:
                    deleted, delete_error = safe_delete(path)
                    if deleted:
                        stats["deleted_nonfunctional_pdf"] += 1
                    else:
                        stats["delete_error"] += 1
                append_jsonl(
                    short_pdf_log,
                    {
                        "kind": "nonfunctional_pdf",
                        "reason": "pdf_reader_error",
                        "path": path,
                        "work_id": work_id,
                        "bytes": size,
                        "min_pages_threshold": min_pages,
                        "error": str(exc),
                        "deleted": deleted,
                        "delete_error": delete_error,
                        "timestamp": time.time(),
                    },
                )
                continue

            if pages < min_pages:
                stats["short_pdf"] += 1
                deleted = False
                delete_error = None
                if delete_short_pdfs:
                    deleted, delete_error = safe_delete(path)
                    if deleted:
                        stats["deleted_short_pdf"] += 1
                    else:
                        stats["delete_error"] += 1
                append_jsonl(
                    short_pdf_log,
                    {
                        "kind": "short_pdf",
                        "path": path,
                        "work_id": work_id,
                        "bytes": size,
                        "pages": pages,
                        "min_pages_threshold": min_pages,
                        "deleted": deleted,
                        "delete_error": delete_error,
                        "timestamp": time.time(),
                    },
                )
                continue

            stats["kept_valid_pdf"] += 1

    return stats


def latest_status_by_record(manifest_jsonl: str) -> Dict[str, Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    for rec in stream_jsonl(manifest_jsonl):
        openalex_id = rec.get("id") or ""
        work_id = rec.get("work_id") or extract_work_id(openalex_id) or ""
        key = work_id or openalex_id
        if not key:
            continue
        latest[key] = rec
    return latest


def input_counts(input_json: str) -> tuple[int, int]:
    total = 0
    has_pdf_link = 0
    for rec in stream_jsonl(input_json):
        total += 1
        if choose_pdf_url(rec):
            has_pdf_link += 1
    return total, has_pdf_link


def write_not_downloaded_csv(
    *,
    input_json: str,
    output_dir: str,
    min_pdf_bytes: int,
    csv_path: str,
) -> int:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    count = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["openalex_id", "pdf_link"])
        for rec in stream_jsonl(input_json):
            openalex_id = rec.get("id") or ""
            if not openalex_id:
                continue
            pdf_link = choose_pdf_url(rec)
            if not pdf_link:
                continue
            work_id = extract_work_id(openalex_id)
            if not work_id:
                writer.writerow([openalex_id, pdf_link])
                count += 1
                continue
            dst = shard_path_for_work(output_dir, work_id)
            if not has_valid_pdf(dst, min_pdf_bytes):
                writer.writerow([openalex_id, pdf_link])
                count += 1
    return count


def build_snapshot_summary(
    *,
    latest_counts: Counter,
    total_records: int,
    has_pdf_link: int,
    valid_pdfs_on_disk_before_cleanup: int,
    valid_pdfs_on_disk_after_cleanup: int,
    short_pdf_scan_stats: Counter,
    min_pages: int,
) -> str:
    downloaded = latest_counts.get("downloaded", 0)
    downloaded_browser = latest_counts.get("downloaded_playwright", 0) + latest_counts.get(
        "downloaded_browser_capture", 0
    )
    skipped = latest_counts.get("skipped_exists", 0) + latest_counts.get("skipped_exists_playwright", 0)
    failed_no_url = latest_counts.get("failed_no_url", 0) + latest_counts.get("failed_no_url_playwright", 0)
    failed_identity = latest_counts.get("failed_identity_mismatch", 0) + latest_counts.get(
        "failed_identity_mismatch_playwright", 0
    )
    failed_too_small = latest_counts.get("failed_too_small", 0) + latest_counts.get("failed_too_small_playwright", 0)
    nonfunctional_total = (
        latest_counts.get("failed_nonfunctional", 0)
        + latest_counts.get("failed_http", 0)
        + latest_counts.get("failed_http_playwright", 0)
    )
    still_blocked_after_fallback = latest_counts.get("failed_http_playwright", 0)
    pending_fallback_like = latest_counts.get("failed_http", 0)

    tracked = (
        downloaded
        + downloaded_browser
        + skipped
        + failed_no_url
        + failed_identity
        + failed_too_small
        + nonfunctional_total
    )
    latest_total = sum(latest_counts.values())
    other_failures = max(0, latest_total - tracked)

    summary_lines = [
        f"Total records: {total_records}",
        f"Has PDF link: {has_pdf_link}",
        f"Total valid PDFs on disk (before cleanup): {valid_pdfs_on_disk_before_cleanup}",
        f"Total valid PDFs on disk (after cleanup): {valid_pdfs_on_disk_after_cleanup}",
        f"Downloaded: {downloaded}",
        f"Downloaded via browser fallback: {downloaded_browser}",
        f"Non-functional links: {nonfunctional_total}",
        f"Still blocked 401/403 after fallback: {still_blocked_after_fallback}",
        f"Rejected as possible wrong PDF (identity mismatch): {failed_identity}",
        f"Too-small PDFs: {failed_too_small}",
        f"Skipped existing valid PDFs: {skipped}",
        f"No PDF link / missing ID: {failed_no_url}",
        f"Other failures: {other_failures}",
        f"Pending fallback-like failures (latest=failed_http): {pending_fallback_like}",
    ]

    if min_pages > 0:
        summary_lines.append(
            f"PDFs with <{min_pages} pages: {short_pdf_scan_stats.get('short_pdf', 0)}"
        )
        summary_lines.append(
            f"Deleted short PDFs (<{min_pages} pages): {short_pdf_scan_stats.get('deleted_short_pdf', 0)}"
        )
        summary_lines.append(
            f"Detected nonfunctional PDFs: "
            f"{short_pdf_scan_stats.get('nonfunctional_too_small_bytes', 0) + short_pdf_scan_stats.get('nonfunctional_bad_pdf_header', 0) + short_pdf_scan_stats.get('nonfunctional_reader_error', 0) + short_pdf_scan_stats.get('nonfunctional_read_head_error', 0)}"
        )
        summary_lines.append(
            f"Deleted nonfunctional PDFs: {short_pdf_scan_stats.get('deleted_nonfunctional_pdf', 0)}"
        )
        summary_lines.append(
            f"PDF delete errors: {short_pdf_scan_stats.get('delete_error', 0)}"
        )
        summary_lines.append(
            f"PDF page-scan scanned files: {short_pdf_scan_stats.get('scanned', 0)}"
        )
    return "SNAPSHOT | " + " | ".join(summary_lines)


def main() -> None:
    input_json = DEFAULT_INPUT_JSON
    output_dir = DEFAULT_OUTPUT_DIR
    manifest_jsonl = DEFAULT_MANIFEST
    min_pdf_bytes = DEFAULT_MIN_PDF_BYTES
    min_pages = MIN_PAGES_REQUIRED
    short_pdf_log = DEFAULT_SHORT_PDF_LOG
    report_json = DEFAULT_REPORT_JSON
    not_downloaded_csv = DEFAULT_NOT_DOWNLOADED_CSV

    if not os.path.exists(manifest_jsonl):
        raise FileNotFoundError(f"Missing manifest: {manifest_jsonl}")
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Missing output dir: {output_dir}")
    if not SKIP_INPUT_SCAN and not os.path.exists(input_json):
        raise FileNotFoundError(f"Missing input JSONL: {input_json}")

    latest = latest_status_by_record(manifest_jsonl)
    latest_counts = Counter(rec.get("status", "<missing>") for rec in latest.values())
    valid_pdfs_on_disk_before_cleanup = count_valid_pdfs_on_disk(output_dir, min_pdf_bytes)
    short_pdf_scan_stats = scan_short_pdfs_on_disk(
        output_dir=output_dir,
        min_pdf_bytes=min_pdf_bytes,
        min_pages=min_pages,
        short_pdf_log=short_pdf_log,
        max_pdf_scan=MAX_PDF_SCAN,
        delete_short_pdfs=DELETE_SHORT_PDFS,
        delete_nonfunctional_pdfs=DELETE_NONFUNCTIONAL_PDFS,
    )
    valid_pdfs_on_disk_after_cleanup = count_valid_pdfs_on_disk(output_dir, min_pdf_bytes)

    if SKIP_INPUT_SCAN:
        total_records = len(latest)
        has_pdf_link = max(0, total_records - latest_counts.get("failed_no_url", 0))
    else:
        total_records, has_pdf_link = input_counts(input_json)
    not_downloaded_csv_rows = write_not_downloaded_csv(
        input_json=input_json,
        output_dir=output_dir,
        min_pdf_bytes=min_pdf_bytes,
        csv_path=not_downloaded_csv,
    )

    downloaded_manifest = latest_counts.get("downloaded", 0) + latest_counts.get("downloaded_playwright", 0)
    short_pdf_count = short_pdf_scan_stats.get("short_pdf", 0)
    nonfunctional_count = (
        short_pdf_scan_stats.get("nonfunctional_too_small_bytes", 0)
        + short_pdf_scan_stats.get("nonfunctional_bad_pdf_header", 0)
        + short_pdf_scan_stats.get("nonfunctional_reader_error", 0)
        + short_pdf_scan_stats.get("nonfunctional_read_head_error", 0)
    )
    downloaded_after_page_filter = valid_pdfs_on_disk_after_cleanup
    blocked_before_fallback = latest_counts.get("failed_http", 0)
    blocked_after_fallback = latest_counts.get("failed_http_playwright", 0)
    blocked_total = blocked_before_fallback + blocked_after_fallback

    report = {
        "timestamp": time.time(),
        "total_records": total_records,
        "has_pdf_link": has_pdf_link,
        "downloaded_on_disk_before_cleanup": valid_pdfs_on_disk_before_cleanup,
        "downloaded_on_disk_after_cleanup": valid_pdfs_on_disk_after_cleanup,
        "downloaded_manifest_latest": downloaded_manifest,
        "filtered_out_lt_2_pages": short_pdf_count,
        "nonfunctional_pdfs_detected": nonfunctional_count,
        "deleted_short_pdfs": short_pdf_scan_stats.get("deleted_short_pdf", 0),
        "deleted_nonfunctional_pdfs": short_pdf_scan_stats.get("deleted_nonfunctional_pdf", 0),
        "downloaded_after_lt_2_filter": downloaded_after_page_filter,
        "blocked_total": blocked_total,
        "blocked_before_fallback_latest_failed_http": blocked_before_fallback,
        "blocked_after_fallback_latest_failed_http_playwright": blocked_after_fallback,
        "latest_status_counts": dict(latest_counts),
        "short_pdf_scan_counts": dict(short_pdf_scan_stats),
        "short_pdf_log": short_pdf_log,
        "not_downloaded_csv": not_downloaded_csv,
        "not_downloaded_csv_rows": not_downloaded_csv_rows,
        "delete_short_pdfs": DELETE_SHORT_PDFS,
        "delete_nonfunctional_pdfs": DELETE_NONFUNCTIONAL_PDFS,
    }
    os.makedirs(os.path.dirname(report_json), exist_ok=True)
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(
        build_snapshot_summary(
            latest_counts=latest_counts,
            total_records=total_records,
            has_pdf_link=has_pdf_link,
            valid_pdfs_on_disk_before_cleanup=valid_pdfs_on_disk_before_cleanup,
            valid_pdfs_on_disk_after_cleanup=valid_pdfs_on_disk_after_cleanup,
            short_pdf_scan_stats=short_pdf_scan_stats,
            min_pages=min_pages,
        )
    )
    print("\nFULL_REPORT")
    print(f"Has PDF links\t{has_pdf_link}")
    print(f"Downloaded (on disk, before cleanup)\t{valid_pdfs_on_disk_before_cleanup}")
    print(f"Downloaded (on disk, after cleanup)\t{valid_pdfs_on_disk_after_cleanup}")
    print(f"Blocked (total)\t{blocked_total}")
    print(f"Blocked before fallback (latest failed_http)\t{blocked_before_fallback}")
    print(f"Blocked after fallback (latest failed_http_playwright)\t{blocked_after_fallback}")
    print(f"Filtered out (<{min_pages} pages)\t{short_pdf_count}")
    print(f"Nonfunctional PDFs detected\t{nonfunctional_count}")
    print(f"Deleted short PDFs\t{short_pdf_scan_stats.get('deleted_short_pdf', 0)}")
    print(f"Deleted nonfunctional PDFs\t{short_pdf_scan_stats.get('deleted_nonfunctional_pdf', 0)}")
    print(f"Downloaded after <{min_pages}-page filter\t{downloaded_after_page_filter}")
    print(f"Not downloaded CSV rows\t{not_downloaded_csv_rows}")
    print(f"Not downloaded CSV\t{not_downloaded_csv}")
    print("\nLATEST_STATUS_COUNTS")
    for status, count in latest_counts.most_common():
        print(f"{status}\t{count}")
    if min_pages > 0:
        print("\nSHORT_PDF_SCAN_COUNTS")
        for status, count in short_pdf_scan_stats.most_common():
            print(f"{status}\t{count}")
        print(f"\nSHORT_PDF_LOG\t{short_pdf_log}")
    print(f"REPORT_JSON\t{report_json}")


if __name__ == "__main__":
    main()
