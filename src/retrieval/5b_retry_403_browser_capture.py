#!/usr/bin/env python3
"""
Retry unresolved PDF links from 5a CSV using authorized browser behavior.

Input:
- CSV with columns: openalex_id,pdf_link

Output:
- Saved PDFs in same sharded layout as 5_download_pdfs.py
- Retry manifest JSONL with per-record attempt details
- Unresolved CSV for another pass
- Report JSON with aggregate counts
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import time
from collections import Counter
from html import unescape
from typing import Any, Dict, Generator, Optional
from urllib.parse import urljoin

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


DEFAULT_INPUT_CSV = "./data/retrieval/merged/5a_not_downloaded_with_links.csv"
DEFAULT_INPUT_JSON = "./data/retrieval/merged/oax_merged_dedup.jsonl"
DEFAULT_OUTPUT_DIR = "./data/retrieval/merged/pdfs"
DEFAULT_RETRY_MANIFEST = "./logs/retrieval/5b_retry_manifest.jsonl"
DEFAULT_UNRESOLVED_CSV = "./data/retrieval/merged/5b_unresolved_after_browser_retry.csv"
DEFAULT_REPORT_JSON = "./logs/retrieval/5b_retry_report.json"
DEFAULT_RUN_LOG = "./logs/retrieval/5b_retry_run.log"
DEFAULT_MIN_PDF_BYTES = 1024
DEFAULT_MIN_PAGES = 2
DEFAULT_TIMEOUT_MS = 45_000
DEFAULT_MAX_NAV_URLS = 30
DEFAULT_CONTEXT_RESTART_EVERY = 750

_CITATION_PDF_RE = re.compile(
    r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_HREF_PDF_RE = re.compile(r'href=["\']([^"\']+\.pdf(?:\?[^"\']*)?)["\']', re.IGNORECASE)
_PAYWALL_HINT_RE = re.compile(
    r"(purchase\s+access|subscribe|institutional\s+access|login\s+to\s+access|sign\s+in\s+to\s+access|rent\s+this\s+article|buy\s+article|access\s+through\s+your\s+institution)",
    re.IGNORECASE,
)


def resolve_headless_mode(requested_headless: bool) -> tuple[bool, str]:
    """
    On compute nodes there is often no X/Wayland display.
    If headed mode is requested without display, force headless and explain why.
    """
    if requested_headless:
        return True, "requested_headless"

    has_x = bool(os.getenv("DISPLAY"))
    has_wayland = bool(os.getenv("WAYLAND_DISPLAY"))
    if has_x or has_wayland:
        return False, "headed_with_display"

    return True, "forced_headless_no_display"


def stream_jsonl(path: str) -> Generator[Dict[str, Any], None, None]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                yield rec


def append_jsonl(path: str, rec: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_report_json(path: str, report: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def configure_logging(log_file: str) -> None:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


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


def candidate_landing_urls(rec: Dict[str, Any]) -> list[str]:
    urls: list[str] = []

    def _push(url: Any) -> None:
        if _ok_url(url):
            urls.append(url.strip())

    pl = rec.get("primary_location") or {}
    boa = rec.get("best_oa_location") or {}
    oa = rec.get("open_access") or {}
    _push(pl.get("landing_page_url"))
    _push(boa.get("landing_page_url"))
    _push(oa.get("oa_url"))
    for loc in rec.get("locations") or []:
        _push(loc.get("landing_page_url"))
    return _dedupe_keep_order(urls)


def build_pdf_candidates(pdf_url: str) -> list[str]:
    raw = pdf_url.strip()
    candidates = [raw]
    if "?" in raw:
        candidates.append(raw.split("?", 1)[0])
    candidates.append(raw.replace("/pdfdirect/", "/pdf/"))
    if raw.startswith("http://"):
        candidates.append("https://" + raw[len("http://") :])
    return _dedupe_keep_order(candidates)


def candidate_pdf_urls_from_landing_html(html_text: str, base_url: str) -> list[str]:
    text = (html_text or "")[:300_000]
    out: list[str] = []
    for m in _CITATION_PDF_RE.finditer(text):
        out.append(urljoin(base_url, unescape(m.group(1)).strip()))
    for m in _HREF_PDF_RE.finditer(text):
        out.append(urljoin(base_url, unescape(m.group(1)).strip()))
    return _dedupe_keep_order(out)


def load_targets_from_csv(csv_path: str, limit: int) -> list[Dict[str, str]]:
    out: list[Dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            openalex_id = (row.get("openalex_id") or row.get("id") or "").strip()
            pdf_link = (row.get("pdf_link") or row.get("url") or "").strip()
            if not openalex_id or not pdf_link:
                continue
            out.append({"openalex_id": openalex_id, "pdf_link": pdf_link})
            if limit > 0 and len(out) >= limit:
                break
    return out


def load_tried_ids(manifest_path: str) -> set[str]:
    tried: set[str] = set()
    if not manifest_path or not os.path.exists(manifest_path):
        return tried
    for rec in stream_jsonl(manifest_path):
        rec_id = rec.get("id")
        if isinstance(rec_id, str) and rec_id:
            tried.add(rec_id)
    return tried


def load_records_for_ids(input_json: str, ids: set[str]) -> Dict[str, Dict[str, Any]]:
    recs: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(input_json):
        return recs
    for rec in stream_jsonl(input_json):
        openalex_id = rec.get("id") or ""
        if openalex_id in ids:
            recs[openalex_id] = rec
    return recs


def detect_paywall_text(html_text: str) -> bool:
    return bool(_PAYWALL_HINT_RE.search(html_text or ""))


def write_pdf_atomically(
    *,
    path: str,
    body: bytes,
    min_pdf_bytes: int,
    min_pages: int,
) -> tuple[bool, str]:
    if len(body) < min_pdf_bytes:
        return False, "too_small_bytes"
    if not body.startswith(b"%PDF"):
        return False, "invalid_pdf_header"

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(body)

    if os.path.getsize(tmp) < min_pdf_bytes:
        os.remove(tmp)
        return False, "too_small_after_write"

    if min_pages > 0 and PdfReader is not None:
        try:
            reader = PdfReader(tmp)
            n_pages = len(reader.pages)
        except Exception:
            os.remove(tmp)
            return False, "pdf_reader_error"
        if n_pages < min_pages:
            os.remove(tmp)
            return False, f"too_few_pages_{n_pages}"

    os.replace(tmp, path)
    return True, "saved"


def try_click_pdfish_elements(page: Any) -> None:
    click_labels = ["PDF", "Download", "Full Text", "View PDF", "Open PDF"]
    for label in click_labels:
        try:
            loc = page.get_by_text(label, exact=False).first
            if loc and loc.is_visible(timeout=600):
                loc.click(timeout=1200)
                page.wait_for_timeout(250)
        except Exception:
            continue


def retry_one_target(
    *,
    target: Dict[str, str],
    rec: Optional[Dict[str, Any]],
    context: Any,
    output_dir: str,
    timeout_ms: int,
    min_pdf_bytes: int,
    min_pages: int,
    max_nav_urls: int,
) -> tuple[str, Optional[str], Optional[str], str, int]:
    openalex_id = target["openalex_id"]
    seed_link = target["pdf_link"]
    work_id = extract_work_id(openalex_id)
    if not work_id:
        return "failed_no_work_id", None, None, "invalid_openalex_id", 0

    dst = shard_path_for_work(output_dir, work_id)
    if has_valid_pdf(dst, min_pdf_bytes):
        return "skipped_exists", None, dst, "already_exists", 0

    queue: list[str] = []
    queue.extend(build_pdf_candidates(seed_link))
    if rec:
        for u in collect_pdf_urls(rec):
            queue.extend(build_pdf_candidates(u))
        queue.extend(candidate_landing_urls(rec))
    queue = _dedupe_keep_order(queue)

    if not queue:
        return "failed_no_url", None, None, "empty_candidate_queue", 0

    attempted_urls: list[str] = []
    blocked_notes: list[str] = []
    invalid_notes: list[str] = []
    paywall_hint = False
    captured: Dict[str, Any] = {"ok": False, "url": None}

    page = context.new_page()

    def on_response(resp: Any) -> None:
        if captured["ok"]:
            return
        try:
            status = int(resp.status)
            url = str(resp.url or "")
            headers = resp.headers or {}
            ctype = (headers.get("content-type") or "").lower()
            lower_url = url.lower()
            pdfish = ("pdf" in ctype) or (".pdf" in lower_url)

            if status in (401, 402, 403, 451) and pdfish and len(blocked_notes) < 8:
                blocked_notes.append(f"{status}@{url}")
                return

            if status != 200 or not pdfish:
                return

            body = resp.body()
            ok, reason = write_pdf_atomically(
                path=dst,
                body=body,
                min_pdf_bytes=min_pdf_bytes,
                min_pages=min_pages,
            )
            if ok:
                captured["ok"] = True
                captured["url"] = url
            else:
                if len(invalid_notes) < 8:
                    invalid_notes.append(f"{reason}@{url}")
        except Exception:
            return

    def on_dialog(dialog: Any) -> None:
        # Some landing pages spawn alert/confirm dialogs; dismiss them so navigation can continue.
        # Swallow detach/closure races because the page may close while handling the dialog event.
        try:
            dialog.dismiss()
        except Exception:
            try:
                dialog.close()
            except Exception:
                return

    page.on("response", on_response)
    page.on("dialog", on_dialog)
    try:
        idx = 0
        while idx < len(queue) and len(attempted_urls) < max_nav_urls and not captured["ok"]:
            url = queue[idx]
            idx += 1
            if not url or url in attempted_urls:
                continue
            attempted_urls.append(url)

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                page.wait_for_timeout(300)
            except Exception:
                continue

            if captured["ok"]:
                break

            try:
                html = page.content()
                if detect_paywall_text(html):
                    paywall_hint = True
                base_url = page.url or url
                discovered = candidate_pdf_urls_from_landing_html(html, base_url)
                for d in discovered:
                    if d not in attempted_urls and d not in queue and len(queue) < max_nav_urls * 3:
                        queue.append(d)
            except Exception:
                pass

            if not captured["ok"]:
                try_click_pdfish_elements(page)

        if captured["ok"]:
            return "downloaded_browser_csv", str(captured["url"]), dst, "captured_pdf_response", len(attempted_urls)

        if paywall_hint:
            detail = "; ".join(blocked_notes[:3]) if blocked_notes else "paywall_text_detected"
            return "failed_paywall", None, None, detail, len(attempted_urls)

        if blocked_notes:
            return "failed_blocked_http", None, None, "; ".join(blocked_notes[:4]), len(attempted_urls)

        if invalid_notes:
            return "failed_invalid_pdf", None, None, "; ".join(invalid_notes[:4]), len(attempted_urls)

        return "failed_no_pdf_found", None, None, "no_pdf_response_captured", len(attempted_urls)
    finally:
        try:
            page.close()
        except Exception:
            pass


def write_unresolved_csv(path: str, rows: list[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["openalex_id", "pdf_link", "last_status", "last_detail"])
        for row in rows:
            writer.writerow(
                [row.get("openalex_id", ""), row.get("pdf_link", ""), row.get("last_status", ""), row.get("last_detail", "")]
            )


def is_closed_target_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "targetclosederror" in text:
        return True
    if "page.handlejavascriptdialog" in text and "not attached to an active page" in text:
        return True
    if "protocolerror" in text and "not attached to an active page" in text:
        return True
    return "has been closed" in text and any(token in text for token in ("browser", "context", "page", "target"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retry unresolved PDF links from 5a CSV using browser capture (authorized-user mode)."
    )
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--input-json", default=DEFAULT_INPUT_JSON)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--retry-manifest-jsonl", default=DEFAULT_RETRY_MANIFEST)
    parser.add_argument("--unresolved-csv", default=DEFAULT_UNRESOLVED_CSV)
    parser.add_argument("--report-json", default=DEFAULT_REPORT_JSON)
    parser.add_argument("--log-file", default=DEFAULT_RUN_LOG)
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows from CSV.")
    parser.add_argument("--headless", dest="headless", action="store_true", default=True)
    parser.add_argument("--headful", dest="headless", action="store_false")
    parser.add_argument("--user-data-dir", default="", help="Optional persistent browser profile dir (for logged-in session).")
    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS)
    parser.add_argument("--min-pdf-bytes", type=int, default=DEFAULT_MIN_PDF_BYTES)
    parser.add_argument("--min-pages", type=int, default=DEFAULT_MIN_PAGES)
    parser.add_argument("--max-nav-urls", type=int, default=DEFAULT_MAX_NAV_URLS)
    parser.add_argument(
        "--restart-context-every",
        type=int,
        default=DEFAULT_CONTEXT_RESTART_EVERY,
        help="Restart browser context every N records to reduce memory growth. 0 disables.",
    )
    parser.add_argument(
        "--skip-tried",
        dest="skip_tried",
        action="store_true",
        default=True,
        help="Skip IDs already present in the retry manifest JSONL.",
    )
    parser.add_argument(
        "--no-skip-tried",
        dest="skip_tried",
        action="store_false",
        help="Disable skipping IDs present in the retry manifest JSONL.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    effective_headless, headless_reason = resolve_headless_mode(args.headless)

    configure_logging(args.log_file)
    logging.info("Starting 5b retry run")
    logging.info(
        "Config | input_csv=%s | output_dir=%s | headless=%s | user_data_dir=%s | limit=%d | min_pages=%d",
        args.input_csv,
        args.output_dir,
        effective_headless,
        args.user_data_dir or "<none>",
        args.limit,
        args.min_pages,
    )
    if headless_reason == "forced_headless_no_display":
        logging.warning(
            "Headed mode requested but no DISPLAY/WAYLAND_DISPLAY detected. "
            "Falling back to headless=True. Use xvfb-run if you need headed mode on this node."
        )
    if tqdm is None:
        logging.warning("tqdm not installed; running without progress bar")

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Missing input CSV: {args.input_csv}")

    targets = load_targets_from_csv(args.input_csv, args.limit)
    logging.info("Loaded CSV targets: %d from %s", len(targets), args.input_csv)
    if not targets:
        return

    if args.skip_tried:
        tried_ids = load_tried_ids(args.retry_manifest_jsonl)
        if tried_ids:
            before = len(targets)
            targets = [row for row in targets if row["openalex_id"] not in tried_ids]
            logging.info(
                "Skipping already-tried IDs from manifest: %d skipped, %d remaining",
                before - len(targets),
                len(targets),
            )

    if args.dry_run:
        logging.info("Dry run only. First 10 rows:")
        for row in targets[:10]:
            logging.info("%s\t%s", row["openalex_id"], row["pdf_link"])
        return

    ids = {row["openalex_id"] for row in targets}
    rec_map = load_records_for_ids(args.input_json, ids)
    if args.input_json and os.path.exists(args.input_json):
        logging.info("Metadata records found in JSONL: %d/%d", len(rec_map), len(ids))
    else:
        logging.info("Input JSONL not found; proceeding with CSV links only.")

    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        raise RuntimeError(
            "Playwright is not available. Install with: pip install playwright && playwright install chromium"
        ) from exc

    stats: Counter = Counter()
    unresolved_rows: list[Dict[str, str]] = []
    started = time.time()

    with sync_playwright() as p:
        browser = None
        context = None

        def launch_context() -> tuple[Any, Any]:
            launch_args = ["--disable-dev-shm-usage", "--disable-gpu"]
            if args.user_data_dir:
                os.makedirs(args.user_data_dir, exist_ok=True)
                ctx = p.chromium.launch_persistent_context(
                    args.user_data_dir,
                    headless=effective_headless,
                    args=launch_args,
                    locale="en-US",
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                    ),
                )
                configure_context(ctx, args.timeout_ms)
                return None, ctx

            br = p.chromium.launch(headless=effective_headless, args=launch_args)
            ctx = br.new_context(
                locale="en-US",
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                ),
            )
            configure_context(ctx, args.timeout_ms)
            return br, ctx

        def close_context_and_browser(ctx: Any, br: Any) -> None:
            try:
                if ctx is not None:
                    ctx.close()
            except Exception:
                pass
            try:
                if br is not None:
                    br.close()
            except Exception:
                pass

        def configure_context(ctx: Any, timeout_ms: int) -> None:
            try:
                ctx.set_default_timeout(timeout_ms)
            except Exception:
                pass

            def _route_handler(route: Any, request: Any) -> None:
                try:
                    rtype = request.resource_type
                    if rtype in {"image", "media", "font"}:
                        route.abort()
                        return
                except Exception:
                    pass
                try:
                    route.continue_()
                except Exception:
                    route.abort()

            try:
                ctx.route("**/*", _route_handler)
            except Exception:
                pass

        browser, context = launch_context()
        context_restarts = 0

        try:
            iterable = tqdm(targets, desc="Retry CSV links", unit="rec") if tqdm is not None else targets
            for i, target in enumerate(iterable, start=1):
                openalex_id = target["openalex_id"]
                rec = rec_map.get(openalex_id)

                work_id = extract_work_id(openalex_id)
                if work_id:
                    dst = shard_path_for_work(args.output_dir, work_id)
                    if has_valid_pdf(dst, args.min_pdf_bytes):
                        status = "skipped_exists"
                        source_url = None
                        local_path = dst
                        detail = "already_exists"
                        attempted_count = 0
                    else:
                        restart_attempts = 0
                        while True:
                            try:
                                status, source_url, local_path, detail, attempted_count = retry_one_target(
                                    target=target,
                                    rec=rec,
                                    context=context,
                                    output_dir=args.output_dir,
                                    timeout_ms=args.timeout_ms,
                                    min_pdf_bytes=args.min_pdf_bytes,
                                    min_pages=args.min_pages,
                                    max_nav_urls=args.max_nav_urls,
                                )
                                break
                            except Exception as exc:
                                if not is_closed_target_error(exc):
                                    raise
                                restart_attempts += 1
                                context_restarts += 1
                                logging.warning(
                                    "Browser target/session detached during id=%s; restarting browser context (restart=%d) | err=%s",
                                    openalex_id,
                                    context_restarts,
                                    exc,
                                )
                                if restart_attempts > 3:
                                    status = "failed_context_closed"
                                    source_url = None
                                    local_path = None
                                    detail = f"context_closed_after_{restart_attempts}_restarts"
                                    attempted_count = 0
                                    break
                                close_context_and_browser(context, browser)
                                try:
                                    browser, context = launch_context()
                                except Exception:
                                    logging.exception("Failed to relaunch browser context after closure")
                                    status = "failed_context_restart"
                                    source_url = None
                                    local_path = None
                                    detail = "context_restart_failed"
                                    attempted_count = 0
                                    break
                                continue
                else:
                    status = "failed_no_work_id"
                    source_url = None
                    local_path = None
                    detail = "invalid_openalex_id"
                    attempted_count = 0

                stats[status] += 1
                append_jsonl(
                    args.retry_manifest_jsonl,
                    {
                        "id": openalex_id,
                        "work_id": extract_work_id(openalex_id),
                        "pdf_link_csv": target["pdf_link"],
                        "source_url": source_url,
                        "local_path": local_path,
                        "status": status,
                        "detail": detail,
                        "attempted_url_count": attempted_count,
                        "timestamp": time.time(),
                    },
                )
                logging.info(
                    "id=%s | status=%s | attempted_urls=%d | source_url=%s | local_path=%s | detail=%s",
                    openalex_id,
                    status,
                    attempted_count,
                    source_url or "",
                    local_path or "",
                    detail,
                )

                if status not in {"downloaded_browser_csv", "skipped_exists"}:
                    unresolved_rows.append(
                        {
                            "openalex_id": openalex_id,
                            "pdf_link": target["pdf_link"],
                            "last_status": status,
                            "last_detail": detail,
                        }
                    )

                if tqdm is not None:
                    iterable.set_postfix(
                        downloaded=stats.get("downloaded_browser_csv", 0),
                        paywall=stats.get("failed_paywall", 0),
                        blocked=stats.get("failed_blocked_http", 0),
                        unresolved=len(unresolved_rows),
                        refresh=False,
                    )

                if i % 25 == 0 or i == len(targets):
                    elapsed = time.time() - started
                    rate = i / elapsed if elapsed > 0 else 0.0
                    logging.info(
                        "Progress %d/%d | rate=%.2f rec/s | downloaded=%d | paywall=%d | blocked=%d | ctx_restarts=%d",
                        i,
                        len(targets),
                        rate,
                        stats.get("downloaded_browser_csv", 0),
                        stats.get("failed_paywall", 0),
                        stats.get("failed_blocked_http", 0),
                        context_restarts,
                    )

                if args.restart_context_every > 0 and i % args.restart_context_every == 0:
                    context_restarts += 1
                    logging.info(
                        "Recycling browser context after %d records (restart=%d)",
                        i,
                        context_restarts,
                    )
                    close_context_and_browser(context, browser)
                    browser, context = launch_context()
        finally:
            close_context_and_browser(context, browser)

    write_unresolved_csv(args.unresolved_csv, unresolved_rows)
    report = {
        "timestamp": time.time(),
        "input_csv": args.input_csv,
        "input_rows": len(targets),
        "retry_manifest_jsonl": args.retry_manifest_jsonl,
        "unresolved_csv": args.unresolved_csv,
        "unresolved_rows": len(unresolved_rows),
        "output_dir": args.output_dir,
        "headless_requested": args.headless,
        "headless_effective": effective_headless,
        "headless_reason": headless_reason,
        "user_data_dir": args.user_data_dir,
        "min_pdf_bytes": args.min_pdf_bytes,
        "min_pages": args.min_pages,
        "max_nav_urls": args.max_nav_urls,
        "restart_context_every": args.restart_context_every,
        "skip_tried": args.skip_tried,
        "browser_context_restarts": context_restarts,
        "status_counts": dict(stats),
    }
    write_report_json(args.report_json, report)

    logging.info("RETRY_SUMMARY")
    for k, v in stats.most_common():
        logging.info("%s\t%d", k, v)
    logging.info("RETRY_MANIFEST\t%s", args.retry_manifest_jsonl)
    logging.info("UNRESOLVED_CSV\t%s", args.unresolved_csv)
    logging.info("REPORT_JSON\t%s", args.report_json)
    logging.info("RUN_LOG\t%s", args.log_file)


if __name__ == "__main__":
    main()
