"""
PDF Downloading with Robust Error Handling and Manifest Logging
- Reads the filtered OpenAlex records (which contain PDF URLs)
- Uses a ThreadPoolExecutor to download PDFs in parallel with retries and backoff
- Validates that the downloaded file is a PDF (Content-Type and magic bytes)
- Saves PDFs in a sharded directory structure based on the OpenAlex work ID
- Maintains a manifest JSONL file that logs the status of each download attempt (success, failure reason, etc.)
- Logs progress and any issues encountered during downloading
"""
import json
import logging
import os
import random
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import unescape
from urllib.parse import urlparse, urljoin
from typing import Optional, Dict, Any, Generator

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

# CONFIG
INPUT_JSON       = "./data/retrieval/merged/oax_merged_dedup.jsonl"
OUTPUT_DIR       = "./data/retrieval/merged/pdfs"
LOG_FILE         = "./logs/retrieval/5_download_pdfs.log"
MANIFEST_JSONL   = "./data/retrieval/merged/pdf_download_manifest.jsonl"

REQUEST_TIMEOUT  = (10, 60)
MAX_RETRIES      = 3
BACKOFF_FACTOR   = 1.0
CHUNK_BYTES      = 1_048_576
SKIP_IF_EXISTS   = True
MIN_PDF_BYTES    = 1024

# MAX_WORKERS: How many PDFs to download at once. 
# Don't go too high (e.g. >50) or publishers might block your IP.
MAX_WORKERS      = int(os.getenv("PDF_MAX_WORKERS", "6"))
HTTP_202_RETRIES = int(os.getenv("PDF_202_RETRIES", "2"))
HTTP_202_WAIT_S  = float(os.getenv("PDF_202_WAIT_S", "1.5"))
ENABLE_PLAYWRIGHT_FALLBACK = os.getenv("PDF_ENABLE_PLAYWRIGHT", "1") == "1"
PLAYWRIGHT_HEADLESS = os.getenv("PDF_PLAYWRIGHT_HEADLESS", "1") == "1"
PLAYWRIGHT_MAX_RECORDS = int(os.getenv("PDF_PLAYWRIGHT_MAX_RECORDS", "0"))  # 0 => all
PLAYWRIGHT_NAV_TIMEOUT_MS = int(os.getenv("PDF_PLAYWRIGHT_NAV_TIMEOUT_MS", "30000"))
PLAYWRIGHT_REQ_TIMEOUT_MS = int(os.getenv("PDF_PLAYWRIGHT_REQ_TIMEOUT_MS", "60000"))

_PMCID_RE = re.compile(r"(https?://pmc\.ncbi\.nlm\.nih\.gov/articles/PMC\d+)", re.IGNORECASE)
_CITATION_PDF_RE = re.compile(
    r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_HREF_PDF_RE = re.compile(r'href=["\']([^"\']+\.pdf(?:\?[^"\']*)?)["\']', re.IGNORECASE)
_DOI_PREFIX_RE = re.compile(r"^doi:\s*", re.IGNORECASE)
_DOI_URL_RE = re.compile(r"^https?://(dx\.)?doi\.org/", re.IGNORECASE)
STRICT_DOI_IDENTITY = os.getenv("PDF_STRICT_DOI_IDENTITY", "1") == "1"

_TITLE_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "by", "for", "from", "in", "into",
    "is", "of", "on", "or", "that", "the", "to", "with", "without", "via",
    "using", "use", "review", "systematic", "meta", "analysis",
}

# Setup
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs(os.path.dirname(MANIFEST_JSONL), exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
    force=True,
)

# Lock for writing to the manifest safely from multiple threads
MANIFEST_LOCK = threading.Lock()
THREAD_LOCAL = threading.local()

# Helpers
def get_session() -> requests.Session:
    sess = requests.Session()
    retry = Retry(
        total=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS*2)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    sess.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    })
    return sess

def get_thread_session() -> requests.Session:
    sess = getattr(THREAD_LOCAL, "session", None)
    if sess is None:
        sess = get_session()
        THREAD_LOCAL.session = sess
    return sess

def extract_work_id(openalex_id: str) -> Optional[str]:
    if not isinstance(openalex_id, str):
        return None
    m = re.search(r"([A-Z]\d+)$", openalex_id.strip())
    return m.group(1).upper() if m else None

def normalize_doi(value: Any) -> Optional[str]:
    if not value:
        return None
    s = str(value).strip().lower()
    s = _DOI_PREFIX_RE.sub("", s)
    s = _DOI_URL_RE.sub("", s)
    return s or None

def extract_doi(rec: Dict[str, Any]) -> Optional[str]:
    doi = normalize_doi(rec.get("doi"))
    if doi:
        return doi
    ids = rec.get("ids") or {}
    if isinstance(ids, dict):
        doi = normalize_doi(ids.get("doi"))
        if doi:
            return doi
    return normalize_doi(rec.get("requested_doi"))

def _normalize_text(s: str) -> str:
    out = re.sub(r"[^a-z0-9\s]+", " ", (s or "").lower())
    return re.sub(r"\s+", " ", out).strip()

def _title_tokens(title: Optional[str], max_tokens: int = 10) -> list[str]:
    if not title:
        return []
    words = _normalize_text(title).split()
    tokens: list[str] = []
    for w in words:
        if len(w) < 5:
            continue
        if w in _TITLE_STOPWORDS:
            continue
        tokens.append(w)
        if len(tokens) >= max_tokens:
            break
    return tokens

def _pdf_text_excerpt(path: str, max_pages: int = 2, max_chars: int = 24_000) -> str:
    if PdfReader is None:
        return ""
    try:
        reader = PdfReader(path)
        pages = reader.pages[:max_pages]
        txt = []
        for page in pages:
            try:
                txt.append(page.extract_text() or "")
            except Exception:
                continue
        return (" ".join(txt))[:max_chars]
    except Exception:
        return ""

def _doi_evidence_in_file(path: str, doi: str, text_hint: str = "") -> bool:
    doi_l = doi.lower()

    if text_hint and doi_l in text_hint.lower():
        return True

    patterns = {
        doi_l,
        f"doi:{doi_l}",
        f"doi.org/{doi_l}",
        doi_l.replace("/", "%2f"),
    }

    try:
        with open(path, "rb") as f:
            blob = f.read(3_000_000).lower()
    except Exception:
        blob = b""

    for p in patterns:
        b = p.encode("utf-8", errors="ignore")
        if b and b in blob:
            return True
    return False

def verify_pdf_identity(
    rec: Dict[str, Any],
    path: str,
    source_url: str,
    final_url: str,
    content_type: str,
    content_disposition: str,
) -> tuple[bool, str]:
    """
    Strict identity check to avoid false positives:
    - If DOI exists: require DOI evidence in URL/headers/content/text.
    - Else: require title-token overlap in extracted PDF text.
    """
    doi = extract_doi(rec)
    title = rec.get("display_name") or rec.get("title") or ""
    text_excerpt = ""

    # Fast evidence from URL/headers first.
    joined_meta = " | ".join(
        [source_url or "", final_url or "", content_type or "", content_disposition or ""]
    ).lower()

    if doi:
        if doi in joined_meta:
            return True, "ok_doi_meta"
        text_excerpt = _pdf_text_excerpt(path)
        if _doi_evidence_in_file(path, doi, text_hint=text_excerpt):
            return True, "ok_doi_content"
        if STRICT_DOI_IDENTITY:
            return False, "failed_identity_mismatch"

    # DOI missing (or relaxed DOI mode): fallback to title evidence.
    tokens = _title_tokens(title)
    if not tokens:
        return False, "failed_identity_mismatch"

    if not text_excerpt:
        text_excerpt = _pdf_text_excerpt(path)
    norm_text = _normalize_text(text_excerpt)
    if not norm_text:
        return False, "failed_identity_mismatch"

    hits = sum(1 for t in tokens if t in norm_text)
    needed = min(4, max(2, len(tokens) // 2))
    if hits >= needed:
        return True, "ok_title_tokens"

    return False, "failed_identity_mismatch"

def _ok_url(url: Any) -> bool:
    return isinstance(url, str) and url.strip() != ""

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

    # Prefer OA-first sources
    boa = (rec.get("best_oa_location") or {})
    _push(boa.get("pdf_url"))

    oa = (rec.get("open_access") or {})
    _push(oa.get("oa_url"))

    repo_oa: list[str] = []
    oa_non_repo: list[str] = []
    other: list[str] = []
    for loc in (rec.get("locations") or []):
        url = loc.get("pdf_url")
        if not _ok_url(url):
            continue
        src = (loc.get("source") or {})
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

    pl = (rec.get("primary_location") or {})
    _push(pl.get("pdf_url"))

    urls.extend(other)
    return _dedupe_keep_order(urls)

def choose_pdf_url(rec: Dict[str, Any]) -> Optional[str]:
    urls = collect_pdf_urls(rec)
    return urls[0] if urls else None

def candidate_pdf_urls_from_landing_html(html_text: str, base_url: str) -> list[str]:
    candidates: list[str] = []
    text = html_text[:300_000]

    for m in _CITATION_PDF_RE.finditer(text):
        candidates.append(urljoin(base_url, unescape(m.group(1)).strip()))

    for m in _HREF_PDF_RE.finditer(text):
        candidates.append(urljoin(base_url, unescape(m.group(1)).strip()))

    return _dedupe_keep_order(candidates)
    pl = (rec.get("primary_location") or {})
    boa = (rec.get("best_oa_location") or {})
    if _ok(pl.get("pdf_url")): return pl.get("pdf_url").strip()
    if _ok(boa.get("pdf_url")): return boa.get("pdf_url").strip()
    for loc in (rec.get("locations") or []):
        if _ok(loc.get("pdf_url")): return loc.get("pdf_url").strip()
    return None

def candidate_landing_urls(rec: Dict[str, Any]) -> list[str]:
    urls: list[str] = []

    def _push(url: Any) -> None:
        if isinstance(url, str) and url.strip():
            urls.append(url.strip())

    pl = (rec.get("primary_location") or {})
    boa = (rec.get("best_oa_location") or {})
    oa = (rec.get("open_access") or {})
    _push(pl.get("landing_page_url"))
    _push(boa.get("landing_page_url"))
    _push(oa.get("oa_url"))

    for loc in (rec.get("locations") or []):
        _push(loc.get("landing_page_url"))

    # Keep order but remove duplicates.
    return _dedupe_keep_order(urls)

def build_pdf_candidates(pdf_url: str) -> list[str]:
    raw = pdf_url.strip()
    candidates = [raw]

    # Some hosts work better without tracking query params.
    if "?" in raw:
        candidates.append(raw.split("?", 1)[0])

    # Wiley often blocks /pdfdirect but allows /pdf for the same DOI path.
    candidates.append(raw.replace("/pdfdirect/", "/pdf/"))

    # Try https if source still uses http.
    if raw.startswith("http://"):
        candidates.append("https://" + raw[len("http://"):])

    raw_no_query = raw.split("?", 1)[0]
    pmc_match = _PMCID_RE.search(raw_no_query)
    if pmc_match:
        pmc_base = pmc_match.group(1)
        candidates.append(f"{pmc_base}/pdf/")
        candidates.append(f"{pmc_base}/?pdf=1")

    # Keep order, remove empties/duplicates.
    return _dedupe_keep_order(candidates)

def _origin(url: str) -> Optional[str]:
    p = urlparse(url)
    if p.scheme and p.netloc:
        return f"{p.scheme}://{p.netloc}/"
    return None

def _headers_for(pdf_url: str, referer: Optional[str]) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    if referer:
        headers["Referer"] = referer
    else:
        origin = _origin(pdf_url)
        if origin:
            headers["Referer"] = origin
    return headers

def warmup_cookies(session: requests.Session, landing_urls: list[str]) -> Optional[str]:
    # Load a landing page first so publisher sets cookies/anti-bot context.
    for u in landing_urls:
        try:
            time.sleep(random.uniform(0.05, 0.2))
            resp = session.get(u, timeout=REQUEST_TIMEOUT, allow_redirects=True)
            if resp.status_code < 500:
                return resp.url if isinstance(resp.url, str) else u
        except requests.RequestException:
            continue
    return landing_urls[0] if landing_urls else None

def discover_pdf_candidates_from_landing(
    session: requests.Session,
    landing_urls: list[str],
    referer: Optional[str],
) -> list[str]:
    discovered: list[str] = []

    for u in landing_urls[:2]:
        try:
            headers = _headers_for(u, referer)
            time.sleep(random.uniform(0.05, 0.2))
            resp = session.get(u, timeout=REQUEST_TIMEOUT, allow_redirects=True, headers=headers)
            if resp.status_code != 200:
                continue
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if "html" not in ctype:
                continue
            discovered.extend(candidate_pdf_urls_from_landing_html(resp.text, resp.url or u))
        except requests.RequestException:
            continue

    return _dedupe_keep_order(discovered)

def looks_like_pdf(resp: requests.Response, url: str) -> bool:
    ct = (resp.headers.get("Content-Type") or "").lower()
    if "pdf" in ct:
        return True
    if url.lower().endswith(".pdf"):
        return True
    return False

def looks_like_pdf_payload(content_type: str, url: str, first_chunk: bytes) -> bool:
    ct = (content_type or "").lower()
    if "pdf" in ct:
        return True
    if url.lower().endswith(".pdf"):
        return True
    if first_chunk.startswith(b"%PDF"):
        return True
    return False

def shard_path_for_work(work_id: str) -> str:
    # One-digit sharding: W1, W2, ..., based on first numeric digit after W.
    shard = "W0"
    if len(work_id) >= 2 and work_id[1].isdigit():
        shard = f"W{work_id[1]}"
    return os.path.join(OUTPUT_DIR, shard, f"{work_id}.pdf")

def stream_jsonl(path: str) -> Generator[Dict[str, Any], None, None]:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logging.warning("Invalid JSONL at line %d in %s", i, path)
                continue
            if isinstance(rec, dict):
                yield rec
            else:
                logging.warning("Non-object JSON at line %d in %s", i, path)

def write_manifest_threadsafe(openalex_id, work_id, pdf_url, local_path, status):
    """Writes to JSONL using a lock to prevent garbled lines."""
    rec = {
        "id": openalex_id,
        "work_id": work_id,
        "pdf_url": pdf_url,
        "local_path": local_path,
        "status": status,
        "timestamp": time.time()
    }
    line = json.dumps(rec, ensure_ascii=False) + "\n"
    with MANIFEST_LOCK:
        with open(MANIFEST_JSONL, "a", encoding="utf-8") as mf:
            mf.write(line)

def request_pdf_response(
    session: requests.Session,
    url: str,
    headers: Dict[str, str],
) -> requests.Response:
    """
    Fetch URL with special retry handling for 202 (accepted/queued) responses.
    Caller must close the returned response (use `with`).
    """
    for attempt in range(HTTP_202_RETRIES + 1):
        time.sleep(random.uniform(0.05, 0.2))
        resp = session.get(
            url,
            stream=True,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
            headers=headers,
        )
        if resp.status_code == 202 and attempt < HTTP_202_RETRIES:
            resp.close()
            time.sleep(HTTP_202_WAIT_S * (attempt + 1))
            continue
        return resp
    return resp

def process_record_playwright(rec: Dict[str, Any], context: Any) -> tuple[str, str]:
    """
    Browser fallback for records that failed with 401/403 in requests path.
    Returns: (status_code_string, openalex_id)
    """
    openalex_id = rec.get("id") or ""
    work_id = extract_work_id(openalex_id)
    pdf_urls = collect_pdf_urls(rec)
    pdf_url = pdf_urls[0] if pdf_urls else None
    landing_urls = candidate_landing_urls(rec)

    if not work_id or not pdf_url:
        write_manifest_threadsafe(openalex_id, work_id, pdf_url, None, "failed_no_url_playwright")
        return "failed_no_url", openalex_id

    dst = shard_path_for_work(work_id)
    if SKIP_IF_EXISTS and os.path.exists(dst) and os.path.getsize(dst) >= MIN_PDF_BYTES:
        write_manifest_threadsafe(openalex_id, work_id, pdf_url, dst, "skipped_exists_playwright")
        return "skipped", openalex_id

    os.makedirs(os.path.dirname(dst), exist_ok=True)

    referer: Optional[str] = None
    discovered: list[str] = []
    http_statuses: list[str] = []
    saw_identity_mismatch = False

    page = None
    try:
        page = context.new_page()
        for landing in landing_urls[:3]:
            try:
                resp = page.goto(landing, wait_until="domcontentloaded", timeout=PLAYWRIGHT_NAV_TIMEOUT_MS)
                if resp and resp.status:
                    if page.url:
                        referer = page.url
                html = page.content()
                discovered.extend(candidate_pdf_urls_from_landing_html(html, page.url or landing))
                time.sleep(random.uniform(0.1, 0.3))
            except Exception:
                continue
    finally:
        if page is not None:
            try:
                page.close()
            except Exception:
                pass

    candidates: list[str] = []
    for u in pdf_urls:
        candidates.extend(build_pdf_candidates(u))
    candidates.extend(discovered)
    candidates = _dedupe_keep_order(candidates)

    for candidate in candidates:
        headers = _headers_for(candidate, referer)
        for attempt in range(HTTP_202_RETRIES + 1):
            try:
                resp = context.request.get(candidate, headers=headers, timeout=PLAYWRIGHT_REQ_TIMEOUT_MS)
            except Exception:
                http_statuses.append(f"EXC@{candidate}")
                break

            status = resp.status
            if status == 202 and attempt < HTTP_202_RETRIES:
                time.sleep(HTTP_202_WAIT_S * (attempt + 1))
                continue

            if status != 200:
                http_statuses.append(f"{status}@{candidate}")
                break

            body = resp.body()
            first_chunk = body[:5]
            ctype = (resp.headers or {}).get("content-type", "")
            if not looks_like_pdf_payload(ctype, candidate, first_chunk):
                try:
                    txt = body[:300_000].decode("utf-8", errors="ignore")
                except Exception:
                    txt = ""
                if txt:
                    new_candidates = candidate_pdf_urls_from_landing_html(txt, candidate)
                    if new_candidates:
                        candidates = _dedupe_keep_order(candidates + new_candidates)
                break

            tmp_path = dst + ".part"
            with open(tmp_path, "wb") as f:
                f.write(body)

            if os.path.getsize(tmp_path) < MIN_PDF_BYTES:
                os.remove(tmp_path)
                write_manifest_threadsafe(openalex_id, work_id, candidate, None, "failed_too_small_playwright")
                return "failed_too_small", openalex_id

            ok_identity, identity_reason = verify_pdf_identity(
                rec=rec,
                path=tmp_path,
                source_url=candidate,
                final_url=getattr(resp, "url", "") or candidate,
                content_type=(resp.headers or {}).get("content-type", ""),
                content_disposition=(resp.headers or {}).get("content-disposition", ""),
            )
            if not ok_identity:
                os.remove(tmp_path)
                write_manifest_threadsafe(openalex_id, work_id, candidate, None, identity_reason + "_playwright")
                saw_identity_mismatch = True
                continue

            os.replace(tmp_path, dst)
            write_manifest_threadsafe(openalex_id, work_id, candidate, dst, "downloaded_playwright")
            return "downloaded_browser", openalex_id

    if http_statuses:
        logging.warning("Playwright failed HTTP attempts for %s | %s", openalex_id, "; ".join(http_statuses[:4]))
    if saw_identity_mismatch:
        return "failed_identity_mismatch", openalex_id
    write_manifest_threadsafe(openalex_id, work_id, pdf_url, None, "failed_http_playwright")
    return "failed_nonfunctional", openalex_id

def run_playwright_fallback(records: list[Dict[str, Any]]) -> list[tuple[str, str]]:
    if not records:
        return []

    if PLAYWRIGHT_MAX_RECORDS > 0:
        records = records[:PLAYWRIGHT_MAX_RECORDS]

    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        logging.warning(
            "Playwright fallback requested but not available. Install with: pip install playwright && playwright install chromium"
        )
        return []

    results: list[tuple[str, str]] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=PLAYWRIGHT_HEADLESS)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            locale="en-US",
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
        )
        try:
            for rec in tqdm(records, desc="Playwright fallback", unit="pdf"):
                status, openalex_id = process_record_playwright(rec, context)
                results.append((openalex_id, status))
        finally:
            try:
                context.close()
            finally:
                browser.close()

    return results

def process_record(rec: Dict[str, Any]):
    """
    Worker function for a single record. 
    Returns: (status_code_string, openalex_id)
    """
    openalex_id = rec.get("id") or ""
    work_id = extract_work_id(openalex_id)
    pdf_urls = collect_pdf_urls(rec)
    pdf_url = pdf_urls[0] if pdf_urls else None
    landing_urls = candidate_landing_urls(rec)

    if not work_id or not pdf_url:
        write_manifest_threadsafe(openalex_id, work_id, pdf_url, None, "failed_no_url")
        return "failed_no_url", openalex_id

    dst = shard_path_for_work(work_id)
    
    # 1. Check Existing
    if SKIP_IF_EXISTS and os.path.exists(dst) and os.path.getsize(dst) >= MIN_PDF_BYTES:
        # We don't write "skipped" to manifest every time to save disk space 
        # unless you really need to audit every run. 
        write_manifest_threadsafe(openalex_id, work_id, pdf_url, dst, "skipped_exists")
        return "skipped", openalex_id

    os.makedirs(os.path.dirname(dst), exist_ok=True)

    # 2. Download
    try:
        session = get_thread_session()
        referer = warmup_cookies(session, landing_urls)
        candidates: list[str] = []
        for u in pdf_urls:
            candidates.extend(build_pdf_candidates(u))
        candidates = _dedupe_keep_order(candidates)
        http_statuses: list[str] = []
        saw_403_or_401 = False
        saw_identity_mismatch = False

        for candidate in candidates:
            headers = _headers_for(candidate, referer)
            resp = request_pdf_response(session, candidate, headers)
            try:
                if resp.status_code != 200:
                    http_statuses.append(f"{resp.status_code}@{candidate}")
                    if resp.status_code in {401, 403}:
                        saw_403_or_401 = True

                    # Some 403 pages work after one more warmup with current referer context.
                    if resp.status_code in {401, 403}:
                        referer = warmup_cookies(session, landing_urls) or referer
                        retry_headers = _headers_for(candidate, referer)
                        resp.close()
                        resp = request_pdf_response(session, candidate, retry_headers)
                        if resp.status_code != 200:
                            http_statuses.append(f"{resp.status_code}@{candidate}")
                            if resp.status_code in {401, 403}:
                                saw_403_or_401 = True
                            continue
                    else:
                        continue

                # PDF Validation (Magic Bytes)
                is_pdf_header = looks_like_pdf(resp, candidate)

                # Peek first 5 bytes
                first_chunk = next(resp.iter_content(chunk_size=5), b"")
                if not is_pdf_header and not first_chunk.startswith(b"%PDF"):
                    write_manifest_threadsafe(openalex_id, work_id, candidate, None, "failed_not_pdf")
                    continue

                # Atomic Write
                tmp_path = dst + ".part"
                with open(tmp_path, "wb") as f:
                    f.write(first_chunk)
                    for chunk in resp.iter_content(chunk_size=CHUNK_BYTES):
                        if chunk:
                            f.write(chunk)

                # Final Size Check
                if os.path.getsize(tmp_path) < MIN_PDF_BYTES:
                    os.remove(tmp_path)
                    write_manifest_threadsafe(openalex_id, work_id, candidate, None, "failed_too_small")
                    return "failed_too_small", openalex_id

                ok_identity, identity_reason = verify_pdf_identity(
                    rec=rec,
                    path=tmp_path,
                    source_url=candidate,
                    final_url=resp.url or candidate,
                    content_type=resp.headers.get("Content-Type") or "",
                    content_disposition=resp.headers.get("Content-Disposition") or "",
                )
                if not ok_identity:
                    os.remove(tmp_path)
                    write_manifest_threadsafe(openalex_id, work_id, candidate, None, identity_reason)
                    saw_identity_mismatch = True
                    continue

                os.replace(tmp_path, dst)
                write_manifest_threadsafe(openalex_id, work_id, candidate, dst, "downloaded")
                return "downloaded", openalex_id
            finally:
                resp.close()

        # Last fallback: parse landing HTML and try discovered pdf URLs.
        discovered = discover_pdf_candidates_from_landing(session, landing_urls, referer)
        for candidate in discovered:
            headers = _headers_for(candidate, referer)
            with request_pdf_response(session, candidate, headers) as resp:
                if resp.status_code != 200:
                    http_statuses.append(f"{resp.status_code}@{candidate}")
                    if resp.status_code in {401, 403}:
                        saw_403_or_401 = True
                    continue

                is_pdf_header = looks_like_pdf(resp, candidate)
                first_chunk = next(resp.iter_content(chunk_size=5), b"")
                if not is_pdf_header and not first_chunk.startswith(b"%PDF"):
                    continue

                tmp_path = dst + ".part"
                with open(tmp_path, "wb") as f:
                    f.write(first_chunk)
                    for chunk in resp.iter_content(chunk_size=CHUNK_BYTES):
                        if chunk:
                            f.write(chunk)

                if os.path.getsize(tmp_path) < MIN_PDF_BYTES:
                    os.remove(tmp_path)
                    write_manifest_threadsafe(openalex_id, work_id, candidate, None, "failed_too_small")
                    return "failed_too_small", openalex_id

                ok_identity, identity_reason = verify_pdf_identity(
                    rec=rec,
                    path=tmp_path,
                    source_url=candidate,
                    final_url=resp.url or candidate,
                    content_type=resp.headers.get("Content-Type") or "",
                    content_disposition=resp.headers.get("Content-Disposition") or "",
                )
                if not ok_identity:
                    os.remove(tmp_path)
                    write_manifest_threadsafe(openalex_id, work_id, candidate, None, identity_reason)
                    saw_identity_mismatch = True
                    continue

                os.replace(tmp_path, dst)
                write_manifest_threadsafe(openalex_id, work_id, candidate, dst, "downloaded")
                return "downloaded", openalex_id

        if http_statuses:
            logging.warning("Failed HTTP attempts for %s | %s", openalex_id, "; ".join(http_statuses[:4]))
            write_manifest_threadsafe(openalex_id, work_id, pdf_url, None, "failed_http")
            if saw_403_or_401:
                return "failed_nonfunctional_403", openalex_id
            if saw_identity_mismatch:
                return "failed_identity_mismatch", openalex_id
            return "failed_nonfunctional", openalex_id

        if saw_identity_mismatch:
            return "failed_identity_mismatch", openalex_id

        write_manifest_threadsafe(openalex_id, work_id, pdf_url, None, "failed_nonfunctional")
        return "failed_nonfunctional", openalex_id

    except Exception:
        # logging.error(f"Error {openalex_id}: {e}") # Optional: keep log clean
        write_manifest_threadsafe(openalex_id, work_id, pdf_url, None, "failed_exception")
        return "failed_nonfunctional", openalex_id

# Main
def main():
    logging.info("Loading records from JSONL...")
    if PdfReader is None:
        logging.warning(
            "pypdf is not installed. Identity validation is stricter and may reject more PDFs. "
            "Install with: pip install pypdf"
        )
    records = list(stream_jsonl(INPUT_JSON))
    total = len(records)

    has_pdf_link = sum(1 for r in records if choose_pdf_url(r))
    logging.info(
        "Loaded %d records. Records with PDF link: %d. Starting threads (Workers=%d)...",
        total,
        has_pdf_link,
        MAX_WORKERS,
    )

    # Stats requested
    stats = {
        "downloaded": 0,
        "downloaded_browser": 0,
        "skipped": 0,
        "failed_nonfunctional": 0,
        "failed_nonfunctional_403": 0,
        "failed_identity_mismatch": 0,
        "failed_too_small": 0,
        "failed_no_url": 0,
        "failed_other": 0,
    }
    fallback_records: list[Dict[str, Any]] = []
    
    # ThreadPool Execution
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_record = {executor.submit(process_record, r): r for r in records}
        
        # Process as they complete
        for future in tqdm(as_completed(future_to_record), total=total, desc="Processing PDFs", unit="pdf"):
            try:
                result, _ = future.result()
                if result in stats:
                    stats[result] += 1
                    if result == "failed_nonfunctional_403":
                        fallback_records.append(future_to_record[future])
                else:
                    stats["failed_other"] += 1
            except Exception as e:
                logging.error(f"Thread Error: {e}")
                stats["failed_other"] += 1

    if ENABLE_PLAYWRIGHT_FALLBACK and fallback_records:
        logging.info(
            "Starting Playwright fallback for %d persistent 401/403 records",
            len(fallback_records),
        )
        browser_results = run_playwright_fallback(fallback_records)
        for _, status in browser_results:
            if stats["failed_nonfunctional_403"] > 0:
                stats["failed_nonfunctional_403"] -= 1
            if status in stats:
                stats[status] += 1
            else:
                stats["failed_other"] += 1

        unresolved = len(fallback_records) - len(browser_results)
        if unresolved > 0:
            logging.warning(
                "Playwright fallback unavailable or partially run. Unresolved fallback records=%d",
                unresolved,
            )
            stats["failed_nonfunctional_403"] += unresolved

    nonfunctional_total = stats["failed_nonfunctional"] + stats["failed_nonfunctional_403"]

    summary_lines = [
        f"Total records: {total}",
        f"Has PDF link: {has_pdf_link}",
        f"Downloaded: {stats['downloaded']}",
        f"Downloaded via browser fallback: {stats['downloaded_browser']}",
        f"Non-functional links: {nonfunctional_total}",
        f"Still blocked 401/403 after fallback: {stats['failed_nonfunctional_403']}",
        f"Rejected as possible wrong PDF (identity mismatch): {stats['failed_identity_mismatch']}",
        f"Too-small PDFs: {stats['failed_too_small']}",
        f"Skipped existing valid PDFs: {stats['skipped']}",
        f"No PDF link / missing ID: {stats['failed_no_url']}",
        f"Other failures: {stats['failed_other']}",
    ]
    summary_text = "SUMMARY | " + " | ".join(summary_lines)

    print("\n" + summary_text)
    logging.info(summary_text)

if __name__ == "__main__":
    main()
