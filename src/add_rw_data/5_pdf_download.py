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
import re
import time
import threading
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# CONFIG
INPUT_JSON = (
    "./data/rw_ds/filtered/unmatched_refs_id_doi_openalex_with_refs_has_pdf.jsonl"
)
OUTPUT_DIR = "./data/rw_ds/filtered/pdfs"
LOG_FILE = "./logs/add_data/pdf_download.log"
MANIFEST_JSONL = "./data/rw_ds/filtered/pdf_download_manifest.jsonl"

REQUEST_TIMEOUT = (10, 60)
MAX_RETRIES = 3
BACKOFF_FACTOR = 1.0
CHUNK_BYTES = 1_048_576
SKIP_IF_EXISTS = True

# MAX_WORKERS: How many PDFs to download at once. Lowered to reduce blocking.
MAX_WORKERS = 4
# Small per-request delay to reduce rate-limiting risk
REQUEST_DELAY = 0.25
# Known paywalled or strict hosts — treat 403 from these as paywalled
PAYWALLED_HOSTS = ["cochranelibrary.com"]

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
    adapter = HTTPAdapter(
        max_retries=retry, pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS * 2
    )
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    sess.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }
    )
    return sess


# Create a global session to reuse TCP connections
SESSION = get_session()


def extract_work_id(openalex_id: str) -> Optional[str]:
    if not isinstance(openalex_id, str):
        return None
    m = re.search(r"/([A-Z]\d+)$", openalex_id.strip())
    return m.group(1) if m else None


def choose_pdf_url(rec: Dict[str, Any]) -> Optional[str]:
    def _ok(url):
        return isinstance(url, str) and url.strip() != ""

    pl = rec.get("primary_location") or {}
    boa = rec.get("best_oa_location") or {}
    if _ok(pl.get("pdf_url")):
        return pl.get("pdf_url").strip()
    if _ok(boa.get("pdf_url")):
        return boa.get("pdf_url").strip()
    for loc in rec.get("locations") or []:
        if _ok(loc.get("pdf_url")):
            return loc.get("pdf_url").strip()
    return None


def looks_like_pdf(resp: requests.Response, url: str) -> bool:
    ct = (resp.headers.get("Content-Type") or "").lower()
    if "pdf" in ct:
        return True
    if url.lower().endswith(".pdf"):
        return True
    return False


def shard_path_for_work(work_id: str) -> str:
    p1 = work_id[:3] if len(work_id) >= 3 else work_id
    p2 = work_id[:6] if len(work_id) >= 6 else work_id
    return os.path.join(OUTPUT_DIR, p1, p2, f"{work_id}.pdf")


def write_manifest_threadsafe(
    openalex_id, work_id, pdf_url, local_path, status, details: dict | None = None
):
    """Writes to JSONL using a lock to prevent garbled lines."""
    rec = {
        "id": openalex_id,
        "work_id": work_id,
        "pdf_url": pdf_url,
        "local_path": local_path,
        "status": status,
        "timestamp": time.time(),
    }
    if details:
        for k, v in details.items():
            rec[k] = v
    line = json.dumps(rec, ensure_ascii=False) + "\n"
    with MANIFEST_LOCK:
        with open(MANIFEST_JSONL, "a", encoding="utf-8") as mf:
            mf.write(line)


def process_record(rec: Dict[str, Any]):
    """
    Worker function for a single record.
    Returns: (status_code_string, openalex_id)
    """
    openalex_id = rec.get("id") or ""
    work_id = extract_work_id(openalex_id)
    pdf_url = choose_pdf_url(rec)

    if not work_id or not pdf_url:
        write_manifest_threadsafe(openalex_id, work_id, pdf_url, None, "failed_no_url")
        return "failed", openalex_id
    dst = shard_path_for_work(work_id)

    # 1. Check Existing
    if SKIP_IF_EXISTS and os.path.exists(dst) and os.path.getsize(dst) > 1024:
        # We don't write "skipped" to manifest every time to save disk space
        # unless you really need to audit every run.
        # write_manifest_threadsafe(openalex_id, work_id, pdf_url, dst, "skipped")
        return "skipped", openalex_id

    os.makedirs(os.path.dirname(dst), exist_ok=True)

    # 2. Download with improved handling
    try:
        # initial attempt
        resp = SESSION.get(
            pdf_url, stream=True, timeout=REQUEST_TIMEOUT, allow_redirects=True
        )
        # If 403, try one more time with PDF-specific accept header and referer
        if resp.status_code == 403:
            host = urllib.parse.urlparse(pdf_url).netloc or ""
            alt_headers = {"Accept": "application/pdf", "Referer": rec.get("id", "")}
            time.sleep(0.1)
            resp2 = SESSION.get(
                pdf_url,
                stream=True,
                timeout=REQUEST_TIMEOUT,
                allow_redirects=True,
                headers=alt_headers,
            )
            if resp2.status_code == 200:
                resp = resp2
            else:
                logging.warning("Failed %s: %s", resp2.status_code, pdf_url)
                details = {
                    "http_status": resp2.status_code,
                    "content_type": resp2.headers.get("Content-Type"),
                }
                if any(h in host for h in PAYWALLED_HOSTS):
                    write_manifest_threadsafe(
                        openalex_id,
                        work_id,
                        pdf_url,
                        None,
                        "failed_paywalled",
                        details=details,
                    )
                else:
                    write_manifest_threadsafe(
                        openalex_id,
                        work_id,
                        pdf_url,
                        None,
                        "failed_http",
                        details=details,
                    )
                return "failed", openalex_id

        if resp.status_code != 200:
            logging.warning("Failed %s: %s", resp.status_code, pdf_url)
            details = {
                "http_status": resp.status_code,
                "content_type": resp.headers.get("Content-Type"),
            }
            write_manifest_threadsafe(
                openalex_id, work_id, pdf_url, None, "failed_http", details=details
            )
            return "failed", openalex_id

        # PDF Validation (Content-Type and Magic Bytes)
        ct = (resp.headers.get("Content-Type") or "").lower()
        is_pdf_header = looks_like_pdf(resp, pdf_url)

        # Peek first 5 bytes
        iter_content = resp.iter_content(chunk_size=5)
        try:
            first_chunk = next(iter_content)
        except StopIteration:
            first_chunk = b""

        if (
            "pdf" not in ct
            and not is_pdf_header
            and not first_chunk.startswith(b"%PDF")
        ):
            # capture snippet for debugging
            snippet = first_chunk[:64].hex() if first_chunk else ""
            logging.warning("Not a PDF (content-type=%s): %s", ct, pdf_url)
            write_manifest_threadsafe(
                openalex_id,
                work_id,
                pdf_url,
                None,
                "failed_not_pdf",
                details={"content_type": ct, "snippet": snippet},
            )
            return "failed", openalex_id

        # Atomic Write
        tmp_path = dst + ".part"
        with open(tmp_path, "wb") as f:
            f.write(first_chunk)
            for chunk in resp.iter_content(chunk_size=CHUNK_BYTES):
                if chunk:
                    f.write(chunk)

        os.replace(tmp_path, dst)

        # Final Size Check
        final_size = os.path.getsize(dst)
        if final_size < 1024:
            os.remove(dst)
            write_manifest_threadsafe(
                openalex_id,
                work_id,
                pdf_url,
                None,
                "failed_too_small",
                details={"final_size": final_size},
            )
            return "failed", openalex_id

        write_manifest_threadsafe(
            openalex_id,
            work_id,
            pdf_url,
            dst,
            "downloaded",
            details={"content_type": ct, "final_size": os.path.getsize(dst)},
        )
        # small polite delay
        time.sleep(REQUEST_DELAY)
        return "downloaded", openalex_id

    except Exception as e:
        logging.exception("Error downloading %s: %s", pdf_url, e)
        write_manifest_threadsafe(
            openalex_id,
            work_id,
            pdf_url,
            None,
            "failed_exception",
            details={"error": str(e)},
        )
        return "failed", openalex_id


# Main
def main():
    # Load Data (supports both JSON array and JSONL)
    logging.info("Loading records...")
    inp = Path(INPUT_JSON)
    if not inp.exists():
        logging.error("Input file not found: %s", INPUT_JSON)
        raise SystemExit(1)

    records = []
    # Peek first non-whitespace char to detect format
    with inp.open("r", encoding="utf-8") as f:
        first_chunk = f.read(1024)
        if not first_chunk:
            logging.error("Input file is empty: %s", INPUT_JSON)
            raise SystemExit(1)
        first_nonws = next((c for c in first_chunk if not c.isspace()), "")

    if first_nonws == "[":
        # standard JSON array
        with inp.open("r", encoding="utf-8") as f:
            records = json.load(f)
    else:
        # assume JSONL - read line by line
        with inp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    logging.warning("Skipping malformed JSON line")
                    continue
                records.append(obj)

    total = len(records)
    logging.info(
        "Loaded %d records. Starting threads (Workers=%d)...", total, MAX_WORKERS
    )

    # Stats
    stats = {"downloaded": 0, "skipped": 0, "failed": 0}

    # ThreadPool Execution
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_id = {
            executor.submit(process_record, r): r.get("id") for r in records
        }

        # Process as they complete
        for future in tqdm(
            as_completed(future_to_id), total=total, desc="Processing PDFs", unit="pdf"
        ):
            try:
                result, _ = future.result()
                stats[result] += 1
            except Exception as e:
                logging.error(f"Thread Error: {e}")
                stats["failed"] += 1

    summary = f"SUMMARY | Total: {total} | Downloaded: {stats['downloaded']} | Skipped: {stats['skipped']} | Failed: {stats['failed']}"
    print("\n" + summary)
    logging.info(summary)


if __name__ == "__main__":
    main()
