import json
import logging
import os
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# CONFIG
INPUT_JSON       = "../../data/filtered/oax_sr_refs_title_doi_pdf_filtered.json"
OUTPUT_DIR       = "../../data/filtered/pdfs"
LOG_FILE         = "../../logs/retrieval/pdf_download.log"
MANIFEST_JSONL   = "../../data/filtered/pdf_download_manifest.jsonl"

REQUEST_TIMEOUT  = (10, 60)
MAX_RETRIES      = 3
BACKOFF_FACTOR   = 1.0
CHUNK_BYTES      = 1_048_576
SKIP_IF_EXISTS   = True

# MAX_WORKERS: How many PDFs to download at once. 
# Don't go too high (e.g. >50) or publishers might block your IP.
MAX_WORKERS      = 10 

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
    adapter = HTTPAdapter(max_retries=retry, pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS*2)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    sess.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    })
    return sess

# Create a global session to reuse TCP connections
SESSION = get_session()

def extract_work_id(openalex_id: str) -> Optional[str]:
    if not isinstance(openalex_id, str): return None
    m = re.search(r"/([A-Z]\d+)$", openalex_id.strip())
    return m.group(1) if m else None

def choose_pdf_url(rec: Dict[str, Any]) -> Optional[str]:
    def _ok(url): return isinstance(url, str) and url.strip() != ""
    pl = (rec.get("primary_location") or {})
    boa = (rec.get("best_oa_location") or {})
    if _ok(pl.get("pdf_url")): return pl.get("pdf_url").strip()
    if _ok(boa.get("pdf_url")): return boa.get("pdf_url").strip()
    for loc in (rec.get("locations") or []):
        if _ok(loc.get("pdf_url")): return loc.get("pdf_url").strip()
    return None

def looks_like_pdf(resp: requests.Response, url: str) -> bool:
    ct = (resp.headers.get("Content-Type") or "").lower()
    if "pdf" in ct: return True
    if url.lower().endswith(".pdf"): return True
    return False

def shard_path_for_work(work_id: str) -> str:
    p1 = work_id[:3] if len(work_id) >= 3 else work_id
    p2 = work_id[:6] if len(work_id) >= 6 else work_id
    return os.path.join(OUTPUT_DIR, p1, p2, f"{work_id}.pdf")

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

    # 2. Download
    try:
        with SESSION.get(pdf_url, stream=True, timeout=REQUEST_TIMEOUT, allow_redirects=True) as resp:
            if resp.status_code != 200:
                logging.warning(f"Failed {resp.status_code}: {pdf_url}")
                write_manifest_threadsafe(openalex_id, work_id, pdf_url, None, "failed_http")
                return "failed", openalex_id

            # PDF Validation (Magic Bytes)
            is_pdf_header = looks_like_pdf(resp, pdf_url)
            
            # Peek first 5 bytes
            first_chunk = next(resp.iter_content(chunk_size=5), b"")
            if not is_pdf_header and not first_chunk.startswith(b"%PDF"):
                write_manifest_threadsafe(openalex_id, work_id, pdf_url, None, "failed_not_pdf")
                return "failed", openalex_id

            # Atomic Write
            tmp_path = dst + ".part"
            with open(tmp_path, "wb") as f:
                f.write(first_chunk)
                for chunk in resp.iter_content(chunk_size=CHUNK_BYTES):
                    if chunk: f.write(chunk)
            
            os.replace(tmp_path, dst)
            
            # Final Size Check
            if os.path.getsize(dst) < 1024:
                os.remove(dst)
                write_manifest_threadsafe(openalex_id, work_id, pdf_url, None, "failed_too_small")
                return "failed", openalex_id
            
            write_manifest_threadsafe(openalex_id, work_id, pdf_url, dst, "downloaded")
            return "downloaded", openalex_id

    except Exception as e:
        # logging.error(f"Error {openalex_id}: {e}") # Optional: keep log clean
        write_manifest_threadsafe(openalex_id, work_id, pdf_url, None, "failed_exception")
        return "failed", openalex_id

# Main
def main():
    # Load Data (Stream generator if file is huge, but list is okay for <500MB)
    logging.info("Loading records...")
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        records = json.load(f)
    
    total = len(records)
    logging.info(f"Loaded {total} records. Starting threads (Workers={MAX_WORKERS})...")

    # Stats
    stats = {"downloaded": 0, "skipped": 0, "failed": 0}
    
    # ThreadPool Execution
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_id = {executor.submit(process_record, r): r.get("id") for r in records}
        
        # Process as they complete
        for future in tqdm(as_completed(future_to_id), total=total, desc="Processing PDFs", unit="pdf"):
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