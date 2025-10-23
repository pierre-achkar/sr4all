import json
import logging
import os
import re
import time
from typing import Optional, Dict, Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# =========================
# CONFIG (edit here)
# =========================
INPUT_JSON       = "../../data/filtered/oax_sr_title_doi_pdf_filtered.json"
OUTPUT_DIR       = "../../data/filtered/pdfs"                     # PDFs root (will shard inside)
LOG_FILE         = "../../logs/retrieval/pdf_download.log"        # human-readable .log
MANIFEST_JSONL   = "../../data/filtered/pdf_download_manifest.jsonl"  # machine-readable manifest
REQUEST_TIMEOUT  = (10, 60)
MAX_RETRIES      = 4
BACKOFF_FACTOR   = 0.8
CHUNK_BYTES      = 1_048_576
PAUSE_BETWEEN    = 0.2
SKIP_IF_EXISTS   = True

# =========================
# Logging setup
# =========================
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

# =========================
# Helpers
# =========================
def get_session() -> requests.Session:
    sess = requests.Session()
    retry = Retry(
        total=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=32)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    sess.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    })
    return sess

def extract_work_id(openalex_id: str) -> Optional[str]:
    # "https://openalex.org/W3022903699" -> "W3022903699"
    if not isinstance(openalex_id, str):
        return None
    m = re.search(r"/([A-Z]\d+)$", openalex_id.strip())
    return m.group(1) if m else None

def choose_pdf_url(rec: Dict[str, Any]) -> Optional[str]:
    def _ok(url):
        return isinstance(url, str) and url.strip() != ""
    pl = (rec.get("primary_location") or {})
    boa = (rec.get("best_oa_location") or {})
    if _ok(pl.get("pdf_url")):
        return pl.get("pdf_url").strip()
    if _ok(boa.get("pdf_url")):
        return boa.get("pdf_url").strip()
    for loc in (rec.get("locations") or []):
        if _ok(loc.get("pdf_url")):
            return loc.get("pdf_url").strip()
    return None

def looks_like_pdf(resp: requests.Response, url: str) -> bool:
    ct = (resp.headers.get("Content-Type") or "").lower()
    if "pdf" in ct:
        return True
    cd = (resp.headers.get("Content-Disposition") or "").lower()
    if ".pdf" in cd:
        return True
    if url.lower().endswith(".pdf"):
        return True
    return False

def save_stream(resp: requests.Response, dst_path: str) -> None:
    # atomic-ish write: write to .part then rename
    tmp_path = dst_path + ".part"
    with open(tmp_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=CHUNK_BYTES):
            if chunk:
                f.write(chunk)
    os.replace(tmp_path, dst_path)

def shard_path_for_work(work_id: str) -> str:
    """
    Shard layout:
      - level 1: first 3 chars, e.g., 'W30'
      - level 2: first 6 chars, e.g., 'W30229'
      -> OUTPUT_DIR/W30/W30229/W3022903699.pdf
    """
    p1 = work_id[:3] if len(work_id) >= 3 else work_id
    p2 = work_id[:6] if len(work_id) >= 6 else work_id
    return os.path.join(OUTPUT_DIR, p1, p2, f"{work_id}.pdf")

def write_manifest_line(openalex_id: str, work_id: Optional[str], pdf_url: Optional[str],
                        local_path: Optional[str], status: str):
    rec = {
        "id": openalex_id,
        "work_id": work_id,
        "pdf_url": pdf_url,
        "local_path": local_path,
        "status": status,  # "downloaded" | "skipped" | "failed"
    }
    with open(MANIFEST_JSONL, "a", encoding="utf-8") as mf:
        mf.write(json.dumps(rec, ensure_ascii=False) + "\n")

# =========================
# Main
# =========================
def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        records = json.load(f)

    session = get_session()

    total = len(records)
    downloaded = 0
    skipped = 0
    failed = 0

    for rec in tqdm(records, total=total, desc="Downloading PDFs", unit="pdf"):
        openalex_id = rec.get("id") or ""
        work_id = extract_work_id(openalex_id)
        pdf_url = choose_pdf_url(rec)

        dst = shard_path_for_work(work_id) if work_id else None
        if dst:
            os.makedirs(os.path.dirname(dst), exist_ok=True)

        def log_result(ok: bool, status: str):
            logging.info(
                f"{openalex_id}, {pdf_url or ''}, downloaded={'yes' if ok else 'no'}, saved_as={dst if dst else ''}"
            )
            write_manifest_line(openalex_id, work_id, pdf_url, dst, status)

        if not work_id or not pdf_url:
            log_result(False, "failed")
            failed += 1
            continue

        if SKIP_IF_EXISTS and dst and os.path.exists(dst) and os.path.getsize(dst) > 0:
            log_result(True, "skipped")
            skipped += 1
            continue

        try:
            with session.get(pdf_url, stream=True, timeout=REQUEST_TIMEOUT, allow_redirects=True) as resp:
                if resp.status_code != 200:
                    log_result(False, "failed")
                    failed += 1
                    continue

                if not looks_like_pdf(resp, pdf_url):
                    first = next(resp.iter_content(chunk_size=5), b"")
                    if not first.startswith(b"%PDF"):
                        log_result(False, "failed")
                        failed += 1
                        continue
                    # write sniffed + rest atomically
                    tmp_path = dst + ".part"
                    with open(tmp_path, "wb") as f:
                        f.write(first)
                        for chunk in resp.iter_content(chunk_size=CHUNK_BYTES):
                            if chunk:
                                f.write(chunk)
                    os.replace(tmp_path, dst)
                else:
                    save_stream(resp, dst)

            if dst and os.path.exists(dst) and os.path.getsize(dst) > 1024:
                log_result(True, "downloaded")
                downloaded += 1
            else:
                if dst and os.path.exists(dst):
                    try:
                        os.remove(dst)
                    except Exception:
                        pass
                log_result(False, "failed")
                failed += 1
        except Exception:
            log_result(False, "failed")
            failed += 1

        time.sleep(PAUSE_BETWEEN)

    summary = f"SUMMARY | total: {total} | downloaded: {downloaded} | skipped(existing): {skipped} | failures: {failed}"
    print(summary)
    logging.info(summary)

if __name__ == "__main__":
    main()
