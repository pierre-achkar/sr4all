import json
import logging
import os
import re
from typing import Dict, Any, Optional, Set, List
from tqdm import tqdm

# =========================
# CONFIG (edit here)
# =========================
INPUT_JSON        = "../../data/filtered/oax_sr_title_doi_pdf_filtered.json"
MANIFEST_JSONL    = "../../data/filtered/pdf_download_manifest.jsonl"
PDFS_ROOT         = "../../data/filtered/pdfs"
OUTPUT_JSON       = "../../data/filtered/oax_sr_title_doi_pdf_downloaded_filtered.json"
LOG_FILE          = "../../logs/retrieval/oax_filter_downloaded.log"

OK_STATUSES: Set[str] = {"downloaded", "skipped"}
MIN_PDF_BYTES = 1024

# =========================
# Logging
# =========================
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

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
def extract_work_id(openalex_id: str) -> Optional[str]:
    """Extract 'Wxxxx' from https://openalex.org/Wxxxx."""
    if not isinstance(openalex_id, str):
        return None
    m = re.search(r"/([A-Z]\d+)$", openalex_id.strip())
    return m.group(1) if m else None

def shard_path_for_work(work_id: str) -> str:
    """Replicate sharded folder layout."""
    p1 = work_id[:3] if len(work_id) >= 3 else work_id
    p2 = work_id[:6] if len(work_id) >= 6 else work_id
    return os.path.join(PDFS_ROOT, p1, p2, f"{work_id}.pdf")

def file_is_valid_pdf(path: str) -> bool:
    try:
        return os.path.exists(path) and os.path.getsize(path) >= MIN_PDF_BYTES
    except Exception:
        return False

def load_manifest_work_ids(manifest_path: str) -> Set[str]:
    """Read manifest JSONL → work_ids whose PDFs exist and status ok."""
    work_ids: Set[str] = set()
    if not os.path.exists(manifest_path):
        logging.info(f"Manifest not found: {manifest_path} (will rely on filesystem scan only)")
        return work_ids
    n = 0
    kept = 0
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                rec = json.loads(line)
            except Exception:
                continue
            status = (rec.get("status") or "").lower()
            wid = rec.get("work_id") or extract_work_id(rec.get("id") or "")
            if not wid:
                continue
            if status in OK_STATUSES:
                pdf_path = shard_path_for_work(wid)
                if file_is_valid_pdf(pdf_path):
                    work_ids.add(wid)
                    kept += 1
    logging.info(f"Manifest parsed: lines={n} | work_ids_with_present_pdf={kept}")
    return work_ids

def scan_filesystem_for_work_ids(root: str) -> Set[str]:
    """Collect valid PDFs from filesystem."""
    found: Set[str] = set()
    if not os.path.isdir(root):
        return found
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".pdf"):
                wid = fn[:-4]
                pdf_path = os.path.join(dirpath, fn)
                if file_is_valid_pdf(pdf_path):
                    found.add(wid)
    logging.info(f"Filesystem scan: valid_pdfs={len(found)}")
    return found

# =========================
# Main
# =========================
def main():
    # 1) Gather valid work IDs (manifest + FS)
    wid_from_manifest = load_manifest_work_ids(MANIFEST_JSONL)
    wid_from_fs = scan_filesystem_for_work_ids(PDFS_ROOT)
    valid_wids = wid_from_manifest | wid_from_fs
    logging.info(f"Total valid work_ids (union manifest+fs): {len(valid_wids)}")

    # 2) Load source JSON
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        records: List[Dict[str, Any]] = json.load(f)

    # 3) Deduplicate and filter by valid_wids
    seen: Set[str] = set()
    kept: List[Dict[str, Any]] = []

    for rec in tqdm(records, total=len(records), desc="Filtering unique downloaded PDFs", unit="rec"):
        openalex_id = rec.get("id") or ""
        wid = extract_work_id(openalex_id)
        if not wid:
            continue
        if wid in seen:
            continue
        if wid in valid_wids:
            kept.append(rec)
            seen.add(wid)

    # 4) Save deduplicated filtered set
    with open(OUTPUT_JSON, "w", encoding="utf-8") as out:
        json.dump(kept, out, ensure_ascii=False, indent=2)

    summary = f"SUMMARY | input={len(records)} | unique_kept={len(kept)} | valid_pdfs={len(valid_wids)}"
    print(summary)
    logging.info(summary)

if __name__ == "__main__":
    main()
