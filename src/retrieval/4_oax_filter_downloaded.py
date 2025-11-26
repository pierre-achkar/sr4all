import json
import logging
import os
import re
from typing import Dict, Any, Optional, Set, List, Generator
from tqdm import tqdm

# =========================
# CONFIG 
# =========================
INPUT_JSON        = "../../data/filtered/oax_sr_refs_title_doi_pdf_filtered.json"
PDFS_ROOT         = "../../data/filtered/pdfs"
OUTPUT_JSON       = "../../data/filtered/oax_sr_refs_title_doi_pdf_downloaded_filtered.json"
LOG_FILE          = "../../logs/retrieval/oax_filter_downloaded.log"

# Minimum bytes to consider a PDF valid (avoids empty 0kb files)
MIN_PDF_BYTES = 1024

# =========================
# Setup
# =========================
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
    force=True,
    filemode="w",
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

def stream_json_list(filepath: str) -> Generator[Dict[str, Any], None, None]:
    """
    Yields items one by one to save memory. 
    Standard json.load is fine for <500MB, but this is safer.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        # If using standard json (loading full list then iterating)
        # For true streaming on massive files, we'd use 'ijson', 
        # but here we just iterate the loaded list to keep the interface clean.
        data = json.load(f)
        for item in data:
            yield item

def get_valid_pdf_ids(root: str) -> Set[str]:
    """
    Scans the filesystem for valid PDFs.
    The filesystem is the SOURCE OF TRUTH.
    """
    valid_ids: Set[str] = set()
    
    if not os.path.isdir(root):
        logging.error(f"PDF Root directory not found: {root}")
        return valid_ids

    logging.info(f"Scanning {root} for valid PDFs...")
    
    # os.walk is efficient. It visits every nested folder.
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".pdf"):
                # Filename format is usually "W123456.pdf"
                work_id = fn[:-4] 
                
                full_path = os.path.join(dirpath, fn)
                
                try:
                    # Check if file is not empty/corrupt (size check)
                    if os.path.getsize(full_path) >= MIN_PDF_BYTES:
                        valid_ids.add(work_id)
                except OSError:
                    pass

    return valid_ids

# =========================
# Main
# =========================
def main():
    # 1. Build the "Allow List" from the Disk
    valid_wids = get_valid_pdf_ids(PDFS_ROOT)
    
    if not valid_wids:
        logging.warning("No valid PDFs found! Check your PDFS_ROOT path.")
        print("WARNING: No valid PDFs found on disk. Output will be empty.")
    else:
        logging.info(f"Found {len(valid_wids)} valid PDFs on disk.")

    # 2. Filter the Input JSON
    kept_records: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()
    
    input_count = 0
    
    # We use a stream generator to avoid memory spikes if input is huge
    for rec in tqdm(stream_json_list(INPUT_JSON), desc="Filtering JSON", unit="rec"):
        input_count += 1
        
        openalex_id = rec.get("id") or ""
        wid = extract_work_id(openalex_id)
        
        # Criteria 1: Must have a valid Work ID
        if not wid:
            continue
            
        # Criteria 2: Must not be a duplicate in this dataset
        if wid in seen_ids:
            continue
            
        # Criteria 3: Must exist in our valid_wids set (The Disk Check)
        if wid in valid_wids:
            kept_records.append(rec)
            seen_ids.add(wid)

    # 3. Save
    with open(OUTPUT_JSON, "w", encoding="utf-8") as out:
        json.dump(kept_records, out, ensure_ascii=False, indent=2)

    # Summary
    summary = (
        f"SUMMARY | Input Records: {input_count} | "
        f"PDFs on Disk: {len(valid_wids)} | "
        f"Matched & Saved: {len(kept_records)}"
    )
    print("\n" + summary)
    logging.info(summary)

    if len(kept_records) == 0:
        print("\n[!] Alert: Resulting JSON is empty. Did the paths match?")

if __name__ == "__main__":
    main()