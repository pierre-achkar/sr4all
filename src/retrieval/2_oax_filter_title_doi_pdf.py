import json
import logging
import os
import re

# --- Config ---
INPUT_JSON  = "../../data/raw/oax_sr_full.json"
OUTPUT_JSON = "../../data/filtered/oax_sr_title_doi_pdf_filtered.json"
LOG_FILE    = "../../logs/retrieval/oax_filter_title_doi_pdf.log"

PHRASES = [
    "systematic review of",
    "systematic review in",
    "systematic review on",
    "systematic literature review of",
    "systematic literature review in",
    "systematic literature review on",
    "a systematic review",
    "a systematic literature review",
]

# --- Setup ---
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
    force=True,
)

def norm_title(s: str) -> str:
    """Lowercase, replace punctuation with spaces, collapse whitespace."""
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"[^\w]+", " ", s)  # punctuation -> space
    s = re.sub(r"\s+", " ", s).strip()
    return s

def title_ok(title: str) -> bool:
    t = norm_title(title)
    return any(p in t for p in PHRASES)

def extract_doi(rec: dict) -> str:
    """Return DOI from 'doi' or fallback 'ids.doi' (empty string if none)."""
    doi = (rec.get("doi") or "").strip()
    if not doi:
        ids = rec.get("ids") or {}
        doi = (ids.get("doi") or "").strip()
    return doi

def has_doi(rec: dict) -> bool:
    return bool(extract_doi(rec))

def has_pdf(rec: dict) -> bool:
    """Check primary_location, best_oa_location, or any locations[].pdf_url."""
    def _ok(url):
        return isinstance(url, str) and url.strip() != ""
    pl = rec.get("primary_location") or {}
    if _ok(pl.get("pdf_url")):
        return True
    boa = rec.get("best_oa_location") or {}
    if _ok(boa.get("pdf_url")):
        return True
    for loc in (rec.get("locations") or []):
        if _ok(loc.get("pdf_url")):
            return True
    return False

# --- Load ---
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    records = json.load(f)

total = len(records)
logging.info(f"Loaded {total} records")

# --- Stage 1: Title filter ---
stage1 = [r for r in records if title_ok(r.get("title") or r.get("display_name"))]
n1 = len(stage1)
logging.info(f"Title match -> kept: {n1} | dropped: {total - n1}")

# --- Stage 2: DOI present (among stage1) ---
stage2 = [r for r in stage1 if has_doi(r)]
n2 = len(stage2)
logging.info(f"DOI present (within title-matched) -> kept: {n2} | dropped: {n1 - n2}")

# --- Stage 3: Has at least one pdf_url (among stage2) ---
stage3 = [r for r in stage2 if has_pdf(r)]
n3 = len(stage3)
logging.info(f"PDF URL present (within title+DOI) -> kept: {n3} | dropped: {n2 - n3}")

# --- Summary ---
kept = n3
filtered_out_total = total - kept

summary = (
    f"SUMMARY | total: {total} | kept: {kept} | filtered_out_total: {filtered_out_total} | "
    f"(title kept: {n1}, doi kept: {n2}, pdf kept: {n3})"
)
print(summary)
logging.info(summary)

# --- Save ---
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(stage3, f, ensure_ascii=False, indent=2)

logging.info(f"Saved filtered JSON -> {OUTPUT_JSON}")