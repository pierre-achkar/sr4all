import json
import logging
import os
import re
import random
import csv

# Config 
INPUT_JSON  = "../../data/raw/oax_sr_full.json"
OUTPUT_JSON = "../../data/filtered/oax_sr_refs_title_doi_pdf_filtered.json"
SAMPLE_CSV  = "../../data/filtered/oax_sr_verification_sample.csv" 
LOG_FILE    = "../../logs/retrieval/oax_filter_refs_title_doi_pdf.log"

# STRICT INCLUSION PHRASES
# Must contain one of these to be considered
STRICT_PHRASES = [
    "systematic review of",
    "systematic review in",
    "systematic review on",
    "systematic review:",
    "systematic literature review of",
    "systematic literature review in",
    "systematic literature review on",
    "systematic literature review:",
    "a systematic review",
    "a systematic literature review",
]

# EXCLUSION PHRASES
# If the title contains these, DROP it (even if it matches above).
# We use specific patterns to avoid dropping "Review of software updates".
EXCLUDE_PHRASES = [
    "systematic review update",       # "A systematic review update"
    "updated systematic review",      # "An updated systematic review"
    ": an update",                    # "Intervention for X: an update"
    ": update",                       # "Intervention for X: update"
    "(update)",                       # "Intervention for X (Update)"
    "review: update",                 # "Systematic Review: Update"
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
    filemode="w",
)

def stream_json_list(filepath):
    """Yields items one by one."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            yield item

def norm_title(s: str) -> str:
    """Lowercase and normalize whitespace, keep punctuation."""
    if not s: return ""
    return re.sub(r"\s+", " ", s.lower()).strip()

def is_excluded_update(title_norm: str) -> bool:
    """Return True if title looks like a review update."""
    for p in EXCLUDE_PHRASES:
        if p in title_norm:
            return True
    return False

def title_is_strict_sr(title: str) -> bool:
    """
    1. Must contain a STRICT_PHRASE.
    2. Must NOT contain an EXCLUDE_PHRASE.
    """
    t = norm_title(title)
    if not t: return False

    # Check Inclusion
    matched_inclusion = any(p in t for p in STRICT_PHRASES)
    if not matched_inclusion:
        return False

    # Check Exclusion
    if is_excluded_update(t):
        return False

    return True

def extract_doi(rec: dict) -> str:
    doi = (rec.get("doi") or "").strip()
    if not doi:
        ids = rec.get("ids") or {}
        doi = (ids.get("doi") or "").strip()
    return doi

def has_pdf(rec: dict) -> bool:
    def _ok(url):
        return isinstance(url, str) and url.strip() != ""
    
    pl = rec.get("primary_location") or {}
    if _ok(pl.get("pdf_url")): return True
    
    boa = rec.get("best_oa_location") or {}
    if _ok(boa.get("pdf_url")): return True
    
    for loc in (rec.get("locations") or []):
        if _ok(loc.get("pdf_url")): return True
    return False

def has_references(rec: dict) -> bool:
    """Check if the actual list of references exists and is not empty."""
    refs = rec.get("referenced_works")
    return isinstance(refs, list) and len(refs) > 0

def is_in_english(rec: dict) -> bool:
    """
    Returns True if:
    1. Language is explicitly English ('en').
    2. Language is MISSING/NULL (we trust the English title match).
    
    Returns False if:
    1. Language is explicitly NOT English (e.g., 'fr', 'es', 'de').
    """
    lang = rec.get("language")
    
    if not lang:
        return True
    
    lang = lang.strip().lower()
    if lang in ["en", "eng", "english"]:
        return True

    return False

# --- Main Processing ---
stats = {
    "total": 0,
    "kept": 0,
    "drop_not_english": 0,
    "drop_no_refs_list": 0,
    "drop_title_strict": 0,
    "drop_is_update": 0,  
    "drop_no_doi": 0,
    "drop_no_pdf": 0
}

filtered_records = []

logging.info("Starting strict filtering (excluding Updates)...")

for rec in stream_json_list(INPUT_JSON):
    stats["total"] += 1

    # 0. Language Check (English Only)
    if not is_in_english(rec):
        stats["drop_not_english"] += 1
        continue
    
    # 1. Check Refs List
    if not has_references(rec):
        stats["drop_no_refs_list"] += 1
        continue

    # 2. Check Title (Inclusion + Exclusion)
    t_main = rec.get("title")
    t_disp = rec.get("display_name")
    
    # Choose which title to check
    title_to_check = t_disp if t_disp else t_main
    t_norm = norm_title(title_to_check)

    # A. Check if it is an update 
    if is_excluded_update(t_norm):
        stats["drop_is_update"] += 1
        continue
        
    # B. Check if it is a Strict SR
    if not any(p in t_norm for p in STRICT_PHRASES):
        stats["drop_title_strict"] += 1
        continue

    # 3. DOI
    if not extract_doi(rec):
        stats["drop_no_doi"] += 1
        continue

    # 4. PDF
    if not has_pdf(rec):
        stats["drop_no_pdf"] += 1
        continue

    filtered_records.append(rec)
    stats["kept"] += 1

# --- Summary ---
summary = (
    f"SUMMARY | Total: {stats['total']} | Kept: {stats['kept']} | "
    f"Drops: EmptyRefList({stats['drop_no_refs_list']}), "
    f"NotEnglish({stats['drop_not_english']}), "
    f"IsUpdate({stats['drop_is_update']}), "
    f"NotStrictTitle({stats['drop_title_strict']}), "
    f"NoDOI({stats['drop_no_doi']}), NoPDF({stats['drop_no_pdf']})"
)

print(summary)
logging.info(summary)

# --- Save JSON ---
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(filtered_records, f, ensure_ascii=False, indent=2)

logging.info(f"Saved filtered JSON -> {OUTPUT_JSON}")

# Generate Verification Sample 
# Extracts 150 random records to CSV for manual checking
if stats["kept"] > 0:
    sample_size = min(150, len(filtered_records))
    sample_records = random.sample(filtered_records, sample_size)
    
    with open(SAMPLE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["openalex_id", "doi", "title", "publication_year", "cited_by_count"])
        for r in sample_records:
            writer.writerow([
                r.get("id"),
                extract_doi(r),
                r.get("display_name") or r.get("title"),
                r.get("publication_year"),
                r.get("cited_by_count")
            ])
    
    print(f"\n[Action Item] Generated verification sample: {SAMPLE_CSV}")