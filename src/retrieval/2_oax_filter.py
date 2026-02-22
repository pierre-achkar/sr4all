"""
Filtering Systematic Review Studies by References List and Title Patterns (Excluding Updates) and is English Only
- Filters studies based on strict inclusion and exclusion phrases in titles
- Checks for presence of references
- Ensures the study is in English 
- Samples a subset for manual verification
- Logs progress and any issues encountered during filtering
"""
import json
import logging
import os
import re
import random
import csv

# Config 
INPUT_JSON  = "./data/retrieval/oax_sr_full.json"
OUTPUT_JSON = "./data/retrieval/filtered/oax_sr_title_english_refs.jsonl"
EMPTY_REFS_JSON = "./data/retrieval/filtered/oax_sr_title_english_empty_refs.jsonl"
SAMPLE_CSV  = "./data/retrieval/filtered/oax_sr_verification_sample.csv" 
LOG_FILE    = "./logs/retrieval/2_oax_filter.log"

# STRICT INCLUSION PHRASES
# Must contain one of these to be considered
STRICT_PHRASES = [
    "systematic review of",
    "systematic review in",
    "systematic review on",
    "systematic literature review of",
    "systematic literature review in",
    "systematic literature review on",
    "systematic literature review:",
    "a systematic review",
    "a systematic literature review",
    "systematic review and meta-analysis",
    "systematic literature review and meta-analysis"
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

def matched_strict_phrase(title: str) -> str:
    """Return the first matched strict phrase in normalized title, else empty string."""
    t = norm_title(title)
    for p in STRICT_PHRASES:
        if p in t:
            return p
    return ""

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
    Returns True only if language is explicitly English.
    Returns False for missing or non-English language values.
    """
    lang = rec.get("language")
    
    if not lang:
        return False
    
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
    "title_matched_total": 0
}

filtered_records = []
empty_refs_records = []
title_match_counts = {p: 0 for p in STRICT_PHRASES}

logging.info("Starting strict filtering ...")

for rec in stream_json_list(INPUT_JSON):
    stats["total"] += 1

    # 0. Language Check (English Only)
    if not is_in_english(rec):
        stats["drop_not_english"] += 1
        continue
    
    # 1. Check Title (Inclusion + Exclusion)
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
    if not title_is_strict_sr(title_to_check):
        stats["drop_title_strict"] += 1
        continue

    matched_phrase = matched_strict_phrase(title_to_check)
    if matched_phrase:
        title_match_counts[matched_phrase] += 1
        stats["title_matched_total"] += 1

    # 2. Check Refs List (after English + title match)
    if not has_references(rec):
        stats["drop_no_refs_list"] += 1
        empty_refs_records.append(rec)
        continue

    # # 3. DOI
    # if not extract_doi(rec):
    #     stats["drop_no_doi"] += 1
    #     continue

    # # 4. PDF
    # if not has_pdf(rec):
    #     stats["drop_no_pdf"] += 1
    #     continue

    filtered_records.append(rec)
    stats["kept"] += 1

# --- Summary ---
summary = (
    f"SUMMARY | Total: {stats['total']} | Kept: {stats['kept']} | "
    f"Drops: EmptyRefList({stats['drop_no_refs_list']}), "
    f"NotEnglish({stats['drop_not_english']}), "
    f"IsUpdate({stats['drop_is_update']}), "
    f"NotStrictTitle({stats['drop_title_strict']}), "
    #f"NoDOI({stats['drop_no_doi']}), NoPDF({stats['drop_no_pdf']})"
)

print(summary)
logging.info(summary)
logging.info(f"TitleMatchedTotal(English+StrictTitle): {stats['title_matched_total']}")
for phrase, count in title_match_counts.items():
    logging.info(f"TitleMatch | phrase='{phrase}' | count={count}")

# --- Save JSON ---
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    for rec in filtered_records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

logging.info(f"Saved filtered JSONL -> {OUTPUT_JSON}" f"| count={len(filtered_records)}")

# --- Save English + StrictTitle + EmptyRefs JSONL ---
with open(EMPTY_REFS_JSON, "w", encoding="utf-8") as f:
    for rec in empty_refs_records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

logging.info(f"Saved empty refs JSONL -> {EMPTY_REFS_JSON} | count={len(empty_refs_records)}")

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
