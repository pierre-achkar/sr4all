import json
import logging
import os
import re

# --- Config ---
INPUT_JSON  = "../../data/raw/openalex_systematic_reviews_full.json"
OUTPUT_JSON = "../../data/filtered/openalex_sr_title_filtered.json"
LOG_FILE    = "../../logs/retrieval/oax_filter_title.log"

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

# --- Load ---
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    records = json.load(f)

total = len(records)
logging.info(f"Loaded {total} records")

# --- Filter ---
filtered = [r for r in records if title_ok(r.get("title") or r.get("display_name") or "")]
kept = len(filtered)
filtered_out = total - kept

print(f"Input: {total} | Kept: {kept} | Filtered out: {filtered_out}")
logging.info(f"Kept {kept} | Filtered out {filtered_out}")

# --- Save ---
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(filtered, f, ensure_ascii=False, indent=2)

logging.info(f"Saved filtered JSON -> {OUTPUT_JSON}")