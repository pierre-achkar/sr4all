#!/usr/bin/env python3
import json
import os
import logging
from typing import Any, Dict, Optional
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# If pypdf isn't installed in your container, install it in the image or overlay.
# pip install pypdf
from pypdf import PdfReader

# =========================
# CONFIG
# =========================
INPUT_JSON  = "../../data/filtered/oax_sr_title_doi_pdf_downloaded_filtered.json"
PDFS_ROOT   = "../../data/filtered/pdfs"     
OUTPUT_DIR  = "../../data/filtered/eda"           # output folder
LOG_FILE    = "../../logs/retrieval/eda.log"

# Plot settings
TOP_N = 15
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

# Sanity threshold to consider a PDF valid
MIN_PDF_BYTES = 1024

# Optional on-disk cache for page counts (avoids recounting on reruns)
PAGECOUNT_CACHE_CSV = os.path.join(OUTPUT_DIR, "page_counts_cache.csv")

# =========================
# Setup
# =========================
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
    force=True,
)

tqdm.pandas()

# =========================
# Helpers
# =========================
def extract_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Pull just what we need; fallback to top-scored topic if primary_topic missing."""
    wid = (rec.get("id") or "").split("/")[-1] or None
    domain   = (rec.get("primary_topic") or {}).get("domain",  {}).get("display_name")
    field    = (rec.get("primary_topic") or {}).get("field",   {}).get("display_name")
    subfield = (rec.get("primary_topic") or {}).get("subfield",{}).get("display_name")

    if not (domain and field and subfield):
        topics = rec.get("topics") or []
        if topics:
            mx = max(topics, key=lambda t: t.get("score", 0))
            domain   = domain   or ((mx.get("domain")   or {}).get("display_name"))
            field    = field    or ((mx.get("field")    or {}).get("display_name"))
            subfield = subfield or ((mx.get("subfield") or {}).get("display_name"))

    return {
        "work_id": wid,
        "referenced_works_count": rec.get("referenced_works_count"),
        "cited_by_count": rec.get("cited_by_count"),
        "domain": domain,
        "field": field,
        "subfield": subfield,
    }

def shard_path_for_work(work_id: str) -> str:
    """Replicate downloader sharding: ROOT/Wxx/Wxxxxxx/Wxxxx.pdf"""
    p1 = work_id[:3] if len(work_id) >= 3 else work_id
    p2 = work_id[:6] if len(work_id) >= 6 else work_id
    return os.path.join(PDFS_ROOT, p1, p2, f"{work_id}.pdf")

def file_is_valid_pdf(path: str) -> bool:
    try:
        return os.path.exists(path) and os.path.getsize(path) >= MIN_PDF_BYTES
    except Exception:
        return False

def count_pages(path: str) -> Optional[int]:
    """Return page count, or None on failure (encrypted/corrupt)."""
    try:
        with open(path, "rb") as f:
            reader = PdfReader(f, strict=False)
            # Some PDFs can be encrypted but still have no password requirement post-open
            if reader.is_encrypted:
                try:
                    reader.decrypt("")  # try empty password
                except Exception:
                    return None
            return len(reader.pages)
    except Exception:
        return None

def load_pagecount_cache() -> Dict[str, int]:
    if not os.path.exists(PAGECOUNT_CACHE_CSV):
        return {}
    try:
        df = pd.read_csv(PAGECOUNT_CACHE_CSV, dtype={"work_id": str, "pages": "Int64"})
        df = df.dropna(subset=["work_id", "pages"])
        return dict(zip(df["work_id"].astype(str), df["pages"].astype(int)))
    except Exception:
        return {}

def save_pagecount_cache(cache: Dict[str, int]) -> None:
    if not cache:
        return
    df = pd.DataFrame({"work_id": list(cache.keys()), "pages": list(cache.values())})
    df.to_csv(PAGECOUNT_CACHE_CSV, index=False)

# =========================
# Main
# =========================
def main():
    logging.info("Loading input JSON…")
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        records = json.load(f)
    logging.info(f"Loaded {len(records)} records")

    # Minimal rows
    rows = [extract_row(r) for r in tqdm(records, desc="Extracting", unit="rec")]
    df = pd.DataFrame(rows)
    for col in ["referenced_works_count", "cited_by_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Page counts (with cache)
    cache = load_pagecount_cache()
    pages: Dict[str, int] = {}
    to_do = []

    for wid in df["work_id"].astype(str):
        if wid in cache:
            pages[wid] = cache[wid]
        else:
            to_do.append(wid)

    if to_do:
        for wid in tqdm(to_do, desc="Counting PDF pages", unit="pdf"):
            pdf_path = shard_path_for_work(wid)
            if not file_is_valid_pdf(pdf_path):
                continue
            n = count_pages(pdf_path)
            if n is not None and n > 0:
                pages[wid] = n

        # update cache on disk
        cache.update(pages)
        save_pagecount_cache(cache)

    # Merge pages into dataframe
    df["pages"] = df["work_id"].map(pages)

    # Save tidy CSV with pages
    tidy_path = os.path.join(OUTPUT_DIR, "works_minimal_with_pages.csv")
    df.to_csv(tidy_path, index=False)

    # =========================
    # Plots (exactly as requested) + pages histogram
    # =========================
    # 1) Histogram of citation counts (log-scaled Y)
    plt.figure(figsize=(7,5))
    df["cited_by_count"].dropna().clip(upper=df["cited_by_count"].quantile(0.99)).plot.hist(bins=60)
    plt.title("Citations (99% capped)")
    plt.xlabel("Citations")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "citations_hist.png"))
    plt.close()

    # 2) Histogram of reference counts
    plt.figure(figsize=(7,5))
    df["referenced_works_count"].dropna().clip(upper=df["referenced_works_count"].quantile(0.99)).plot.hist(bins=60)
    plt.title("References (99% capped)")
    plt.xlabel("References")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "references_hist.png"))
    plt.close()

    # 3) Top 15 domains by record count
    top_domains = df["domain"].fillna("NA").value_counts().head(TOP_N)
    plt.figure(figsize=(8,6))
    top_domains.sort_values().plot(kind="barh")
    plt.title(f"Top {TOP_N} Domains by Count")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "domains_top15.png"))
    plt.close()

    # 4) Top 15 fields by record count
    top_fields = df["field"].fillna("NA").value_counts().head(TOP_N)
    plt.figure(figsize=(8,6))
    top_fields.sort_values().plot(kind="barh")
    plt.title(f"Top {TOP_N} Fields by Count")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "fields_top15.png"))
    plt.close()

    # 5) Histogram of PDF page counts
    plt.figure(figsize=(7,5))
    df["pages"].dropna().astype(int).clip(upper=df["pages"].quantile(0.99)).plot.hist(bins=60)
    plt.title("PDF Page Counts (99% capped)")
    plt.xlabel("Pages")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "pages_hist.png"))
    plt.close()

    logging.info("Saved plots to %s", PLOT_DIR)
    print(f"✅ Done. Outputs in {OUTPUT_DIR}")
    print(f"- works_minimal_with_pages.csv")
    print(f"- plots/citations_hist.png")
    print(f"- plots/references_hist.png")
    print(f"- plots/domains_top15.png")
    print(f"- plots/fields_top15.png")
    print(f"- plots/pages_hist.png")

if __name__ == "__main__":
    main()
