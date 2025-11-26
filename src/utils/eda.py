import json
import os
import logging
from typing import Any, Dict, Optional
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from pypdf import PdfReader

# =========================
# CONFIG
# =========================
INPUT_JSON  = "../../data/filtered/oax_sr_slim.json"
PDFS_ROOT   = "../../data/filtered/pdfs"     
OUTPUT_DIR  = "../../data/filtered/eda"           
LOG_FILE    = "../../logs/utils/eda.log"

# Plot settings
TOP_N = 15
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

# Sanity threshold to consider a PDF valid
MIN_PDF_BYTES = 1024

# Cache for page counts
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

# Seaborn Aesthetic Setup
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
# palette = sns.color_palette("viridis", as_cmap=False) 

tqdm.pandas()

# =========================
# Helpers
# =========================
def extract_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Pull just what we need; fallback to top-scored topic if primary_topic missing."""
    wid = (rec.get("id") or "").split("/")[-1] or None
    
    # Check if 'primary_topic' is directly in root or inside a structure 
    # (Adjust based on your specific slim JSON structure)
    # Based on previous steps, 'field' and 'domain' are at root in slim JSON
    domain   = rec.get("domain")
    field    = rec.get("field")
    subfield = rec.get("subfield")

    return {
        "work_id": wid,
        "referenced_works_count": rec.get("referenced_works_count"),
        "cited_by_count": rec.get("cited_by_count"),
        "domain": domain,
        "field": field,
        "subfield": subfield,
    }

def shard_path_for_work(work_id: str) -> str:
    p1 = work_id[:3] if len(work_id) >= 3 else work_id
    p2 = work_id[:6] if len(work_id) >= 6 else work_id
    return os.path.join(PDFS_ROOT, p1, p2, f"{work_id}.pdf")

def file_is_valid_pdf(path: str) -> bool:
    try:
        return os.path.exists(path) and os.path.getsize(path) >= MIN_PDF_BYTES
    except Exception:
        return False

def count_pages(path: str) -> Optional[int]:
    try:
        with open(path, "rb") as f:
            reader = PdfReader(f, strict=False)
            if reader.is_encrypted:
                try:
                    reader.decrypt("") 
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
    if not cache: return
    df = pd.DataFrame({"work_id": list(cache.keys()), "pages": list(cache.values())})
    df.to_csv(PAGECOUNT_CACHE_CSV, index=False)

# =========================
# Plotting Functions
# =========================
def plot_hist(series: pd.Series, title: str, filename: str, xlabel: str, log_scale=False, color="teal"):
    """Generates a clean Seaborn histogram."""
    plt.figure(figsize=(8, 5))
    
    # 99th percentile clipping for better visualization of the mass
    cap = series.quantile(0.99)
    data = series.clip(upper=cap)
    
    sns.histplot(data, bins=50, kde=True, color=color, edgecolor="w", linewidth=0.5)
    
    plt.title(title, fontweight="bold", pad=15)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    
    if log_scale:
        plt.yscale("log")
        plt.ylabel("Frequency (Log Scale)")

    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=300)
    plt.close()

def plot_barh(series: pd.Series, title: str, filename: str, palette="viridis"):
    """Generates a horizontal bar chart with value annotations."""
    plt.figure(figsize=(10, 8))
    
    # Create plot
    ax = sns.barplot(x=series.values, y=series.index, hue=series.index, palette=palette, legend=False)
    
    # Styling
    plt.title(title, fontweight="bold", pad=15)
    plt.xlabel("Count")
    plt.ylabel("") # Index labels are self-explanatory
    
    # Annotate bars with counts
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        ax.text(width + (width * 0.01), p.get_y() + p.get_height() / 2, 
                f'{int(width)}', ha='left', va='center', fontsize=10, color="#333333")
    
    sns.despine(left=True, bottom=False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=300)
    plt.close()

# =========================
# Main
# =========================
def main():
    logging.info("Loading input JSON…")
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        records = json.load(f)
    logging.info(f"Loaded {len(records)} records")

    # 1. Extraction
    rows = [extract_row(r) for r in tqdm(records, desc="Extracting Metadata", unit="rec")]
    df = pd.DataFrame(rows)
    for col in ["referenced_works_count", "cited_by_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2. PDF Page Counting
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

        cache.update(pages)
        save_pagecount_cache(cache)

    df["pages"] = df["work_id"].map(pages)

    # Save minimal CSV
    tidy_path = os.path.join(OUTPUT_DIR, "works_minimal_with_pages.csv")
    df.to_csv(tidy_path, index=False)

    # =========================
    # 3. Visualization
    # =========================
    logging.info("Generating plots...")

    # A. Citations (Log Scale Y is usually best for citations)
    plot_hist(
        df["cited_by_count"].dropna(), 
        title="Distribution of Citations (99% capped)", 
        filename="citations_hist.png", 
        xlabel="Citations",
        log_scale=True,
        color="#34495e"
    )

    # B. References
    plot_hist(
        df["referenced_works_count"].dropna(), 
        title="Distribution of References (99% capped)", 
        filename="references_hist.png", 
        xlabel="Number of References",
        color="#2ecc71"
    )

    # C. PDF Pages
    plot_hist(
        df["pages"].dropna(), 
        title="Distribution of PDF Page Counts", 
        filename="pages_hist.png", 
        xlabel="Pages",
        color="#e74c3c"
    )

    # D. Top Domains
    top_domains = df["domain"].fillna("Unknown").value_counts().head(TOP_N)
    plot_barh(top_domains, f"Top {TOP_N} Domains", "domains_top15.png", palette="mako")

    # E. Top Fields
    top_fields = df["field"].fillna("Unknown").value_counts().head(TOP_N)
    plot_barh(top_fields, f"Top {TOP_N} Fields", "fields_top15.png", palette="rocket")

    # F. Non-Medicine Stats
    num_non_medicine = df[df["field"] != "Medicine"].shape[0]
    logging.info(f"Number of works with field not 'Medicine': {num_non_medicine}")

    print(f"\n✅ Done. Outputs in {OUTPUT_DIR}")
    print(f"   - works_minimal_with_pages.csv")
    print(f"   - plots/citations_hist.png")
    print(f"   - plots/references_hist.png")
    print(f"   - plots/domains_top15.png")
    print(f"   - plots/fields_top15.png")
    print(f"   - plots/pages_hist.png")

if __name__ == "__main__":
    main()