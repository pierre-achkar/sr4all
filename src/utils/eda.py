import json
import os
import logging
from typing import Any, Dict
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# CONFIG
# =========================
INPUT_JSONL = "data/final/sr4all_full_normalized_year_range.jsonl"
OUTPUT_DIR  = "data/final/eda"
LOG_FILE    = "logs/utils/eda.log"
FIELD_DIST_JSON = os.path.join(OUTPUT_DIR, "field_distributions.json")
FIELD_VALUE_COUNTS = os.path.join(OUTPUT_DIR, "fields_value_counts.json")

# Plot settings
TOP_N = 15
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")


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
    """Pull just what we need for EDA."""
    return {
        "field": rec.get("field"),
        "subfield": rec.get("subfield"),
    }

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
    logging.info("Loading input JSONL…")
    records = []
    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    logging.info(f"Loaded {len(records)} records")

    # Compute and save field distributions (presence / missing counts)
    def compute_field_distributions(recs):
        total = len(recs)
        keys = set()
        for r in recs:
            if isinstance(r, dict):
                keys.update(r.keys())

        stats = {}
        for k in sorted(keys):
            present = 0
            for r in recs:
                v = r.get(k) if isinstance(r, dict) else None
                if v not in (None, "", [], {}, False):
                    present += 1
            missing = total - present
            pct = (present / total) if total else 0.0
            stats[k] = {
                "present": present,
                "missing": missing,
                "total": total,
                "pct_present": round(pct, 4)
            }
        return stats

    logging.info("Computing field distributions...")
    field_stats = compute_field_distributions(records)
    try:
        with open(FIELD_DIST_JSON, "w", encoding="utf-8") as fo:
            json.dump(field_stats, fo, ensure_ascii=False, indent=2)
        logging.info(f"Wrote field distributions to: {FIELD_DIST_JSON}")
        print(f"Saved field distributions JSON to: {FIELD_DIST_JSON}")
    except Exception as e:
        logging.error(f"Failed to write field distributions: {e}")

    # 1. Extraction
    rows = [extract_row(r) for r in tqdm(records, desc="Extracting Metadata", unit="rec")]
    df = pd.DataFrame(rows)
    # =========================
    # 2. Visualization
    # =========================
    logging.info("Generating plots...")

    # A. Top Fields
    top_fields = df["field"].fillna("Unknown").value_counts().head(TOP_N)
    plot_barh(top_fields, f"Top {TOP_N} Fields", "fields_top15.png", palette="rocket")

    # Save full distribution of `field` values to JSON
    try:
        full_counts = df["field"].fillna("Unknown").value_counts().to_dict()
        with open(FIELD_VALUE_COUNTS, "w", encoding="utf-8") as fo:
            json.dump(full_counts, fo, ensure_ascii=False, indent=2)
        logging.info(f"Wrote full field value counts to: {FIELD_VALUE_COUNTS}")
        print(f"Saved full 'field' value counts to: {FIELD_VALUE_COUNTS}")
    except Exception as e:
        logging.error(f"Failed to write field value counts: {e}")

    # B. Top Subfields
    top_subfields = df["subfield"].fillna("Unknown").value_counts().head(TOP_N)
    plot_barh(top_subfields, f"Top {TOP_N} Subfields", "subfields_top15.png", palette="mako")

    print(f"\n✅ Done. Outputs in {OUTPUT_DIR}")
    print(f"   - plots/fields_top15.png")
    print(f"   - plots/subfields_top15.png")

if __name__ == "__main__":
    main()