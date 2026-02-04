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

    # B. Top Subfields
    top_subfields = df["subfield"].fillna("Unknown").value_counts().head(TOP_N)
    plot_barh(top_subfields, f"Top {TOP_N} Subfields", "subfields_top15.png", palette="mako")

    print(f"\n✅ Done. Outputs in {OUTPUT_DIR}")
    print(f"   - plots/fields_top15.png")
    print(f"   - plots/subfields_top15.png")

if __name__ == "__main__":
    main()