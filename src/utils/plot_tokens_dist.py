import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# -----------------------------------------------------------------------------
# 1. CONFIGURATION
# -----------------------------------------------------------------------------
class Config:
    INPUT_PARQUET = Path("/home/fhg/pie65738/projects/sr4all/data/sr4all/token_counts.parquet")
    OUTPUT_PLOT = Path("/home/fhg/pie65738/projects/sr4all/plots/token_distribution_boxplot.png")
    
    # Analysis Thresholds
    CONTEXT_LIMIT_1 = 28000  # Qwen Original Limit for 32k models
    CONTEXT_LIMIT = 121000   # Critical Limit for 128k models
    
    # Visual settings
    FIG_SIZE = (12, 6)
    DPI = 300

# Setup Logging
Config.OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Plotter")

# -----------------------------------------------------------------------------
# 2. PLOTTING LOGIC
# -----------------------------------------------------------------------------
def main():
    if not Config.INPUT_PARQUET.exists():
        print(f"Error: Parquet file not found at {Config.INPUT_PARQUET}")
        print("Run the token check script first.")
        return

    print("Loading data...")
    df = pd.read_parquet(Config.INPUT_PARQUET)
    tokens = df["token_count"]

    # Calculate Statistics for the report
    Q1 = tokens.quantile(0.25)
    Q3 = tokens.quantile(0.75)
    IQR = Q3 - Q1
    upper_whisker = Q3 + 1.5 * IQR
    
    extreme_outliers = tokens[tokens > upper_whisker]
    over_limit = tokens[tokens > Config.CONTEXT_LIMIT]

    print("-" * 40)
    print(f"Total Docs:       {len(tokens)}")
    print(f"Median:           {tokens.median():.0f}")
    print(f"Q3 (75%):         {Q3:.0f}")
    print(f"IQR:              {IQR:.0f}")
    print(f"Whisker Limit:    {upper_whisker:.0f} (Traditional Outlier Threshold)")
    print("-" * 40)
    print(f"Statistical Outliers: {len(extreme_outliers)} docs ({len(extreme_outliers)/len(tokens):.1%} of corpus)")
    print(f"Original Limit (>28k): {len(tokens[tokens > Config.CONTEXT_LIMIT_1])} docs ({len(tokens[tokens > Config.CONTEXT_LIMIT_1])/len(tokens):.2%} of corpus)")
    print(f"CRITICAL (>121k):     {len(over_limit)} docs ({len(over_limit)/len(tokens):.2%} of corpus)")
    print("-" * 40)

    # Plotting
    print("Generating plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=Config.FIG_SIZE)

    # Plot 1: Linear Scale (Shows the extremes)
    ax1.boxplot(tokens, vert=True, patch_artist=True, boxprops=dict(facecolor="lightblue"))
    ax1.set_title("Linear Scale (Impact of Extreme Outliers)")
    ax1.set_ylabel("Token Count")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Plot 2: Log Scale (Shows the body)
    ax2.boxplot(tokens, vert=True, patch_artist=True, boxprops=dict(facecolor="lightgreen"))
    ax2.set_yscale("log")
    ax2.set_title("Log Scale (Distribution Body)")
    ax2.set_ylabel("Token Count (Log)")
    ax2.grid(True, linestyle="--", alpha=0.6, which="both")

    # Annotate the Context Window Limit
    ax2.axhline(y=32768, color='r', linestyle='--', label='32k Context')
    ax2.axhline(y=Config.CONTEXT_LIMIT, color='orange', linestyle='--', label='128k Context')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(Config.OUTPUT_PLOT, dpi=Config.DPI)
    print(f"Plot saved to: {Config.OUTPUT_PLOT}")

if __name__ == "__main__":
    main()