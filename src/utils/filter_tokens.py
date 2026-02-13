"""
This script filters the SR4All dataset based on token counts. It reads a Parquet file
containing file paths and their corresponding token counts, identifies files that exceed a specified token threshold (e.g., 121k tokens),
and moves those files to a separate "rejected" directory for isolation.
The remaining valid files are saved in a new Parquet file sorted by token count to optimize VLLM throughput during training.
Detailed logging is maintained throughout the process to track moved files, missing files, and any errors encountered.
"""

import shutil
import pandas as pd
import logging
from pathlib import Path
from tqdm import tqdm


# -----------------------------------------------------------------------------
# 1. CONFIGURATION
# -----------------------------------------------------------------------------
class Config:
    # INPUTS
    INPUT_PARQUET = Path("./data/sr4all/token_counts.parquet")

    # OUTPUTS
    CLEAN_PARQUET = Path("./data/sr4all/clean_corpus.parquet")
    REJECTED_DIR = Path("./data/sr4all/rejected_over_121k")

    # THRESHOLDS
    # 131k (Model) - 10k (Output Buffer) = 121k Max Input
    MAX_TOKENS = 121000

    # LOGGING
    LOG_FILE = Path("/logs/filtering.log")


# Setup Logging
Config.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=Config.LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("Filter")


# -----------------------------------------------------------------------------
# 2. CORE LOGIC
# -----------------------------------------------------------------------------
def main():
    if not Config.INPUT_PARQUET.exists():
        print(f"Error: Input file {Config.INPUT_PARQUET} not found.")
        return

    # Ensure rejection dump exists
    Config.REJECTED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading token counts...")
    df = pd.read_parquet(Config.INPUT_PARQUET)
    total_docs = len(df)

    # 1. SPLIT DATA
    # -------------------------------------------------------------------------
    mask_rejected = df["token_count"] > Config.MAX_TOKENS
    rejected_df = df[mask_rejected].copy()
    valid_df = df[~mask_rejected].copy()

    print(f"Total Documents:    {total_docs}")
    print(f"Valid (<121k):      {len(valid_df)}")
    print(f"Rejected (>121k):   {len(rejected_df)} ({len(rejected_df)/total_docs:.2%})")

    # 2. MOVE REJECTED FILES
    # -------------------------------------------------------------------------
    print("\nMoving rejected files to isolation...")

    moved_count = 0
    missing_count = 0

    for _, row in tqdm(rejected_df.iterrows(), total=len(rejected_df), unit="file"):
        src_path = Path(row["file_path"])

        # We flatten the structure in the rejected folder (just ID.md)
        # to avoid recreating the complex W10/W1005 tree structure.
        dest_path = Config.REJECTED_DIR / f"{row['doc_id']}.md"

        try:
            if src_path.exists():
                shutil.move(str(src_path), str(dest_path))
                moved_count += 1
                logger.info(
                    f"MOVED: {src_path} -> {dest_path} | Tokens: {row['token_count']}"
                )
            else:
                missing_count += 1
                logger.warning(f"MISSING: Could not find source file {src_path}")
        except Exception as e:
            logger.error(f"FAILED to move {src_path}: {e}")

    # 3. SAVE CLEAN LIST (SORTED)
    # -------------------------------------------------------------------------
    print("\nSorting valid corpus by length (Short -> Long)...")
    # Sorting improves VLLM throughput by minimizing padding in batches
    valid_df = valid_df.sort_values(by="token_count", ascending=True)

    valid_df.to_parquet(Config.CLEAN_PARQUET, index=False)

    print("-" * 40)
    print(f"Operation Complete.")
    print(f"Moved:       {moved_count} files")
    print(f"Missing:     {missing_count} files (already moved?)")
    print(f"Clean List:  {Config.CLEAN_PARQUET}")
    print(f"Rejected Dir:{Config.REJECTED_DIR}")
    print("-" * 40)


if __name__ == "__main__":
    main()
