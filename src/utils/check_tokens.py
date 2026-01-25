import sys
import json
import logging
import numpy as np
import pandas as pd  
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
from prompts import SYSTEM_PROMPT, USER_TEMPLATE_RAW

# -----------------------------------------------------------------------------
# 1. CONFIGURATION
# -----------------------------------------------------------------------------
class Config:
    INPUT_DIR = Path("/home/fhg/pie65738/projects/sr4all/data/sr4all/md")
    OUTPUT_STATS = Path("/home/fhg/pie65738/projects/sr4all/data/sr4all/token_stats.json")
    OUTPUT_PARQUET = Path("/home/fhg/pie65738/projects/sr4all/data/sr4all/token_counts.parquet") 
    MODEL_PATH = "Qwen/Qwen3-32B" 
    MAX_FILES_TO_CHECK = None 
    LOG_FILE = Path("/home/fhg/pie65738/projects/sr4all/logs/extraction/token_check.log")

# Setup Logging
Config.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=Config.LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    filemode="w"
)
logger = logging.getLogger("TokenChecker")

# -----------------------------------------------------------------------------
# 2. MAIN LOGIC
# -----------------------------------------------------------------------------
def main():
    if not Config.INPUT_DIR.exists():
        print(f"Error: Input directory {Config.INPUT_DIR} does not exist.")
        return

    # Sanity Check
    if "{TEXT}" not in USER_TEMPLATE_RAW:
        print("CRITICAL ERROR: '{TEXT}' placeholder missing from USER_TEMPLATE_RAW in prompts.py")
        sys.exit(1)

    print(f"Loading Tokenizer: {Config.MODEL_PATH}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH, trust_remote_code=True)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load tokenizer. {e}")
        return

    print("Scanning for Markdown files (recursive)...")
    all_files = list(Config.INPUT_DIR.rglob("*.md"))
    total_files = len(all_files)
    print(f"Found {total_files} files.")

    data_records = [] # <--- Store tuples (id, tokens)

    print("\nStarting Tokenization...")
    for md_file in tqdm(all_files, unit="doc"):
        try:
            content = md_file.read_text(encoding="utf-8", errors="replace")
            user_content = USER_TEMPLATE_RAW.replace("{TEXT}", content)
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]
            full_text = tokenizer.apply_chat_template(messages, tokenize=False)
            
            tokens = tokenizer.encode(full_text, add_special_tokens=False)
            count = len(tokens)
            
            # Store Data
            data_records.append({
                "doc_id": md_file.stem,       # e.g., "W10005962"
                "file_path": str(md_file),    # Keep full path for loading later
                "token_count": count
            })
            
        except Exception as e:
            logger.error(f"Failed to process {md_file.name}: {e}")

    if not data_records:
        print("No files processed successfully.")
        return

    # -----------------------------------------------------------------------------
    # 3. SAVE PARQUET & STATS
    # -----------------------------------------------------------------------------
    df = pd.DataFrame(data_records)
    
    # Save detailed parquet
    df.to_parquet(Config.OUTPUT_PARQUET, index=False)
    print(f"\nSaved per-document counts to: {Config.OUTPUT_PARQUET}")

    # Calculate Stats
    counts_arr = df["token_count"].values
    stats = {
        "total_docs": int(len(counts_arr)),
        "min": int(np.min(counts_arr)),
        "max": int(np.max(counts_arr)),
        "mean": int(np.mean(counts_arr)),
        "median": int(np.median(counts_arr)),
        "p90": int(np.percentile(counts_arr, 90)),
        "p95": int(np.percentile(counts_arr, 95)),
        "p99": int(np.percentile(counts_arr, 99))
    }

    print("\n" + "="*40)
    print(f"RESULTS ({len(counts_arr)} docs)")
    print("="*40)
    print(f" Mean:   {stats['mean']}")
    print(f" Median: {stats['median']}")
    print(f" P95:    {stats['p95']}")
    print(f" P99:    {stats['p99']}")
    print(f" Max:    {stats['max']}")
    print("="*40)

    with open(Config.OUTPUT_STATS, "w") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()