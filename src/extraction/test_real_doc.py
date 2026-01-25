import sys
import json
import logging
from pathlib import Path
import time

# Ensure we can import from src
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inference_engine import QwenInference

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
# Update this path to one of your real markdown files
TEST_FILE_PATH = Path("/home/fhg/pie65738/projects/sr4all/data/sr4all/md/W10/W10005/W10005962.md") 

# Update this to your model path
MODEL_PATH = "Qwen/Qwen3-32B" 

def main():
    # 1. Load File
    if not TEST_FILE_PATH.exists():
        print(f"❌ Error: File not found at {TEST_FILE_PATH}")
        print("Please edit TEST_FILE_PATH in the script to point to a real .md file.")
        return

    print(f"📖 Reading {TEST_FILE_PATH}...")
    text = TEST_FILE_PATH.read_text(encoding="utf-8", errors="replace")
    print(f"   Size: {len(text)} characters")

    # 2. Initialize Engine
    print("\n🚀 Initializing H100 Inference Engine...")
    try:
        engine = QwenInference(MODEL_PATH, tensor_parallel=2)
    except Exception as e:
        print(f"❌ Initialization Failed: {e}")
        return

    # 3. Run Extraction
    print("\n🧠 Extracting Data (Thinking disabled, Schema enforced)...")
    start_time = time.perf_counter()
    
    result_json, raw_output = engine.generate(text)
    
    duration = time.perf_counter() - start_time
    print(f"   Finished in {duration:.2f} seconds.")

    # 4. Show Results
    print("\n" + "="*60)
    print("EXTRACTION RESULTS")
    print("="*60)
    
    if result_json:
        # Print formatted JSON
        print(json.dumps(result_json, indent=2))
        
        # Save for inspection
        output_file = TEST_FILE_PATH.with_suffix(".json")
        output_file.write_text(json.dumps(result_json, indent=2), encoding="utf-8")
        print(f"\n✅ Saved result to: {output_file}")
    else:
        print("❌ Extraction Failed.")
        print("Raw Output Dump:")
        print(raw_output)

if __name__ == "__main__":
    main()