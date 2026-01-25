import shutil
import logging
from pathlib import Path
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 1. CONFIGURATION (Must match your previous script)
# -----------------------------------------------------------------------------
REJECTED_DIR = Path("/home/fhg/pie65738/projects/sr4all/data/sr4all/rejected/non_english")
ORIGINAL_DIR = Path("/home/fhg/pie65738/projects/sr4all/data/sr4all/md")

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Restorer")

def main():
    if not REJECTED_DIR.exists():
        print(f"Directory not found: {REJECTED_DIR}")
        return

    print(f"Scanning {REJECTED_DIR}...")
    files_to_restore = list(REJECTED_DIR.rglob("*.md"))
    
    if not files_to_restore:
        print("No files found to restore.")
        return

    print(f"Found {len(files_to_restore)} files to move back.")
    
    moved_count = 0
    
    for src_path in tqdm(files_to_restore, desc="Restoring"):
        try:
            # Calculate original path
            relative_path = src_path.relative_to(REJECTED_DIR)
            dest_path = ORIGINAL_DIR / relative_path
            
            # Ensure parent exists
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Move
            shutil.move(str(src_path), str(dest_path))
            moved_count += 1
            
        except Exception as e:
            logger.error(f"Failed to restore {src_path.name}: {e}")

    print("-" * 30)
    print(f"Restoration Complete.")
    print(f"Moved back: {moved_count} files")
    print("-" * 30)

    # Optional: Remove empty rejected dir if clean
    try:
        if not any(REJECTED_DIR.iterdir()):
            REJECTED_DIR.rmdir()
            print("Removed empty rejected directory.")
    except:
        pass

if __name__ == "__main__":
    main()