"""
This script uses FastText to detect the language of the text in the markdown files. It is
designed to be noise-resistant by aggressively stripping HTML tags, LaTeX artifacts, URLs, 
and markdown syntax before classification. It checks both the head (abstract) and middle (body) 
of the document to catch cases where the abstract might be in English but the body is not. 
Files that are classified as non-English with high confidence are moved to a separate "rejected" directory for manual review.
"""
import shutil
import logging
import math
import re
from pathlib import Path
from tqdm import tqdm
import fasttext
from huggingface_hub import hf_hub_download

# -----------------------------------------------------------------------------
# 1. CONFIGURATION
# -----------------------------------------------------------------------------
CONFIG = {
    "input_dir": Path("/data/sr4all/md"),
    "rejected_dir": Path("/data/sr4all/rejected/non_english"),
    "log_file": Path("/logs/extraction/language_filter_fasttext.log"),
    "min_probability": 0.85,
    "sample_size": 4000
}

# Setup Logging
CONFIG["log_file"].parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=CONFIG["log_file"],
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    filemode="w"
)
logger = logging.getLogger("FastTextFilter")

# -----------------------------------------------------------------------------
# 2. MODEL LOADER
# -----------------------------------------------------------------------------
def load_fasttext_model():
    model_path = Path("lid.176.bin")
    if not model_path.exists():
        try:
            downloaded = hf_hub_download(
                repo_id="facebook/fasttext-language-identification", 
                filename="model.bin"
            )
            shutil.copy(downloaded, model_path)
        except Exception as e:
            print(f"Download failed: {e}")
            raise e
    fasttext.FastText.eprint = lambda x: None
    return fasttext.load_model(str(model_path))

# -----------------------------------------------------------------------------
# 3. CLEANING LOGIC (UPDATED)
# -----------------------------------------------------------------------------
class LanguageValidator:
    def __init__(self):
        self.model = load_fasttext_model()
        
        # Pre-compile regex for speed
        self.re_html = re.compile(r'<[^>]+>')           # Remove <div...>, <table...>
        self.re_md = re.compile(r'[#\*\_`\$\[\]]')      # Remove #, *, _, `, $, [, ]
        self.re_latex = re.compile(r'\^\{\d+\}')        # Remove ^{1} reference markers
        self.re_url = re.compile(r'http[s]?://\S+')     # Remove URLs
        self.re_spaces = re.compile(r'\s+')             # Collapse whitespace

    def clean_text(self, text: str) -> str:
        """
        Aggressively strips markup to reveal the underlying human language.
        """
        # 1. Remove HTML tags (tables/divs)
        text = self.re_html.sub(' ', text)
        # 2. Remove LaTeX/Math artifacts
        text = self.re_latex.sub(' ', text)
        # 3. Remove URLs
        text = self.re_url.sub(' ', text)
        # 4. Remove Markdown/Special chars
        text = self.re_md.sub(' ', text)
        # 5. Collapse multiple spaces/newlines into single space
        return self.re_spaces.sub(' ', text).strip()

    def check_file(self, filepath: Path) -> bool:
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()

            if len(raw_text) < 100:
                return False 

            # Zone 1: Head (Abstract)
            # Take a slightly larger chunk to account for stripped HTML
            head_chunk = raw_text[:CONFIG["sample_size"] * 2] 
            if not self._check_chunk(head_chunk, "HEAD", filepath.name):
                return False

            # Zone 2: Middle (Body)
            if len(raw_text) > CONFIG["sample_size"] * 4:
                mid_idx = math.floor(len(raw_text) / 2)
                mid_chunk = raw_text[mid_idx : mid_idx + (CONFIG["sample_size"] * 2)]
                if not self._check_chunk(mid_chunk, "BODY", filepath.name):
                    return False

            return True

        except Exception as e:
            logger.error(f"Read Error {filepath.name}: {e}")
            return False

    def _check_chunk(self, text_chunk: str, zone: str, fname: str) -> bool:
        clean = self.clean_text(text_chunk)
        
        # If cleaning removed everything (e.g. just a giant table), fail safe (or skip)
        if len(clean) < 50:
            return True # Assume it's okay if we can't find text, let the LLM handle it.

        labels, scores = self.model.predict(clean, k=1)
        lang_code = labels[0].replace("__label__", "")
        score = scores[0]

        # Check generic English (en) or Latinized English (eng_Latn)
        if "en" not in lang_code:
            logger.info(f"Reject {fname} [{zone}]: Detected {lang_code} ({score:.2f})")
            return False
        
        if score < CONFIG["min_probability"]:
            logger.info(f"Reject {fname} [{zone}]: Low English Confidence ({score:.2f})")
            return False
            
        return True

# -----------------------------------------------------------------------------
# 4. EXECUTION
# -----------------------------------------------------------------------------
def main():
    if not CONFIG["input_dir"].exists():
        print("Input directory not found.")
        return

    CONFIG["rejected_dir"].mkdir(parents=True, exist_ok=True)
    
    print("Loading FastText Model...")
    validator = LanguageValidator()

    files = list(CONFIG["input_dir"].rglob("*.md"))
    print(f"Scanning {len(files)} files...")

    kept = 0
    moved = 0

    for f in tqdm(files, desc="Noise-Resistant Filter"):
        is_english = validator.check_file(f)
        
        if not is_english:
            rel_path = f.relative_to(CONFIG["input_dir"])
            dest = CONFIG["rejected_dir"] / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(dest))
            moved += 1
        else:
            kept += 1

    logging.info("-" * 40)
    logging.info(f"Filter Complete.")
    logging.info(f"Kept:     {kept}")
    logging.info(f"Rejected: {moved}")
    logging.info(f"Logs:     {CONFIG['log_file']}")
    logging.info("-" * 40)

if __name__ == "__main__":
    main()