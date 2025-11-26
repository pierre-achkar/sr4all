import os
import time
import fitz  # PyMuPDF
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from chandra.model import InferenceManager
from chandra.model.schema import BatchInputItem

# ---------- CONFIG ----------
PDF_PATH = "/home/fhg/pie65738/projects/sr4all/data/filtered/pdfs/W10/W10005/W10005962.pdf"
OUTPUT_MD = "output.md"

SCALE = 2.0    # image render scale; increase to 2.5 if small fonts
CHUNK = 8      # batch size; increase to 10-12 if GPU VRAM free
RETRIES = 3
BACKOFF = 1.8
# ----------------------------

os.environ.setdefault("CHANDRA_VLLM_SERVER", "http://localhost:8000/v1")
os.environ.setdefault("CHANDRA_VLLM_MODEL", "chandra")

manager = InferenceManager(method="vllm")


def yield_batches(doc, chunk, scale):
    M = fitz.Matrix(scale, scale)
    buf = []
    for page_idx in range(len(doc)):
        page = doc.load_page(page_idx)
        pix = page.get_pixmap(matrix=M, colorspace=fitz.csRGB, alpha=False)
        img = Image.frombuffer("RGB", (pix.width, pix.height), pix.samples, "raw", "RGB", 0, 1)
        buf.append(BatchInputItem(image=img, prompt_type="ocr_layout"))
        if len(buf) == chunk:
            yield buf
            buf = []
    if buf:
        yield buf


def safe_generate(batch):
    for attempt in range(RETRIES):
        try:
            return manager.generate(batch)
        except Exception:
            if attempt == RETRIES - 1:
                raise
            time.sleep(BACKOFF ** attempt)


def main():
    pdf = Path(PDF_PATH)
    out = Path(OUTPUT_MD)
    part = out.with_suffix(out.suffix + ".part")

    doc = fitz.open(pdf.as_posix())
    total_batches = (len(doc) + CHUNK - 1) // CHUNK

    with part.open("w", encoding="utf-8") as f:
        for batch in tqdm(yield_batches(doc, CHUNK, SCALE), total=total_batches, desc="OCR"):
            results = safe_generate(batch)
            for r in results:
                f.write(r.markdown)
                f.write("\n\n---\n\n")

    doc.close()
    part.replace(out)
    print(f"✅ Saved → {out.resolve()}")


if __name__ == "__main__":
    main()
