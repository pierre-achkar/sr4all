import os
import base64
from io import BytesIO

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_no_anchoring_v4_yaml_prompt

# Optional: use PyPDF2 just to count pages
try:
    from PyPDF2 import PdfReader
    HAVE_PYPDF2 = True
except Exception:
    HAVE_PYPDF2 = False


# =========================
# Configuration (edit here)
# =========================
PDF_PATH = "/home/fhg/pie65738/projects/sr4all/data/filtered/pdfs/W10/W10005/W10005962.pdf"                     
OUTPUT_MD = "./olmocr_output.md"             # Single Markdown output
MODEL_ID = "allenai/olmOCR-2-7B-1025-FP8"
PROCESSOR_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# Image render settings
TARGET_LONGEST_DIM = 1288                    # 1025–1536 is typical; tradeoff quality vs speed/mem

# Generation settings
MAX_NEW_TOKENS = 512
DO_SAMPLE = False                            # deterministic by default
TEMPERATURE = 0.1

# Optional: page range (1-indexed, inclusive). Set to None to auto-detect full doc.
PAGE_START = None
PAGE_END = None
# =========================


def count_pages(pdf_path: str) -> int:
    if HAVE_PYPDF2:
        reader = PdfReader(pdf_path)
        return len(reader.pages)
    # Fallback: probe until render fails (slower)
    n = 1
    while True:
        try:
            _ = render_pdf_to_base64png(pdf_path, n, target_longest_image_dim=64)
            n += 1
        except Exception:
            break
    return n - 1


def build_inputs(processor, image_b64: str):
    """Build chat-formatted inputs for one page."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": build_no_anchoring_v4_yaml_prompt()},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}" }},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    main_image = Image.open(BytesIO(base64.b64decode(image_b64)))

    inputs = processor(
        text=[text],
        images=[main_image],
        padding=True,
        return_tensors="pt",
    )
    return inputs


def generate_page_markdown(model, processor, device, image_b64: str) -> str:
    inputs = build_inputs(processor, image_b64)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            temperature=TEMPERATURE,
            max_new_tokens=MAX_NEW_TOKENS,
            num_return_sequences=1,
            do_sample=DO_SAMPLE,
        )

    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = output[:, prompt_length:]
    page_text = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
    return page_text.strip()


def main():
    # 1) Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Using device: {device}")

    # 2) Download sample if PDF not present (matches your original demo)
    if not os.path.exists(PDF_PATH):
        import urllib.request
        print(f"[info] Downloading sample PDF to {PDF_PATH}")
        urllib.request.urlretrieve("https://olmocr.allenai.org/papers/olmocr.pdf", PDF_PATH)

    # 3) Page range
    total_pages = count_pages(PDF_PATH)
    start = PAGE_START if PAGE_START is not None else 1
    end = PAGE_END if PAGE_END is not None else total_pages
    if start < 1 or end < start or end > total_pages:
        raise ValueError(f"Invalid page range: {start}-{end} for a {total_pages}-page PDF")

    print(f"[info] PDF pages: {total_pages}. Processing pages {start}..{end}")

    # 4) Load model & processor once
    print(f"[info] Loading model: {MODEL_ID}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16
    ).eval().to(device)

    print(f"[info] Loading processor: {PROCESSOR_ID}")
    processor = AutoProcessor.from_pretrained(PROCESSOR_ID)

    # 5) Iterate pages
    md_chunks = []
    for page_num in range(start, end + 1):
        print(f"[info] Rendering page {page_num}...")
        image_b64 = render_pdf_to_base64png(
            PDF_PATH,
            page_num,
            target_longest_image_dim=TARGET_LONGEST_DIM
        )

        print(f"[info] Generating text for page {page_num}...")
        page_md = generate_page_markdown(model, processor, device, image_b64)

        # Wrap per-page content
        md_section = f"## Page {page_num}\n\n{page_md}\n"
        md_chunks.append(md_section)

    # 6) Write single Markdown file
    full_md = "# OLM-OCR Extraction\n\n" + "\n\n---\n\n".join(md_chunks) + "\n"
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(full_md)

    print(f"[done] Wrote Markdown to: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
