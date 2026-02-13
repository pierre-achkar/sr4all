"""
Final Dataset Joiner.

Joins OpenAlex slim metadata with extraction results.

Input 1: oax_sr_slim.json (list of OpenAlex records)
Input 2: sr4all_final_0.jsonl (flattened extraction output)
Output: final_joined.jsonl
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
OAX_JSONL = Path("./data/filtered/oax_sr_slim_abstract_coverage.jsonl")
EXTRACTION_JSONL = Path(
    "./data/sr4all/extraction_v1/intermediate/sr4all_intermediate_all_corrected.jsonl"
)
OUTPUT_JSONL = Path("./data/final/sr4all_full.jsonl")
LOG_FILE = Path("./logs/final_ds/sr4all_full.log")
UNMATCHED_EXTRACTION_IDS = Path("./data/final/unmatched_extraction_ids_all.txt")
# Setup Logging
OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger("FinalJoiner")


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def normalize_openalex_id(openalex_id: Optional[str]) -> Optional[str]:
    """Return the OpenAlex ID without the https://openalex.org/ prefix."""
    if not openalex_id or not isinstance(openalex_id, str):
        return None
    openalex_id = openalex_id.strip()
    if not openalex_id:
        return None
    if "/" in openalex_id:
        return openalex_id.rsplit("/", 1)[-1]
    return openalex_id


# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
def main() -> None:
    if not OAX_JSONL.exists():
        logger.error(f"Input file not found: {OAX_JSONL}")
        return
    if not EXTRACTION_JSONL.exists():
        logger.error(f"Input file not found: {EXTRACTION_JSONL}")
        return

    logger.info("Loading OpenAlex slim records...")
    with open(OAX_JSONL, "r", encoding="utf-8") as f:
        oax_records = [json.loads(line) for line in f]

    oax_by_id: Dict[str, Dict[str, Any]] = {}
    for rec in oax_records:
        rec_id = normalize_openalex_id(rec.get("id"))
        if rec_id:
            oax_by_id[rec_id] = rec

    logger.info(f"OpenAlex records loaded: {len(oax_records)}")
    logger.info(f"OpenAlex records indexed: {len(oax_by_id)}")

    total_extraction = 0
    matched = 0
    unmatched_extraction = 0
    missing_doc_id = 0
    unmatched_extraction_ids = []

    with open(EXTRACTION_JSONL, "r", encoding="utf-8") as fin, open(
        OUTPUT_JSONL, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            try:
                extraction_rec = json.loads(line)
            except Exception as e:
                logger.warning(f"Skipping bad JSON line: {e}")
                continue

            total_extraction += 1
            doc_id = extraction_rec.get("doc_id")
            if not doc_id:
                unmatched_extraction += 1
                missing_doc_id += 1
                continue

            oax_rec = oax_by_id.get(str(doc_id))
            if not oax_rec:
                unmatched_extraction += 1
                unmatched_extraction_ids.append(str(doc_id))
                continue

            merged = {**oax_rec, **extraction_rec}
            merged.pop("file_path", None)
            merged.pop("doc_id", None)
            fout.write(json.dumps(merged, ensure_ascii=False) + "\n")
            matched += 1

    unmatched_oax = len(oax_by_id) - matched

    UNMATCHED_EXTRACTION_IDS.parent.mkdir(parents=True, exist_ok=True)
    with open(UNMATCHED_EXTRACTION_IDS, "w", encoding="utf-8") as f:
        for _id in unmatched_extraction_ids:
            f.write(f"{_id}\n")

    logger.info("-" * 40)
    logger.info("JOIN COMPLETE")
    logger.info(f"OpenAlex total:          {len(oax_records)}")
    logger.info(f"Extraction total:        {total_extraction}")
    logger.info(f"Matched:                 {matched}")
    logger.info(f"Unmatched (extraction):  {unmatched_extraction}")
    logger.info(f"Unmatched (openalex):    {unmatched_oax}")
    logger.info(f"Missing doc_id:          {missing_doc_id}")
    logger.info(f"Unmatched IDs file:      {UNMATCHED_EXTRACTION_IDS}")
    logger.info(f"Output file:             {OUTPUT_JSONL}")
    logger.info("-" * 40)


if __name__ == "__main__":
    main()
