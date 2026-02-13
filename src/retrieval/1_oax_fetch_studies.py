"""
Fetching Systematic Review Studies from OpenAlex
- Uses the OpenAlex API to search for works with "systematic review" in the title or abstract
- Handles pagination and rate limits to retrieve all relevant studies
- Saves the raw results to JSON files for downstream processing (deduplication, filtering, etc.)
- Logs progress and any issues encountered during fetching
"""
import requests
import json
import os
import re
import logging
from tqdm import tqdm

# Config
PHRASES = [
    "systematic review",
    "systematic literature review",
]
WORK_TYPE = "review"
PER_PAGE = 200                 # OpenAlex max
MAX_RESULTS = None             # <-- test cap; set to None to fetch ALL
OUTPUT_PREFIX = "./data/raw/oax_sr_full"
LOG_FILE = "./logs/retrieval/oax_fetch_studies.log"
MAILTO = "email@example.com"
SHARD_SIZE = 10_000            # save progress every N items

# Logging
_LOGGER = None

def _get_logger():
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER

    log_path = LOG_FILE
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger("oax_fetch_studies")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)

    logger.propagate = False
    _LOGGER = logger
    return logger

# =========================
# Helpers
# =========================
def _normalize_title(title: str) -> str:
    if not title:
        return ""
    t = title.lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\s]", "", t)
    return t.strip()

def _deduplicate(records):
    seen_doi, seen_id, seen_title = set(), set(), set()
    out = []
    for w in records:
        doi = w.get("doi")
        oid = w.get("id")
        tnorm = _normalize_title(w.get("title"))
        if doi and doi in seen_doi:
            continue
        if oid and oid in seen_id:
            continue
        if (not doi) and (not oid) and tnorm and tnorm in seen_title:
            continue
        if doi: seen_doi.add(doi)
        if oid: seen_id.add(oid)
        if (not doi) and (not oid) and tnorm: seen_title.add(tnorm)
        out.append(w)
    return out

def _write_shard(shard_idx, buffer):
    path = f"{OUTPUT_PREFIX}.part{shard_idx:03d}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(buffer, f, ensure_ascii=False, indent=2)
    _get_logger().info("Saved shard %03d with %d works to %s", shard_idx, len(buffer), path)
    return path

def _merge_shards(shard_paths):
    merged = []
    for p in shard_paths:
        with open(p, "r", encoding="utf-8") as f:
            merged.extend(json.load(f))
    return merged

# Core fetch
def fetch_openalex_full(or_query: str, work_type: str, per_page: int = 200, max_results=None):
    base_url = "https://api.openalex.org/works"
    cursor = "*"
    total_count = None
    pulled = 0

    params_common = {
        "filter": f"title_and_abstract.search:{or_query},type:{work_type}",
        "per-page": per_page,
    }
    if MAILTO:
        params_common["mailto"] = MAILTO

    shard_paths = []
    buffer = []
    shard_idx = 0

    with tqdm(unit="works", desc="Fetching") as pbar:
        while True:
            params = dict(params_common)
            params["cursor"] = cursor

            r = requests.get(base_url, params=params, timeout=120)
            r.raise_for_status()
            data = r.json()

            if total_count is None:
                total_count = data.get("meta", {}).get("count", 0)
                pbar.total = total_count
                _get_logger().info("OpenAlex meta.count=%s for query '%s'", total_count, or_query)

            works = data.get("results", [])
            if not works:
                break

            # trim last page if testing
            if max_results is not None and pulled + len(works) > max_results:
                works = works[:max(0, max_results - pulled)]

            # add to buffer
            for w in works:
                buffer.append(w)
                pulled += 1
                if len(buffer) >= SHARD_SIZE:
                    shard_paths.append(_write_shard(shard_idx, buffer))
                    shard_idx += 1
                    buffer = []

            pbar.update(len(works))

            if max_results is not None and pulled >= max_results:
                break

            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break

    # write final (possibly partial) shard
    if buffer:
        shard_paths.append(_write_shard(shard_idx, buffer))

    return total_count or 0, pulled, shard_paths

# Run
if __name__ == "__main__":
    or_query = "|".join([f'"{p}"' for p in PHRASES])

    total_found, pulled, shard_paths = fetch_openalex_full(
        or_query=or_query,
        work_type=WORK_TYPE,
        per_page=PER_PAGE,
        max_results=MAX_RESULTS,
    )

    logger = _get_logger()
    logger.info(
        "Run summary: meta.count=%s pulled=%s shards=%s",
        total_found,
        pulled,
        len(shard_paths),
    )

    print(f"Total found by OpenAlex (meta.count): {total_found}")
    print(f"Pulled (before dedupe): {pulled}")
    print(f"Wrote {len(shard_paths)} shard(s):")
    for p in shard_paths:
        print("  -", p)

    # merge -> dedupe -> write final json list
    merged = _merge_shards(shard_paths)
    merged_count = len(merged)
    deduped = _deduplicate(merged)
    deduped_count = len(deduped)

    final_path = f"{OUTPUT_PREFIX}.json"
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(deduped, f, ensure_ascii=False, indent=2)

    logger.info(
        "Deduplicated %s merged works down to %s unique works (removed %s)",
        merged_count,
        deduped_count,
        merged_count - deduped_count,
    )
    logger.info("Saved merged JSON list to: %s", final_path)

    print(f"After dedupe: {deduped_count}")
    print(f"Saved merged JSON list to: {final_path}")

