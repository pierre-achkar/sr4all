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
import time
import random
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment from src/.env
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

# Config
PHRASES = [
    "systematic review",
    "systematic literature review",
]
WORK_TYPE = "review"
PER_PAGE = 200                 
MAX_RESULTS = None            
OUTPUT_PREFIX = "./data/raw/oax_sr_full"
LOG_FILE = "./logs/retrieval/1_oax_fetch_studies.log"
MAILTO = os.getenv("OPENALEX_EMAIL")     
SHARD_SIZE = 10_000            # save progress every N items
REQUEST_TIMEOUT = 120
MAX_REQUEST_RETRIES = 10
MAX_RATE_LIMIT_RETRIES = 1000
BASE_BACKOFF_SECONDS = 2
MAX_BACKOFF_SECONDS = 90
RESUME_IF_SHARDS_EXIST = True
CHECKPOINT_PATH = f"{OUTPUT_PREFIX}.checkpoint.json"

def _load_openalex_api_keys():
    keys = []
    raw_list = os.getenv("OPENALEX_API_KEYS", "")
    if raw_list:
        keys.extend([k.strip() for k in raw_list.split(",") if k.strip()])

    for env_name, value in os.environ.items():
        if env_name == "OPENALEX_API_KEYS":
            continue
        if not env_name.startswith("OPENALEX_API_KEY"):
            continue
        key = value.strip()
        if key:
            keys.append(key)

    # de-duplicate while preserving order
    seen = set()
    unique_keys = []
    for k in keys:
        if k in seen:
            continue
        seen.add(k)
        unique_keys.append(k)
    return unique_keys

OPENALEX_API_KEYS = _load_openalex_api_keys()

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
        handler = logging.FileHandler(log_path, mode="w")
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
    filtered_doi = 0
    filtered_id = 0
    filtered_title = 0

    for w in records:
        doi = w.get("doi")
        oid = w.get("id")
        tnorm = _normalize_title(w.get("title"))
        if doi and doi in seen_doi:
            filtered_doi += 1
            continue
        if oid and oid in seen_id:
            filtered_id += 1
            continue
        if (not doi) and (not oid) and tnorm and tnorm in seen_title:
            filtered_title += 1
            continue
        if doi: seen_doi.add(doi)
        if oid: seen_id.add(oid)
        if (not doi) and (not oid) and tnorm: seen_title.add(tnorm)
        out.append(w)
    stats = {
        "input_total": len(records),
        "output_total": len(out),
        "filtered_doi": filtered_doi,
        "filtered_id": filtered_id,
        "filtered_title": filtered_title,
        "filtered_total": filtered_doi + filtered_id + filtered_title,
    }
    return out, stats

def _write_shard(shard_idx, buffer):
    path = f"{OUTPUT_PREFIX}.part{shard_idx:03d}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
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

def _discover_existing_shards():
    base = Path(OUTPUT_PREFIX)
    shard_glob = f"{base.name}.part*.json"
    shard_files = []

    for p in base.parent.glob(shard_glob):
        m = re.search(r"\.part(\d+)\.json$", p.name)
        if not m:
            continue
        shard_files.append((int(m.group(1)), str(p)))

    shard_files.sort(key=lambda x: x[0])
    ordered_paths = [p for _, p in shard_files]
    next_shard_idx = (shard_files[-1][0] + 1) if shard_files else 0
    return ordered_paths, next_shard_idx

def _count_records_in_shards(shard_paths):
    total = 0
    for p in shard_paths:
        with open(p, "r", encoding="utf-8") as f:
            total += len(json.load(f))
    return total

def _load_checkpoint():
    if not os.path.exists(CHECKPOINT_PATH):
        return None
    try:
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except (OSError, json.JSONDecodeError) as exc:
        _get_logger().warning("Could not read checkpoint file %s: %s", CHECKPOINT_PATH, type(exc).__name__)
        return None

def _save_checkpoint(next_cursor, saved_count, shard_idx):
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    payload = {
        "next_cursor": next_cursor,
        "saved_count": saved_count,
        "next_shard_idx": shard_idx,
        "updated_at_epoch": int(time.time()),
    }
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def _clear_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)

def _request_openalex_with_retries(session, url, params, api_keys=None, key_index=0):
    logger = _get_logger()
    transient_attempt = 0
    rate_limit_attempt = 0
    while True:
        params_for_attempt = dict(params)
        if api_keys:
            params_for_attempt["api_key"] = api_keys[key_index % len(api_keys)]
            key_index += 1

        try:
            response = session.get(url, params=params_for_attempt, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            transient_attempt += 1
            if transient_attempt >= MAX_REQUEST_RETRIES:
                raise RuntimeError(
                    f"OpenAlex request failed after {MAX_REQUEST_RETRIES} retries: {type(exc).__name__}"
                ) from exc

            sleep_for = min(MAX_BACKOFF_SECONDS, BASE_BACKOFF_SECONDS * (2 ** (transient_attempt - 1)))
            sleep_for += random.uniform(0, 1.0)
            logger.warning(
                "OpenAlex request error (%s). Retry %s/%s in %.1fs",
                type(exc).__name__,
                transient_attempt,
                MAX_REQUEST_RETRIES,
                sleep_for,
            )
            time.sleep(sleep_for)
            continue

        if response.status_code == 200:
            return response, key_index

        if response.status_code == 429:
            rate_limit_attempt += 1
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    sleep_for = min(MAX_BACKOFF_SECONDS, float(retry_after))
                except ValueError:
                    sleep_for = min(MAX_BACKOFF_SECONDS, BASE_BACKOFF_SECONDS * (2 ** min(rate_limit_attempt - 1, 6)))
            else:
                sleep_for = min(MAX_BACKOFF_SECONDS, BASE_BACKOFF_SECONDS * (2 ** min(rate_limit_attempt - 1, 6)))
            sleep_for += random.uniform(0, 1.0)
            logger.warning(
                "OpenAlex rate-limited (429). Retry %s/%s in %.1fs",
                rate_limit_attempt,
                MAX_RATE_LIMIT_RETRIES,
                sleep_for,
            )
            if rate_limit_attempt >= MAX_RATE_LIMIT_RETRIES:
                raise RuntimeError(
                    f"OpenAlex request failed with HTTP 429 after {MAX_RATE_LIMIT_RETRIES} retries"
                )
            time.sleep(sleep_for)
            continue

        if 500 <= response.status_code < 600:
            transient_attempt += 1
            if transient_attempt >= MAX_REQUEST_RETRIES:
                break
            sleep_for = min(MAX_BACKOFF_SECONDS, BASE_BACKOFF_SECONDS * (2 ** (transient_attempt - 1)))
            sleep_for += random.uniform(0, 1.0)
            logger.warning(
                "OpenAlex returned HTTP %s. Retry %s/%s in %.1fs",
                response.status_code,
                transient_attempt,
                MAX_REQUEST_RETRIES,
                sleep_for,
            )
            time.sleep(sleep_for)
            continue

        break

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"OpenAlex request failed with HTTP {response.status_code} after {transient_attempt} transient retries"
        ) from exc
    raise RuntimeError(f"OpenAlex request failed after {MAX_REQUEST_RETRIES} retries")

# Core fetch
def fetch_openalex_full(or_query: str, work_type: str, per_page: int = 200, max_results=None):
    base_url = "https://api.openalex.org/works"
    cursor = "*"
    total_count = None
    pulled = 0

    params_common = {
        "filter": f"title.search:{or_query},type:{work_type}",
        "per-page": per_page,
    }
    if MAILTO:
        params_common["mailto"] = MAILTO

    shard_paths = []
    shard_idx = 0
    remaining_to_skip = 0
    skipped_logged_milestone = 0
    if RESUME_IF_SHARDS_EXIST:
        existing_shards, next_shard_idx = _discover_existing_shards()
        if existing_shards:
            existing_count = _count_records_in_shards(existing_shards)
            shard_paths.extend(existing_shards)
            shard_idx = next_shard_idx
            remaining_to_skip = existing_count
            pulled = existing_count
            _get_logger().info(
                "Resume mode: found %s shard(s), already downloaded=%s, next shard index=%s",
                len(existing_shards),
                existing_count,
                shard_idx,
            )
            checkpoint = _load_checkpoint()
            if checkpoint:
                cp_count = checkpoint.get("saved_count")
                cp_cursor = checkpoint.get("next_cursor")
                cp_shard_idx = checkpoint.get("next_shard_idx")
                if cp_count == existing_count and cp_cursor and cp_shard_idx == shard_idx:
                    cursor = cp_cursor
                    remaining_to_skip = 0
                    _get_logger().info(
                        "Resume checkpoint loaded: starting from saved cursor with saved_count=%s",
                        cp_count,
                    )
                else:
                    _get_logger().info(
                        "Checkpoint ignored due to mismatch (checkpoint_count=%s shard_count=%s checkpoint_shard_idx=%s current_shard_idx=%s)",
                        cp_count,
                        existing_count,
                        cp_shard_idx,
                        shard_idx,
                    )

    buffer = []
    key_index = 0
    if OPENALEX_API_KEYS:
        _get_logger().info("Using %s OpenAlex API key(s) in rotation", len(OPENALEX_API_KEYS))
    else:
        _get_logger().info("No OpenAlex API key configured; running without api_key param")

    session = requests.Session()
    with tqdm(unit="works", desc="Fetching") as pbar:
        while True:
            if max_results is not None and pulled >= max_results:
                break

            params = dict(params_common)
            params["cursor"] = cursor

            r, key_index = _request_openalex_with_retries(
                session,
                base_url,
                params,
                api_keys=OPENALEX_API_KEYS,
                key_index=key_index,
            )
            data = r.json()

            if total_count is None:
                total_count = data.get("meta", {}).get("count", 0)
                pbar.total = total_count
                if pulled:
                    pbar.n = min(pulled, total_count)
                    pbar.refresh()
                _get_logger().info("OpenAlex meta.count=%s for query '%s'", total_count, or_query)

            works = data.get("results", [])
            if not works:
                break

            # Resume behavior: skip works that were already written in previous runs.
            if remaining_to_skip > 0:
                skip_now = min(remaining_to_skip, len(works))
                if skip_now:
                    works = works[skip_now:]
                    remaining_to_skip -= skip_now
                    skipped_so_far = pulled - remaining_to_skip
                    if skipped_so_far - skipped_logged_milestone >= 10000:
                        skipped_logged_milestone = skipped_so_far
                        _get_logger().info(
                            "Resume replay progress: skipped=%s/%s",
                            skipped_so_far,
                            pulled,
                        )

            # trim last page if testing
            if max_results is not None and pulled + len(works) > max_results:
                works = works[:max(0, max_results - pulled)]

            # add to buffer and flush on page boundaries for stable checkpointing.
            if works:
                buffer.extend(works)
                pulled += len(works)

            pbar.update(len(works))

            if max_results is not None and pulled >= max_results:
                break

            next_cursor = data.get("meta", {}).get("next_cursor")
            if len(buffer) >= SHARD_SIZE:
                shard_paths.append(_write_shard(shard_idx, buffer))
                shard_idx += 1
                buffer = []
                if next_cursor:
                    _save_checkpoint(next_cursor, pulled, shard_idx)

            cursor = next_cursor
            if not next_cursor:
                break

    # write final (possibly partial) shard
    if buffer:
        shard_paths.append(_write_shard(shard_idx, buffer))
        shard_idx += 1
    _clear_checkpoint()

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
    deduped, dedupe_stats = _deduplicate(merged)
    deduped_count = len(deduped)

    final_path = f"{OUTPUT_PREFIX}.json"
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(deduped, f, ensure_ascii=False, indent=2)

    logger.info("Found by OpenAlex (meta.count): %s", total_found)
    logger.info("Downloaded works (before dedupe): %s", pulled)
    logger.info(
        "Deduplication results: input=%s output=%s removed_total=%s",
        dedupe_stats["input_total"],
        dedupe_stats["output_total"],
        dedupe_stats["filtered_total"],
    )
    logger.info("Duplicates filtered by DOI: %s", dedupe_stats["filtered_doi"])
    logger.info("Duplicates filtered by OpenAlex ID: %s", dedupe_stats["filtered_id"])
    logger.info("Duplicates filtered by normalized title: %s", dedupe_stats["filtered_title"])
    logger.info("Saved merged JSON list to: %s", final_path)

    print(f"After dedupe: {deduped_count}")
    print(f"Saved merged JSON list to: {final_path}")
