"""
Fetch OpenAlex query IDs asynchronously with rate limiting and retries.
"""

import asyncio
import json
import logging
import os
import time
import random
from typing import Dict, List, Optional, Tuple, Set
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
import importlib
import aiohttp
from tqdm import tqdm


def _load_dotenv():
    try:
        module = importlib.import_module("dotenv")
        module.load_dotenv()
    except Exception:
        return


_load_dotenv()

# ---------------- CONFIGURATION ----------------
CONFIG = {
    "INPUT_FILE": "./data/final_old/with_oax/oax_count_buckets/bucket_5k_50k.jsonl",
    "OUTPUT_FILE": "./data/final_old/with_oax/oax_count_buckets/bucket_5k_50k_with_ids.jsonl",
    "LOG_FILE": "./logs/query_id_fetch.log",
    "EMAIL": os.getenv("OPENALEX_EMAIL_2", ""),
    "API_KEY": os.getenv("OPENALEX_API_KEY_2"),
    "MAX_CONCURRENT_REQUESTS": 4,
    "REQUEST_TIMEOUT": aiohttp.ClientTimeout(total=120),
    "MAX_RETRIES": 5,
    "BACKOFF_BASE": 2.0,
    "MIN_REQUEST_INTERVAL_SECONDS": 0.5,
    "PER_PAGE": 200,
    "SELECT_FIELDS": "id",
    "SAVE_INTERVAL": 20,
    "VERBOSE_LOGGING": True,
}

# Setup logging
os.makedirs(os.path.dirname(CONFIG["LOG_FILE"]), exist_ok=True)
logging.basicConfig(
    filename=CONFIG["LOG_FILE"],
    filemode="w",
    level=logging.DEBUG if CONFIG.get("VERBOSE_LOGGING") else logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("oax_query_ids")


class RateLimiter:
    def __init__(self, min_interval_seconds: float):
        self._min_interval = min_interval_seconds
        self._lock = asyncio.Lock()
        self._last_request_ts = 0.0

    async def wait(self):
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request_ts
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)
            self._last_request_ts = time.monotonic()


def _prepare_oax_url(base_url: str, cursor: str) -> str:
    parsed = urlparse(base_url)
    query = parse_qs(parsed.query)
    query.update(
        {
            "per-page": [str(CONFIG["PER_PAGE"])],
            "select": [CONFIG["SELECT_FIELDS"]],
            "cursor": [cursor],
            "mailto": [CONFIG["EMAIL"]] if CONFIG["EMAIL"] else [],
            "api_key": [CONFIG["API_KEY"]] if CONFIG["API_KEY"] else [],
        }
    )
    return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))


async def fetch_with_retry(session, url, rate_limiter, semaphore, record_id):
    for attempt in range(CONFIG["MAX_RETRIES"]):
        async with semaphore:
            await rate_limiter.wait()
            try:
                async with session.get(url) as resp:
                    remaining = resp.headers.get("X-RateLimit-Remaining")
                    if resp.status == 429:
                        wait = int(resp.headers.get("Retry-After", 5))
                        actual_wait = wait + random.uniform(0.5, 2.0)
                        logger.warning(
                            f"ID: {record_id} | 429 Limit hit. Sleeping {actual_wait:.2f}s"
                        )
                        await asyncio.sleep(actual_wait)
                        continue

                    resp.raise_for_status()
                    return await resp.json()

            except Exception as e:
                wait_time = (CONFIG["BACKOFF_BASE"] ** attempt) + random.uniform(0, 1)
                logger.error(
                    f"ID: {record_id} | Attempt {attempt+1} failed: {e}. Retrying in {wait_time:.2f}s"
                )
                await asyncio.sleep(wait_time)
    return None


async def fetch_worker(
    queue,
    session,
    semaphore,
    results_map,
    rate_limiter,
    expected_counts,
    completed_queue,
    completed_set,
    completed_lock,
):
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break

        record_idx, url_idx, base_url, record_id = item
        all_ids, error, cursor = [], None, "*"

        logger.info(f"START: Fetching results for ID: {record_id} (Idx: {record_idx})")

        try:
            while True:
                target_url = _prepare_oax_url(base_url, cursor)
                data = await fetch_with_retry(
                    session, target_url, rate_limiter, semaphore, record_id
                )

                if data is None:
                    logger.error(
                        f"DISCARD: Failed all retries for ID: {record_id}. Clearing results."
                    )
                    error = "max_retries_exceeded"
                    all_ids = []
                    break

                results = data.get("results", [])
                all_ids.extend([r.get("id") for r in results if r.get("id")])

                next_cursor = data.get("meta", {}).get("next_cursor")
                if not next_cursor or not results:
                    break
                cursor = next_cursor

            if not error:
                logger.info(f"SUCCESS: Retrieved {len(all_ids)} IDs for {record_id}")

        except Exception as e:
            logger.error(f"CRITICAL: Unexpected error for {record_id}: {e}")
            error = f"unexpected_error: {str(e)}"
            all_ids = []

        results_map[record_idx][url_idx] = (all_ids, error)

        if len(results_map[record_idx]) == expected_counts.get(record_idx, 0):
            async with completed_lock:
                if record_idx not in completed_set:
                    completed_set.add(record_idx)
                    await completed_queue.put(record_idx)
        queue.task_done()


async def main():
    semaphore = asyncio.Semaphore(CONFIG["MAX_CONCURRENT_REQUESTS"])
    queue = asyncio.Queue()
    results_map = {}
    records_cache = {}
    rate_limiter = RateLimiter(CONFIG["MIN_REQUEST_INTERVAL_SECONDS"])
    expected_counts: Dict[int, int] = {}
    completed_queue: asyncio.Queue = asyncio.Queue()
    completed_set: Set[int] = set()
    completed_lock = asyncio.Lock()
    pending_records = {"count": 0}
    pending_lock = asyncio.Lock()
    done_event = asyncio.Event()
    input_done = {"value": False}

    if not os.path.exists(CONFIG["INPUT_FILE"]):
        print(f"Error: {CONFIG['INPUT_FILE']} not found.")
        return

    # Count total for TQDM
    with open(CONFIG["INPUT_FILE"], "r") as f:
        total_records = sum(1 for _ in f)

    processed_ids: Set[str] = set()
    if os.path.exists(CONFIG["OUTPUT_FILE"]):
        with open(CONFIG["OUTPUT_FILE"], "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec_id = rec.get("id")
                if rec_id:
                    processed_ids.add(rec_id)

    resume_lines = len(processed_ids)

    pbar = tqdm(
        total=total_records, initial=resume_lines, desc="Processing OpenAlex Records"
    )

    async def writer_task(f_out):
        while True:
            record_idx = await completed_queue.get()
            if record_idx is None:
                completed_queue.task_done()
                break

            rec = records_cache.pop(record_idx, None)
            res_bundle = results_map.pop(record_idx, None)
            if rec is None or res_bundle is None:
                completed_queue.task_done()
                continue

            ordered_res = [res_bundle[i] for i in range(len(res_bundle))]
            rec["oax_query_ids"] = [r[0] for r in ordered_res]
            rec["oax_query_id_errors"] = [r[1] for r in ordered_res]
            f_out.write(json.dumps(rec) + "\n")
            logger.info("SAVED: record_idx=%d id=%s", record_idx, rec.get("id"))

            if record_idx % CONFIG["SAVE_INTERVAL"] == 0:
                f_out.flush()

            pbar.update(1)

            async with pending_lock:
                pending_records["count"] -= 1
                if pending_records["count"] == 0 and input_done["value"]:
                    done_event.set()

            completed_queue.task_done()

    async with aiohttp.ClientSession(timeout=CONFIG["REQUEST_TIMEOUT"]) as session:
        workers = [
            asyncio.create_task(
                fetch_worker(
                    queue,
                    session,
                    semaphore,
                    results_map,
                    rate_limiter,
                    expected_counts,
                    completed_queue,
                    completed_set,
                    completed_lock,
                )
            )
            for _ in range(CONFIG["MAX_CONCURRENT_REQUESTS"])
        ]

        with open(CONFIG["INPUT_FILE"], "r") as f_in, open(
            CONFIG["OUTPUT_FILE"], "a"
        ) as f_out:
            writer = asyncio.create_task(writer_task(f_out))

            for record_idx, line in enumerate(f_in, start=resume_lines):
                last_record_idx = record_idx
                if not line.strip():
                    pbar.update(1)
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping invalid JSON at line %d", record_idx)
                    pbar.update(1)
                    continue
                record_id = record.get("id", "Unknown")
                if record_id in processed_ids:
                    pbar.update(1)
                    continue
                urls = record.get("oax_query", [])
                results_map[record_idx] = {}
                records_cache[record_idx] = record
                expected_counts[record_idx] = len(urls)

                async with pending_lock:
                    pending_records["count"] += 1

                if urls:
                    for url_idx, url in enumerate(urls):
                        await queue.put((record_idx, url_idx, url, record_id))
                else:
                    async with completed_lock:
                        if record_idx not in completed_set:
                            completed_set.add(record_idx)
                            await completed_queue.put(record_idx)

            await queue.join()
            input_done["value"] = True
            async with pending_lock:
                if pending_records["count"] == 0:
                    done_event.set()
            await done_event.wait()

            await completed_queue.put(None)
            await writer

        for _ in workers:
            await queue.put(None)
        await asyncio.gather(*workers)
        pbar.close()
        logger.info("Completed successfully")


if __name__ == "__main__":
    asyncio.run(main())
