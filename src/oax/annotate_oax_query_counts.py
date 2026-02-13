"""
Annotate OpenAlex query URLs with result counts asynchronously.
"""

import asyncio
import aiohttp
import json
import logging
import time
import importlib
import os
from typing import Dict, List, Tuple, Optional
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse


def _get_tqdm():
    try:
        module = importlib.import_module("tqdm")
        return module.tqdm
    except Exception:  # pragma: no cover

        def _noop(iterable, **kwargs):
            return iterable

        return _noop


# ---------------- CONFIGURATION ----------------
def _load_dotenv():
    try:
        module = importlib.import_module("dotenv")
        module.load_dotenv()
    except Exception:
        return


_load_dotenv()
INPUT_FILE = (
    "./data/final_old/with_oax/sr4all_full_normalized_boolean_with_year_range_oax.jsonl"
)
OUTPUT_FILE = "./data/final_old/with_oax/sr4all_full_normalized_boolean_with_year_range_oax_with_counts.jsonl"
LOG_FILE = "./logs/oax/query_count_annotate.log"

EMAIL = "ex@example.com"
API_KEY = os.getenv("OPENALEX_API_KEY")
MAX_CONCURRENT_REQUESTS = 10
REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=60)
MAX_RETRIES = 4
BACKOFF_BASE_SECONDS = 1.5
MIN_REQUEST_INTERVAL_SECONDS = 0.35
SAVE_INTERVAL = 50

# -------------------------------------------------
logging.basicConfig(
    filename=LOG_FILE,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("oax_query_counts")


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


def _prepare_oax_url(url: str) -> str:
    """
    Ensures mailto and per-page=1 are set. Preserves existing params.
    """
    if not isinstance(url, str) or not url.strip():
        return ""

    parsed = urlparse(url)
    query = parse_qs(parsed.query)

    # Overwrite per-page to minimize payload
    query["per-page"] = ["1"]

    # Add mailto (as per OpenAlex requirements)
    if EMAIL:
        query["mailto"] = [EMAIL]

    # Add API key if provided
    if API_KEY:
        query["api_key"] = [API_KEY]

    new_query = urlencode(query, doseq=True)
    return urlunparse(parsed._replace(query=new_query))


async def _fetch_count(
    session: aiohttp.ClientSession,
    url: str,
    semaphore: asyncio.Semaphore,
    rate_limiter: RateLimiter,
) -> Tuple[int, Optional[str]]:
    """
    Fetches meta.count from OpenAlex for a single prepared URL.
    """
    if not url:
        return 0, None

    for attempt in range(MAX_RETRIES):
        async with semaphore:
            try:
                await rate_limiter.wait()
                async with session.get(url) as response:
                    if response.status == 429:
                        retry_after = response.headers.get("Retry-After")
                        if retry_after and retry_after.isdigit():
                            sleep_s = int(retry_after)
                        else:
                            sleep_s = BACKOFF_BASE_SECONDS * (attempt + 1)
                        logger.warning(
                            "Rate limit hit. Sleeping %.1fs: %s", sleep_s, url
                        )
                        await asyncio.sleep(sleep_s)
                        continue

                    response.raise_for_status()
                    data = await response.json()
                    return int(data.get("meta", {}).get("count", 0)), None
            except Exception as exc:
                sleep_s = BACKOFF_BASE_SECONDS * (attempt + 1)
                logger.warning(
                    "Error fetching %s (attempt %d/%d): %s",
                    url,
                    attempt + 1,
                    MAX_RETRIES,
                    exc,
                )
                await asyncio.sleep(sleep_s)

    logger.error("Failed after retries: %s", url)
    return 0, "fetch_failed"


async def _count_queries_for_record(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    rate_limiter: RateLimiter,
    urls: List[str],
    cache: Dict[str, int],
) -> Tuple[List[int], List[Optional[str]]]:
    """
    Returns counts aligned with the input URL list. Uses cache when possible.
    """
    prepared_urls = [_prepare_oax_url(u) for u in urls]
    counts: List[int] = [0] * len(prepared_urls)
    errors: List[Optional[str]] = [None] * len(prepared_urls)

    tasks: List[Tuple[int, asyncio.Task]] = []
    for idx, url in enumerate(prepared_urls):
        if not url:
            counts[idx] = 0
            errors[idx] = None
            continue
        if url in cache:
            counts[idx] = cache[url]
            errors[idx] = None
            continue
        tasks.append(
            (
                idx,
                asyncio.create_task(
                    _fetch_count(session, url, semaphore, rate_limiter)
                ),
            )
        )

    if tasks:
        results = await asyncio.gather(*[t for _, t in tasks])
        for (idx, _), (count, err) in zip(tasks, results):
            counts[idx] = count
            errors[idx] = err
            cache[prepared_urls[idx]] = count

    return counts, errors


async def main():
    logger.info("Starting OpenAlex query count annotation.")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    rate_limiter = RateLimiter(MIN_REQUEST_INTERVAL_SECONDS)
    cache: Dict[str, int] = {}

    headers = {
        "User-Agent": f"mailto:{EMAIL}",
    }

    processed = 0

    def _count_lines(path: str) -> int:
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)

    total_lines = _count_lines(INPUT_FILE)

    resume_lines = 0
    if os.path.exists(OUTPUT_FILE):
        resume_lines = _count_lines(OUTPUT_FILE)
        if resume_lines > 0:
            logger.info("Resuming from %d already written records.", resume_lines)

    tqdm = _get_tqdm()

    async with aiohttp.ClientSession(
        headers=headers, timeout=REQUEST_TIMEOUT
    ) as session:
        with open(INPUT_FILE, "r", encoding="utf-8") as f_in, open(
            OUTPUT_FILE, "a", encoding="utf-8"
        ) as f_out:
            if resume_lines:
                for _ in range(resume_lines):
                    next(f_in, None)

            for line in tqdm(
                f_in,
                total=total_lines,
                initial=resume_lines,
                desc="Annotating records",
                unit="rec",
            ):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping invalid JSON line.")
                    continue

                urls = record.get("oax_query")
                if not isinstance(urls, list):
                    urls = []

                counts, errors = await _count_queries_for_record(
                    session, semaphore, rate_limiter, urls, cache
                )
                record["oax_query_counts"] = counts
                record["oax_query_errors"] = errors

                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                processed += 1

                if processed % SAVE_INTERVAL == 0:
                    f_out.flush()
                    os.fsync(f_out.fileno())

                if processed % 500 == 0:
                    logger.info("Processed %d records", processed)

    logger.info("Completed. Total records processed: %d", processed)


if __name__ == "__main__":
    asyncio.run(main())
