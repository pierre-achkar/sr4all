"""
Checks Abstract Coverage in OpenAlex Records
- Reads slimmed OpenAlex records
- Checks if each record has an abstract using OpenAlex API
- Caches results in SQLite to avoid redundant API calls
- Saves results with abstract coverage info to JSONL
- Logs progress and any issues encountered during checking
"""
import json
import logging
import requests
import sqlite3
import time
import os
import importlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from threading import Lock
from dataclasses import dataclass
from typing import List, Dict, Set, Any
from pathlib import Path
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

_DOTENV_LOADED: Path | None = None
_THROTTLE_LOCK = Lock()
_THROTTLE_UNTIL = 0.0


def _load_dotenv() -> Path | None:
    try:
        module = importlib.import_module("dotenv")
        env_path = Path(__file__).resolve().parents[1] / ".env"
        if env_path.exists():
            module.load_dotenv(dotenv_path=env_path)
            return env_path
    except Exception:
        return None
    return None


def _get_rate_headers(resp: requests.Response) -> dict:
    return {
        "Retry-After": resp.headers.get("Retry-After"),
        "X-RateLimit-Limit": resp.headers.get("X-RateLimit-Limit"),
        "X-RateLimit-Remaining": resp.headers.get("X-RateLimit-Remaining"),
        "X-RateLimit-Reset": resp.headers.get("X-RateLimit-Reset"),
    }


def _apply_global_throttle(seconds: float) -> None:
    global _THROTTLE_UNTIL
    if seconds <= 0:
        return
    now = time.time()
    with _THROTTLE_LOCK:
        _THROTTLE_UNTIL = max(_THROTTLE_UNTIL, now + seconds)


def _sleep_if_throttled() -> None:
    with _THROTTLE_LOCK:
        until = _THROTTLE_UNTIL
    now = time.time()
    if until > now:
        time.sleep(until - now)


def _parse_retry_after(resp: requests.Response) -> float | None:
    retry_after = resp.headers.get("Retry-After")
    if not retry_after:
        return None
    try:
        return float(retry_after)
    except ValueError:
        return None


# ==========================================
# 1. Configuration
# ==========================================
_DOTENV_LOADED = _load_dotenv()


@dataclass
class Config:
    # PATHS
    input_file: str = "./data/filtered/no_ft_subset/oax_sr_slim_no_ft.jsonl"
    output_file: str = "./data/filtered/no_ft_subset/oax_sr_slim_no_ft_abstract_coverage.jsonl"
    cache_db: str = "./data/filtered/no_ft_subset/cache_refs.db" 
    log_file: str = "./logs/retrieval/abstract_coverage_no_ft.log"

    # API
    email: str = os.getenv("OPENALEX_EMAIL_4", "pieer.achkar@imw.fraunhofer.de")
    api_key: str = os.getenv("OPENALEX_API_KEY_4", "")
    base_url: str = "https://api.openalex.org/works"
    batch_size: int = 50
    max_workers: int = 4
    max_in_flight: int = 8

# ==========================================
# 2. Database (Cache) Manager
# ==========================================
class RefCache:
    """Simple wrapper around SQLite to persist availability checks."""
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._setup()
    
    def _setup(self):
        # Table: id (W123) -> has_abstract (1 or 0)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS refs (
                id TEXT PRIMARY KEY,
                has_abstract INTEGER
            )
        """)
        self.conn.commit()

    def get_existing_ids(self) -> Set[str]:
        """Returns a set of all IDs currently in the cache."""
        self.cursor.execute("SELECT id FROM refs")
        return {row[0] for row in self.cursor.fetchall()}

    def get_map(self, id_list: List[str]) -> Dict[str, bool]:
        """Returns {id: bool} for the requested list from DB."""
        if not id_list: return {}
        placeholders = ','.join('?' * len(id_list))
        # Note: SQLite limits distinct variables, but for lookup we iterating logic later
        # Optimization: We usually just load the whole cache into memory for the final calculation 
        # if it fits (1M ints is small), or query in chunks.
        # Here we will load all into a Dict for the final pass since 10M items is < 1GB RAM.
        pass 

    def load_all_into_memory(self) -> Dict[str, bool]:
        """Loads entire cache to Dict for fast final processing."""
        self.cursor.execute("SELECT id, has_abstract FROM refs")
        return {row[0]: bool(row[1]) for row in self.cursor.fetchall()}

    def save_batch(self, results: Dict[str, bool]):
        """Writes a batch of results to disk."""
        data = [(k, 1 if v else 0) for k, v in results.items()]
        self.cursor.executemany(
            "INSERT OR IGNORE INTO refs (id, has_abstract) VALUES (?, ?)", 
            data
        )
        self.conn.commit()

    def close(self):
        self.conn.close()

# ==========================================
# 3. Network Logic
# ==========================================
def clean_id(url_or_id: str) -> str:
    return url_or_id.replace("https://openalex.org/", "")

def is_retryable(ex):
    return isinstance(ex, requests.HTTPError) and ex.response.status_code in [429, 500, 502, 503, 504]

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception(is_retryable))
def fetch_batch(ids: List[str], config: Config) -> Dict[str, bool]:
    _sleep_if_throttled()
    clean_ids = [clean_id(i) for i in ids]
    id_str = "|".join(clean_ids)
    
    params = {
        "filter": f"openalex_id:{id_str}",
        "select": "id,abstract_inverted_index",
        "per_page": config.batch_size
    }
    if config.api_key:
        params["api_key"] = config.api_key
    
    resp = requests.get(config.base_url, params=params, headers={"User-Agent": f"mailto:{config.email}"})
    if resp.status_code != 200:
        headers = _get_rate_headers(resp)
        logging.warning(
            "OpenAlex status=%s batch=%d headers=%s",
            resp.status_code,
            len(ids),
            headers,
        )
        if resp.status_code == 429:
            wait_s = _parse_retry_after(resp)
            if wait_s is None:
                wait_s = 10.0
            _apply_global_throttle(wait_s)
            time.sleep(wait_s)
        resp.raise_for_status()
    
    results = resp.json().get('results', [])
    
    # Map results
    batch_map = {}
    found_ids = set()
    
    for item in results:
        sid = clean_id(item['id'])
        found_ids.add(sid)
        # Check if abstract exists
        batch_map[sid] = bool(item.get('abstract_inverted_index'))
    
    # Handle deleted/missing works (OpenAlex didn't return them)
    for requested in clean_ids:
        if requested not in found_ids:
            batch_map[requested] = False
            
    return batch_map

# ==========================================
# 4. Main Pipeline
# ==========================================
def main():
    conf = Config()
    
    # Setup Logging
    Path(conf.log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=conf.log_file, level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    
    print(f"--- Starting Coverage Audit ---")
    print(f"Cache DB: {conf.cache_db}")
    if _DOTENV_LOADED:
        logger.info("Loaded .env from %s", _DOTENV_LOADED)
    else:
        logger.info("No .env loaded (expected at src/.env)")
    logger.info("OpenAlex API key present: %s", bool(conf.api_key))
    
    # 1. LOAD DATA
    print("Loading input data...")
    input_path = Path(conf.input_file)
    with open(input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    # 2. IDENTIFY UNIQUE REFS
    all_refs = set()
    for entry in data:
        all_refs.update([clean_id(r) for r in entry.get('referenced_works', [])])
    
    print(f"Total Unique References needed: {len(all_refs)}")
    logger.info(f"Total Unique References: {len(all_refs)}")

    # 3. CHECK CACHE
    db = RefCache(conf.cache_db)
    cached_ids = db.get_existing_ids()
    to_fetch = list(all_refs - cached_ids)
    
    print(f"Already in cache: {len(cached_ids)}")
    print(f"Need to fetch: {len(to_fetch)}")
    
    # 4. FETCH MISSING (If any)
    if to_fetch:
        chunks = [to_fetch[i:i + conf.batch_size] for i in range(0, len(to_fetch), conf.batch_size)]
        pending = deque(chunks)
        in_flight: dict = {}

        pbar = tqdm(total=len(chunks), desc="Fetching from OpenAlex", unit="batch")
        with ThreadPoolExecutor(max_workers=conf.max_workers) as ex:
            while pending or in_flight:
                while pending and len(in_flight) < conf.max_in_flight:
                    batch = pending.popleft()
                    fut = ex.submit(fetch_batch, batch, conf)
                    in_flight[fut] = batch

                done, _ = wait(in_flight.keys(), timeout=0.1, return_when=FIRST_COMPLETED)
                for fut in done:
                    batch = in_flight.pop(fut)
                    try:
                        results = fut.result()
                        db.save_batch(results)
                    except Exception as e:
                        logger.error(f"Batch failed: {e}")
                        fallback = {uid: False for uid in batch}
                        db.save_batch(fallback)
                    pbar.update(1)
        pbar.close()

    # 5. CALCULATE SCORES
    print("Loading cache to memory for scoring...")
    availability_map = db.load_all_into_memory()
    db.close()
    
    stats_buckets = {
        "90+": 0,
        "80-89": 0,
        "70-79": 0,
        "60-69": 0,
        "<60": 0
    }
    
    print("Enriching surveys...")
    with open(conf.output_file, 'w', encoding='utf-8') as fout:
        for entry in tqdm(data, desc="Writing Output"):
            refs = [clean_id(r) for r in entry.get('referenced_works', [])]
            total = len(refs)
            
            if total == 0:
                coverage = 0.0
                valid = 0
            else:
                valid = sum(1 for r in refs if availability_map.get(r, False))
                coverage = valid / total
            
            # Bucket Stats
            if coverage >= 0.90: stats_buckets["90+"] += 1
            elif coverage >= 0.80: stats_buckets["80-89"] += 1
            elif coverage >= 0.70: stats_buckets["70-79"] += 1
            elif coverage >= 0.60: stats_buckets["60-69"] += 1
            else: stats_buckets["<60"] += 1

            # Update Object
            entry['references_abstract_coverage'] = {
                "ratio": round(coverage, 4),
                "valid_refs": valid,
                "total_refs": total
            }
            
            fout.write(json.dumps(entry) + "\n")

    # 6. REPORT
    print("\n" + "="*30)
    print("FINAL STATISTICS")
    print("="*30)
    logger.info("FINAL STATISTICS")
    
    total_docs = len(data)
    for k, v in stats_buckets.items():
        pct = (v / total_docs) * 100 if total_docs > 0 else 0
        msg = f"Coverage {k}%: {v} docs ({pct:.1f}%)"
        print(msg)
        logger.info(msg)

if __name__ == "__main__":
    main()
