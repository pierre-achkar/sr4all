import json
import logging
import requests
import sqlite3
import time
from dataclasses import dataclass
from typing import List, Dict, Set, Any
from pathlib import Path
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

# ==========================================
# 1. Configuration
# ==========================================
@dataclass
class Config:
    # PATHS
    input_file: str = "/home/fhg/pie65738/projects/sr4all/data/filtered/oax_sr_slim.json"
    output_file: str = "/home/fhg/pie65738/projects/sr4all/data/filtered/oax_sr_slim_abstract_coverage.jsonl"
    cache_db: str = "/home/fhg/pie65738/projects/sr4all/data/filtered/cache_refs.db" 
    log_file: str = "logs/retreival/abstract_coverage.log"

    # API
    email: str = "pierre.achkar@imw.fraunhofer.de"
    base_url: str = "https://api.openalex.org/works"
    batch_size: int = 50

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
    clean_ids = [clean_id(i) for i in ids]
    id_str = "|".join(clean_ids)
    
    params = {
        "filter": f"openalex_id:{id_str}",
        "select": "id,abstract_inverted_index",
        "per_page": config.batch_size
    }
    
    resp = requests.get(config.base_url, params=params, headers={"User-Agent": f"mailto:{config.email}"})
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
    
    # 1. LOAD DATA
    print("Loading input data...")
    input_path = Path(conf.input_file)
    with open(input_path, 'r', encoding='utf-8') as f:
        # Detect if list or jsonl
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            data = json.load(f)
        else:
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
        
        for batch in tqdm(chunks, desc="Fetching from OpenAlex"):
            try:
                results = fetch_batch(batch, conf)
                db.save_batch(results)
            except Exception as e:
                logger.error(f"Batch failed: {e}")
                # Save as False to prevent infinite retry loops on broken batches
                # Robustness choice: treat network errors here as "no abstract" for now
                fallback = {uid: False for uid in batch}
                db.save_batch(fallback)

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