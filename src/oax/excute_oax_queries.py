import asyncio
import aiohttp
import hashlib
import os
import re
import time
from pathlib import Path

# CONFIGURATION
EMAIL = "pieer.achkar@imw.fraunhofer.de"  # REQUIRED for polite pool
API_KEY = os.getenv("OPENALEX_API_KEY", "")
MAX_CONCURRENT_REQUESTS = 15  # Limit active HTTP connections (not just queries)
PER_PAGE = 200
OUTPUT_DIR = Path("../../data/oax_ids")
REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=120, sock_connect=30, sock_read=60)
MAX_RETRIES = 5
RETRY_BACKOFF_SECONDS = 2.0
SHARDING_THRESHOLD = 2000  # If results < 2000, don't bother splitting by year

def _query_to_filename(query: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", query.strip())
    safe = safe[:60].strip("_") or "query"
    qhash = hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]
    return f"ids_{safe}_{qhash}.txt"

async def fetch_json(session, url, params):
    """Robust fetcher with retries."""
    retries = 0
    while True:
        try:
            async with session.get(url, params=params, timeout=REQUEST_TIMEOUT) as response:
                if response.status in {429, 500, 502, 503, 504}:
                    retries += 1
                    if retries > MAX_RETRIES:
                        print(f"!! Retry limit reached. Status {response.status} | Params: {params}")
                        return None
                    await asyncio.sleep(RETRY_BACKOFF_SECONDS * retries)
                    continue
                
                if response.status != 200:
                    print(f"!! Error {response.status} for {url}")
                    return None
                
                return await response.json()
        except Exception as e:
            retries += 1
            if retries > MAX_RETRIES:
                print(f"!! Exception: {e}")
                return None
            await asyncio.sleep(RETRY_BACKOFF_SECONDS * retries)

async def get_year_shards(session, query):
    """
    Scouts the API to determine the distribution of works by year.
    Returns a list of filter strings (e.g., ['publication_year:2023', ...]) 
    or [None] if the query is small enough to run sequentially.
    """
    base_url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "group_by": "publication_year",
        "mailto": EMAIL
    }
    if API_KEY:
        params["api_key"] = API_KEY
    
    data = await fetch_json(session, base_url, params)
    if not data:
        # If the group_by query fails, fallback to standard sequential fetch
        return [None]

    groups = data.get("group_by", [])
    total_count = data.get("meta", {}).get("count", 0)
    
    # Optimization: Don't shard small queries. The overhead isn't worth it.
    if total_count < SHARDING_THRESHOLD:
        return [None]

    shards = []
    # OpenAlex returns top 200 groups. For years, this covers basically everything relevant.
    for group in groups:
        year = group.get("key")
        # 'unknown' years are returned as strings, valid years as ints usually
        if year and str(year).lower() != "unknown":
            shards.append(f"publication_year:{year}")
            
    print(f"Query: '{query}' | Total: {total_count} | Sharding into {len(shards)} concurrent year-tasks.")
    return shards

async def fetch_shard_worker(session, query, filter_param, queue, semaphore):
    """
    Worker: Fetches IDs for a specific shard (query + year filter) 
    and pushes results into the shared queue.
    """
    async with semaphore:
        base_url = "https://api.openalex.org/works"
        cursor = "*"
        
        params = {
            "search": query,
            "select": "id",
            "per-page": PER_PAGE,
            "cursor": cursor,
        }
        if filter_param:
            # If we have a year shard, add it to the filter
            params["filter"] = filter_param
        
        if EMAIL: params["mailto"] = EMAIL
        if API_KEY: params["api_key"] = API_KEY

        while cursor:
            params["cursor"] = cursor
            data = await fetch_json(session, base_url, params)
            
            if not data:
                break
            
            results = data.get("results", [])
            if not results:
                break
                
            # Efficiently push to queue
            for item in results:
                if item.get("id"):
                    queue.put_nowait(item["id"])
            
            cursor = data.get("meta", {}).get("next_cursor")

async def file_writer(output_path, queue):
    """
    Consumer: Reads from the queue and writes to disk safely.
    Stops when it receives None.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    
    with open(output_path, "w", encoding="utf-8") as f:
        while True:
            work_id = await queue.get()
            if work_id is None:
                queue.task_done()
                break
            
            f.write(work_id + "\n")
            count += 1
            queue.task_done()
            
    return count

async def process_query(session, query, global_semaphore):
    """
    Orchestrator for a single query.
    1. Scouts shards.
    2. Spawns workers.
    3. Spawns writer.
    4. Joins them.
    """
    start_time = time.perf_counter()
    output_path = OUTPUT_DIR / _query_to_filename(query)
    
    # 1. Scout
    shards = await get_year_shards(session, query)
    
    # 2. Setup Pipeline
    # Maxsize prevents memory explosion if network is much faster than disk (unlikely here)
    queue = asyncio.Queue(maxsize=10000) 
    writer_task = asyncio.create_task(file_writer(output_path, queue))
    
    # 3. Launch Workers
    # We pass the GLOBAL semaphore to ensure we don't exceed rate limits 
    # even if we have 50 year-shards active.
    worker_tasks = []
    for shard in shards:
        task = asyncio.create_task(fetch_shard_worker(session, query, shard, queue, global_semaphore))
        worker_tasks.append(task)
        
    # 4. Wait for completion
    await asyncio.gather(*worker_tasks) # Wait for all network IO to finish
    
    # 5. Shutdown Writer
    await queue.put(None) # Sentinel signal
    total_ids = await writer_task
    
    duration_s = time.perf_counter() - start_time
    print(f"DONE: {query} | IDs: {total_ids} | Time: {duration_s:.2f}s | Saved: {output_path.name}")

async def main(queries):
    # This semaphore controls TOTAL concurrent HTTP requests across ALL queries and shards
    global_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    # TCPConnector limit=0 means "unlimited connections" (governed by semaphore instead)
    connector = aiohttp.TCPConnector(limit=0) 
    
    async with aiohttp.ClientSession(connector=connector) as session:
        # We launch all queries immediately. 
        # The semaphore inside the workers will prevent the API from exploding.
        tasks = [process_query(session, q, global_semaphore) for q in queries]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    example_queries = [
        "machine learning",
        "natural language processing",
        "computer vision",
    ]
    asyncio.run(main(example_queries))