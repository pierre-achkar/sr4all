import asyncio
import aiohttp
from urllib.parse import quote

# ---------------- CONFIGURATION ----------------
EMAIL = "pieer.achkar@imw.fraunhofer.de"  # REQUIRED for polite pool (higher limits)
MAX_CONCURRENT_REQUESTS = 10      # 10 is safe for polite pool; can push to 20ish
BASE_URL = "https://api.openalex.org/works"

# Your list of queries (search terms, filters, or concept IDs)
QUERIES = [
    "search=natural language processing",
    "filter=institutions.id:I136199984",
    "search=deep learning&filter=publication_year:2024"
]

async def fetch_count_for_query(session, query_str, semaphore):
    """
    Fetches only the total result count for a single query string.
    """
    async with semaphore:
        encoded_query = query_str  # Assume user strings are mostly safe
        
        # per-page=1 minimizes payload; count is in meta.count
        url = f"{BASE_URL}?{encoded_query}&per-page=1"
        
        try:
            async with session.get(url) as response:
                if response.status == 429:
                    print(f"Rate limit hit for {query_str}, sleeping...")
                    await asyncio.sleep(2)
                    return await fetch_count_for_query(session, query_str, semaphore)
                
                response.raise_for_status()
                data = await response.json()
                count = data.get("meta", {}).get("count", 0)
                return query_str, count
                
        except Exception as e:
            print(f"Error processing '{query_str}': {e}")
            return query_str, 0

async def main():
    headers = {
        "User-Agent": f"mailto:{EMAIL}"
    }
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = []
        for q in QUERIES:
            task = fetch_count_for_query(session, q, semaphore)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        for query, count in results:
            print(f"Query: {query} -> Expected count: {count}")

if __name__ == "__main__":
    asyncio.run(main())