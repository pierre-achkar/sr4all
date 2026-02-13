import json
import re
import requests
import time

# Configuration
INPUT_FILE = "./data/rw_ds/seed_collection/overall.jsonl"
OUTPUT_FILE = "./data/rw_ds/seed_collection/overall_doi.jsonl"


def get_doi_from_url(url):
    """Attempt to find a DOI pattern directly in the URL string."""
    doi_pattern = re.compile(r"10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+")
    match = doi_pattern.search(url)
    return match.group(0).rstrip("/") if match else None


def get_doi_from_pubmed(url):
    """Extract PMID and fetch DOI from NCBI API."""
    pmid_match = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", url)
    if not pmid_match:
        return None

    pmid = pmid_match.group(1)
    api_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={pmid}&retmode=json"
    try:
        response = requests.get(api_url, timeout=10)
        data = response.json()
        ids = data["result"][pmid]["articleids"]
        return next(i["value"] for i in ids if i["idtype"] == "doi")
    except Exception:
        return None


def get_doi_from_title(title):
    """Fallback: Search Crossref API by title (useful for ResearchGate)."""
    if not title:
        return None
    api_url = "https://api.crossref.org/works"
    params = {"query.title": title, "rows": 1}
    try:
        response = requests.get(api_url, params=params, timeout=10)
        items = response.json().get("message", {}).get("items", [])
        return items[0].get("DOI") if items else None
    except Exception:
        return None


def process_jsonl():
    with open(INPUT_FILE, "r") as f_in, open(OUTPUT_FILE, "w") as f_out:
        for line in f_in:
            entry = json.loads(line)
            url = entry.get("link_to_review", "")
            title = entry.get("title", "")
            doi = None

            # Step 1: Check if DOI is in URL (Jammi, BMC, etc.)
            doi = get_doi_from_url(url)

            # Step 2: If PubMed, use PMID lookup
            if not doi and "pubmed.ncbi" in url:
                doi = get_doi_from_pubmed(url)
                time.sleep(0.4)  # Rate limit for NCBI

            # Step 3: If still no DOI (ResearchGate or missing link), use Title
            if not doi:
                doi = get_doi_from_title(title)
                time.sleep(0.2)  # Rate limit for Crossref

            # Save minimal output
            output = {"id": entry.get("id"), "doi": doi}
            f_out.write(json.dumps(output) + "\n")
            print(f"Processed ID {entry.get('id')}: {doi}")


if __name__ == "__main__":
    process_jsonl()
