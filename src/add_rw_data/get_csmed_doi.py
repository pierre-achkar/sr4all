import requests
import json


def resolve_cochrane_doi(cochrane_id, verify_latest=False):
    # Standard prefix for Cochrane reviews
    base_doi = f"10.1002/14651858.{cochrane_id}"

    if not verify_latest:
        return base_doi

    # Optional: Resolve via redirect to find the latest '.pubX' version
    url = f"https://doi.org/{base_doi}"
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        # Extract DOI from the final URL (e.g., .../CD010254.pub2/full)
        final_url = response.url
        if "10.1002" in final_url:
            return final_url.split("/doi/")[1].split("/full")[0]
        return base_doi
    except Exception:
        return base_doi


# Process
# read json file
with open("./data/rw_ds/csmed/pcs.json", "r") as f:
    dataset = json.load(f)


for entry in dataset["data"]:
    c_id = entry["cochrane_id"]
    # We use verify_latest=True to get the specific publication suffix
    entry["doi"] = resolve_cochrane_doi(c_id, verify_latest=True)

print(dataset["data"])

# save to new json file
with open("./data/rw_ds/csmed/pcs_with_dois.json", "w") as f:
    json.dump(dataset, f, indent=4)
