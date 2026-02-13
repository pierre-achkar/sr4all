import re
import json

# Configuration / Dataset
# dataset = [
#     {
#         "id": 1,
#         "url": "http://onlinelibrary.wiley.com/doi/10.1002/14651858.CD011145.pub2/full"
#     },
#     {
#         "id": 2,
#         "url": "http://onlinelibrary.wiley.com/doi/10.1002/14651858.CD004944.pub3/full"
#     }
# ]

def extract_dois(data):
    # Regex breakdown:
    # (?<=/doi/) -> Lookbehind for '/doi/'
    # (.*?)      -> Non-greedy capture of the DOI string
    # (?=/full|/abstract|/pdf|$) -> Lookahead for common Wiley URL suffixes or end of string
    doi_pattern = re.compile(r'(?<=/doi/)(.*?)(?=/full|/abstract|/pdf|$)')
    
    results = []
    for item in data:
        match = doi_pattern.search(item['url'])
        doi = match.group(0) if match else None
        
        results.append({
            "id": item['id'],
            "doi": doi
        })
    
    return results

# Execution
# read jsonl file
with open('/home/fhg/pie65738/pie65738/projects/oax_count/data/seed_sr/sr.json', 'r') as f:
    dataset = json.load(f)

processed_data = extract_dois(dataset)
# Save the results to a new JSON file
with open('/home/fhg/pie65738/pie65738/projects/oax_count/data/seed_sr/sr_with_dois.json', 'w') as f:
    json.dump(processed_data, f, indent=4)