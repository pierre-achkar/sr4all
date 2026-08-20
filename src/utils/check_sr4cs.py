import csv
import json

# Load sr4cs sr_dois (bare DOIs, lowercased)
with open('/home/fhg/pie65738/projects/sr4all/data_old/rw_ds/sr4cs.json') as f:
    sr4cs = json.load(f)

sr_dois = {r['sr_doi'].lower().strip() for r in sr4cs if r.get('sr_doi')}
print(f'Total sr4cs records: {len(sr4cs)}')
print(f'Unique sr_dois: {len(sr_dois)}')

# Load sr4all_full dois (strip prefix, lowercase)
prefix = 'https://doi.org/'
full_dois = set()
with open('/home/fhg/pie65738/projects/sr4all/data_old/release/sr4all_full.jsonl') as f:
    for line in f:
        rec = json.loads(line)
        doi = rec.get('doi')
        if doi:
            bare = doi.lower().strip()
            if bare.startswith(prefix):
                bare = bare[len(prefix):]
            full_dois.add(bare)

print(f'Total sr4all_full records: 301871')
print(f'Unique dois in sr4all_full: {len(full_dois)}')

matched = sr_dois & full_dois
not_found = sorted(sr_dois - full_dois)
print(f'sr_dois found in sr4all_full: {len(matched)} / {len(sr_dois)}')
print(f'sr_dois NOT found: {len(not_found)}')

out_path = '/home/fhg/pie65738/projects/sr4all/data_old/rw_ds/sr4cs_not_in_sr4all.csv'
with open(out_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['sr_doi'])
    writer.writerows([[doi] for doi in not_found])
print(f'Written not-found DOIs to {out_path}')