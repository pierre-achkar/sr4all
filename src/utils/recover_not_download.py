import json

download_manifest_path = "/home/fhg/pie65738/projects/sr4all/data/rw_ds/filtered/pdf_download_manifest.jsonl"
full_data = "/home/fhg/pie65738/projects/sr4all/data/raw/oax_sr_full.json"

# read download manifest
with open(download_manifest_path, "r") as f:
    download_manifest = [json.loads(line) for line in f]

# read full data (JSON, not JSONL)
with open(full_data, "r") as f:
    full_data = json.load(f)

# get set of rows from manifest where "status" is not "downloaded"
not_downloaded_ids = set()
for item in download_manifest:
    if item["status"] != "downloaded":
        not_downloaded_ids.add(item["id"])

print(f"Number of items in download manifest: {len(download_manifest)}")
print(f"Number of items not downloaded: {len(not_downloaded_ids)}")

# filter full data to get items that were not downloaded
not_downloaded_data = [item for item in full_data if item["id"] in not_downloaded_ids]

print(f"Number of items in full data: {len(full_data)}")
print(f"Number of matching items in full data that were not downloaded: {len(not_downloaded_data)}")

# save not downloaded data
output_path = "/home/fhg/pie65738/projects/sr4all/data/filtered/no_ft_subset/not_downloaded_data.jsonl"
with open(output_path, "w") as f:
    for item in not_downloaded_data:
        f.write(json.dumps(item) + "\n")

print(f"Saved not downloaded data to {output_path}")
print("Done!")
