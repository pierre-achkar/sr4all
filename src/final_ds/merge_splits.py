import json

ft_split_path = "./data/final/sr4all_ft.jsonl"
no_ft_split_path = "./data/final/sr4all_no_ft.jsonl"
output_path = "./data/final/sr4all_merged.jsonl"

# load ft split
with open(ft_split_path, "r") as f:
    ft_data = [json.loads(line) for line in f]

# load no ft split
with open(no_ft_split_path, "r") as f:
    no_ft_data = [json.loads(line) for line in f]

print(f"Number of items in ft split: {len(ft_data)}")
print(f"Number of items in no ft split: {len(no_ft_data)}")

# merge splits
merged_data = ft_data + no_ft_data

print(f"Number of items in merged data (before deduplication): {len(merged_data)}")

# deduplicate merged data based on "id" field
seen_ids = set()
deduplicated_data = []
for item in merged_data:
    if item["id"] not in seen_ids:
        deduplicated_data.append(item)
        seen_ids.add(item["id"])

# save merged data
with open(output_path, "w") as f:
    for item in deduplicated_data:
        f.write(json.dumps(item) + "\n")

print(f"Saved merged data to {output_path}")
print(f"Number of items in merged data (after deduplication): {len(deduplicated_data)}")
print("Done!")
