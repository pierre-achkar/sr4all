import json

PATH_1="./data/final_old/with_oax/oax_count_buckets/bucket_1_50k_with_ids.jsonl"
PATH_2="./data/final_old/with_oax/oax_count_buckets/bucket_50k_250k_with_ids.jsonl"

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

data_1 = read_jsonl(PATH_1)
data_2 = read_jsonl(PATH_2)

print(f"Data 1 count: {len(data_1)}")
print(f"Data 2 count: {len(data_2)}")

# Combine and deduplicate by "id"
combined_data = {rec["id"]: rec for rec in data_1 + data_2}
combined_list = list(combined_data.values())    

print(f"Combined unique count: {len(combined_list)}")
# Save combined data
OUTPUT_PATH = "./data/final_old/with_oax/oax_count_buckets/bucket_1_250k_with_ids.jsonl"
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for rec in combined_list:
        f.write(json.dumps(rec) + "\n")

