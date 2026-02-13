import json

PATH_1="/home/fhg/pie65738/projects/sr4all/data/final/sr4all_merged.jsonl"
PATH_2="/home/fhg/pie65738/projects/sr4all/data/final/benchmark_data.jsonl"

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
OUTPUT_PATH = "/home/fhg/pie65738/projects/sr4all/data/final/sr4all_full.jsonl"
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for rec in combined_list:
        f.write(json.dumps(rec) + "\n")

