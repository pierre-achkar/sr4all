import json

PATH_1 = "./data/final_old/with_boolean/final/sr4all_full_normalized_boolean_mapping_merged_2_with_year_range.jsonl"
PATH_2 = "./data/final_old/with_boolean/final/sr4all_full_normalized_keywords_only_mapping_merged_2_with_year_range.jsonl"


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


filtered_list = [
    {"id": rec["id"], "boolean_queries": rec.get("boolean_queries")}
    for rec in combined_list
]

print(f"Filtered count (including null boolean_queries): {len(filtered_list)}")
# drop records where boolean_queries is null
filtered_list = [rec for rec in filtered_list if rec["boolean_queries"] is not None]
print(f"Filtered count (non-null boolean_queries): {len(filtered_list)}")

print(f"Combined unique count: {len(filtered_list)}")
# Save combined data
OUTPUT_PATH = "./data/final_old/with_boolean/final/sr4all_normalised_queries.jsonl"
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for rec in filtered_list:
        f.write(json.dumps(rec) + "\n")
