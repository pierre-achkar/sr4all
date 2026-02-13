import json

file_path = "/home/fhg/pie65738/projects/sr4all/data/release/sr4all_full_split_search_keywords_oax.jsonl"
output_path = "/home/fhg/pie65738/projects/sr4all/data/release/sr4all_full_split_search_keywords_oax_deduped.jsonl"

ids = set()
duplicates = 0

with open(file_path, "r", encoding="utf-8") as fin:
    lines = fin.readlines()
print(f"Total lines before dedupe: {len(lines)}")

with open(output_path, "w", encoding="utf-8") as fout:
    for line_num, line in enumerate(lines, 1):
        try:
            obj = json.loads(line)
            doc_id = obj.get("id")
            if doc_id is not None:
                if doc_id in ids:
                    duplicates += 1
                    continue  # skip duplicate
                ids.add(doc_id)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Line {line_num}: JSON decode error: {e}")

print(
    f"Deduplication complete. {duplicates} duplicate rows dropped. Output: {output_path}"
)
print(f"Total unique IDs: {len(ids)}")
