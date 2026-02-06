import json


# read both files
def read_jsonl(path: str):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


# Original Split
PATH_1 = "/home/fhg/pie65738/projects/sr4all/data/final/sr4all_full_normalized_year_range_search_has_boolean.jsonl"
PATH_2 = "/home/fhg/pie65738/projects/sr4all/data/final/sr4all_full_normalized_year_range_search_keywords_only.jsonl"

print("\nOriginal Splits:")
records_3 = read_jsonl(PATH_1)
records_4 = read_jsonl(PATH_2)

print(f"Records in Boolean : {len(records_3)}")
print(f"Records in Keywords Only: {len(records_4)}")


# After norm, postprocessing
PATH_3 = "/home/fhg/pie65738/projects/sr4all/data/final/with_boolean/final/sr4all_full_normalized_boolean_mapping_merged_2_with_year_range.jsonl"
PATH_4 = "/home/fhg/pie65738/projects/sr4all/data/final/with_boolean/final/sr4all_full_normalized_keywords_only_mapping_merged_2_with_year_range.jsonl"

# hwo many records in each file
records_3 = read_jsonl(PATH_3)
records_4 = read_jsonl(PATH_4)

print("After norm, postprocessing:")
print(f"Records in Boolean : {len(records_3)}")
print(f"Records in Keywords Only: {len(records_4)}")
# how many records have "boolean_queries": null in each file
null_count_1 = sum(1 for r in records_3 if r.get("boolean_queries") is None)
null_count_2 = sum(1 for r in records_4 if r.get("boolean_queries") is None)

print(f"Records with null boolean_queries in Boolean : {null_count_1}")
print(f"Records with null boolean_queries in Keywords Only: {null_count_2}")

# OAX query counts
PATH_5 = "/home/fhg/pie65738/projects/sr4all/data/final/with_oax/sr4all_full_normalized_boolean_with_year_range_oax_with_counts.jsonl"
PATH_6= "/home/fhg/pie65738/projects/sr4all/data/final/with_oax/sr4all_full_normalized_keywords_only_oax_with_year_range_oax_with_counts.jsonl"

print("\nOAX Confor:")
records_5 = read_jsonl(PATH_5)
records_6 = read_jsonl(PATH_6)

print(f"Records in Boolean : {len(records_5)}")
print(f"Records in Keywords Only: {len(records_6)}")

def summarize_oax_errors(records):
    total = len(records)
    missing_errors = 0
    no_queries = 0
    ok_records = 0
    failed_records = 0
    error_items = 0

    for r in records:
        urls = r.get("oax_query")
        if not isinstance(urls, list) or len(urls) == 0:
            no_queries += 1

        errors = r.get("oax_query_errors")
        if not isinstance(errors, list):
            missing_errors += 1
            continue

        if errors and all(e is None for e in errors):
            ok_records += 1
        elif any(e is not None for e in errors):
            failed_records += 1
            error_items += sum(1 for e in errors if e is not None)

    return {
        "total": total,
        "missing_errors": missing_errors,
        "no_queries": no_queries,
        "ok_records": ok_records,
        "failed_records": failed_records,
        "error_items": error_items,
    }

def get_failed_oax_records(records):
    failed_ids = []
    for r in records:
        errors = r.get("oax_query_errors")
        if isinstance(errors, list) and any(e is not None for e in errors):
            rec_id = r.get("id") or r.get("doc_id") or r.get("rec_id")
            failed_ids.append(rec_id)
    return failed_ids

summary_1 = summarize_oax_errors(records_5)
summary_2 = summarize_oax_errors(records_6)

print("OAX error summary (Boolean):")
print(summary_1)
print("OAX error summary (Keywords Only):")
print(summary_2)

failed_ids_1 = get_failed_oax_records(records_5)
failed_ids_2 = get_failed_oax_records(records_6)

print("Raw failed OAX records (Boolean):", len(failed_ids_1))
print("Raw failed OAX records (Keywords Only):", len(failed_ids_2))

# Write failed IDs to text files (one ID per line)
FAILED_IDS_1 = "/home/fhg/pie65738/projects/sr4all/data/final/with_oax/debug_errors/failed_oax_ids_boolean.txt"
FAILED_IDS_2 = "/home/fhg/pie65738/projects/sr4all/data/final/with_oax/debug_errors/failed_oax_ids_keywords_only.txt"

with open(FAILED_IDS_1, "w", encoding="utf-8") as f:
    for rid in failed_ids_1:
        if rid is not None:
            f.write(str(rid) + "\n")

with open(FAILED_IDS_2, "w", encoding="utf-8") as f:
    for rid in failed_ids_2:
        if rid is not None:
            f.write(str(rid) + "\n")

print("Wrote failed ID lists:")
print(FAILED_IDS_1)
print(FAILED_IDS_2)