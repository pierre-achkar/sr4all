import pandas as pd
import json

PATH = "/home/fhg/pie65738/projects/sr4all/data/sr4all/clean_corpus.parquet"
df = pd.read_parquet(PATH)

PROCESSED = "/home/fhg/pie65738/projects/sr4all/data/sr4all/extraction_v1/raw_candidates.jsonl"
with open(PROCESSED, "r") as f:
    processed_ids = set()
    for line in f:
        item = json.loads(line)
        processed_ids.add(str(item.get("doc_id") or item.get("id")))

print(f"Total documents in parquet: {len(df)}")
print(f"Total processed IDs: {len(processed_ids)}")
        
# Create a new DataFrame with unprocessed documents
unprocessed_df = df[~df['doc_id'].astype(str).isin(processed_ids)]
print(f"Total unprocessed documents: {len(unprocessed_df)}")

# schuffle the unprocessed dataframe
unprocessed_df = unprocessed_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split in 2 equal parts to distribute workload by randomally assigning documents
mid_index = len(unprocessed_df) // 2
part1 = unprocessed_df.iloc[:mid_index].reset_index(drop=True)
part2 = unprocessed_df.iloc[mid_index:].reset_index(drop=True)
print(f"Part 1 documents: {len(part1)}")
print(f"Part 2 documents: {len(part2)}")


# Save to parquet files
part1.to_parquet("/home/fhg/pie65738/projects/sr4all/data/sr4all/unprocessed_part1.parquet")
part2.to_parquet("/home/fhg/pie65738/projects/sr4all/data/sr4all/unprocessed_part2.parquet")