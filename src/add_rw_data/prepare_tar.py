import pandas as pd

df = pd.read_csv("./data/rw_ds/clef/tar_19_18_17.csv")

print(f"Before deduplication: {len(df)} rows")

# deduplicate by doi, keeping the first occurrence
df = df.drop_duplicates(subset=["doi"], keep="first")

print(f"After deduplication: {len(df)} rows")

# save to parquet
output_path = "./data/rw_ds/clef/tar_19_18_17.parquet"
df.to_parquet(output_path, index=False)
print(f"Saved deduplicated data to {output_path}")
