import pandas as pd

temp = pd.read_parquet(
    "./data/rw_ds/autobool/temporal1000-00000-of-00001_with_dois.parquet"
)
test = pd.read_parquet("./data/rw_ds/autobool/test-00000-of-00001_with_dois.parquet")
train = pd.read_parquet("./data/rw_ds/autobool/train-00000-of-00001_with_dois.parquet")

print(f"temp: {len(temp)} rows")
print(f"test: {len(test)} rows")
print(f"train: {len(train)} rows")

combined = pd.concat([temp, test, train], ignore_index=True)
print(f"Combined: {len(combined)} rows")

combined.to_parquet("./data/rw_ds/autobool/autobool_with_dois.parquet", index=False)
