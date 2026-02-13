import json
import os
import pandas as pd
import logging

# Config: edit paths as needed
sr4all_jsonl = "./data/final/sr4all_merged.jsonl"
refs_df = "./data/rw_ds/raw/refs_id_doi.parquet"
log_file = "./logs/add_data/check_availability_in_sr4all.log"
# output parquet paths
matches_parquet = "./data/rw_ds/raw/matched_refs_sr4all.parquet"
non_matches_parquet = "./data/rw_ds/raw/unmatched_refs_sr4all.parquet"


# check how many refs in refs_df are present in sr4all_jsonl based on doi matching, and log the results
def main():
    logger = logging.getLogger("check_availability_in_sr4all")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file, mode="w")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.handlers = []
    logger.addHandler(handler)
    logger.propagate = False

    # load refs_df
    df_refs = pd.read_parquet(refs_df)
    if "doi" not in df_refs.columns:
        raise SystemExit("Missing 'doi' column in refs_df")
    df_refs["doi"] = df_refs["doi"].str.strip().str.lower()

    def normalize_doi(val):
        if not isinstance(val, str):
            return None
        v = val.strip().lower()
        if v.startswith("https://doi.org/"):
            v = v[len("https://doi.org/") :]
        return v

    # load sr4all_jsonl and extract normalized dois
    sr4all_dois = set()
    with open(sr4all_jsonl, "r", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            doi = obj.get("doi")
            norm = normalize_doi(doi)
            if norm:
                sr4all_dois.add(norm)

    # check availability and log matched sources with their CSV source and SR4All id
    total_refs = len(df_refs)
    available_refs = 0

    # Build a mapping from normalized DOI to SR4All id
    sr4all_doi_to_id = {}
    with open(sr4all_jsonl, "r", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            doi = obj.get("doi")
            sr_id = obj.get("id")
            norm = normalize_doi(doi)
            if norm and sr_id is not None:
                sr4all_doi_to_id[norm] = sr_id

    matches = []
    non_matches = []
    for idx, row in df_refs.iterrows():
        csv_source = row.get("source", "")
        csv_doi = row.get("doi")
        csv_norm = normalize_doi(csv_doi)
        sr4all_id = sr4all_doi_to_id.get(csv_norm) if csv_norm else None
        if sr4all_id:
            available_refs += 1
            entry = {"doi": csv_doi, "source": csv_source, "openalex_id": sr4all_id}
            matches.append(entry)
            logger.info(
                f"Matched: CSV source='{csv_source}', CSV DOI='{csv_doi}', SR4All id='{sr4all_id}'"
            )
        else:
            entry = {"doi": csv_doi, "source": csv_source}
            non_matches.append(entry)

    # write matches and non-matches to parquet
    try:
        out_dir = os.path.dirname(matches_parquet)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        df_matches = pd.DataFrame(matches)
        df_non = pd.DataFrame(non_matches)
        # ensure columns and types
        if not df_matches.empty:
            df_matches = df_matches[["doi", "source", "openalex_id"]]
        if not df_non.empty:
            df_non = df_non[["doi", "source"]]
        df_matches.to_parquet(matches_parquet, index=False)
        df_non.to_parquet(non_matches_parquet, index=False)
        logger.info(f"Wrote {len(df_matches)} matches to {matches_parquet}")
        logger.info(f"Wrote {len(df_non)} non-matches to {non_matches_parquet}")
    except Exception as e:
        logger.info(f"Failed to write parquet outputs: {e}")

    logger.info(f"Total refs: {total_refs}")
    logger.info(f"Available in SR4All: {available_refs}")
    logger.info(f"Availability percentage: {available_refs / total_refs:.2%}")


if __name__ == "__main__":
    main()
