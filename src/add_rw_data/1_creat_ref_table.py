import json
import logging
from pathlib import Path
import pandas as pd
from typing import Iterable

# Config: edit paths as needed
Config = {
    "sys_rev_path": "./data/rw_ds/sys_rev_17/sr_with_dois.json",
    "seed_path": "./data/rw_ds/seed_collection/overall_doi.jsonl",
    "csmed_1_path": "./data/rw_ds/csmed/e2cs_with_dois.json",
    "csmed_2_path": "./data/rw_ds/csmed/pcs_with_dois.json",
    "autobool_path": "./data/rw_ds/autobool/autobool_with_dois.parquet",
    "clef_path": "./data/rw_ds/clef/tar_19_18_17.parquet",
    "output_parquet": "./data/rw_ds/raw/refs_id_doi.parquet",
    "log_path": "./logs/add_data/creat_ref_table.log",
}


def read_sys_rev(path: str) -> Iterable[tuple[str, str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    # expecting a list of objects with 'id' and 'doi'
    for item in data:
        _id = item.get("id")
        doi = item.get("doi")
        if _id is None or doi is None:
            continue
        yield str(_id), doi, "sys_rev"


def read_jsonl(path: str) -> Iterable[tuple[str, str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            _id = obj.get("id")
            doi = obj.get("doi")
            if _id is None or doi is None:
                continue
            yield str(_id), doi, "seed"


def read_csmed(path: str, source: str = "csmed") -> Iterable[tuple[str, str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    obj = json.loads(p.read_text())
    # expecting {'data': [ ... ]}
    data = obj.get("data") if isinstance(obj, dict) else None
    if not isinstance(data, list):
        return []
    for item in data:
        # prefer explicit 'id' if present, otherwise use 'cochrane_id' or index
        _id = item.get("id") or item.get("cochrane_id")
        doi = item.get("doi")
        if _id is None or doi is None:
            continue
        yield str(_id), doi, source


def read_autobool(path: str) -> Iterable[tuple[str, str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    df = pd.read_parquet(p)
    for _, row in df.iterrows():
        title = row.get("title")
        doi = row.get("doi")
        if pd.isna(title) or pd.isna(doi):
            continue
        yield str(title), str(doi), "autobool"


def read_clef(path: str) -> Iterable[tuple[str, str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    df = pd.read_parquet(p)
    for _, row in df.iterrows():
        cochrane_id = row.get("cochrane_id")
        doi = row.get("doi")
        if pd.isna(cochrane_id) or pd.isna(doi):
            continue
        yield str(cochrane_id), str(doi), "clef"


def main() -> None:
    # prepare logging
    log_path = Path(Config.get("log_path"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(log_path),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        filemode="w",
    )
    logger = logging.getLogger(__name__)

    rows: list[tuple[str, str, str]] = []

    rows.extend(read_sys_rev(Config["sys_rev_path"]))
    rows.extend(read_jsonl(Config["seed_path"]))
    rows.extend(read_csmed(Config["csmed_1_path"], source="csmed"))
    rows.extend(read_csmed(Config["csmed_2_path"], source="csmed"))
    rows.extend(read_autobool(Config["autobool_path"]))
    rows.extend(read_clef(Config["clef_path"]))

    if not rows:
        raise SystemExit("No id/doi pairs found from configured sources.")

    raw_df = pd.DataFrame(rows, columns=["id", "doi", "source"])
    # summary per source (before dedup)
    logger.info("Per-source counts (raw rows):")
    for src, grp in raw_df.groupby("source"):
        raw_rows = len(grp)
        raw_unique_dois = grp["doi"].astype(str).nunique()
        logger.info(" - %s: %d rows, %d unique DOIs", src, raw_rows, raw_unique_dois)

    # compute exclusive DOIs per source
    src_dois = {src: set(g["doi"]) for src, g in raw_df.groupby("source")}
    logger.info("Per-source exclusive DOI counts:")
    all_srcs = list(src_dois.keys())
    for src in all_srcs:
        others = (
            set().union(*(src_dois[s] for s in all_srcs if s != src))
            if len(all_srcs) > 1
            else set()
        )
        exclusive = {d for d in src_dois[src] if d not in others}
        logger.info(" - %s: %d DOIs unique to this source", src, len(exclusive))

    # normalize DOIs and clean, then deduplicate by DOI (keep first occurrence)
    df = raw_df.copy()
    df["id"] = df["id"].astype(str)
    df["doi"] = df["doi"].astype(str).str.strip()
    # normalize DOI case to lower for deduplication
    df["doi_norm"] = df["doi"].str.lower()
    df = df.dropna(subset=["doi_norm"])
    df = df.drop_duplicates(subset=["doi_norm"], keep="first")
    # remove helper column
    df = df.drop(columns=["doi_norm"])

    # counts per source in final table
    logger.info("Per-source counts in final table (deduplicated by DOI):")
    for src, cnt in df.groupby("source").size().items():
        logger.info(" - %s: %d rows", src, cnt)

    out_parquet = Path(Config["output_parquet"])
    df.to_parquet(out_parquet, index=False)
    logger.info("Wrote %d unique DOIs to %s", len(df), out_parquet)


if __name__ == "__main__":
    main()
