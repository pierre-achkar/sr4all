from __future__ import annotations
import random
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# Configuration: edit these values instead of using command-line args
Config = {
    "input": "./data/rw_ds/autobool/train-00000-of-00001.parquet",
    "output": "./data/rw_ds/autobool/train-00000-of-00001_with_dois.parquet",
    "checkpoint": "./data/rw_ds/autobool/train-00000-of-00001_doi_checkpoint.parquet",
    "pmid_col": "pmid",
    "title_col": "title",
    "sleep": 0.34,
    "timeout": 30.0,
    "retries": 6,
    "backoff_base": 1.0,
    "save_every": 50,
}


def fetch_doi(
    pmid: str,
    session: requests.Session,
    timeout: float,
    retries: int,
    backoff_base: float,
) -> str | None:
    url = "https://api.ncbi.nlm.nih.gov/lit/ctxp/v1/pubmed/"
    params = {"format": "csl", "id": pmid}
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("DOI")
            if resp.status_code in {404}:
                return None
            if resp.status_code not in {408, 429, 500, 502, 503, 504}:
                return None
        except requests.RequestException:
            pass

        if attempt < retries:
            backoff = backoff_base * (2 ** (attempt - 1))
            jitter = random.uniform(0, 0.25)
            time.sleep(backoff + jitter)
    return None


def main() -> None:
    input_path = Path(Config["input"])
    output_path = Path(Config["output"])
    checkpoint_path = Path(Config["checkpoint"])

    df = pd.read_parquet(input_path)
    if Config["pmid_col"] not in df.columns:
        raise SystemExit(f"Missing PMID column: {Config['pmid_col']!r}")
    if Config["title_col"] not in df.columns:
        raise SystemExit(f"Missing title column: {Config['title_col']!r}")

    pmids = df[Config["pmid_col"]].astype(str).tolist()
    doi_map: dict[str, str | None] = {}
    if checkpoint_path.exists():
        chk = pd.read_parquet(checkpoint_path)
        if "pmid" in chk.columns and "doi" in chk.columns:
            doi_map = dict(zip(chk["pmid"].astype(str), chk["doi"]))

    with requests.Session() as session:
        for idx, pmid in enumerate(
            tqdm(pmids, desc="Fetching DOIs", unit="pmid"), start=1
        ):
            if pmid in doi_map:
                continue
            doi_map[pmid] = fetch_doi(
                pmid,
                session=session,
                timeout=Config["timeout"],
                retries=Config["retries"],
                backoff_base=Config["backoff_base"],
            )
            time.sleep(Config["sleep"])
            if Config["save_every"] and idx % Config["save_every"] == 0:
                pd.DataFrame(
                    {"pmid": list(doi_map.keys()), "doi": list(doi_map.values())}
                ).to_parquet(checkpoint_path, index=False)

    df_out = pd.DataFrame(
        {
            "title": df[Config["title_col"]],
            "doi": df[Config["pmid_col"]].astype(str).map(doi_map),
        }
    )

    if output_path.suffix.lower() == ".csv":
        df_out.to_csv(output_path, index=False)
    else:
        df_out.to_parquet(output_path, index=False)


if __name__ == "__main__":
    main()
