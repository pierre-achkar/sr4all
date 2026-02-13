"""
Fetch OpenAlex works by DOI list.
- Reads CSV with a 'doi' column
- Queries OpenAlex for each DOI
- Writes results to JSONL (one record per line)
- Writes placeholder records for missing DOIs
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import random
import re
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

_DOTENV_LOADED: Path | None = None


def _load_dotenv() -> Path | None:
    try:
        module = importlib.import_module("dotenv")
        env_path = Path(__file__).resolve().parents[1] / ".env"
        if env_path.exists():
            module.load_dotenv(dotenv_path=env_path)
            return env_path
    except Exception:
        return None
    return None


# =========================
# Config
# =========================
_DOTENV_LOADED = _load_dotenv()

Config = {
    "input_path": "./data/rw_ds/raw/unmatched_refs_sr4all.parquet",
    "doi_col": "doi",
    "output_jsonl": "./data/rw_ds/raw/unmatched_refs_id_doi_openalex.jsonl",
    "log_file": "./logs/add_data/fetch_oax_doi.log",
    "EMAIL": os.getenv("OPENALEX_EMAIL_5", ""),
    "API_KEY": os.getenv("OPENALEX_API_KEY_5"),
    "resume": True,
    "sleep": 0.2,
    "timeout": 60.0,
    "retries": 6,
    "backoff_base": 1.0,
    "batch_size": 50,
}


# =========================
# Helpers
# =========================
_DOI_RE_PREFIX = re.compile(r"^doi:\s*", re.IGNORECASE)
_DOI_RE_URL = re.compile(r"^https?://(dx\.)?doi\.org/", re.IGNORECASE)


def normalize_doi(doi: str | None) -> str | None:
    if doi is None:
        return None
    s = str(doi).strip().lower()
    s = _DOI_RE_PREFIX.sub("", s)
    s = _DOI_RE_URL.sub("", s)
    return s or None


def _get_logger() -> logging.Logger:
    log_path = Path(Config["log_file"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("fetch_oax_doi")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def _is_retryable(status_code: int) -> bool:
    return status_code in {408, 429, 500, 502, 503, 504}


def _build_params() -> dict:
    params: dict[str, str] = {}
    email = Config.get("EMAIL")
    api_key = Config.get("API_KEY")
    if email:
        params["mailto"] = email
    if api_key:
        params["api_key"] = api_key
    return params


def _sleep_retry_after(resp: requests.Response) -> float | None:
    retry_after = resp.headers.get("Retry-After")
    if not retry_after:
        return None
    try:
        return float(retry_after)
    except ValueError:
        return None


def fetch_batch(
    dois: list[str],
    session: requests.Session,
    timeout: float,
    retries: int,
    backoff_base: float,
    logger: logging.Logger,
) -> dict[str, dict]:
    # Batch DOI lookup: https://api.openalex.org/works?filter=doi:doi1|doi2|...&per-page=50
    params = _build_params()
    params["filter"] = "doi:" + "|".join(dois)
    params["per-page"] = str(len(dois))
    base_url = "https://api.openalex.org/works"

    for attempt in range(1, retries + 1):
        try:
            resp = session.get(base_url, params=params, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                out: dict[str, dict] = {}
                for rec in results:
                    rdoi = normalize_doi(rec.get("doi"))
                    if rdoi:
                        out[rdoi] = rec
                return out

            logger.warning(
                "OpenAlex batch status=%s attempt=%d size=%d",
                resp.status_code,
                attempt,
                len(dois),
            )

            if not _is_retryable(resp.status_code):
                return {}

            retry_after = _sleep_retry_after(resp)
            if retry_after is not None:
                time.sleep(retry_after)
                continue
        except requests.RequestException:
            logger.warning(
                "OpenAlex batch request failed (attempt=%d size=%d)", attempt, len(dois)
            )

        if attempt < retries:
            backoff = backoff_base * (2 ** (attempt - 1))
            jitter = random.uniform(0, 0.25)
            time.sleep(backoff + jitter)

    return {}


def load_existing_dois(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            doi = normalize_doi(obj.get("doi") or obj.get("requested_doi"))
            if doi:
                out.add(doi)
    return out


def load_done_ids(path: Path) -> set[str]:
    """Load a simple checkpoint file with one DOI per line."""
    if not path.exists():
        return set()
    out: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            v = line.strip()
            if v:
                nd = normalize_doi(v)
                if nd:
                    out.add(nd)
    return out


def save_done_ids(path: Path, ids: set[str]) -> None:
    """Save checkpoint file (one DOI per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for doi in sorted(ids):
            fh.write(doi + "\n")
    tmp.replace(path)


def _chunked(seq: list[str], size: int) -> list[list[str]]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


# =========================
# Main
# =========================
def main() -> None:
    logger = _get_logger()
    if _DOTENV_LOADED:
        logger.info("Loaded .env from %s", _DOTENV_LOADED)
    else:
        logger.info("No .env loaded (expected at src/.env)")
    logger.info("OpenAlex API key present: %s", bool(Config.get("API_KEY")))

    input_path = Path(Config["input_path"])
    output_path = Path(Config["output_jsonl"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # checkpoint file to speed up resume and avoid re-scanning large output
    checkpoint_path = output_path.parent / (output_path.name + ".checkpoint")

    df = pd.read_parquet(input_path)
    if Config["doi_col"] not in df.columns:
        raise SystemExit(f"Missing DOI column: {Config['doi_col']!r}")

    raw_dois = df[Config["doi_col"]].astype(str).tolist()
    dois = []
    for d in raw_dois:
        nd = normalize_doi(d)
        if nd:
            dois.append(nd)

    # De-duplicate while preserving order
    seen = set()
    unique_dois = []
    for d in dois:
        if d in seen:
            continue
        seen.add(d)
        unique_dois.append(d)

    if Config.get("resume", True):
        # prefer a lightweight checkpoint file when present
        checkpoint_ids = load_done_ids(checkpoint_path)
        if checkpoint_ids:
            already = checkpoint_ids
        else:
            already = load_existing_dois(output_path)
    else:
        already = set()

    pending = [d for d in unique_dois if d not in already]

    logger.info("Total input DOIs: %d", len(raw_dois))
    logger.info("Normalized unique DOIs: %d", len(unique_dois))
    logger.info("Already in output: %d", len(already))
    logger.info("Pending to fetch: %d", len(pending))

    if not pending:
        print("No new DOIs to fetch.")
        return

    fetched = 0
    missing = 0
    written = 0
    # track newly processed count since last checkpoint write
    since_last_save = 0

    open_mode = "a" if Config.get("resume", True) else "w"
    batch_size = max(1, int(Config.get("batch_size", 50)))
    batches = _chunked(pending, batch_size)
    with requests.Session() as session, output_path.open(
        open_mode, encoding="utf-8"
    ) as out_f:
        for batch in tqdm(batches, desc="Fetching OpenAlex", unit="batch"):
            results = fetch_batch(
                batch,
                session=session,
                timeout=Config["timeout"],
                retries=Config["retries"],
                backoff_base=Config["backoff_base"],
                logger=logger,
            )

            for d in batch:
                rec = results.get(d)
                if rec:
                    rec.setdefault("requested_doi", d)
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fetched += 1
                else:
                    out_f.write(
                        json.dumps(
                            {"requested_doi": d, "status": "not_found"},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    missing += 1
                written += 1

                # mark processed and flush
                already.add(d)
                since_last_save += 1

                # checkpoint every 50 processed DOIs
                if since_last_save >= 50:
                    try:
                        save_done_ids(checkpoint_path, already)
                        logger.info(
                            "Saved checkpoint %s (%d processed)",
                            checkpoint_path,
                            len(already),
                        )
                    except Exception:
                        logger.exception(
                            "Failed to write checkpoint %s", checkpoint_path
                        )
                    since_last_save = 0

            out_f.flush()
            time.sleep(Config["sleep"])

        # final checkpoint write for any remaining processed DOIs
        if since_last_save > 0:
            try:
                save_done_ids(checkpoint_path, already)
                logger.info(
                    "Saved final checkpoint %s (%d processed)",
                    checkpoint_path,
                    len(already),
                )
            except Exception:
                logger.exception("Failed to write final checkpoint %s", checkpoint_path)

    logger.info("Fetched: %d", fetched)
    logger.info("Missing: %d", missing)
    logger.info("Wrote JSONL records: %d", written)

    print(f"Fetched: {fetched}")
    print(f"Missing: {missing}")
    print(f"Wrote JSONL records: {written} to {output_path}")


if __name__ == "__main__":
    main()
