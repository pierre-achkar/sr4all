"""
Slim OpenAlex JSONL records for downstream use.
- Reads merged OpenAlex JSONL
- Writes slim JSONL
- Keeps missing values as null
- Preserves keywords and concepts as separate fields
- Uses the same PDF link selection logic as 5_download_pdfs.py
- Logs field coverage (absolute + percentage)
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, Generator, Optional

from tqdm import tqdm

# =========================
# CONFIG
# =========================
INPUT_JSONL = "./data/retrieval/merged/oax_merged_dedup.jsonl"
OUTPUT_JSONL = "./data/retrieval/merged/oax_slim.jsonl"
LOG_FILE = "./logs/retrieval/6_oax_slim.log"

OUTPUT_FIELDS = [
    "id",
    "title",
    "doi",
    "abstract",
    "year",
    "type",
    "source",
    "cited_by_count",
    "referenced_works_count",
    "referenced_works",
    "pdf_url",
    "language",
    "field",
    "subfield",
    "topics",
    "keywords",
    "concepts",
    "authors",
]


# =========================
# Setup
# =========================
os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
    force=True,
    filemode="w",
)


def stream_jsonl(path: str) -> Generator[Dict[str, Any], None, None]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logging.warning("Invalid JSON at line %d in %s", line_no, path)
                continue
            if not isinstance(obj, dict):
                logging.warning("Non-object JSON at line %d in %s", line_no, path)
                continue
            yield obj


def _clean_str(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    s = value.strip()
    return s if s else None


def _dedupe_keep_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _normalize_str_list(values: Any) -> Optional[list[str]]:
    if not isinstance(values, list):
        return None
    out: list[str] = []
    for v in values:
        s = _clean_str(v)
        if s:
            out.append(s)
    out = _dedupe_keep_order(out)
    return out or None


def _extract_display_names(values: Any) -> Optional[list[str]]:
    if not isinstance(values, list):
        return None
    out: list[str] = []
    for item in values:
        if not isinstance(item, dict):
            continue
        name = _clean_str(item.get("display_name"))
        if name:
            out.append(name)
    out = _dedupe_keep_order(out)
    return out or None


def reconstruct_abstract(inverted_index: Any) -> Optional[str]:
    """Reconstruct abstract from OpenAlex abstract_inverted_index."""
    if not isinstance(inverted_index, dict) or not inverted_index:
        return None

    max_index = -1
    for positions in inverted_index.values():
        if not isinstance(positions, list):
            continue
        for pos in positions:
            if isinstance(pos, int) and pos >= 0:
                max_index = max(max_index, pos)

    if max_index < 0:
        return None

    text_list = [""] * (max_index + 1)
    for word, positions in inverted_index.items():
        if not isinstance(word, str) or not isinstance(positions, list):
            continue
        for pos in positions:
            if isinstance(pos, int) and 0 <= pos <= max_index:
                text_list[pos] = word

    raw_text = " ".join(token for token in text_list if token)
    clean_text = re.sub(r"^Abstract\s+", "", raw_text, flags=re.IGNORECASE).strip()
    return clean_text or None


def simplify_authors(authorships: Any) -> Optional[list[Dict[str, Any]]]:
    if not isinstance(authorships, list):
        return None

    simple_authors: list[Dict[str, Any]] = []
    for auth in authorships:
        if not isinstance(auth, dict):
            continue

        a_profile = auth.get("author") or {}
        author_id = _clean_str(a_profile.get("id")) if isinstance(a_profile, dict) else None
        author_name = _clean_str(a_profile.get("display_name")) if isinstance(a_profile, dict) else None

        inst_names: list[str] = []
        institutions = auth.get("institutions") or []
        if isinstance(institutions, list):
            for inst in institutions:
                if not isinstance(inst, dict):
                    continue
                inst_name = _clean_str(inst.get("display_name"))
                if inst_name:
                    inst_names.append(inst_name)

        if not inst_names:
            raw_affs = auth.get("raw_affiliation_strings") or []
            if isinstance(raw_affs, list):
                for aff in raw_affs:
                    aff_s = _clean_str(aff)
                    if aff_s:
                        inst_names.append(aff_s)

        inst_names = _dedupe_keep_order(inst_names)
        affiliations = inst_names or None

        if author_id is None and author_name is None and affiliations is None:
            continue

        simple_authors.append(
            {
                "id": author_id,
                "name": author_name,
                "affiliations": affiliations,
            }
        )

    return simple_authors or None


def _ok_url(url: Any) -> bool:
    return isinstance(url, str) and url.strip() != ""


def collect_pdf_urls(rec: Dict[str, Any]) -> list[str]:
    """
    Same selection order as in 5_download_pdfs.py:
    best_oa_location.pdf_url -> open_access.oa_url -> OA locations -> primary_location.pdf_url -> other locations
    """
    urls: list[str] = []

    def _push(url: Any) -> None:
        if _ok_url(url):
            urls.append(url.strip())

    boa = rec.get("best_oa_location") or {}
    _push(boa.get("pdf_url"))

    oa = rec.get("open_access") or {}
    _push(oa.get("oa_url"))

    repo_oa: list[str] = []
    oa_non_repo: list[str] = []
    other: list[str] = []

    for loc in (rec.get("locations") or []):
        if not isinstance(loc, dict):
            continue
        url = loc.get("pdf_url")
        if not _ok_url(url):
            continue

        src = loc.get("source") or {}
        src_type = (src.get("type") or "").lower() if isinstance(src, dict) else ""
        is_oa = bool(loc.get("is_oa"))

        if is_oa and src_type == "repository":
            repo_oa.append(url.strip())
        elif is_oa:
            oa_non_repo.append(url.strip())
        else:
            other.append(url.strip())

    urls.extend(repo_oa)
    urls.extend(oa_non_repo)

    pl = rec.get("primary_location") or {}
    _push(pl.get("pdf_url"))

    urls.extend(other)
    return _dedupe_keep_order(urls)


def extract_pdf_link(rec: Dict[str, Any]) -> Optional[str]:
    urls = collect_pdf_urls(rec)
    return urls[0] if urls else None


def _is_filled(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) > 0
    return True


def process_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    primary_topic = rec.get("primary_topic") or {}
    field_obj = primary_topic.get("field") if isinstance(primary_topic, dict) else None
    subfield_obj = primary_topic.get("subfield") if isinstance(primary_topic, dict) else None

    field = _clean_str(field_obj.get("display_name")) if isinstance(field_obj, dict) else None
    subfield = _clean_str(subfield_obj.get("display_name")) if isinstance(subfield_obj, dict) else None

    title = _clean_str(rec.get("display_name")) or _clean_str(rec.get("title"))
    source = _clean_str((((rec.get("primary_location") or {}).get("source") or {}).get("display_name")))
    doi = _clean_str(rec.get("doi"))
    abstract = reconstruct_abstract(rec.get("abstract_inverted_index"))
    pdf_url = extract_pdf_link(rec)
    language = _clean_str(rec.get("language"))

    topics = _extract_display_names(rec.get("topics"))
    keywords = _extract_display_names(rec.get("keywords"))
    concepts = _extract_display_names(rec.get("concepts"))
    authors = simplify_authors(rec.get("authorships"))

    publication_year = rec.get("publication_year")
    year = publication_year if isinstance(publication_year, int) else None

    cited = rec.get("cited_by_count")
    cited_by_count = cited if isinstance(cited, (int, float)) else None

    ref_count_raw = rec.get("referenced_works_count")
    referenced_works_count = ref_count_raw if isinstance(ref_count_raw, int) else None
    referenced_works = _normalize_str_list(rec.get("referenced_works"))

    return {
        "id": _clean_str(rec.get("id")),
        "title": title,
        "doi": doi,
        "abstract": abstract,
        "year": year,
        "type": _clean_str(rec.get("type")),
        "source": source,
        "cited_by_count": cited_by_count,
        "referenced_works_count": referenced_works_count,
        "referenced_works": referenced_works,
        "pdf_url": pdf_url,
        "language": language,
        "field": field,
        "subfield": subfield,
        "topics": topics,
        "keywords": keywords,
        "concepts": concepts,
        "authors": authors,
    }


def main() -> None:
    print(f"Reading from JSONL: {INPUT_JSONL}")
    logging.info("Starting slimming from %s", INPUT_JSONL)

    total = 0
    field_filled_counts = {field: 0 for field in OUTPUT_FIELDS}

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as out_f:
        for rec in tqdm(stream_jsonl(INPUT_JSONL), desc="Slimming Records", unit="rec"):
            total += 1
            try:
                slim_rec = process_record(rec)
            except Exception as e:
                logging.error("Error processing %s: %s", rec.get("id"), e)
                continue

            out_f.write(json.dumps(slim_rec, ensure_ascii=False) + "\n")

            for field in OUTPUT_FIELDS:
                if _is_filled(slim_rec.get(field)):
                    field_filled_counts[field] += 1

    logging.info("Slimming finished | total_records=%d | output=%s", total, OUTPUT_JSONL)
    for field in OUTPUT_FIELDS:
        filled = field_filled_counts[field]
        pct = (filled / total * 100.0) if total else 0.0
        missing = total - filled
        logging.info(
            "Field coverage | field=%s | filled=%d | missing=%d | pct=%.2f",
            field,
            filled,
            missing,
            pct,
        )

    print(f"Saved {total} records to: {OUTPUT_JSONL}")
    print(f"Coverage stats logged to: {LOG_FILE}")


if __name__ == "__main__":
    main()
