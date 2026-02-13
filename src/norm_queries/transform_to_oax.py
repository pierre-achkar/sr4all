"""
Transforms the boolean queries in the dataset into OpenAlex API query URLs.
This involves cleaning the boolean queries according to OpenAlex's syntax requirements,
and then constructing the appropriate API URLs with search and filter parameters.
The script also logs any transformations made to the queries, especially around wildcard handling,
to ensure transparency in how the original boolean queries are being modified for OpenAlex compatibility.
"""

import json
import urllib.parse
import re
import logging

# Configuration parameters directly in code
INPUT_FILE = "./data/final/with_boolean/final/sr4all_full_normalized_boolean_mapping_merged_2_with_year_range.jsonl"
OUTPUT_FILE = (
    "./data/final/with_oax/sr4all_full_normalized_boolean_with_year_range_oax.jsonl"
)
LOG_FILE = "./logs/oax/query_transform.log"

# Setup logging with mode 'w' to overwrite each session
logging.basicConfig(
    filename=LOG_FILE,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def process_boolean_query(q):
    """
    Cleans the query according to OpenAlex official docs.
    Logs any transformations made to wildcard phrases.
    """
    original_q = q

    # 1. Identify and de-quote phrases with wildcards
    # Example: "open pred*" -> open pred
    quoted_wildcards = re.findall(r'"([^"]*\*[^"]*)"', q)
    for phrase in quoted_wildcards:
        clean_phrase = phrase.replace("*", "")
        q = q.replace(f'"{phrase}"', clean_phrase)
        logging.info(f'De-quoted wildcard phrase: ["{phrase}"] -> [{clean_phrase}]')

    # 2. Handle standalone wildcards and log if found
    if "*" in q:
        q = q.replace("*", "")
        logging.info(f"Removed standalone wildcard(s) in query: {original_q[:50]}...")

    # 3. Ensure Operators are UPPERCASE
    q = re.sub(r"\band\b", "AND", q, flags=re.IGNORECASE)
    q = re.sub(r"\bor\b", "OR", q, flags=re.IGNORECASE)
    q = re.sub(r"\bnot\b", "NOT", q, flags=re.IGNORECASE)

    return q.strip()


def build_oax_urls(item):
    base_url = "https://api.openalex.org/works"

    # Year logic: single year is upper limit (to_publication_date)
    year_raw = str(item.get("year_range_normalized", ""))
    year_filters = []

    if "-" in year_raw:
        try:
            start, end = year_raw.split("-")
            year_filters.append(f"from_publication_date:{start.strip()}-01-01")
            year_filters.append(f"to_publication_date:{end.strip()}-12-31")
        except ValueError:
            logging.error(f"Malformed year range: {year_raw}")
    elif year_raw and year_raw.isdigit():
        year_filters.append(f"to_publication_date:{year_raw}-12-31")

    filter_str = ",".join(year_filters)

    urls = []
    raw_queries = item.get("boolean_queries")
    if not isinstance(raw_queries, list):
        if raw_queries not in (None, ""):
            logging.warning(
                f"Invalid boolean_queries type: {type(raw_queries)}; coercing to empty list."
            )
        raw_queries = []

    for raw_q in raw_queries:
        clean_q = process_boolean_query(raw_q)
        encoded_q = urllib.parse.quote(clean_q)

        # Build URL with top-level search and filter parameters
        full_url = f"{base_url}?search={encoded_q}"
        if filter_str:
            full_url += f"&filter={filter_str}"

        urls.append(full_url)

    return urls


def main():
    logging.info("Starting OpenAlex URL generation.")
    try:
        count = 0
        with open(INPUT_FILE, "r") as f_in, open(OUTPUT_FILE, "w") as f_out:
            for line in f_in:
                if not line.strip():
                    continue

                data = json.loads(line)
                data["oax_query"] = build_oax_urls(data)
                f_out.write(json.dumps(data) + "\n")
                count += 1
        logging.info(f"Successfully processed {count} records.")
        print(f"Processing complete. Check {LOG_FILE} for details.")

    except FileNotFoundError:
        logging.error(f"Input file {INPUT_FILE} not found.")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    main()
