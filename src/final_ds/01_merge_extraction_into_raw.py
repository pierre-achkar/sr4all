"""Merge flattened extraction values into the OpenAlex slim dataset."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional


EXTRACTION_JSONL = Path(
	"./data/extraction/repaired_fact_checked/repaired_fact_checked_merged.jsonl"
)
OAX_JSONL = Path("./data/retrieval/merged/oax_slim.jsonl")
OUTPUT_JSONL = Path("./data/final_ds/oax_slim_with_extraction.jsonl")
SAMPLE_JSONL = Path("./data/final_ds/oax_slim_with_extraction_sample_5.jsonl")
LOG_FILE = Path("./logs/final_ds/merge_extraction_into_raw.log")

OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
SAMPLE_JSONL.parent.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s | %(levelname)s | %(message)s",
	handlers=[logging.FileHandler(LOG_FILE, mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger("MergeExtractionIntoRaw")


def normalize_id(identifier: Optional[str]) -> Optional[str]:
	"""Return the final W... component from a dataset identifier."""
	if not isinstance(identifier, str) or not identifier.strip():
		return None
	return identifier.strip().rsplit("/", 1)[-1]


def strip_evidence(extraction: Dict[str, Any]) -> Dict[str, Any]:
	"""Keep extracted values while removing evidence and verification metadata."""
	flattened: Dict[str, Any] = {}
	for field, field_data in extraction.items():
		if isinstance(field_data, dict) and "value" in field_data:
			flattened[field] = field_data["value"]
		elif field == "exact_boolean_queries" and isinstance(field_data, list):
			flattened[field] = [
				{
					key: item[key]
					for key in ("boolean_query_string", "database_source")
					if key in item
				}
				for item in field_data
				if isinstance(item, dict)
			]
		else:
			flattened[field] = field_data
	return flattened


def main() -> None:
	if not EXTRACTION_JSONL.exists() or not OAX_JSONL.exists():
		logger.error(
			"Missing input file(s): extraction=%s, OpenAlex=%s",
			EXTRACTION_JSONL,
			OAX_JSONL,
		)
		return

	extraction_by_id: Dict[str, Dict[str, Any]] = {}
	extraction_records = 0
	duplicate_extraction_ids = 0
	with EXTRACTION_JSONL.open("r", encoding="utf-8") as extraction_file:
		for line_number, line in enumerate(extraction_file, start=1):
			try:
				record = json.loads(line)
			except json.JSONDecodeError as error:
				logger.warning("Skipping invalid extraction line %d: %s", line_number, error)
				continue

			extraction_records += 1
			record_id = normalize_id(record.get("doc_id"))
			if not record_id:
				continue
			if record_id in extraction_by_id:
				duplicate_extraction_ids += 1
			extraction_by_id[record_id] = strip_evidence(record.get("extraction") or {})

	raw_records = 0
	matched_records = 0
	sample_records = 0
	with OAX_JSONL.open("r", encoding="utf-8") as oax_file, OUTPUT_JSONL.open(
		"w", encoding="utf-8"
	) as output_file, SAMPLE_JSONL.open("w", encoding="utf-8") as sample_file:
		for line_number, line in enumerate(oax_file, start=1):
			try:
				raw_record = json.loads(line)
			except json.JSONDecodeError as error:
				logger.warning("Skipping invalid OpenAlex line %d: %s", line_number, error)
				continue

			raw_records += 1
			merged_record = dict(raw_record)
			record_id = normalize_id(raw_record.get("id"))
			extracted_fields = extraction_by_id.get(record_id) if record_id else None
			if extracted_fields is not None:
				merged_record.update(extracted_fields)
				matched_records += 1
				if sample_records < 5:
					sample_file.write(
						json.dumps(merged_record, ensure_ascii=False) + "\n"
					)
					sample_records += 1
			output_file.write(json.dumps(merged_record, ensure_ascii=False) + "\n")

	logger.info("MERGE COMPLETE")
	logger.info("Extraction records:       %d", extraction_records)
	logger.info("Unique extraction IDs:    %d", len(extraction_by_id))
	logger.info("Duplicate extraction IDs: %d", duplicate_extraction_ids)
	logger.info("OpenAlex records:         %d", raw_records)
	logger.info("Matched records:          %d", matched_records)
	logger.info("Unmatched OpenAlex:       %d", raw_records - matched_records)
	logger.info("Output:                   %s", OUTPUT_JSONL)
	logger.info("Five-row sample:          %s", SAMPLE_JSONL)


if __name__ == "__main__":
	main()
