"""Merge flattened extraction values into the OpenAlex slim dataset."""

import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Iterator, Optional


EXTRACTION_JSONL = Path(
	"./data/extraction/repaired_fact_checked/repaired_fact_checked_merged.jsonl"
)
OAX_JSONL = Path("./data/retrieval/merged/oax_slim.jsonl")
OUTPUT_JSONL = Path("./data/final_ds/oax_slim_with_extraction.jsonl")
SAMPLE_JSONL = Path("./data/final_ds/oax_slim_with_extraction_sample_5.jsonl")
LOG_FILE = Path("./logs/final_ds/merge_extraction_into_raw.log")

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


def _iter_records(
	path: Path, id_field: str
) -> Iterator[tuple[int, str, Dict[str, Any]]]:
	with path.open("r", encoding="utf-8") as stream:
		for line_number, line in enumerate(stream, start=1):
			if not line.strip():
				continue
			location = f"{path} at line {line_number}"
			try:
				record = json.loads(line)
			except json.JSONDecodeError as error:
				raise ValueError(f"{location}: invalid JSON: {error.msg}") from error
			if not isinstance(record, dict):
				raise ValueError(f"{location}: expected an object")
			record_id = normalize_id(record.get(id_field))
			if not record_id:
				raise ValueError(f"{location}: missing or invalid {id_field}")
			yield line_number, record_id, record


def main() -> int:
	if not EXTRACTION_JSONL.is_file() or not OAX_JSONL.is_file():
		logger.error(
			"Missing input file(s) or non-file input: extraction=%s, OpenAlex=%s",
			EXTRACTION_JSONL,
			OAX_JSONL,
		)
		return 1

	extraction_by_id: Dict[str, Dict[str, Any]] = {}
	extraction_records = 0
	duplicate_extraction_ids = 0
	raw_records = 0
	matched_records = 0
	sample_records = 0
	try:
		for line_number, record_id, record in _iter_records(EXTRACTION_JSONL, "doc_id"):
			location = f"{EXTRACTION_JSONL} at line {line_number}"
			extraction = record.get("extraction")
			if extraction is None:
				extraction = {}
			if not isinstance(extraction, dict):
				raise ValueError(f"{location}: extraction must be an object or null")
			if "id" in extraction:
				raise ValueError(f"{location}: extraction must not replace the OpenAlex id")
			queries = extraction.get("exact_boolean_queries")
			if queries is not None and (
				not isinstance(queries, list)
				or any(
					query is not None and not isinstance(query, dict)
					for query in queries
				)
			):
				raise ValueError(f"{location}: expected a list of query objects or nulls")

			extraction_records += 1
			if record_id in extraction_by_id:
				duplicate_extraction_ids += 1
			extraction_by_id[record_id] = strip_evidence(extraction)

		OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
		SAMPLE_JSONL.parent.mkdir(parents=True, exist_ok=True)
		with (
			TemporaryDirectory(prefix=".merge-output-", dir=OUTPUT_JSONL.parent) as output_dir,
			TemporaryDirectory(prefix=".merge-sample-", dir=SAMPLE_JSONL.parent) as sample_dir,
		):
			temporary_output = Path(output_dir) / OUTPUT_JSONL.name
			temporary_sample = Path(sample_dir) / SAMPLE_JSONL.name
			with temporary_output.open("w", encoding="utf-8") as output_file, \
				temporary_sample.open("w", encoding="utf-8") as sample_file:
				for _, record_id, raw_record in _iter_records(OAX_JSONL, "id"):
					raw_records += 1
					merged_record = dict(raw_record)
					extracted_fields = extraction_by_id.get(record_id)
					if extracted_fields is not None:
						merged_record.update(extracted_fields)
						matched_records += 1
						if sample_records < 5:
							sample_file.write(
								json.dumps(merged_record, ensure_ascii=False) + "\n"
							)
							sample_records += 1
					output_file.write(
						json.dumps(merged_record, ensure_ascii=False) + "\n"
					)

			temporary_sample.replace(SAMPLE_JSONL)
			temporary_output.replace(OUTPUT_JSONL)
	except (OSError, ValueError) as error:
		logger.error("Merge failed: %s", error)
		return 1

	logger.info("MERGE COMPLETE")
	logger.info("Extraction records:       %d", extraction_records)
	logger.info("Unique extraction IDs:    %d", len(extraction_by_id))
	logger.info("Duplicate extraction IDs: %d", duplicate_extraction_ids)
	logger.info("OpenAlex records:         %d", raw_records)
	logger.info("Matched records:          %d", matched_records)
	logger.info("Unmatched OpenAlex:       %d", raw_records - matched_records)
	logger.info("Output:                   %s", OUTPUT_JSONL)
	logger.info("Five-row sample:          %s", SAMPLE_JSONL)
	return 0


if __name__ == "__main__":
	LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s | %(levelname)s | %(message)s",
		handlers=[
			logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
			logging.StreamHandler(),
		],
	)
	raise SystemExit(main())
