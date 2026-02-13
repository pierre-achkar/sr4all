"""
Flatten oax_query_ids (list of lists) into unique list per record.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

DEFAULT_INPUT = Path(
    "./data/final_old/with_oax/oax_count_buckets/bucket_1_250k_with_ids.jsonl"
)
DEFAULT_OUTPUT = Path(
    "./data/final_old/with_oax/oax_count_buckets/bucket_1_250k_with_ids_flattened.jsonl"
)


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _unique_union(lists: Sequence[List]) -> List:
    seen = set()
    out = []
    for lst in lists:
        for item in lst:
            if item not in seen:
                seen.add(item)
                out.append(item)
    return out


def flatten_oax_query_ids(value: Optional[object]) -> Optional[object]:
    if not isinstance(value, list):
        return value

    list_items = [item for item in value if isinstance(item, list)]
    if not list_items:
        return value

    return _unique_union(list_items)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Flatten oax_query_ids (list of lists) into a unique list per record. "
            "If multiple lists exist, their union is used."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    changed = 0

    with args.output.open("w", encoding="utf-8") as f_out:
        for rec in iter_jsonl(args.input):
            total += 1
            original = rec.get("oax_query_ids")
            flattened = flatten_oax_query_ids(original)
            if flattened is not original:
                changed += 1
                rec = dict(rec)
                rec["oax_query_ids"] = flattened
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Processed {total} records")
    print(f"Updated oax_query_ids in {changed} records")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
