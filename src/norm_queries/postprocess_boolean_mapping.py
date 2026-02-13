"""
Post-processes the repaired boolean mapping to fix common syntax issues and filter out meaningless queries.
This is a final cleanup step to improve the quality of the repaired boolean queries before merging them back
into the main dataset.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any  # Added Any here

# ========================
# Config
# ========================
CONFIG = {
    "input_jsonl": Path(
        "./data/final/with_boolean/repaired/sr4all_full_normalized_boolean_repaired_mapping_2.jsonl"
    ),
    "output_jsonl": Path(
        "./data/final/with_boolean/repaired_fixed/sr4all_full_normalized_boolean_repaired_fixed_mapping_2.jsonl"
    ),
    "log_file": Path("./logs/oax/postprocess_boolean_mapping_repaired_2.log"),
}

# ========================
# Logging
# ========================
CONFIG["log_file"].parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(CONFIG["log_file"], mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("lucene_postprocess")

_BOOL_OPS = {"AND", "OR", "NOT"}

# ========================
# Logic Helpers
# ========================


def _tokenize_query(text: str) -> List[Dict]:
    """Tokenize with slash protection and quote awareness."""
    if not isinstance(text, str):
        return []
    tokens = []
    pattern = re.compile(r'("[^"\\]*(?:\\.[^"\\]*)*")|([()])|([^\s()]+)')
    for match in pattern.finditer(text):
        quoted, paren, word = match.groups()
        if quoted:
            tokens.append({"type": "QUOTED", "val": quoted})
        elif paren:
            tokens.append({"type": "PAREN", "val": paren})
        elif word:
            u_word = word.upper()
            if u_word in _BOOL_OPS:
                tokens.append({"type": "OPERATOR", "val": u_word})
            else:
                processed_word = word.replace("/", "\\/") if "/" in word else word
                tokens.append({"type": "TERM", "val": processed_word})
    return tokens


def _is_meaningful_query(text: Any) -> bool:
    """Filters out bullshit queries composed only of single-letter placeholders."""
    if not isinstance(text, str):
        return False
    tokens = _tokenize_query(text)
    terms = [t["val"] for t in tokens if t["type"] in ("TERM", "QUOTED")]
    if not terms:
        return False
    # Filter: At least one term must be longer than 1 char (ignoring quotes/escapes)
    meaningful = [w for w in terms if len(w.strip('"').replace("\\", "")) > 1]
    return len(meaningful) > 0


def _strip_onion_layers(res: str) -> str:
    """Simplifies ((Term)) to (Term)."""
    while True:
        new_res = re.sub(r"\(\(([^()]+)\)\)", r"(\1)", res)
        if new_res == res:
            break
        res = new_res
    return res


def _fix_lucene_syntax(text: Optional[str]) -> Tuple[str, bool]:
    if not isinstance(text, str) or not text.strip():
        return "", False

    original = text.strip()
    text = re.sub(r"\s+", " ", original.replace("$", ""))

    tokens = _tokenize_query(text)
    processed_tokens = []
    buffer = []

    def flush():
        if not buffer:
            return
        if len(buffer) > 1:
            combined = " ".join(buffer)
            if "*" in combined or "?" in combined:
                processed_tokens.append({"type": "PAREN", "val": "("})
                for idx, b in enumerate(buffer):
                    processed_tokens.append({"type": "TERM", "val": b})
                    if idx < len(buffer) - 1:
                        processed_tokens.append({"type": "OPERATOR", "val": "AND"})
                processed_tokens.append({"type": "PAREN", "val": ")"})
            else:
                processed_tokens.append({"type": "QUOTED", "val": f'"{combined}"'})
        else:
            term = buffer[0].lstrip("*?")
            if term:
                processed_tokens.append({"type": "TERM", "val": term})
        buffer.clear()

    for tok in tokens:
        if tok["type"] == "TERM":
            buffer.append(tok["val"])
        else:
            flush()
            processed_tokens.append(tok)
    flush()

    cleaned_tokens = []
    for t in processed_tokens:
        if t["type"] == "OPERATOR":
            if not cleaned_tokens or (cleaned_tokens[-1]["val"] == "("):
                continue
            if cleaned_tokens[-1]["type"] == "OPERATOR":
                cleaned_tokens[-1] = t
                continue
        cleaned_tokens.append(t)
    while cleaned_tokens and cleaned_tokens[-1]["type"] == "OPERATOR":
        cleaned_tokens.pop()

    res = " ".join([t["val"] for t in cleaned_tokens])

    res = re.sub(r"\(\s+", "(", res)
    res = re.sub(r"\s+\)", ")", res)
    res = re.sub(r"\)\s+\)", "))", res)
    res = re.sub(r"\(\s+\(", "((", res)
    res = re.sub(r"\)(AND|OR|NOT)", r") \1", res)
    res = re.sub(r"(AND|OR|NOT)\(", r"\1 (", res)

    open_c, close_c = res.count("("), res.count(")")
    if open_c > close_c:
        res += ")" * (open_c - close_c)
    elif close_c > open_c:
        for _ in range(close_c - open_c):
            res = res.replace(")", "", 1)

    final_query = _strip_onion_layers(res.strip())
    return final_query, final_query != original


def is_valid(text: str) -> bool:
    if not text or re.match(r"^[\s()]*$", text):
        return False
    tokens = _tokenize_query(text)
    return any(t["type"] in ("TERM", "QUOTED") for t in tokens)


# ========================
# Main
# ========================


def main():
    input_path = CONFIG["input_jsonl"]
    output_path = CONFIG["output_jsonl"]
    if not input_path.exists():
        return
    temp_path = output_path.with_suffix(".tmp")

    m = {"records": 0, "nulled": 0, "q_total": 0, "q_rep": 0, "q_drop": 0}

    with input_path.open("r", encoding="utf-8") as fin, temp_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            m["records"] += 1
            try:
                rec = json.loads(line)
            except:
                continue

            queries = rec.get("boolean_queries")
            if not isinstance(queries, list):
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            # Substance Filter: Guard against NoneType elements
            meaningful = [
                q for q in queries if isinstance(q, str) and _is_meaningful_query(q)
            ]

            if not meaningful:
                rec["boolean_queries"] = None
                m["nulled"] += 1
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            has_any_operator = any(
                any(t["type"] == "OPERATOR" for t in _tokenize_query(mq))
                for mq in meaningful
            )

            if not has_any_operator:
                seen = set()
                unique_terms = []
                for mq in meaningful:
                    fixed_q, _ = _fix_lucene_syntax(mq)
                    if fixed_q and fixed_q.lower() not in seen:
                        unique_terms.append(fixed_q)
                        seen.add(fixed_q.lower())

                if unique_terms:
                    rec["boolean_queries"] = [f"({' OR '.join(unique_terms)})"]
                    m["q_rep"] += 1
                else:
                    rec["boolean_queries"] = None
                    m["nulled"] += 1
            else:
                fixed_list = []
                for mq in meaningful:
                    m["q_total"] += 1
                    fixed_q, repaired = _fix_lucene_syntax(mq)
                    if is_valid(fixed_q):
                        fixed_list.append(fixed_q)
                        if repaired:
                            m["q_rep"] += 1
                    else:
                        m["q_drop"] += 1

                rec["boolean_queries"] = fixed_list if fixed_list else None
                if not fixed_list:
                    m["nulled"] += 1

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    temp_path.replace(output_path)
    logger.info(
        f"Done. Records: {m['records']} | Nulled: {m['nulled']} | Repaired Queries: {m['q_rep']} | Dropped: {m['q_drop']}"
    )


if __name__ == "__main__":
    main()
