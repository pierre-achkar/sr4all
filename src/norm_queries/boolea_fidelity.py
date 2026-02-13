"""
Boolean fidelity evaluation for the boolean split only.

What it measures (purely automatic, no retrieval):
1) Term fidelity: overlap between original and normalised term sets
   - precision / recall / F1 (set-based)
   - also unigram-view to avoid phrase-vs-token artefacts
2) Operator fidelity: edit-similarity of operator streams (AND/OR/NOT)
3) Structure fidelity: edit-similarity of "skeleton" strings (terms -> t, keep () and ops)
4) Parentheses well-formedness (balanced) + max depth diff
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

# ========================
# Config
# ========================
CONFIG = {
    "original_boolean_jsonl": Path(
        "./data/final/sr4all_full_normalized_year_range_search_has_boolean.jsonl"
    ),
    "normalized_boolean_jsonl": Path(
        "./data/final/with_boolean/final/sr4all_full_normalized_boolean_mapping_merged_2_with_year_range.jsonl"
    ),
    "out_aggregate_json": Path(
        "./data/final/with_boolean/boolean_fidelity_aggregate.json"
    ),
    "out_per_record_jsonl": Path(
        "./data/final/with_boolean/boolean_fidelity_per_record.jsonl"
    ),
    # Canonicalisation toggles
    "strip_fields": True,
    "strip_metadata": True,  # years / filters
    "repair_original_parentheses": True,  # balance-only repair for original before structure metrics
    "min_term_len": 2,
}

# ========================
# Logging
# ========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("boolean_fidelity")


# ========================
# IO
# ========================
def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_map_by_id(path: Path) -> Dict[str, Dict]:
    out = {}
    for rec in iter_jsonl(path):
        rid = rec.get("id")
        if isinstance(rid, str) and rid:
            out[rid] = rec
    return out


# ========================
# Canonicalisation (same as before, kept compact)
# ========================
_BOOL_WORDS = {"AND", "OR", "NOT"}

SMART_QUOTES = {
    "“": '"',
    "”": '"',
    "„": '"',
    "‟": '"',
    "’": "'",
    "‘": "'",
    "‚": "'",
    "‛": "'",
}

_FIELD_PREFIX_PATTERNS = [
    r"\bTITLE-ABS-KEY\s*\(",
    r"\bTITLE\s*\(",
    r"\bABS(?:TRACT)?\s*\(",
    r"\bKEY\s*\(",
    r"\bAUTHKEY\s*\(",
    r"\bAUTHOR-NAME\s*\(",
    r"\bTS\s*=\s*\(",
    r"\bTI\s*=\s*\(",
    r"\bAB\s*=\s*\(",
    r"\bALL\s*=\s*\(",
]

_PUBMED_TAG_RE = re.compile(r"(\[[a-zA-Z]+\])")
_YEAR_RANGE_RE = re.compile(r"\b(19\d{2}|20\d{2})\s*[-–]\s*(19\d{2}|20\d{2})\b")
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")

TOKEN_RE = re.compile(r'("([^"\\]*(?:\\.[^"\\]*)*)")|([()])|([^\s()]+)')


def normalize_quotes(text: str) -> str:
    for k, v in SMART_QUOTES.items():
        text = text.replace(k, v)
    return text


def strip_pubmed_tags(text: str) -> str:
    return _PUBMED_TAG_RE.sub("", text)


def strip_field_scoping(text: str) -> str:
    for pat in _FIELD_PREFIX_PATTERNS:
        text = re.sub(pat, "(", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\b(TI|AB|TS|TITLE|ABS|KEY|AUTHKEY|AU)\s*:\s*", "", text, flags=re.IGNORECASE
    )
    return text


def strip_metadata(text: str) -> str:
    text = re.sub(r"\bLIMIT-TO\s*\([^)]*\)", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bPUBYEAR\s*[<>=]+\s*\d{4}", " ", text, flags=re.IGNORECASE)
    text = _YEAR_RANGE_RE.sub(" ", text)
    text = _YEAR_RE.sub(" ", text)
    return text


def normalize_bool_ops(text: str) -> str:
    text = re.sub(
        r"\b(and|or|not)\b", lambda m: m.group(1).upper(), text, flags=re.IGNORECASE
    )
    text = re.sub(r"\b(NEAR|W)\s*/\s*\d+\b", "AND", text, flags=re.IGNORECASE)
    return text


def canonicalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = normalize_quotes(text)
    text = strip_pubmed_tags(text)
    if CONFIG["strip_fields"]:
        text = strip_field_scoping(text)
    if CONFIG["strip_metadata"]:
        text = strip_metadata(text)
    text = normalize_bool_ops(text)
    text = text.replace("$", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[Dict[str, str]]:
    toks = []
    for m in TOKEN_RE.finditer(text):
        q_full, q_inner, paren, word = m.groups()
        if q_full:
            toks.append({"type": "PHRASE", "val": q_inner})
        elif paren:
            toks.append({"type": "PAREN", "val": paren})
        elif word:
            up = word.upper()
            if up in _BOOL_WORDS:
                toks.append({"type": "OP", "val": up})
            else:
                toks.append({"type": "TERM", "val": word})
    return toks


def clean_term(t: str) -> str:
    t = t.strip().lower()
    t = t.replace("–", "-")
    t = re.sub(r"^[^\w\*\?]+|[^\w\*\?]+$", "", t)  # keep * and ?
    return t


def terms_set(text: str) -> List[str]:
    out = []
    for tok in tokenize(text):
        if tok["type"] in ("TERM", "PHRASE"):
            ct = clean_term(tok["val"])
            if len(ct) >= CONFIG["min_term_len"]:
                out.append(ct)
    return out


def terms_unigrams(text: str) -> List[str]:
    """More forgiving view: phrases contribute their unigrams too."""
    out = []
    for tok in tokenize(text):
        if tok["type"] == "TERM":
            ct = clean_term(tok["val"])
            if len(ct) >= CONFIG["min_term_len"]:
                out.append(ct)
        elif tok["type"] == "PHRASE":
            phrase = clean_term(tok["val"])
            if phrase:
                # keep phrase as a unit AND unigrams
                out.append(phrase)
                for w in phrase.split():
                    w = clean_term(w)
                    if len(w) >= CONFIG["min_term_len"]:
                        out.append(w)
    return out


def ops_stream(text: str) -> List[str]:
    return [t["val"] for t in tokenize(text) if t["type"] == "OP"]


def skeleton_tokens(text: str) -> List[str]:
    out = []
    for t in tokenize(text):
        if t["type"] in ("TERM", "PHRASE"):
            out.append("t")
        else:
            out.append(t["val"])
    return out


def paren_stats(text: str) -> Dict[str, Any]:
    depth = 0
    max_depth = 0
    balanced = True
    for t in tokenize(text):
        if t["type"] == "PAREN":
            if t["val"] == "(":
                depth += 1
                max_depth = max(max_depth, depth)
            else:
                depth -= 1
                if depth < 0:
                    balanced = False
                    depth = 0
    if depth != 0:
        balanced = False
    return {"balanced": balanced, "max_depth": max_depth}


def _tokenize_preserve_quotes(text: str) -> List[Dict[str, str]]:
    """Tokenize but keep quoted phrases with quotes intact."""
    toks = []
    for m in TOKEN_RE.finditer(text):
        q_full, _q_inner, paren, word = m.groups()
        if q_full:
            toks.append({"type": "PHRASE", "val": q_full})
        elif paren:
            toks.append({"type": "PAREN", "val": paren})
        elif word:
            toks.append({"type": "TERM", "val": word})
    return toks


def balance_parentheses_only(text: str) -> str:
    """Balance parentheses without changing token order. Removes extra ')' and appends missing ')'."""
    toks = _tokenize_preserve_quotes(text)
    out = []
    depth = 0
    for t in toks:
        if t["type"] == "PAREN":
            if t["val"] == "(":
                depth += 1
                out.append("(")
            else:
                if depth > 0:
                    depth -= 1
                    out.append(")")
                else:
                    # drop unmatched ')'
                    continue
        else:
            out.append(t["val"])
    if depth > 0:
        out.extend([")"] * depth)
    return " ".join(out)


# ========================
# Metrics
# ========================
def prf(
    orig_terms: List[str], norm_terms: List[str]
) -> Tuple[float, float, float, int, int]:
    so, sn = set(orig_terms), set(norm_terms)
    if not so and not sn:
        return 1.0, 1.0, 1.0, 0, 0
    inter = len(so & sn)
    prec = inter / len(sn) if sn else 0.0
    rec = inter / len(so) if so else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    added = len(sn - so)
    dropped = len(so - sn)
    return prec, rec, f1, added, dropped


def levenshtein(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m]


def edit_sim(a: List[str], b: List[str]) -> float:
    if not a and not b:
        return 1.0
    denom = max(len(a), len(b))
    if denom == 0:
        return 1.0
    return 1.0 - (levenshtein(a, b) / denom)


def summarize(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0}
    xs = sorted(xs)
    n = len(xs)

    def pct(p: float) -> float:
        i = int(round((n - 1) * p))
        return xs[i]

    median = xs[n // 2] if n % 2 == 1 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])
    return {"mean": sum(xs) / n, "median": median, "p25": pct(0.25), "p75": pct(0.75)}


# ========================
# Extract query strings
# ========================
def original_boolean_string(rec: Dict[str, Any]) -> str:
    b = rec.get("exact_boolean_queries")
    if not isinstance(b, list) or not b:
        return ""
    parts = []
    for item in b:
        if isinstance(item, dict):
            s = item.get("boolean_query_string")
            if isinstance(s, str) and s.strip():
                parts.append(s.strip())
    if not parts:
        return ""
    return parts[0] if len(parts) == 1 else "(" + ") OR (".join(parts) + ")"


def normalized_boolean_string(rec: Dict[str, Any]) -> str:
    bqs = rec.get("boolean_queries")
    if not isinstance(bqs, list) or not bqs:
        return ""
    clean = [q.strip() for q in bqs if isinstance(q, str) and q.strip()]
    if not clean:
        return ""
    return clean[0] if len(clean) == 1 else "(" + ") OR (".join(clean) + ")"


# ========================
# Main
# ========================
def compute_for_population(pairs: List[Tuple[str, str, str]]) -> Dict[str, Any]:
    """
    pairs: list of (id, orig_raw, norm_raw) where norm_raw may be "" for null.
    """
    # accumulators
    precs, recs, f1s = [], [], []
    precs_u, recs_u, f1s_u = [], [], []  # unigram view
    op_sims, skel_sims = [], []
    orig_bal, norm_bal = 0, 0
    depth_diffs = []
    added_counts, dropped_counts = [], []

    for rid, o_raw, n_raw in pairs:
        o_can = canonicalize(o_raw)
        n_can = canonicalize(n_raw)

        if CONFIG["repair_original_parentheses"]:
            o_can = balance_parentheses_only(o_can)

        o_terms = terms_set(o_can)
        n_terms = terms_set(n_can)
        prec, rec, f1, added, dropped = prf(o_terms, n_terms)

        o_terms_u = terms_unigrams(o_can)
        n_terms_u = terms_unigrams(n_can)
        prec_u, rec_u, f1_u, _, _ = prf(o_terms_u, n_terms_u)

        o_ops = ops_stream(o_can)
        n_ops = ops_stream(n_can)
        op_sim = edit_sim(o_ops, n_ops)

        o_skel = skeleton_tokens(o_can)
        n_skel = skeleton_tokens(n_can)
        sk_sim = edit_sim(o_skel, n_skel)

        o_par = paren_stats(o_can)
        n_par = paren_stats(n_can)

        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
        precs_u.append(prec_u)
        recs_u.append(rec_u)
        f1s_u.append(f1_u)
        op_sims.append(op_sim)
        skel_sims.append(sk_sim)
        orig_bal += int(o_par["balanced"])
        norm_bal += int(n_par["balanced"])
        depth_diffs.append(n_par["max_depth"] - o_par["max_depth"])
        added_counts.append(added)
        dropped_counts.append(dropped)

    return {
        "term_precision": summarize(precs),
        "term_recall": summarize(recs),
        "term_f1": summarize(f1s),
        "term_precision_unigram_view": summarize(precs_u),
        "term_recall_unigram_view": summarize(recs_u),
        "term_f1_unigram_view": summarize(f1s_u),
        "operator_seq_similarity": summarize(op_sims),
        "skeleton_similarity": summarize(skel_sims),
        "paren_balance_rate": {
            "original": (orig_bal / len(pairs)) if pairs else 0.0,
            "normalized": (norm_bal / len(pairs)) if pairs else 0.0,
        },
        "max_depth_diff": {
            "mean": (sum(depth_diffs) / len(depth_diffs)) if depth_diffs else 0.0,
            "min": min(depth_diffs) if depth_diffs else 0,
            "max": max(depth_diffs) if depth_diffs else 0,
        },
        "added_terms": {
            "mean": (sum(added_counts) / len(added_counts)) if added_counts else 0.0,
            "median": (
                sorted(added_counts)[len(added_counts) // 2] if added_counts else 0
            ),
        },
        "dropped_terms": {
            "mean": (
                (sum(dropped_counts) / len(dropped_counts)) if dropped_counts else 0.0
            ),
            "median": (
                sorted(dropped_counts)[len(dropped_counts) // 2]
                if dropped_counts
                else 0
            ),
        },
    }


def main():
    orig = load_map_by_id(CONFIG["original_boolean_jsonl"])
    norm = load_map_by_id(CONFIG["normalized_boolean_jsonl"])

    common_ids = sorted(set(orig.keys()) & set(norm.keys()))

    logger.info("Loaded %d original records", len(orig))
    logger.info("Loaded %d normalized records", len(norm))
    logger.info("Matched %d records by id", len(common_ids))

    # Build pair lists
    all_pairs = []
    executable_pairs = []
    null_norm = 0

    per_path = CONFIG["out_per_record_jsonl"]
    per_path.parent.mkdir(parents=True, exist_ok=True)

    with per_path.open("w", encoding="utf-8") as fout:
        for rid in tqdm(common_ids, desc="Scoring records", unit="rec"):
            o_raw = original_boolean_string(orig[rid])
            n_raw = normalized_boolean_string(norm[rid])

            all_pairs.append((rid, o_raw, n_raw))
            if not n_raw:
                null_norm += 1
            else:
                executable_pairs.append((rid, o_raw, n_raw))

            # Write per-record (lightweight; enough to debug outliers)
            o_can = canonicalize(o_raw)
            n_can = canonicalize(n_raw)
            if CONFIG["repair_original_parentheses"]:
                o_can_rep = balance_parentheses_only(o_can)
            else:
                o_can_rep = o_can

            o_terms = terms_set(o_can_rep)
            n_terms = terms_set(n_can)
            prec, rec, f1, added, dropped = prf(o_terms, n_terms)

            o_terms_u = terms_unigrams(o_can_rep)
            n_terms_u = terms_unigrams(n_can)
            prec_u, rec_u, f1_u, _, _ = prf(o_terms_u, n_terms_u)

            o_ops = ops_stream(o_can_rep)
            n_ops = ops_stream(n_can)
            op_sim = edit_sim(o_ops, n_ops)

            o_skel = skeleton_tokens(o_can_rep)
            n_skel = skeleton_tokens(n_can)
            sk_sim = edit_sim(o_skel, n_skel)

            fout.write(
                json.dumps(
                    {
                        "id": rid,
                        "orig_raw": o_raw,
                        "norm_raw": n_raw,
                        "term_precision": round(prec, 4),
                        "term_recall": round(rec, 4),
                        "term_f1": round(f1, 4),
                        "term_precision_unigram_view": round(prec_u, 4),
                        "term_recall_unigram_view": round(rec_u, 4),
                        "term_f1_unigram_view": round(f1_u, 4),
                        "added_terms": added,
                        "dropped_terms": dropped,
                        "operator_seq_similarity": round(op_sim, 4),
                        "skeleton_similarity": round(sk_sim, 4),
                        "orig_paren_balanced_raw": paren_stats(o_can)["balanced"],
                        "orig_paren_balanced_repaired": paren_stats(o_can_rep)[
                            "balanced"
                        ],
                        "norm_paren_balanced": paren_stats(n_can)["balanced"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    agg = {
        "counts": {
            "original_records": len(orig),
            "normalized_records": len(norm),
            "matched_records": len(common_ids),
            "normalized_empty_or_null": null_norm,
            "executable_pairs": len(executable_pairs),
        },
        "settings": {
            "strip_fields": CONFIG["strip_fields"],
            "strip_metadata": CONFIG["strip_metadata"],
            "repair_original_parentheses": CONFIG["repair_original_parentheses"],
            "min_term_len": CONFIG["min_term_len"],
        },
        "metrics_all_matched": compute_for_population(all_pairs),
        "metrics_executable_only": compute_for_population(executable_pairs),
    }

    out_path = CONFIG["out_aggregate_json"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)

    logger.info("Done.")
    logger.info("Matched: %d", len(common_ids))
    logger.info("Normalized null/empty: %d", null_norm)
    logger.info("Executable-only: %d", len(executable_pairs))
    logger.info("Wrote per-record: %s", per_path)
    logger.info("Wrote aggregate: %s", out_path)


if __name__ == "__main__":
    main()
