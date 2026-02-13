"""
Evaluate retrieval vs. relevant referenced works by id match.
"""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_RELEVANT = Path(
    "./data/final_old/sr4all_full_normalized_year_range_search_has_boolean.jsonl"
)
DEFAULT_RETRIEVED = Path(
    "./data/final_old/with_oax/oax_count_buckets/bucket_1_250k_with_ids_flattened.jsonl"
)
DEFAULT_OUTPUT = Path(
    "./data/final_old/with_oax/res/bucket_1_250k_retrieval_eval.jsonl"
)
DEFAULT_SUMMARY = Path(
    "./data/final_old/with_oax/res/bucket_1_250k_retrieval_eval_summary.json"
)
DEFAULT_PLOT = Path(
    "./data/final_old/with_oax/res/bucket_1_250k_retrieval_eval_recall_hist.png"
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


def _to_set(value: Optional[object]) -> Set[str]:
    if not isinstance(value, list):
        return set()
    return {item for item in value if isinstance(item, str)}


def _safe_div(numer: float, denom: float) -> float:
    return numer / denom if denom else 0.0


def _f_beta(precision: float, recall: float, beta: float) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta_sq = beta * beta
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def load_relevant(path: Path) -> Dict[str, Set[str]]:
    relevant: Dict[str, Set[str]] = {}
    for rec in iter_jsonl(path):
        rec_id = rec.get("id")
        if not isinstance(rec_id, str) or not rec_id:
            continue
        relevant[rec_id] = _to_set(rec.get("referenced_works"))
    return relevant


def load_retrieved(path: Path) -> Dict[str, Set[str]]:
    retrieved: Dict[str, Set[str]] = {}
    for rec in iter_jsonl(path):
        rec_id = rec.get("id")
        if not isinstance(rec_id, str) or not rec_id:
            continue
        retrieved[rec_id] = _to_set(rec.get("oax_query_ids"))
    return retrieved


def main() -> None:
    relevant_path = DEFAULT_RELEVANT
    retrieved_path = DEFAULT_RETRIEVED
    output_path = DEFAULT_OUTPUT
    summary_path = DEFAULT_SUMMARY
    plot_path = DEFAULT_PLOT

    if not relevant_path.exists():
        raise FileNotFoundError(f"Relevant file not found: {relevant_path}")
    if not retrieved_path.exists():
        raise FileNotFoundError(f"Retrieved file not found: {retrieved_path}")

    relevant = load_relevant(relevant_path)
    retrieved = load_retrieved(retrieved_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    total_tp = 0
    total_retrieved = 0
    total_relevant = 0
    matched_ids = 0
    macro_precision_sum = 0.0
    macro_recall_sum = 0.0
    macro_f1_sum = 0.0
    macro_f3_sum = 0.0
    recall_values: List[float] = []

    with output_path.open("w", encoding="utf-8") as f_out:
        for rec_id, rel_set in relevant.items():
            if rec_id not in retrieved:
                continue
            matched_ids += 1
            ret_set = retrieved[rec_id]
            tp = len(rel_set.intersection(ret_set))
            ret_count = len(ret_set)
            rel_count = len(rel_set)

            precision = _safe_div(tp, ret_count)
            recall = _safe_div(tp, rel_count)
            f1 = _f_beta(precision, recall, 1.0)
            f3 = _f_beta(precision, recall, 3.0)

            recall_values.append(recall)

            total_tp += tp
            total_retrieved += ret_count
            total_relevant += rel_count

            macro_precision_sum += precision
            macro_recall_sum += recall
            macro_f1_sum += f1
            macro_f3_sum += f3

            out = {
                "id": rec_id,
                "relevant_count": rel_count,
                "retrieved_count": ret_count,
                "true_positives": tp,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "f3": f3,
            }
            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")

    micro_precision = _safe_div(total_tp, total_retrieved)
    micro_recall = _safe_div(total_tp, total_relevant)
    micro_f1 = _f_beta(micro_precision, micro_recall, 1.0)
    micro_f3 = _f_beta(micro_precision, micro_recall, 3.0)

    macro_precision = _safe_div(macro_precision_sum, matched_ids)
    macro_recall = _safe_div(macro_recall_sum, matched_ids)
    macro_f1 = _safe_div(macro_f1_sum, matched_ids)
    macro_f3 = _safe_div(macro_f3_sum, matched_ids)

    summary = {
        "matched_ids": matched_ids,
        "total_relevant": total_relevant,
        "total_retrieved": total_retrieved,
        "true_positives": total_tp,
        "micro": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
            "f3": micro_f3,
        },
        "macro": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
            "f3": macro_f3,
        },
    }

    with summary_path.open("w", encoding="utf-8") as f_sum:
        json.dump(summary, f_sum, ensure_ascii=False, indent=2)

    print(f"Matched ids: {matched_ids}")
    print(f"Per-id results: {output_path}")
    print(f"Summary: {summary_path}")

    if recall_values:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 5))
        plt.hist(
            recall_values, bins=30, range=(0, 1), color="#4C78A8", edgecolor="white"
        )
        plt.title("Recall Distribution")
        plt.xlabel("Recall")
        plt.ylabel("Count")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Recall plot saved to: {plot_path}")
    else:
        print("Recall plot skipped (no matched ids).")


if __name__ == "__main__":
    main()
