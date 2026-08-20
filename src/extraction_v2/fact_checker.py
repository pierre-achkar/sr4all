"""
Fact checker: is an extracted value actually supported by its verbatim span?

Wraps Bespoke-MiniCheck-7B (vLLM-backed, Apache-2.0 package; see Tang et al.,
EMNLP 2024). Two design points that matter:

  * MiniCheck's interface is MiniCheck(document, SENTENCE). The README is explicit
    that it is a sentence-level model and that multi-sentence claims should be
    split first. So a bare value is off-distribution: sending the claim "2610",
    or "False", or "adults" produces a near-meaningless judgment. Every value is
    therefore VERBALIZED into a proposition before it is checked.
  * Automatic prefix caching only pays off when consecutive pairs share a
    document, so pairs are sorted by span before dispatch and restored to input
    order afterwards. One span with twelve criteria then costs one span encode.

Probabilities are always returned. The pass/fail boundary belongs to the caller,
so it can be re-tuned without a second pass over the corpus.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import nltk
import torch

# NLTK resources must be provisioned by the job environment. Do not attempt a
# network download while a production stage is importing this module.
for _res in ("tokenizers/punkt_tab", "tokenizers/punkt"):
    try:
        nltk.data.find(_res)
    except LookupError:
        raise RuntimeError(
            f"Missing NLTK resource {_res}. Set NLTK_DATA to a prepared writable cache."
        )

try:
    from minicheck.minicheck import MiniCheck
except ImportError:
    raise ImportError(
        "MiniCheck not found. Install with: "
        "pip install 'minicheck[llm] @ git+https://github.com/Liyan06/MiniCheck.git@main'"
    )

logger = logging.getLogger("FactChecker")


# --------------------------------------------------------------------------- #
# Claim verbalization
# --------------------------------------------------------------------------- #
def _year_range(value: str) -> str:
    m = re.match(r"\s*(\d{4})\s*[-–—to]+\s*(\d{4}|present)\s*$", str(value), re.I)
    if m:
        return f"The literature search covered publications from {m.group(1)} to {m.group(2)}."
    return f"The literature search was restricted to the period {value}."


VERBALIZERS = {
    "objective": lambda v: f"The objective of this review is {str(v).rstrip('.')}.",
    "research_questions": lambda v: f"This review addresses the research question: {v}",
    "n_studies_initial": lambda v: f"The database search initially identified {v} records.",
    "n_studies_final": lambda v: f"A total of {v} studies were included in this review.",
    "year_range": _year_range,
    "snowballing": lambda v: (
        "The authors searched for additional studies by citation chasing or by "
        "checking the reference lists of included studies."
    ),
    "keywords_used": lambda v: f'The search used the term "{v}".',
    "databases_used": lambda v: f"The authors searched the database {v}.",
    "inclusion_criteria": lambda v: f"Studies were eligible for inclusion if {v}.",
    "exclusion_criteria": lambda v: f"Studies were excluded if {v}.",
    # exact_boolean_queries: the query STRING is checked verbatim by the alignment
    # stage, which is the right test for a copied artifact. Entailment adds nothing
    # there, so only the database attribution is fact-checked.
    "database_source": lambda v: f"The reported search query was run in {v}.",
}


def verbalize(field: str, value: Any) -> Optional[str]:
    """Turn a field value into a single-sentence claim, or None if not checkable."""
    fn = VERBALIZERS.get(field)
    if fn is None or value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    text = " ".join(str(fn(value)).split())
    return text if len(text) > 1 else None


@dataclass
class Claim:
    """One (span, sentence) pair plus enough provenance to write the result back."""

    doc_idx: int                  # index into the record batch
    path: Tuple[Any, ...]         # location of the node inside `extraction`
    field: str
    kind: str                     # "field" | "list_item" | "query_database"
    span: str
    sentence: str
    index: Optional[int] = None   # position within a list value
    span_quality: str = "UNKNOWN"  # VERIFIED / NOISY, from the alignment stage


# --------------------------------------------------------------------------- #
class FactChecker:
    def __init__(
        self,
        model_name: str = "Bespoke-MiniCheck-7B",
        chunk_size: int = 128,
        cache_dir: str = "./ckpts",
        enable_prefix_caching: bool = True,
    ):
        self.chunk_size = chunk_size
        self.model_name = model_name
        self.enable_prefix_caching = enable_prefix_caching
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; MiniCheck requires the Slurm-assigned GPU.")
        torch.cuda.set_device(0)
        logger.info(f"Using CUDA device 0: {torch.cuda.get_device_name(0)}")
        logger.info(f"Initializing FactChecker with {model_name}...")
        self.scorer = MiniCheck(
            model_name=model_name,
            enable_prefix_caching=enable_prefix_caching,
            cache_dir=cache_dir,
        )
        logger.info("MiniCheck loaded.")

    def config(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "chunk_size": self.chunk_size,
            "enable_prefix_caching": self.enable_prefix_caching,
        }

    def score_claims(self, claims: Sequence[Claim]) -> List[Dict[str, Any]]:
        """
        Returns one result per claim, in the order given:
            {"status": "OK"|"ERROR", "support_probability": float,
             "label": int|None, "error": str|None}

        A chunk that raises marks ONLY its own claims as ERROR. The caller must
        not treat an ERROR claim as checked, or a transient GPU failure becomes
        permanent silent data loss.
        """
        if not claims:
            return []

        results: List[Optional[Dict[str, Any]]] = [None] * len(claims)
        # Group identical spans together so prefix caching can reuse the encode.
        order = sorted(range(len(claims)), key=lambda i: (claims[i].span, i))

        for start in range(0, len(order), self.chunk_size):
            idxs = order[start : start + self.chunk_size]
            docs = [claims[i].span for i in idxs]
            sents = [claims[i].sentence for i in idxs]
            try:
                labels, probs, _, _ = self.scorer.score(docs=docs, claims=sents)
                for i, label, prob in zip(idxs, labels, probs):
                    results[i] = {
                        "status": "OK",
                        "support_probability": float(prob),
                        "label": int(label),
                        "error": None,
                    }
            except Exception as e:  # noqa: BLE001
                logger.error(
                    f"MiniCheck failed on chunk {start // self.chunk_size} "
                    f"({len(idxs)} claims): {e}"
                )
                for i in idxs:
                    results[i] = {
                        "status": "ERROR",
                        "support_probability": None,
                        "label": None,
                        "error": str(e)[:300],
                    }

        return [
            result if result is not None else {
                "status": "ERROR",
                "support_probability": None,
                "label": None,
                "error": "MiniCheck returned no result for this claim",
            }
            for result in results
        ]