"""
Fact Checker Module (MiniCheck Wrapper).

This module uses the 'Bespoke-MiniCheck-7B' model to verify if an extracted 'value'
is strictly supported by its 'verbatim_source'.
"""

import logging
import os
from typing import List, Dict, Any, Tuple
import nltk  # <--- Added import

# --- NLTK Auto-Fix: Ensure tokenizers are present ---
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
# ----------------------------------------------------

# Try importing MiniCheck
try:
    from minicheck.minicheck import MiniCheck
except ImportError:
    raise ImportError(
        "MiniCheck not found. Install with: "
        "pip install 'minicheck[llm] @ git+https://github.com/Liyan06/MiniCheck.git@main'"
    )

logger = logging.getLogger("FactChecker")


class FactChecker:
    def __init__(
        self,
        model_name: str = "Bespoke-MiniCheck-7B",
        batch_size: int = 128,
        cache_dir: str = "./ckpts",
    ):
        """
        Args:
            model_name (str): Model to use. Defaults to SOTA 'Bespoke-MiniCheck-7B'.
            batch_size (int): Internal chunk size for processing large lists.
            cache_dir (str): Where to store the downloaded model weights.
        """
        self.chunk_size = batch_size

        logger.info(f"Initializing FactChecker with {model_name}...")

        self.scorer = MiniCheck(
            model_name=model_name, enable_prefix_caching=True, cache_dir=cache_dir
        )
        logger.info("MiniCheck Model loaded successfully.")

    def verify_batch(self, pairs: List[Tuple[str, Any]]) -> List[Dict]:
        """
        Verifies a batch of (Source, Value) pairs.
        """
        if not pairs:
            return []

        docs = []
        claims = []

        for source, value in pairs:
            docs.append(str(source))
            if isinstance(value, list):
                # Convert list to string representation for checking
                claims.append(", ".join(map(str, value)))
            else:
                claims.append(str(value))

        results = []

        try:
            # chunking loop to avoid OOM or batch limits
            for i in range(0, len(docs), self.chunk_size):
                chunk_docs = docs[i : i + self.chunk_size]
                chunk_claims = claims[i : i + self.chunk_size]

                # Run Inference (Removed 'batch_size' arg)
                pred_labels, raw_probs, _, _ = self.scorer.score(
                    docs=chunk_docs, claims=chunk_claims
                )

                # Collect Results
                for label, prob in zip(pred_labels, raw_probs):
                    results.append(
                        {
                            "status": "PASS" if label == 1 else "FAIL",
                            "support_probability": float(prob),
                            "label": int(label),
                        }
                    )

            return results

        except Exception as e:
            logger.error(f"MiniCheck Inference Failed: {e}")
            # FAIL SAFE: Return error dicts for all inputs
            return [
                {"status": "ERROR", "support_probability": 0.0, "error": str(e)}
                for _ in pairs
            ]


# -----------------------------------------------------------------------------
# TEST HARNESS
# -----------------------------------------------------------------------------
# if __name__ == "__main__":
#     # Prevent OOM during test
#     os.environ["VLLM_GPU_MEMORY_UTILIZATION"] = "0.7"

#     print("Testing FactChecker Module...")

#     checker = FactChecker()

#     test_pairs = [
#         # Case 1: Supported
#         ("We identified 2500 records from PubMed.", 2500),

#         # Case 2: Contradicted (Hallucination)
#         ("We identified 2500 records from PubMed.", 50),

#         # Case 3: List Support
#         ("The search included Medline, Embase and Scopus.", ["Medline", "Embase"]),
#     ]

#     results = checker.verify_batch(test_pairs)

#     for (src, val), res in zip(test_pairs, results):
#         print(f"\nSource: {src}")
#         print(f"Value:  {val}")
#         print(f"Result: {res['status']} ({res['support_probability']:.4f})")
