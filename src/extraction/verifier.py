"""
Alignment Verifier Module (Pipeline Ready).

This module validates that extracted quotes exist in the noisy OCR source text.
It returns a VerificationResult object containing stats and a CLEANED version of the data.
"""

import logging
import re
import copy
from typing import Dict, List, Any
from dataclasses import dataclass
from difflib import SequenceMatcher

try:
    from rapidfuzz import fuzz
except ImportError:
    class _FallbackFuzz:
        @staticmethod
        def _ratio(left: str, right: str) -> float:
            return SequenceMatcher(None, left, right).ratio() * 100

        @classmethod
        def partial_ratio(cls, left: str, right: str) -> float:
            if len(left) > len(right):
                left, right = right, left
            if not left:
                return 100.0
            if left in right:
                return 100.0

            matcher = SequenceMatcher(None, left, right, autojunk=False)
            best = 0.0
            for block in matcher.get_matching_blocks():
                start = max(block[1] - block[0], 0)
                window = right[start : start + len(left)]
                if not window:
                    continue
                best = max(best, cls._ratio(left, window))
            return best

        @classmethod
        def token_set_ratio(cls, left: str, right: str) -> float:
            left_tokens = set(left.split())
            right_tokens = set(right.split())
            if not left_tokens or not right_tokens:
                return 0.0

            common = left_tokens & right_tokens
            left_diff = left_tokens - common
            right_diff = right_tokens - common
            common_text = " ".join(sorted(common))
            left_text = " ".join(sorted(common | left_diff))
            right_text = " ".join(sorted(common | right_diff))
            return max(
                cls._ratio(common_text, left_text),
                cls._ratio(common_text, right_text),
                cls._ratio(left_text, right_text),
            )

    fuzz = _FallbackFuzz()

# Setup Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("AlignmentVerifier")


@dataclass
class VerificationResult:
    is_valid: bool
    score: float  # 0.0 to 1.0
    errors: List[str]
    cleaned_data: Dict  # Data with invalid fields set to None


class AlignmentVerifier:
    def __init__(self, threshold: int = 65, min_len: int = 5):
        """
        Args:
            threshold (int): Minimum alignment score (0-100) to accept a match.
            min_len (int): Minimum quote length to trigger fuzzy matching.
        """
        self.threshold = threshold
        self.min_len = min_len

    def _clean_ocr(self, text: str) -> str:
        """Normalizes OCR/layout noise before quote alignment."""
        if not text:
            return ""
        text = text.lower()
        text = text.translate(
            str.maketrans(
                {
                    "\u2010": "-",
                    "\u2011": "-",
                    "\u2012": "-",
                    "\u2013": "-",
                    "\u2014": "-",
                    "\u2212": "-",
                    "\u2018": "'",
                    "\u2019": "'",
                    "\u201c": '"',
                    "\u201d": '"',
                    "\ufb00": "ff",
                    "\ufb01": "fi",
                    "\ufb02": "fl",
                    "\ufb03": "ffi",
                    "\ufb04": "ffl",
                }
            )
        )
        text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
        text = re.sub(r"\s*\|\s*", " ", text)
        text = re.sub(r"\s*/\s*", "/", text)
        return " ".join(text.split()).strip()

    def verify(self, data: Dict, ocr_text: str) -> VerificationResult:
        """
        Verifies alignment and returns stats + a cleaned copy of the data.
        """
        clean_doc = self._clean_ocr(ocr_text)

        # Working copy to modify (nullify bad fields)
        cleaned_data = copy.deepcopy(data)
        errors = []

        # Stats counters
        self.total_quotes = 0
        self.failed_quotes = 0

        if not clean_doc:
            # Fail safely if doc is empty
            return VerificationResult(False, 0.0, ["Empty Document"], cleaned_data)

        # Recursive check & clean
        self._check_and_clean("root", cleaned_data, clean_doc, errors)

        # Calculate Score
        score = 1.0
        if self.total_quotes > 0:
            score = (self.total_quotes - self.failed_quotes) / self.total_quotes

        return VerificationResult(
            is_valid=(self.failed_quotes == 0),
            score=score,
            errors=errors,
            cleaned_data=cleaned_data,
        )

    def _check_and_clean(self, path: str, item: Any, clean_doc: str, errors: List[str]):
        """Recursively traverses JSON, checking and nuking invalid evidence."""
        if isinstance(item, dict):
            # Is this an Evidence Node? (Has value + verbatim_source)
            if "verbatim_source" in item:
                self.total_quotes += 1
                is_valid = self._verify_field(path, item, clean_doc, errors)

                if not is_valid:
                    self.failed_quotes += 1
                    # THE NUKE: Set failed fields to None in the cleaned copy.
                    # Boolean query objects have no generic `value` key, so clean
                    # their actual payload fields instead of leaving an unsupported
                    # query string behind.
                    if "boolean_query_string" in item:
                        item["boolean_query_string"] = None
                        item["database_source"] = None
                    elif "value" in item:
                        item["value"] = None
                    item["verbatim_source"] = None

            # Recurse
            for k, v in item.items():
                if k != "verbatim_source":
                    self._check_and_clean(f"{path}.{k}", v, clean_doc, errors)

        elif isinstance(item, list):
            for i, sub in enumerate(item):
                self._check_and_clean(f"{path}[{i}]", sub, clean_doc, errors)

    def _verify_field(
        self, path: str, item: Dict, clean_doc: str, errors: List[str]
    ) -> bool:
        """Performs the check for a single field."""
        quote = item.get("verbatim_source")
        val = item.get("value")
        boolean_query = item.get("boolean_query_string")

        # 1. Pass if empty/null (nothing to verify)
        if not quote:
            # Boolean query objects do not use the generic `value` key, but they
            # still require a source span to be evidence anchored.
            if boolean_query not in [None, "", []]:
                errors.append(
                    f"[{path}] Boolean query '{boolean_query}' exists but source is null."
                )
                return False

            # If value exists but source is missing, that's a fail (unless value is also null)
            if val not in [None, [], False]:
                errors.append(f"[{path}] Value '{val}' exists but source is null.")
                return False
            return True

        clean_quote = self._clean_ocr(quote)

        # 2. Short quote check (Exact Match Required)
        if len(clean_quote) < self.min_len:
            if clean_quote in clean_doc:
                return True
            errors.append(f"[{path}] Short quote '{clean_quote}' not found exactly.")
            return False

        # 3. Exact match (Fast)
        if clean_quote in clean_doc:
            return True

        # 4. Fuzzy match (Slow & Robust)
        partial_score = fuzz.partial_ratio(clean_quote, clean_doc)
        token_score = fuzz.token_set_ratio(clean_quote, clean_doc)
        token_threshold = max(self.threshold + 20, 85)
        score = max(partial_score, token_score)
        if partial_score >= self.threshold or (
            partial_score >= self.threshold - 10 and token_score >= token_threshold
        ):
            return True

        errors.append(
            f"[{path}] Verbatim mismatch (partial={partial_score:.1f}%, token={token_score:.1f}%): '{quote[:30]}...'"
        )
        return False
