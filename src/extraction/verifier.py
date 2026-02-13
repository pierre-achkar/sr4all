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
from rapidfuzz import fuzz

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
    def __init__(self, threshold: int = 80, min_len: int = 10):
        """
        Args:
            threshold (int): Minimum alignment score (0-100) to accept a match.
            min_len (int): Minimum quote length to trigger fuzzy matching.
        """
        self.threshold = threshold
        self.min_len = min_len

    def _clean_ocr(self, text: str) -> str:
        """Normalizes OCR text (fixes hyphens, collapses whitespace)."""
        if not text:
            return ""
        text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
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
                    # THE NUKE: Set failed fields to None in the cleaned copy
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

        # 1. Pass if empty/null (nothing to verify)
        if not quote:
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
        score = fuzz.partial_ratio(clean_quote, clean_doc)
        if score >= self.threshold:
            return True

        errors.append(f"[{path}] Verbatim mismatch ({score:.1f}%): '{quote[:30]}...'")
        return False
