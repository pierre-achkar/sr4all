"""Alignment verifier: does each extracted span actually occur in the OCR text?

Design notes (rewrite):
  * THREE outcomes per span, not two. Only NOT_FOUND is evidence of a bad
    extraction. NOISY means the span was located but the local text is garbled;
    the value is kept and annotated so the downstream fact-check can arbitrate.
    Nulling noisy spans discards recoverable extractions.
  * Scores are LOCALIZED. partial_ratio against a whole 200k-character document
    is length-blind in both directions: long spans fail on one garble, short
    spans match something by construction. We locate the best window with
    partial_ratio_alignment, then score ratio(quote, window).
  * Thresholds are BANDED by span length, since longer spans tolerate more
    absolute noise at the same semantic fidelity.
  * No document-level all-or-nothing. is_valid means "no NOT_FOUND span";
    per-field decisions are what downstream code should read.
  * token_set_ratio against the whole document is gone. It compares token SETS,
    so it returned ~100 for almost any quote and silently lowered the effective
    threshold by 10 points while contributing nothing but runtime.
"""

import copy
import logging
import re
from dataclasses import dataclass, field as dc_field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

try:
    from rapidfuzz import fuzz

    HAS_RAPIDFUZZ = True
except ImportError:  # pragma: no cover - fallback path
    HAS_RAPIDFUZZ = False

logger = logging.getLogger("AlignmentVerifier")

# --------------------------------------------------------------------------- #
# Decisions
# --------------------------------------------------------------------------- #
VERIFIED = "VERIFIED"          # span located, alignment good
NOISY = "NOISY"                # span located, local text garbled: keep, annotate
NOT_FOUND = "NOT_FOUND"        # no plausible location: null the value
MISSING_SOURCE = "MISSING_SOURCE"  # value present, no span: null the value
SPAN_TOO_SHORT = "SPAN_TOO_SHORT"  # span too short to ground anything
EMPTY = "EMPTY"                # nothing extracted, nothing to verify

NULLING_DECISIONS = {NOT_FOUND, MISSING_SOURCE, SPAN_TOO_SHORT}

# (min_span_chars, verified_at, noisy_at) evaluated longest-first.
# Recall-oriented defaults: alignment is only a provenance check before the
# downstream fact-check stage. Located OCR-degraded spans are retained as NOISY;
# only clearly poor matches are nulled.
DEFAULT_BANDS: Tuple[Tuple[int, float, float], ...] = (
    (200, 85.0, 55.0),
    (80, 90.0, 60.0),
    (0, 95.0, 70.0),
)
MIN_SPAN_CHARS = 8


@dataclass
class SpanCheck:
    path: str
    decision: str
    score: float
    quote_len: int
    quote: Optional[str] = None
    window: Optional[str] = None
    window_start: Optional[int] = None
    band: Optional[str] = None
    note: Optional[str] = None

    def row(self, doc_id: str) -> Dict[str, Any]:
        """Flat record for the calibration dump."""
        return {
            "doc_id": doc_id, "path": self.path, "decision": self.decision,
            "score": round(self.score, 2), "quote_len": self.quote_len,
            "band": self.band, "quote": self.quote, "window": self.window,
            "window_start": self.window_start, "note": self.note,
            "label": None,  # to be filled by hand during calibration
        }


@dataclass
class VerificationResult:
    is_valid: bool                  # no NOT_FOUND span
    score: float                    # verified / spans actually checked
    cleaned_data: Dict
    checks: List[SpanCheck] = dc_field(default_factory=list)
    counts: Dict[str, int] = dc_field(default_factory=dict)
    nulled_fields: List[str] = dc_field(default_factory=list)
    errors: List[str] = dc_field(default_factory=list)


# --------------------------------------------------------------------------- #
def _is_empty(v: Any) -> bool:
    """Careful: `0 == False` in Python, so never test membership in [None, [], False]."""
    if v is None:
        return True
    if isinstance(v, bool):
        return v is False
    if isinstance(v, str):
        return not v.strip()
    if isinstance(v, (list, dict)):
        return len(v) == 0
    return False


class AlignmentVerifier:
    def __init__(
        self,
        bands: Tuple[Tuple[int, float, float], ...] = DEFAULT_BANDS,
        min_span_chars: int = MIN_SPAN_CHARS,
        null_noisy: bool = False,
        keep_window_chars: int = 300,
    ):
        self.bands = bands
        self.min_span_chars = min_span_chars
        self.null_noisy = null_noisy          # leave False: stage 3 arbitrates
        self.keep_window_chars = keep_window_chars

    # ------------------------------------------------------------ normalizing
    _LIGATURES = {
        "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
        "\u2212": "-", "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl",
    }

    def clean(self, text: str) -> str:
        """
        Applied symmetrically to quote and document. The prompt tells the model to
        copy LaTeX residue and reference markers verbatim, so both sides must be
        stripped of them or long spans lose points for artifacts alone.
        """
        if not text:
            return ""
        text = text.lower().translate(str.maketrans(self._LIGATURES))
        text = re.sub(r"<[^>]{1,200}>", " ", text)              # html tags
        text = re.sub(r"\$[^$]{0,200}?\$", " ", text)           # inline latex
        text = re.sub(r"\\[a-z]+\{?|\}", " ", text)             # stray latex commands
        text = re.sub(r"\^?\[\d+(?:[-,–]\s*\d+)*\]", " ", text)  # [12], [21-33]
        text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)      # hyphenation
        text = re.sub(r"[|*#>]+", " ", text)                    # markdown furniture
        text = re.sub(r"\s*/\s*", "/", text)
        return " ".join(text.split()).strip()

    # ---------------------------------------------------------------- scoring
    def _band_for(self, n: int) -> Tuple[float, float, str]:
        for min_len, verified_at, noisy_at in self.bands:
            if n >= min_len:
                return verified_at, noisy_at, f">={min_len}"
        return 100.0, 100.0, "unbanded"

    def _locate(self, quote: str, doc: str) -> Tuple[float, str, Optional[int]]:
        """Best-matching window in doc, and ratio(quote, window). Localized."""
        if HAS_RAPIDFUZZ:
            al = fuzz.partial_ratio_alignment(quote, doc)
            if al is None:
                return 0.0, "", None
            window = doc[al.dest_start : al.dest_end]
            return fuzz.ratio(quote, window), window, al.dest_start
        # fallback: slide over candidate offsets anchored on matching blocks
        matcher = SequenceMatcher(None, quote, doc, autojunk=False)
        best, best_win, best_at = 0.0, "", None
        for a, b, _size in matcher.get_matching_blocks():
            start = max(b - a, 0)
            window = doc[start : start + len(quote)]
            if not window:
                continue
            s = SequenceMatcher(None, quote, window).ratio() * 100
            if s > best:
                best, best_win, best_at = s, window, start
        return best, best_win, best_at

    def _check_span(self, path: str, quote: Optional[str], has_value: bool,
                    clean_doc: str) -> SpanCheck:
        if not quote or not quote.strip():
            if has_value:
                return SpanCheck(path, MISSING_SOURCE, 0.0, 0,
                                 note="value present but span is null")
            return SpanCheck(path, EMPTY, 0.0, 0)

        cq = self.clean(quote)
        n = len(cq)
        if n < self.min_span_chars:
            return SpanCheck(path, SPAN_TOO_SHORT, 0.0, n, quote=quote,
                             note=f"span shorter than {self.min_span_chars} chars")

        verified_at, noisy_at, band = self._band_for(n)
        if cq in clean_doc:
            return SpanCheck(path, VERIFIED, 100.0, n, quote=quote, band=band,
                             window_start=clean_doc.find(cq), note="exact")

        score, window, at = self._locate(cq, clean_doc)
        decision = (
            VERIFIED if score >= verified_at
            else NOISY if score >= noisy_at
            else NOT_FOUND
        )
        return SpanCheck(
            path, decision, score, n, quote=quote,
            window=window[: self.keep_window_chars], window_start=at, band=band,
        )

    # ------------------------------------------------------------- traversal
    def verify(self, data: Dict, ocr_text: str, doc_id: str = "") -> VerificationResult:
        clean_doc = self.clean(ocr_text)
        cleaned = copy.deepcopy(data)
        checks: List[SpanCheck] = []
        nulled: List[str] = []
        errors: List[str] = []

        if not clean_doc:
            return VerificationResult(False, 0.0, cleaned, errors=["EMPTY_DOCUMENT"])

        self._walk("", cleaned, clean_doc, checks, nulled, errors)

        counts: Dict[str, int] = {}
        for c in checks:
            counts[c.decision] = counts.get(c.decision, 0) + 1
        graded = [c for c in checks if c.decision != EMPTY]
        score = (
            sum(1 for c in graded if c.decision == VERIFIED) / len(graded)
            if graded else 1.0
        )
        return VerificationResult(
            is_valid=counts.get(NOT_FOUND, 0) == 0,
            score=score, cleaned_data=cleaned, checks=checks, counts=counts,
            nulled_fields=nulled, errors=errors,
        )

    def _walk(self, path: str, item: Any, clean_doc: str, checks: List[SpanCheck],
              nulled: List[str], errors: List[str]) -> None:
        if isinstance(item, dict):
            if "verbatim_source" in item:
                is_query = "boolean_query_string" in item
                payload = (
                    item.get("boolean_query_string") if is_query else item.get("value")
                )
                chk = self._check_span(
                    path or "root", item.get("verbatim_source"),
                    not _is_empty(payload), clean_doc,
                )

                # A query is supposed to be copied character for character, so it
                # should itself occur in the document. Cheap extra precision signal.
                if is_query and chk.decision in (VERIFIED, NOISY) and payload:
                    cq = self.clean(str(payload))
                    if len(cq) >= 8 and cq not in clean_doc:
                        q_score, _w, _a = self._locate(cq, clean_doc)
                        if q_score < 85.0:
                            chk.note = (
                                f"query string not found in document "
                                f"(best {q_score:.1f}%)"
                            )
                            chk.decision = NOISY if chk.decision == VERIFIED else chk.decision

                checks.append(chk)
                if chk.decision in NULLING_DECISIONS or (
                    self.null_noisy and chk.decision == NOISY
                ):
                    if is_query:
                        item["boolean_query_string"] = None
                        item["database_source"] = None
                    elif "value" in item:
                        item["value"] = None
                    item["verbatim_source"] = None
                    item["verification"] = chk.decision
                    nulled.append(chk.path)
                    errors.append(f"[{chk.path}] {chk.decision} ({chk.score:.1f}%)")
                else:
                    item["verification"] = chk.decision
                    item["verification_score"] = round(chk.score, 1)

            for k, v in item.items():
                if k not in ("verbatim_source", "verification", "verification_score"):
                    self._walk(f"{path}.{k}" if path else k, v, clean_doc,
                               checks, nulled, errors)

        elif isinstance(item, list):
            for i, sub in enumerate(item):
                self._walk(f"{path}[{i}]", sub, clean_doc, checks, nulled, errors)