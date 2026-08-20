"""
Pydantic schema for search-strategy extraction. Drives vLLM constrained decoding,
so this file — not the prompt — is what the model is actually forced to emit.

Two properties matter and are easy to lose:
  * Property ORDER. Grammar backends emit keys in schema order, so verbatim_source
    must be declared before value in every model, or the prompt's evidence-first
    instruction has no effect.
  * REQUIREDNESS. A field with a default is not in `required`, and the grammar then
    permits omitting it. No field here carries a default.

Cross-field checks (value present but no span) are deliberately NOT validators:
raising would discard a whole document's extraction over one bad field. Use
fields_missing_source() in the verbatim-check stage instead.
"""

from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr

T = TypeVar("T")


# -----------------------------------------------------------------------------
# 1. Evidence anchoring
# -----------------------------------------------------------------------------
class Evidence(BaseModel, Generic[T]):
    """Extracted value plus the span that supports it. Span is emitted first."""

    model_config = ConfigDict(extra="forbid")

    verbatim_source: Optional[StrictStr] = Field(
        description="Contiguous exact substring of the document supporting the value. "
        "Null if the field is not reported."
    )
    value: Optional[T] = Field(
        description="The extracted value. Null if not reported, ambiguous, or only "
        "partially specified."
    )


class BoolEvidence(BaseModel):
    """Snowballing: absence is a finding (false), not a null."""

    model_config = ConfigDict(extra="forbid")

    verbatim_source: Optional[StrictStr] = Field(
        description="Span reporting the method. Null when value is false."
    )
    value: StrictBool = Field(
        description="True only if reported as a search method for this review."
    )


class BooleanQueryItem(BaseModel):
    """One printed query, or one numbered step of a step-wise strategy."""

    model_config = ConfigDict(extra="forbid")

    verbatim_source: StrictStr = Field(
        description="Span containing the query exactly as printed."
    )
    boolean_query_string: StrictStr = Field(
        description="The query copied character for character."
    )
    database_source: Optional[List[StrictStr]] = Field(
        description="Databases the query is explicitly attached to, or null if generic."
    )


# -----------------------------------------------------------------------------
# 2. Root schema — declaration order must match the prompt's output block
# -----------------------------------------------------------------------------
class ReviewExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    objective: Evidence[StrictStr]
    research_questions: Evidence[List[StrictStr]]
    n_studies_initial: Evidence[StrictInt]
    n_studies_final: Evidence[StrictInt]
    year_range: Evidence[StrictStr]
    snowballing: BoolEvidence
    keywords_used: Evidence[List[StrictStr]]
    exact_boolean_queries: List[BooleanQueryItem]
    databases_used: Evidence[List[StrictStr]]
    inclusion_criteria: Evidence[List[StrictStr]]
    exclusion_criteria: Evidence[List[StrictStr]]


# -----------------------------------------------------------------------------
# 3. Stage-2 helper: report, do not raise
# -----------------------------------------------------------------------------
_LIST_FIELDS = {
    "research_questions", "keywords_used", "databases_used",
    "inclusion_criteria", "exclusion_criteria",
}


def _has_data(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, str):
        return bool(v.strip())
    if isinstance(v, list):
        return len(v) > 0
    return True


def fields_missing_source(extraction: Dict[str, Any]) -> List[str]:
    """Field names that carry a value but no supporting span. Never raises."""
    bad = []
    for name, field in extraction.items():
        if name == "exact_boolean_queries":
            bad += [
                f"exact_boolean_queries[{i}]"
                for i, q in enumerate(field or [])
                if _has_data(q.get("boolean_query_string"))
                and not _has_data(q.get("verbatim_source"))
            ]
            continue
        if not isinstance(field, dict):
            continue
        if name == "snowballing":
            if field.get("value") is True and not _has_data(field.get("verbatim_source")):
                bad.append(name)
            continue
        if _has_data(field.get("value")) and not _has_data(field.get("verbatim_source")):
            bad.append(name)
    return bad