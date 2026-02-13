from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class RawQueryItem(BaseModel):
    id: str = Field(..., description="Unique ID to track this query line.")
    raw_string: str = Field(..., description="The original messy query string.")


class TransformationInput(BaseModel):
    queries: List[RawQueryItem] = Field(
        default_factory=list, description="Optional list of raw boolean query strings."
    )
    keywords: Optional[List[str]] = Field(
        None, description="Optional list of keywords for keyword-only normalization."
    )


class NormalizedQuery(BaseModel):
    id: str = Field(..., description="Must match the input ID.")

    # CHANGED: Now Optional. If invalid, this will be None.
    boolean_query: Optional[str] = Field(
        None,
        description="The clean, field-agnostic Boolean string. Null if input is invalid.",
    )

    # CHANGED: 'skipped' is clearer than 'empty'
    status: Literal["valid", "skipped"] = "valid"
    error_reason: Optional[str] = None


class TransformationOutput(BaseModel):
    results: List[NormalizedQuery]
