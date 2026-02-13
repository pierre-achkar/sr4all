from typing import List, Optional, TypeVar, Generic
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# 1. Generics for the "Evidence Anchoring" Pattern
# -----------------------------------------------------------------------------
T = TypeVar("T")


class Evidence(BaseModel, Generic[T]):
    """
    Container for extracted values that forces the model to cite its source.
    """

    value: Optional[T] = Field(None, description="The extracted information value.")
    verbatim_source: Optional[str] = Field(
        None,
        description="The exact span of text from the document that supports this value. Returns null if not found.",
    )


# -----------------------------------------------------------------------------
# 2. Specific Sub-Structures
# -----------------------------------------------------------------------------
class BooleanQueryItem(BaseModel):
    """
    Represents a specific boolean query string found in the text.
    """

    boolean_query_string: Optional[str] = Field(
        None, description="The exact boolean query string."
    )
    database_source: Optional[List[str]] = Field(
        None, description="The database this query was run on (e.g., PubMed)."
    )
    verbatim_source: Optional[str] = Field(
        None, description="The text span supporting this query."
    )


# -----------------------------------------------------------------------------
# 3. Root Schema (FLATTENED)
# -----------------------------------------------------------------------------
class ReviewExtraction(BaseModel):
    """
    Root schema for extracting structured methodology from a Systematic Review.
    Matches the flat JSON structure of the prompt instructions.
    """

    # --- Objective & Questions ---
    objective: Optional[Evidence[str]] = Field(
        None, description="The primary objective or aim of the systematic review."
    )
    research_questions: Optional[Evidence[List[str]]] = Field(
        None, description="Specific research questions (RQs) listed by the authors."
    )

    # --- Search Metrics ---
    n_studies_initial: Optional[Evidence[int]] = Field(
        None, description="Number of studies initially identified/retrieved."
    )
    n_studies_final: Optional[Evidence[int]] = Field(
        None, description="Number of studies included in the final review."
    )
    year_range: Optional[Evidence[str]] = Field(
        None,
        description="The range of years covered by the search (e.g., '1990-2023').",
    )

    # --- Methodology ---
    snowballing: Optional[Evidence[bool]] = Field(
        None, description="True if citation chaining (snowballing) was used."
    )
    keywords_used: Optional[Evidence[List[str]]] = Field(
        None, description="List of individual keywords reported."
    )
    databases_used: Optional[Evidence[List[str]]] = Field(
        None, description="List of databases searched (e.g., MEDLINE, PsycINFO)."
    )

    # --- Queries ---
    # List of complex objects (kept as list of dicts in JSON)
    exact_boolean_queries: Optional[List[BooleanQueryItem]] = Field(
        None, description="List of exact boolean search strings reported."
    )

    # --- Criteria ---
    inclusion_criteria: Optional[Evidence[List[str]]] = Field(
        None, description="List of specific inclusion criteria."
    )
    exclusion_criteria: Optional[Evidence[List[str]]] = Field(
        None, description="List of specific exclusion criteria."
    )
