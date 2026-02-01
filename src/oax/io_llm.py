from pydantic import BaseModel, Field
from typing import List, Optional, Union


class LLMQueryItem(BaseModel):
    boolean_query_string: str
    database_source: Optional[Union[str, List[str]]] = None


class LLMInput(BaseModel):
    queries: List[LLMQueryItem] = Field(default_factory=list)
    keywords: Optional[List[str]] = None


class LLMOutput(BaseModel):
    oax_boolean_queries: List[Optional[str]] = Field(
        default_factory=list,
        description="List of OpenAlex /works query fragments in input order.",
    )

   

