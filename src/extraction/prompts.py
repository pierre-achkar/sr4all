SYSTEM_PROMPT = """
You are a specialist information extraction assistant specialising in extracting key information from research papers, 
focusing particularly on systematic reviews. You prioritise accuracy and verifiability, ensuring that all extracted data is directly supported by the source text.
"""

USER_TEMPLATE_RAW = """
# CONTEXT
For building structured datasets from systematic reviews, we need to accurately extract specific methodological details related to their research objectives and search strategies.
You will be given a plain text in markdown as input, which represents the full content of a systematic review paper and serves as the only source of truth for extraction.

# TASK
Your task is to extract the information stated explicitly in the text to a fixed JSON schema. 
Precision and verifiability are more important than completeness. AVOID inferring, normalising or paraphrasing content beyond what is clearly stated in the text.
Each extracted field MUST be accompanied by a verbatim source span copied directly from the text. The verbatim source should be long enough to clearly justify the extracted value.
If information is missing, ambiguous or only partially specified, return the appropriate empty or null value, as defined below.


# INSTRUCTIONS
- Extract information ONLY if it is clearly stated or directly derived from the text. AVOID inferring or guessing missing details.
- Extract the value exactly as it appears, but you may normalize formatting (e.g., "two hundred" -> 200).
- Preserve the original order of list items as they appear in the document.
- All schema keys MUST be present. Never skip fields.
- Numeric fields must be integers. If the text says "approx 200", extract 200.
- AVOID summing sub-counts (e.g. "10 from A, 20 from B") unless a total is explicitly reported
- Missing Information: If specific information for a field is not mentioned or ambiguous in the text, return 'null'.
- Specific Rules for Boolean Queries:
    1. Specific Databases: If a query is explicitly linked to a database (e.g., "PubMed: (A AND B)"), list that database.
    2. Generic Query: If a query is generic or applied to all, set `database_source` to null.
    3. Referenced Queries: If queries are defined in steps where later queries reference earlier ones (e.g. "1. Query A", "2. Query B", "3. #1 AND #2"), extract each step as a separate entry. Extract the string exactly as written (e.g., "1 AND 2") so it matches the source document verbatim.
- Return ONLY the final JSON; NEVER include explanations, checklists, summaries or validation text.

# TARGET FIELDS
- Objective: The primary aim of the systematic review (e.g., "to evaluate the effectiveness of X on Y").
- Research Questions: Explicitly stated research questions (e.g., "What is the impact of X on Y?").
- Search Strategy:
    - n_studies_initial: Number of studies initially identified (e.g., 1234).
    - n_studies_final: Number of studies included after screening (e.g., 56).
    - year_range: Temporal search window (e.g., "2001–2022").
    - snowballing: Whether citation chasing, reference checking, or backward/forward snowballing is mentioned (e.g., true or false).
    - keywords_used: Keywords reported by the authors (e.g., ["keyword1", "keyword2"]).
    - exact_boolean_queries:
        - boolean_query_string: Exact Boolean query as written (e.g., "(X OR Y) AND Z").
        - database_source: Database where the query was used (e.g., "PubMed").
        - verbatim_source: Exact text span supporting this query (e.g., "The following search was conducted in PubMed: (X OR Y) AND Z").
    - databases_used: Databases searched (e.g., ["PubMed", "Scopus"]).
    - inclusion_criteria: Inclusion rules (e.g., ["studies published after 2000", "randomized controlled trials"]).
    - exclusion_criteria: Exclusion rules (e.g., ["non-English studies", "case reports"]).

Each field must include a verbatim_source copied directly from the text that justifies the extracted value.
For list-valued fields, use one shared verbatim_source representing the entire list.

# OUTPUT FORMAT
Return only the following JSON structure:

{
"objective": {
    "value": <string or null>,
    "verbatim_source": <string or null>
},
"research_questions": {
    "value": <list of strings or null>,
    "verbatim_source": <string or null>
},
"n_studies_initial": {
    "value": <int or null>,
    "verbatim_source": <string or null>
},
"n_studies_final": {
    "value": <int or null>,
    "verbatim_source": <string or null>
},
"year_range": {
    "value": <string or null>,
    "verbatim_source": <string or null>
},
"snowballing": {
    "value": <true or false>,
    "verbatim_source": <string or null>
},
"keywords_used": {
    "value": <list of strings or null>,
    "verbatim_source": <string or null>
},
"exact_boolean_queries": [
    {
    "boolean_query_string": <string or null>,
    "database_source": <list of strings or null>,
    "verbatim_source": <string or null>
    }
],
"databases_used": {
    "value": <list of strings or null>,
    "verbatim_source": <string or null>
},
"inclusion_criteria": {
    "value": <list of strings or null>,
    "verbatim_source": <string or null>
},
"exclusion_criteria": {
    "value": <list of strings or null>,
    "verbatim_source": <string or null>
    }
}

# INPUT TEXT
{TEXT}

# OUTPUT
"""
