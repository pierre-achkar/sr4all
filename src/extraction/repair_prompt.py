"""
Repair Prompt Templates
----------------------
Generates focused prompts that direct the LLM's attention to missing fields.
"""

from typing import List

# -----------------------------------------------------------------------------
# 1. SYSTEM PROMPT (The Identity)
# -----------------------------------------------------------------------------
REPAIR_SYSTEM_PROMPT = """
You are a Quality Assurance and Data Recovery Specialist for systematic reviews analysis.
Your specialty is to recover missing methodological details that were missed in a previous extraction pass.
You prioritize precision above all else: if a specific detail is not present in the text, you must explicitly acknowledge it as null rather than guessing or inferring.
"""

# -----------------------------------------------------------------------------
# 2. USER PROMPT TEMPLATES (The Task)
# -----------------------------------------------------------------------------
BASE_INSTRUCTIONS = """
# CONTEXT
For building structured datasets from systematic reviews, we need to accurately extract specific methodological details related to their research objectives and search strategies.
You will be given a plain text in markdown as input, which represents the full content of a systematic review paper and serves as the only source of truth for extraction, 
as well as a list of fields that were previously missing or invalid.
We previously analyzed this document, but the following fields were missing or invalid:
{MISSING_KEYS_LIST}

# TASK
Re-read the provided text specifically looking for these missing details.
Extract the information stated explicitly in the text to a fixed JSON schema. 
Precision and verifiability are more important than completeness. AVOID inferring, normalising or paraphrasing content beyond what is clearly stated in the text.
Each extracted field MUST be accompanied by a verbatim source span copied directly from the text. The verbatim source should be long enough to clearly justify the extracted value.
If information is missing, ambiguous or only partially specified, return the appropriate empty or null value, as defined below.

# INSTRUCTIONS
1. FOCUS: Search the text ONLY for the "TARGET FIELDS" listed below. Ignore other sections.
2. EXTRACTION:
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
3. OUTPUT:
   - You must output the full JSON object structure.
   - LEAVE non-requested fields as null.
   - Return ONLY the final JSON; NEVER include explanations, checklists, summaries or validation text.


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
"""

# Modular definitions to insert dynamically
FIELD_INSTRUCTIONS = {
    "objective": 'objective: The primary aim of the systematic review (e.g., "to evaluate the effectiveness of X on Y").',
    "research_questions": 'research_questions: Explicitly stated research questions (e.g., "What is the impact of X on Y?").',
    "n_studies_initial": "n_studies_initial: Number of studies initially identified (e.g., 1234).",
    "n_studies_final": "n_studies_final: Number of studies included after screening (e.g., 56).",
    "year_range": 'year_range: Temporal search window (e.g., "2001–2022").',
    "snowballing": "snowballing: Whether citation chasing, reference checking, or backward/forward snowballing is mentioned (e.g., true or false)",
    "keywords_used": 'keywords_used: Keywords reported by the authors (e.g., ["keyword1", "keyword2"]).',
    "databases_used": 'databases_used: Databases searched (e.g., ["PubMed", "Scopus"]).',
    "inclusion_criteria": 'inclusion_criteria: Inclusion rules (e.g., ["studies published after 2000", "randomized controlled trials"]).',
    "exclusion_criteria": 'exclusion_criteria: Exclusion rules (e.g., ["non-English studies", "case reports"]).',
    "exact_boolean_queries": """exact_boolean_queries: A list of search query objects. 
    Each object must have:
      - 'boolean_query_string': Exact Boolean query as written (e.g., "(X OR Y) AND Z").
      - 'database_source': Database where the query was used (e.g., "PubMed").
      - 'verbatim_source': Exact text span supporting this query (e.g., "The following search was conducted in PubMed: (X OR Y) AND Z").""",
}


# -----------------------------------------------------------------------------
# 3. FACTORY FUNCTION
# -----------------------------------------------------------------------------
def get_repair_user_prompt(doc_text: str, missing_keys: List[str]) -> str:
    """
    Constructs the dynamic USER prompt based on missing keys.
    """
    # 1. Select specific field definitions
    specific_instrs = []
    for key in missing_keys:
        if key in FIELD_INSTRUCTIONS:
            specific_instrs.append(f"- {FIELD_INSTRUCTIONS[key]}")

    # Fallback (Safety net)
    if not specific_instrs:
        specific_instrs = ["- Review the text for the missing fields."]

    # 2. Format the Prompt
    user_prompt = f"""
    {BASE_INSTRUCTIONS.replace("{MISSING_KEYS_LIST}", ", ".join(missing_keys))}

    # TARGET FIELDS (FOCUS HERE)
    {chr(10).join(specific_instrs)}

    # INPUT TEXT
    {doc_text}

    # OUTPUT
    """
    return user_prompt


# -----------------------------------------------------------------------------


def main():
    print("Running Test Repair Prompt Generation...")

    FILE_PATH = "/home/fhg/pie65738/projects/sr4all/test.json"
    # Load Test Record
    with open(FILE_PATH, "r") as f:
        import json

        TEST_RECORD = json.load(f)

    # 1. Detect Missing Keys
    data = TEST_RECORD["extraction"]
    missing = []

    for k, v in data.items():
        # Case A: Simple Null
        if v is None:
            missing.append(k)

        # Case B: Evidence Node Null
        elif isinstance(v, dict) and "value" in v and v["value"] is None:
            missing.append(k)

        # Case C: Complex List (exact_boolean_queries)
        elif k == "exact_boolean_queries":
            if v is None or not isinstance(v, list):
                missing.append(k)
            elif len(v) == 0:
                missing.append(k)
            else:
                # Check ghost object
                first_item = v[0]
                if (
                    isinstance(first_item, dict)
                    and first_item.get("boolean_query_string") is None
                ):
                    missing.append(k)

    print(f"\n[Detection Result] Found {len(missing)} missing keys:")
    print(missing)

    # 2. Generate Prompt
    # We use a fake text since we don't have the file
    fake_doc_text = "FAKE DOCUMENT TEXT CONTENT: This is a study about IER..."

    prompt = get_repair_user_prompt(fake_doc_text, missing)

    print("\n" + "=" * 80)
    print("GENERATED PROMPT")
    print("=" * 80)
    print(prompt)
    print("=" * 80)

    # Validation
    expected_missing = [
        "research_questions",
        "n_studies_initial",
        "year_range",
        "keywords_used",
        "exact_boolean_queries",  # Should be caught by Case C
        "inclusion_criteria",
        "exclusion_criteria",
    ]

    # snowballing is False (not None), so it should NOT be missing
    # objective and n_studies_final have values, so NOT missing

    assert set(missing) == set(
        expected_missing
    ), f"Mismatch! Expected {expected_missing}, got {missing}"
    print(
        "\n✅ TEST PASSED: Detection logic correctly identified all missing fields (including the complex list)."
    )


if __name__ == "__main__":
    main()
