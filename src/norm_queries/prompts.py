from jinja2 import Template
from textwrap import dedent
import sys
from pathlib import Path

# Ensure we can import from src
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from transform_queries import schemas


class TransformerToSimplePrompts:
    SYSTEM = (
        "You are an expert information retrieval specialist. "
        "Your task is to translate raw bibliographic queries targeting different databases "
        "into syntactically correct boolean strings executable. "
        "You must preserve the intended Boolean structure (AND/OR/NOT, parentheses) as far as possible under the rules you receive, "
        "and you never invent unrelated concepts or add new topical terms. "
        "You ONLY respond with the requested JSON object and nothing else."
    )

    USER_TEMPLATE = dedent("""
        # Context
        {%- if queries%}
        You will receive one or more Boolean search strings written by experts targeting specific bibliographic databases
        (e.g., Scopus, Web of Science, Dimensions, Lens, PubMed).
        {%- endif %}
        {%- if keywords and not queries %}
        You will receive ONLY a list of search keywords used to search bibliographic databases.
        {%- endif %}

        # Task
        {%- if queries %}
        Translate the input boolean search string(s) into a clean, field-agnostic Boolean string. Ensure precise, rules-conformant transformation.
        Follow the instructions below carefully.
        {%- endif %}
        {%- if keywords and not queries %}
        Build a single clean Boolean string that logically combines the keywords. Ensure precise, rules-conformant construction.
        Follow the instructions below carefully.
        {%- endif %}

        # Instructions and Rules
        {%- if queries %}
        ## 1. Field Stripping 
        - Your output must be field-agnostic.
        - Remove specific field tags like `TITLE(...)`, `ABS(...)`, `TITLE-ABS-KEY(...)`, or `ti(...)`.
        - Keep the content inside those tags.
          - Example: `TITLE(vaccine) AND AB(hesitancy)` → `vaccine AND hesitancy`
        
        ## 2. Metadata Removal
        - Remove ALL non-topical limits.
        - Years: Remove `PUBYEAR`, `LIMIT-TO(PUBYEAR...)`, or `2010-2020`.
        - Language/Type: Remove `LIMIT-TO(LANGUAGE...)`, `DOCTYPE(...)`, `SRCTYPE(...)`.
        - Result: The output must contain ONLY the topical keywords and boolean operators.

        ## 3. Boolean Logic & Syntax
        - Operators: MUST be UPPERCASE (`AND`, `OR`, `NOT`).
        - Parentheses: Preserve the original grouping logic exactly.
        - Phrases: ALWAYS wrap multi-word terms in double quotes (e.g., `"machine learning"`).
        - Proximity: Convert `NEAR/n` or `W/n` to a logical AND.
          - Example: `("lung cancer" W/3 diagnosis)` → `"lung cancer" AND diagnosis`

        ## 4. Special Edge Cases (Numeric Sets)
        - If the input consists exclusively of numbers, set IDs (e.g., `1`, `2#`, `#3`), and operators:
          - You MUST return `boolean_query: null` and `status: "skipped"`.
          - These are references to previous search history which we do not have. They are invalid.
          - Example: `(1 OR 2) AND 3` → null.

        ## 5. Wildcards in Phrases 
        - If a phrase contains a wildcard, remove the quotes and use `AND`.
          - Input: `"auto* vehicles"`
          - Output: `auto* AND vehicles`
          - Reason: This ensures the wildcard actually functions as a prefix search.
        {%- endif %}

        {%- if keywords and not queries %}
        ## Guidelines for Keywords
        - Logic: Combine synonyms/related terms with `OR`. Combine distinct concepts with `AND`.
        - Phrases: ALWAYS wrap multi-word terms in double quotes (e.g., `"machine learning"`).
        - No Fields: Do not add any field prefixes.
        {%- endif %}

        ## General Formatting
        - Quotes: Normalize all smart quotes (Start/End) to standard ASCII double quotes `"`.
        - Wildcards: Keep asterisks `*` (e.g., `randomiz*`). Do not remove them.

        # Output (strict)
        Return ONLY a valid JSON object matching this schema:
        {
          "results": [
             {
               "id": "<matches input id>",
               "boolean_query": "<the clean, field-agnostic string>",
               "status": "valid" | "skipped",
               "error_reason": "<optional string>"
             }
          ]
        }

        # Examples
        {%- if queries %}
        ## Example 1
        Input: "\"Condition Monitoring\" OR \"Wind Turbine\" OR \"Data Mining Approaches\" OR \"Fault Diagnosis\""
        Output: { "id": "...", "boolean_query": "(\"Condition Monitoring\" OR \"Wind Turbine\" OR \"Data Mining Approaches\" OR \"Fault Diagnosis\")", "status": "valid" }

        ## Example 2
        Input: "TITLE-ABS-KEY (((indoor OR enclosed) AND (occupancy) AND (environmental OR environment) AND (sensor OR variables OR parameters)))"
        Output: { "id": "...", "boolean_query": "(((indoor OR enclosed) AND (occupancy) AND (environmental OR environment) AND (sensor OR variables OR parameters)))", "status": "valid" }

        ## Example 3
        Input: "dashboard OR whiteboard... Hospital, 1# AND 2# AND 3#"
        Output: { "id": "...", "boolean_query": "((dashboard OR whiteboard OR \"status board\" OR \"Electronic tracking board\" OR visualization OR \"presentation format\" OR \"display format\" OR \"performance measurement system\") AND (Design OR capability OR feature OR character OR attributes OR function OR usability OR content) AND (Hospital))", "status": "valid" }

        ## Example 4
        Input: "TITLE Forecast OR Predict* AND Energy OR Power OR electricity."
        Output: { "id": "...", "boolean_query": "((Forecast OR Predict*) AND (Energy OR Power OR Electricity))", "status": "valid" }

        ## Example 5
        Input: "TI = (Internet of Things OR IoT) AND TS = (Authentication OR Authorization OR Identity OR Access Control) NOT TS = (Hardware OR Cryptography OR Protocol OR RFID OR Physical OR Network) NOT TS = (Survey OR Study) AND TS = Security"
        Output: { "id": "...", "boolean_query": "((\"Internet of Things\" OR IoT) AND (Authentication OR Authorization OR Identity OR \"Access Control\") AND Security AND NOT (Hardware OR Cryptography OR Protocol OR RFID OR Physical OR Network OR Survey OR Study))", "status": "valid" }

        ## Example 6
        Input: "(\"information diffusion\") OR (\"influence analysis\") OR (\"influence maximization\") OR (\"user influence\")"
        Output: { "id": "...", "boolean_query": "(\"information diffusion\" OR \"influence analysis\" OR \"influence maximization\" OR \"user influence\")", "status": "valid" }

        ## Example 7
        Input: "(gamif* OR gameful OR \"game elements\" OR \"game mechanics\" OR \"game dynamics\" OR \"game components\" OR \"game aesthetics\") AND (education OR educational OR learning OR teaching OR course OR syllabus OR syllabi OR curriculum OR curricula) AND (framework OR method OR design OR model OR approach OR theory OR strategy)"
        Output: { "id": "...", "boolean_query": "((gamif* OR gameful OR \"game elements\" OR \"game mechanics\" OR \"game dynamics\" OR \"game components\" OR \"game aesthetics\") AND (education OR educational OR learning OR teaching OR course OR syllabus OR syllabi OR curriculum OR curricula) AND (framework OR method OR design OR model OR approach OR theory OR strategy))", "status": "valid" }

        ## Example 8
        Input: "covid-19, sars-cov-2, coronavirus, genetic variation, gene, genome-wide association study, polymorphisms, single nucleotide, genetic association, genetic susceptibility, genotype, human host, genotype, covid-19 outcome modelling, covid-19 severity modelling, machine learning for covid-19 modelling, covid-19 prediction using genomic data"
        Output: { "id": "...", "boolean_query": "((\"covid-19\" OR \"sars-cov-2\" OR coronavirus OR \"genetic variation\" OR gene OR \"genome-wide association study\" OR polymorphisms OR \"single nucleotide\" OR \"genetic association\" OR \"genetic susceptibility\" OR genotype OR \"human host\" OR \"covid-19 outcome modelling\" OR \"covid-19 severity modelling\" OR \"machine learning for covid-19 modelling\" OR \"covid-19 prediction using genomic data\"))", "status": "valid" }

        ## Example 9
        Input: "PUBYEAR > 2015 AND (TITLE(\"log\") AND TITLE-ABS-KEY(\"log analysis\")) AND (AUTHKEY(\"analysis\") OR AUTHKEY (\"retrieval\") OR AUTHKEY (\"recovery\") OR AUTHKEY (\"mining\") OR AUTHKEY (\"reverse engineering\") OR AUTHKEY (\"detection\")) AND (LIMIT-TO(SUBJAREA, \"COMP\")) AND (LIMIT-TO(LANGUAGE, \"English\"))"
        Output: { "id": "...", "boolean_query": "((log) AND (\"log analysis\") AND ((analysis) OR (retrieval) OR (recovery) OR (mining) OR (\"reverse engineering\") OR (detection)))", "status": "valid" }
        
        ## Example 10 (Wildcards in Phrases)
        Input: "warehous* AND (\"digital transformation\" OR \"technolog*\" OR \"4.0\" OR \"smart\" OR \"auto* vehicles\")"
        Output: { "id": "...", "boolean_query": "warehous* AND (\"digital transformation\" OR technolog* OR \"4.0\" OR smart OR (auto* AND vehicles))", "status": "valid" }
        
        ## Example 11 (Numeric Set Trap)
        Input: "(1 OR 2) AND (3 OR 4)"
        Output: { "id": "...", "boolean_query": null, "status": "skipped", "error_reason": "Numeric set reference" }
        {%- endif %}

        {%- if keywords and not queries %}
        ## Example 1
        Inputs:
        - \"Condition Monitoring\"
        - \"Wind Turbine\"
        - \"Data Mining Approaches\"
        - \"Fault Diagnosis\"
        
        Output: { "id": "...", "boolean_query": "(\"Condition Monitoring\" OR \"Wind Turbine\" OR \"Data Mining Approaches\" OR \"Fault Diagnosis\")", "status": "valid" }

        ## Example 2
        Inputs:
        - indoor 
        - enclosed 
        - occupancy
        - environmental
        - environment
        - sensor
        - variables
        - parameters

        Output: { "id": "...", "boolean_query": "((indoor OR enclosed) AND (occupancy) AND (environmental OR environment) AND (sensor OR variables OR parameters))", "status": "valid" }

        ## Example 3
        Inputs:
        - Forecast 
        - Predict
        - Energy
        - Power
        - electricity

        Output: { "id": "...", "boolean_query": "((Forecast OR Predict*) AND (Energy OR Power OR Electricity))", "status": "valid" }
        {%- endif %}

        # Inputs (ordered list)
        {%- if queries %}
        The following are the Boolean query string(s) to be transformed:
        {%- for q in queries %}
        Item {{ loop.index }}
        ID: {{ q.id }}
        Raw String:
        {{ q.raw_string }}
        
        {%- endfor %}
        {%- endif %}

        {%- if keywords and not queries %}
        The following are the keyword terms to be logically combined:
        {%- for k in keywords %}
        - {{ k }}
        {%- endfor %}
        {%- endif %}

        # Output
        Provide the JSON object as specified above.
        """).strip()

    @staticmethod
    def render(data: schemas.TransformationInput) -> tuple[str, str]:
        t = Template(TransformerToSimplePrompts.USER_TEMPLATE)
        return TransformerToSimplePrompts.SYSTEM, t.render(**data.model_dump())


if __name__ == "__main__":

    # # 1. Test only with queries
    # test_data = schemas.TransformationInput(
    #     queries=[
    #         schemas.RawQueryItem(
    #             id="test1",
    #             raw_string='TITLE-ABS-KEY (((indoor OR enclosed) AND (occupancy) AND (environmental OR environment) AND (sensor OR variables OR parameters)))'
    #         )
    #     ]
    # )

    # system_prompt, user_prompt = TransformerToSimplePrompts.render(test_data)
    # print("=== SYSTEM PROMPT ===")
    # print(system_prompt)
    # print("\n=== USER PROMPT ===")
    # print(user_prompt)

    # 2. Test only with keywords
    test_data = schemas.TransformationInput(
        queries=[], keywords=["Forecast", "Predict*", "Energy", "Power", "electricity"]
    )
    system_prompt, user_prompt = TransformerToSimplePrompts.render(test_data)
    print("\n\n=== SYSTEM PROMPT ===")
    print(system_prompt)
    print("\n=== USER PROMPT ===")
    print(user_prompt)
