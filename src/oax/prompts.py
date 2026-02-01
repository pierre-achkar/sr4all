from jinja2 import Template
from textwrap import dedent

import io_llm


class TransformerToOAXPrompts:
    SYSTEM = (
        "You are an expert information retrieval specialist skilled in bibliographic databases and query languages. "
        "You transform Boolean search queries from different bibliographic databases into valid OpenAlex /works query fragments with strict syntax compliance."
        "You can also build a query fragment from a list of keywords when no Boolean query is available."
        "You value precision, clarity, and adherence to instructions."
        "You ONLY respond with the requested JSON object."
    )

    USER_TEMPLATE = dedent(
        """
        # Context
        {%- if queries and keywords %}
        You will receive one or more bibliographic Boolean search strings written by experts targeting specific databases (e.g., Scopus, Web of Science, Dimensions, Lens, PubMed).
        Additionally, you will receive a list of search keywords used to search bibliographic databases. 
        {%- endif %}
        {%- if queries and not keywords %}
        You will receive one or more bibliographic Boolean search strings written by experts targeting specific databases (e.g., Scopus, Web of Science, Dimensions, Lens, PubMed).
        {%- endif %}
        {%- if keywords and not queries %}
        You will receive ONLY a list of search keywords used to search bibliographic databases.
        {%- endif %}

        # Task
        {%- if queries and keywords %}
        Translate the input Boolean string(s) into OpenAlex-compatible Boolean fragment(s) suitable for querying the `/works` endpoint using the 'search=' parameters, ensuring precise, rules-conformant transformation.
        {%- endif %}
        {%- if queries and not keywords %}
        Translate the input Boolean string(s) into OpenAlex-compatible Boolean fragment(s) suitable for querying the `/works` endpoint using the 'search=' parameters, ensuring precise, rules-conformant transformation.
        {%- endif %}
        {%- if keywords and not queries %}
        Build a single OpenAlex fragment suitable for querying the `/works` endpoint using the 'search=' parameters that logically combines all keywords using the appropriate syntax.
        {%- endif %}

        # Instructions and Rules
        {%- if queries %}
        ## General Guidelines when transforming Boolean queries
        - First, understand the semantics of the input query/queries, including operators, grouping of parentheses, and term relationships.
        - If multiple input queries are semantically identical but target different databases, output ONYL one representative fragment.
        - If multiple input queries are NOT semantically identical, output ONE fragment per input item, in the same order.
        - Each fragment must be a valid OpenAlex /works Boolean query fragment targeting the `search=` parameter.
        - Preserve the EXACT Boolean logic and parentheses. Operators must be UPPERCASE (AND/OR/NOT).
        - Keep only ASCII double quotes `"` that already exist; normalize any smart/single quotes to ASCII double quotes `"`.
        - For phrases (multiple words within quotes), always retain the quotes. If a phrase is unquoted in the input, add quotes in the output.
        - For single terms (single words), do NOT add quotes.
        - Replace proximity operators (NEAR/n, W/n, ADJ, NEXT) with AND.
        - Strip non-keyword limits/metadata (e.g., years, languages, doc types, LIMIT-TO, PUBYEAR, SRCTYPE, MH:, SU:, etc.).
        - As openalex do not support wildcards, remove any asterisks `*` from terms and expressions. To recover the loss, expand the wildcard up to three variants (e.g., `comput*` -> `computer OR computing OR computation`).
        - If only combination of numbers is given (e.g., 1# AND 2#), return an null fragment.
        {%- endif %}
        {%- if keywords and not queries %}
        ## Guidelines for building a query from keywords
        - Combine all keywords into a single OpenAlex /works Boolean fragment targeting the `search=` parameter.
        - Use the right Boolean operators (AND/OR/NOT) and parentheses to logically connect the keywords.
        - For phrases (multiple words within quotes), always retain the quotes. If a phrase is unquoted in the input, add quotes in the output.
        - For single terms (single words), do NOT add quotes.
        - As openalex do not support wildcards, remove any asterisks `*` from terms and expressions. To recover the loss, expand the wildcard up to three variants (e.g., `comput*` -> `computer OR computing OR computation`).
        {%- endif %}
        
        # Examples
        {%- if queries %}
        ## Example 1
        ### Original Query
        "Condition Monitoring" OR "Wind Turbine" OR "Data Mining Approaches" OR "Fault Diagnosis"

        ### OAX Query
        search=(\"Condition Monitoring\" OR \"Wind Turbine\" OR \"Data Mining Approaches\" OR \"Fault Diagnosis\")

        ## Example 2
        ### Original Query
        "TITLE-ABS-KEY (((indoor OR enclosed) AND (occupancy) AND (environmental OR environment) AND (sensor OR variables OR parameters)))"

        ### OAX Query
        search=(((indoor OR enclosed) AND (occupancy) AND (environmental OR environment) AND (sensor OR variables OR parameters)))

        ## Example 3
        ### Original Query
        dashboard OR whiteboard OR status board OR Electronic tracking board OR visualization OR presentation format OR display format OR performance measurement system,
        Design OR capability OR feature OR character OR attributes OR function OR usability OR content",
        Hospital,
        1# AND 2# AND 3#

        ### OAX Query
        search=((dashboard OR whiteboard OR \"status board\" OR \"Electronic tracking board\" OR visualization OR \"presentation format\" OR \"display format\" OR \"performance measurement system\") AND (Design OR capability OR feature OR character OR attributes OR function OR usability OR content) AND (Hospital))

        ## Example 4
        ### Original Query
        TITLE Forecast OR Predict. AND Energy OR Power OR electricity.

        ### OAX Query
        search=((Forecast OR Prediction OR Predictable OR Predicting OR Predicted OR Forecasting) AND (Energy OR Power OR Electricity))

        ## Example 5
        ### Original Query
        TI = (Internet of Things OR IoT) AND TS = (Authentication OR Authorization OR Identity OR Access Control) NOT TS = (Hardware OR Cryptography OR Protocol OR RFID OR Physical OR Network) NOT TS = (Survey OR Study) AND TS = Security

        ### OAX Query
        search=((\"Internet of Things\" OR IoT) AND (Authentication OR Authorization OR Identity OR \"Access Control\") AND Security AND NOT (Hardware OR Cryptography OR Protocol OR RFID OR Physical OR Network OR Survey OR Study))

        ## Example 6
        ### Original Query
        (\"information diffusion\") OR (\"influence analysis\") OR (\"influence maximization\") OR (\"user influence\")

        ### OAX Query
        search=(\"information diffusion\" OR \"influence analysis\" OR \"influence maximization\" OR \"user influence\")

        ## Example 7
        ### Original Query
        (gamif* OR gameful OR \"game elements\" OR \"game mechanics\" OR \"game dynamics\" OR \"game components\" OR \"game aesthetics\") AND (education OR educational OR learning OR teaching OR course OR syllabus OR syllabi OR curriculum OR curricula) AND (framework OR method OR design OR model OR approach OR theory OR strategy)

        ### OAX Query
        search=((gamify OR gamifying OR gamification OR gamif OR gameful OR \"game elements\" OR \"game mechanics\" OR \"game dynamics\" OR \"game components\" OR \"game aesthetics\") AND (education OR educational OR learning OR teaching OR course OR syllabus OR syllabi OR curriculum OR curricula) AND (framework OR method OR design OR model OR approach OR theory OR strategy))

        ## Example 8
        ### Original Query
        covid-19, sars-cov-2, coronavirus, genetic variation, gene, genome-wide association study, polymorphisms, single nucleotide, genetic association, genetic susceptibility, genotype, human host, genotype, covid-19 outcome modelling, covid-19 severity modelling, machine learning for covid-19 modelling, covid-19 prediction using genomic data  

        ### OAX Query
        search=((\"covid-19\" OR \"sars-cov-2\" OR coronavirus OR \"genetic variation\" OR gene OR \"genome-wide association study\" OR polymorphisms OR \"single nucleotide\" OR \"genetic association\" OR \"genetic susceptibility\" OR genotype OR \"human host\" OR \"covid-19 outcome modelling\" OR \"covid-19 severity modelling\" OR \"machine learning for covid-19 modelling\" OR \"covid-19 prediction using genomic data\"))

        ## Example 9
        ### Original Query
        PUBYEAR > 2015 AND (TITLE(\"log\") AND TITLE-ABS-KEY(\"log analysis\")) AND (AUTHKEY(\"analysis\") OR AUTHKEY (\"retrieval\") OR AUTHKEY (\"recovery\") OR AUTHKEY (\"mining\") OR AUTHKEY (\"reverse engineering\") OR AUTHKEY (\"detection\")) AND (LIMIT-TO(SUBJAREA, \"COMP\")) AND (LIMIT-TO(LANGUAGE, \"English\"))

        ### OAX Query
        search=((log) AND (\"log analysis\") AND ((analysis) OR (retrieval) OR (recovery) OR (mining) OR (\"reverse engineering\") OR (detection)))

        ## Example 10
        ### Original Query
        "warehous* AND (\"digital transformation\" OR \"technolog*\" OR \"4.0\" OR \"smart\" OR \"intelligent\" OR \"IoT\" OR \"Internet of Things\" OR \"robots\" OR \"RFID\" OR \"cloud\" OR \"AMR\" OR \"Artificial intelligence\" OR \"AI\" OR \"blockchain\" OR \"big data\" OR \"CPS\" OR \"Cyber Physical\" OR \"augmented reality\" OR \"digital twin\" OR \"AGV\" OR \"auto* vehicles\" OR \"5G\" OR \"fourth industrial revolution\")

        ### OAX Query
        search=((warehouse OR warehouses OR warehousing) AND (\"digital transformation\" OR technology OR technologies OR technological OR \"Industry 4.0\" OR smart OR intelligent OR IoT OR \"Internet of Things\" OR robots OR RFID OR cloud OR AMR OR \"artificial intelligence\" OR AI OR blockchain OR \"big data\" OR CPS OR \"cyber physical\" OR \"cyber-physical\" OR \"augmented reality\" OR \"digital twin\" OR AGV OR \"automated vehicles\" OR \"autonomous vehicles\" OR \"automatic vehicles\" OR 5G OR \"fourth industrial revolution\"))
        {%- endif %}

        {%- if keywords and not queries %}
        ## Example 1
        ### Keywords
        - \"Condition Monitoring\"
        - \"Wind Turbine\"
        - \"Data Mining Approaches\"
        - \"Fault Diagnosis\"

        ### OAX Query
        search=(\"Condition Monitoring\" OR \"Wind Turbine\" OR \"Data Mining Approaches\" OR \"Fault Diagnosis\")

        ## Example 2
        ### Keywords
        - indoor 
        - enclosed 
        - occupancy
        - environmental
        - environment
        - sensor
        - variables
        - parameters

        ### OAX Query
        search=((indoor OR enclosed) AND (occupancy) AND (environmental OR environment) AND (sensor OR variables OR parameters))

        ## Example 3
        ### Keywords
        - Forecast 
        - Predict
        - Energy
        - Power
        - electricity

        ### OAX Query
        search=((Forecast OR Prediction OR Predictable OR Predicting OR Predicted OR Forecasting) AND (Energy OR Power OR Electricity))
        {%- endif %}

        # Inputs (ordered list)
        {%- if queries %}
        The following are the Boolean query string(s) to be transformed:
        {%- for q in queries %}
        {{ loop.index }}. Boolean query string:
        {{ q.boolean_query_string }}
        {%- if q.database_source %}
        Database source: {{ q.database_source }}
        {%- endif %}
        {%- endfor %}
        {%- endif %}

        {%- if keywords %}
        The following are the keyword terms to be logically combined:
        {%- for k in keywords %}
        - {{ k }}
        {%- endfor %}
        {%- endif %}

        # Output
        Return only a valid JSON object that matches this schema:
        {"oax_boolean_queries": ["<fragment 1>", "<fragment 2>", ...]}
        """
    ).strip()

    @staticmethod
    def render(data: io_llm.LLMInput) -> tuple[str, str]:
        t = Template(TransformerToOAXPrompts.USER_TEMPLATE)
        return TransformerToOAXPrompts.SYSTEM, t.render(**data.model_dump())


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Example usage
    input_data = io_llm.LLMInput(
        queries=[
            io_llm.LLMQueryItem(
                boolean_query_string='"Condition Monitoring" OR "Wind Turbine" OR "Data Mining Approaches" OR "Fault Diagnosis"',
                database_source="Scopus",
                
            ),
            io_llm.LLMQueryItem(
                boolean_query_string='TITLE-ABS-KEY (((indoor OR enclosed) AND (occupancy) AND (environmental OR environment) AND (sensor OR variables OR parameters)))',
                database_source="Web of Science"
            ),
        ],
        keywords=[
            "machine learning",
            "data science",
            "artificial intelligence"
        ]
    )

    system_prompt, user_prompt = TransformerToOAXPrompts.render(input_data)
    print("System Prompt:\n", system_prompt)
    print("\nUser Prompt:\n", user_prompt)