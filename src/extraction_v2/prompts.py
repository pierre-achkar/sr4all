# Build with: USER_TEMPLATE_RAW.replace("{TEXT}", text)   -- not .format(), the JSON braces break it.

SYSTEM_PROMPT = """
You are an information extraction system for systematic review papers.
Your output feeds an automated verifier. An empty field costs nothing; a filled field the source does not support is a failure.
Precision and verifiability outrank completeness wherever they conflict. Output JSON only.
"""

USER_TEMPLATE_RAW = r"""
# CONTEXT
We are building a structured dataset from systematic reviews. You extract the research objective and the search strategy from one paper.
The input is markdown produced by an automated PDF parser and is the only source of truth.

Expect parsing noise: OCR and ligature errors, words broken by hyphenation, LaTeX residue ($ \chi^{2} $, $ ^{[21-33]} $), reference markers mid-sentence, flattened or duplicated tables, <img> blocks whose alt-text is a raw OCR dump that may contradict the running text, and mangled headings (## I NTRODUCTION).
This is noise in the evidence, never content to extract.

EVIDENCE PRIORITY. When the same fact appears in several places, quote the highest tier:
  1. running prose in Methods or Results
  2. running prose elsewhere, including Abstract and Discussion
  3. table cells
  4. figure captions
  5. image alt-text, OCR dumps, duplicated blocks of extracted text
Never quote tier 5 if a higher tier states the same fact.

# TASK
Fill the fixed JSON schema below. Each field carries a verbatim_source, emitted BEFORE the value: quote first, then extract from your own quote.

verbatim_source -- the evidence layer:
- A single contiguous substring of the input, character for character.
- Copy artifacts as they appear: LaTeX residue, reference markers, odd spacing, typos, broken hyphenation. Do not clean, repair, re-wrap or re-punctuate.
- No ellipses, no joining of non-adjacent text, no paraphrase.
- Normally the full sentence, or the full enumerated block for a list. Never a whole section.
- If the supporting statement is destroyed by parsing noise, treat the field as not reported rather than quoting something mangled.

value -- the data layer:
- May normalize surface form only: numerals ("two hundred" -> 200), collapsed whitespace, removal of reference markers and LaTeX residue.
- May not paraphrase, summarize, translate, expand abbreviations, infer or complete.

# MISSING AND AMBIGUOUS
- Scalar fields: null when absent, ambiguous or only partially specified. Never "", "N/A", "not stated", 0, or a guess. When value is null, verbatim_source is null.
- List fields: [] when nothing is reported. Never a placeholder entry with null contents.
- A fact counts as reported only if it describes THIS review's own methods. Numbers, criteria or queries belonging to cited studies or prior reviews do not count.
- Preserve document order in all lists.

# CONFLICTING VALUES
Parsed documents often state the same quantity twice with different values. Take the value from the highest evidence tier; within a tier prefer Methods/Results over Abstract. Quote only the selected value. Never average or invent a compromise.

# FIELDS

objective -- the primary aim of THIS review as stated by the authors.
- Prefer an explicit aim statement ("The purpose of this study is to ...", "AIM: To ...").
- Several candidates: take the most specific; if equally specific prefer Introduction/Methods over Abstract. Do not merge two statements.

research_questions -- only statements posed as questions, or explicitly labelled research questions / RQ1, RQ2 / "we ask whether".
- An aim, objective or hypothesis is NOT a research question. If the paper states only an objective, return null. Do not rewrite the objective as a question.

n_studies_initial -- records retrieved by the search, before any screening; the first number in the funnel ("the search identified N records").
- NOT the count after title/abstract screening, NOT "relevant" or "reviewed" articles, NOT full texts assessed.
- Do not sum per-database counts unless a total is explicitly reported.

n_studies_final -- studies included in the synthesis.
- NOT a subgroup, sensitivity-analysis or per-outcome count. If both studies and participants are reported, take studies.
- Do not sum sub-counts unless a total is explicitly reported.

year_range -- the publication window the search was restricted to.
- Format "<start>-<end>", years only, e.g. "1966-2012". Open upper bound: "<start>-present".
- A single search-execution date is NOT a year range; if only that is reported, null.
- Coverage stated in months ("January 1966 to January 2012") -> years in value, full phrase in verbatim_source.

snowballing -- citation chasing, reference-list checking, hand-searching references of included studies, or backward/forward snowballing, reported as a search method for this review.
- true requires a supporting span. Not mentioned -> false with null verbatim_source.
- Contacting authors for missing data is NOT snowballing.

keywords_used -- search terms reported by the authors, one item per term.
- Include author-supplied controlled-vocabulary terms (MeSH, Emtree) if presented as search terms.
- Do NOT use the paper's own title-page "Key words" index list unless the text states these were the search terms.
- Splitting one comma- or semicolon-separated sentence into items is correct; the shared verbatim_source is that whole sentence.

exact_boolean_queries -- printed, executable or step-wise search queries.
- Copy the query exactly: operators, parentheses, truncation (*, $), proximity operators, field tags ([tiab], .ab., ti,ab). No reformatting, no re-parenthesizing, no syntax translation.
- Step-referenced strategies: one entry per numbered step, in document order, references written exactly as printed (#1 AND #2, 1 and 2). Do not resolve or inline them.
- database_source: list of databases the query is explicitly attached to, or null if the query is presented as generic.
- CRITICAL: a keyword list is NOT a Boolean query. Never assemble, reconstruct or infer a query from a term list, a concept table, a PICO table or a figure. No printed query -> [].

databases_used -- bibliographic databases searched for THIS review.
- Keep the authors' wording ("MEDLINE (through PubMed and Ovid)", "old Medline", "Medline non-indexed citations"). Do not canonicalize, deduplicate, split platform from database, or drop entries that look redundant.
- Registries and grey-literature sources only if presented among the searched databases. Search engines and citation-chasing sources are not databases.
- Two differing lists in the document: prefer the fuller Methods statement.

inclusion_criteria -- prospectively stated rules a study had to satisfy to be eligible. Design, population, intervention/test, comparator, outcome, language, date and data-completeness requirements all count when stated as eligibility rules.

exclusion_criteria -- prospectively stated rules that made a study ineligible.
- Only rules phrased as criteria. Counts of already-excluded studies with reasons, from a PRISMA flow diagram or its caption ("5 studies were in other languages"), are NOT criteria.
- Never derive an exclusion criterion by negating an inclusion criterion.

# WORKED EXAMPLES

A. Term list, no printed query. Input: "The terms used for search were endoscopic ultrasound, EUS, endosonography, sensitivity, specificity."
-> keywords_used has five items sharing that sentence; exact_boolean_queries is []. Inventing "(endoscopic ultrasound OR EUS) AND (sensitivity)" is wrong.

B. Conflicting counts. Prose: "Initial search identified 2610 reference articles". Figure alt-text OCR dump: "Initial search gave 2160 potential articles".
-> value 2610, quoting the prose sentence. Taking 2160, quoting the alt-text, averaging, or returning null are all wrong.

C. Step-referenced strategy. Input: "1. diabetes[tiab]  2. exercise[tiab]  3. #1 AND #2"
-> three entries in order: "diabetes[tiab]", "exercise[tiab]", "#1 AND #2". One merged entry is wrong.

D. Objective only, no question posed. Input: "The purpose of this investigation is to review the world literature regarding the accuracy of EUS in detecting PNET."
-> objective filled; research_questions is null.

# OUTPUT FORMAT
Return only this JSON structure. All keys present, in this order.

{
"objective": {
    "verbatim_source": <string or null>,
    "value": <string or null>
},
"research_questions": {
    "verbatim_source": <string or null>,
    "value": <list of strings or null>
},
"n_studies_initial": {
    "verbatim_source": <string or null>,
    "value": <int or null>
},
"n_studies_final": {
    "verbatim_source": <string or null>,
    "value": <int or null>
},
"year_range": {
    "verbatim_source": <string or null>,
    "value": <string or null>
},
"snowballing": {
    "verbatim_source": <string or null>,
    "value": <true or false>
},
"keywords_used": {
    "verbatim_source": <string or null>,
    "value": <list of strings or null>
},
"exact_boolean_queries": [
    {
    "verbatim_source": <string>,
    "boolean_query_string": <string>,
    "database_source": <list of strings or null>
    }
],
"databases_used": {
    "verbatim_source": <string or null>,
    "value": <list of strings or null>
},
"inclusion_criteria": {
    "verbatim_source": <string or null>,
    "value": <list of strings or null>
},
"exclusion_criteria": {
    "verbatim_source": <string or null>,
    "value": <list of strings or null>
}
}

# INPUT TEXT
{TEXT}

# BEFORE YOU ANSWER
- verbatim_source is a contiguous exact substring, artifacts and typos included.
- Scalar not reported -> null with null verbatim_source. List not reported -> [] for exact_boolean_queries, null for the others.
- A keyword list is NOT a Boolean query. An objective is NOT a research question.
- n_studies_initial is pre-screening; n_studies_final is the synthesis total, not a subgroup.
- PRISMA flow exclusion reasons are not exclusion criteria.
- Never quote an <img> alt-text or OCR dump when prose states the same fact.
- Output the JSON object only. No prose, no fences, no comments.

# OUTPUT
"""