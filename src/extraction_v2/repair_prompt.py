"""
Prompt templates for the repair stage: targeted re-extraction of fields that are
still missing after extraction, alignment and fact-checking.

Design notes:
  * FIELD_RULES carries the SAME tightened semantics as the first-pass prompt.
    Repair fires only on fields the first pass got wrong, so it is the worst
    place to relax the definitions. Any edit here must be mirrored in
    extraction/prompts.py (see docstring note at the bottom of that file).
  * The prompt asks for the requested fields ONLY, and subset_schema() produces a
    matching grammar so constrained decoding cannot emit the other nine. This is
    the actual perturbation that makes a second greedy pass worth running: less
    instruction noise, a narrower target, and negative evidence.
  * NEGATIVE EVIDENCE: when a field is null because a previous pass proposed
    something that failed verification, the rejected value and span are shown and
    ruled out. Without this, greedy decoding tends to propose the same rejected
    value again.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

REPAIR_SYSTEM_PROMPT = """\
You are an information extraction system for systematic review papers, working on a
second pass over fields an earlier pass failed to extract.

Most of these fields are absent from the document. Returning null is the correct and
expected answer, and costs nothing. A value the document does not support is a failure.
Do not lower your standard because a field is being re-asked.

Output JSON only."""


# --------------------------------------------------------------------------- #
# Field rules. Must stay semantically identical to extraction/prompts.py.
# --------------------------------------------------------------------------- #
FIELD_RULES: Dict[str, str] = {
    "objective": """\
The primary aim of THIS review, as stated by the authors.
- Prefer an explicit aim statement ("The purpose of this study is to ...", "AIM: To ...").
- If several statements qualify, take the most specific one; if equally specific, prefer
  Introduction/Methods over Abstract. Do not merge two statements.""",

    "research_questions": """\
Only statements posed as questions, or explicitly labelled as research questions /
RQ1, RQ2 / "we ask whether".
- An aim, objective or hypothesis is NOT a research question. If the paper states only
  an objective, return null. Do not rewrite the objective as a question.""",

    "n_studies_initial": """\
Records retrieved by the search, before any screening; the first number in the funnel
("the search identified N records").
- NOT the count after title/abstract screening, NOT "relevant" or "reviewed" articles,
  NOT full texts assessed.
- Do not sum per-database counts unless a total is explicitly reported.""",

    "n_studies_final": """\
Studies included in the synthesis.
- NOT a subgroup, sensitivity-analysis or per-outcome count. If both studies and
  participants are reported, take studies.
- Do not sum sub-counts unless a total is explicitly reported.""",

    "year_range": """\
The publication window the search was restricted to.
- Format "<start>-<end>", years only, e.g. "1966-2012". Open upper bound: "<start>-present".
- A single search-execution date is NOT a year range; if only that is reported, null.
- Coverage stated in months ("January 1966 to January 2012") -> years in value, the full
  phrase in verbatim_source.""",

    "snowballing": """\
Citation chasing, reference-list checking, hand-searching references of included studies,
or backward/forward snowballing, reported as a search method for this review.
- true requires a supporting span. Not mentioned -> false with null verbatim_source.
- Contacting authors for missing data is NOT snowballing.""",

    "keywords_used": """\
Search terms reported by the authors, one item per term.
- Include author-supplied controlled-vocabulary terms (MeSH, Emtree) if presented as
  search terms.
- Do NOT use the paper's own title-page "Key words" index list unless the text states
  these were the search terms.
- Splitting one comma- or semicolon-separated sentence into items is correct; the shared
  verbatim_source is that whole sentence.""",

    "exact_boolean_queries": """\
Printed, executable or step-wise search queries.
- Copy the query exactly: operators, parentheses, truncation (*, $), proximity operators,
  field tags ([tiab], .ab., ti,ab). No reformatting, no re-parenthesizing, no syntax
  translation.
- Step-referenced strategies: one entry per numbered step, in document order, references
  written exactly as printed (#1 AND #2, 1 and 2). Do not resolve or inline them.
- database_source: list of databases the query is explicitly attached to, or null if the
  query is presented as generic.
- CRITICAL: a keyword list is NOT a Boolean query. Never assemble, reconstruct or infer a
  query from a term list, a concept table, a PICO table or a figure. No printed query -> [].""",

    "databases_used": """\
Bibliographic databases searched for THIS review.
- Keep the authors' wording ("MEDLINE (through PubMed and Ovid)", "old Medline",
  "Medline non-indexed citations"). Do not canonicalize, deduplicate, split platform from
  database, or drop entries that look redundant.
- Registries and grey-literature sources only if presented among the searched databases.
  Search engines and citation-chasing sources are not databases.""",

    "inclusion_criteria": """\
Prospectively stated rules a study had to satisfy to be eligible. Design, population,
intervention/test, comparator, outcome, language, date and data-completeness requirements
all count when stated as eligibility rules.""",

    "exclusion_criteria": """\
Prospectively stated rules that made a study ineligible.
- Only rules phrased as criteria. Counts of already-excluded studies with reasons, from a
  PRISMA flow diagram or its caption ("5 studies were in other languages"), are NOT
  criteria.
- Never derive an exclusion criterion by negating an inclusion criterion.""",
}

# Schema block shown to the model, one entry per field.
_EV = '"{f}": {{ "verbatim_source": <string or null>, "value": {t} }}'
SCHEMA_LINES: Dict[str, str] = {
    "objective": _EV.format(f="objective", t="<string or null>"),
    "research_questions": _EV.format(f="research_questions", t="<list of strings or null>"),
    "n_studies_initial": _EV.format(f="n_studies_initial", t="<int or null>"),
    "n_studies_final": _EV.format(f="n_studies_final", t="<int or null>"),
    "year_range": _EV.format(f="year_range", t="<string or null>"),
    "snowballing": _EV.format(f="snowballing", t="<true or false>"),
    "keywords_used": _EV.format(f="keywords_used", t="<list of strings or null>"),
    "databases_used": _EV.format(f="databases_used", t="<list of strings or null>"),
    "inclusion_criteria": _EV.format(f="inclusion_criteria", t="<list of strings or null>"),
    "exclusion_criteria": _EV.format(f="exclusion_criteria", t="<list of strings or null>"),
    "exact_boolean_queries": (
        '"exact_boolean_queries": [ { "verbatim_source": <string>, '
        '"boolean_query_string": <string>, "database_source": <list of strings or null> } ]'
    ),
}

REPAIRABLE_FIELDS = tuple(FIELD_RULES)


BASE_INSTRUCTIONS = r"""
# CONTEXT
You are given one systematic review paper as markdown produced by an automated PDF parser.
It is the only source of truth. Expect parsing noise: OCR errors, hyphenation across line
breaks, LaTeX residue ($ \chi^{2} $, $ ^{[21-33]} $), reference markers mid-sentence,
flattened tables, and <img> blocks whose alt-text is a raw OCR dump that may contradict the
running text. This is noise in the evidence, never content to extract.

An earlier pass did not produce a usable value for the fields listed below. Extract only
those fields. Most of them are genuinely absent from the paper; null is the expected answer.

# EVIDENCE RULES
- Every populated field carries verbatim_source: one contiguous exact substring of the
  markdown above, copied character for character, artifacts and typos included.
- No ellipses, no stitched quotes, no paraphrase, no cleanup.
- Normally the full sentence, or the full enumerated block for a list. Never a whole section.
- If value is null, verbatim_source is null.
- Prefer running prose in Methods or Results over tables, figure captions, and especially
  over <img> alt-text or OCR dumps.

# VALUE RULES
- value may normalize surface form only: numerals ("two hundred" -> 200), collapsed
  whitespace, removal of reference markers and LaTeX residue. Never paraphrase, summarize,
  translate, expand abbreviations or complete.
- Scalars: null when absent, ambiguous or only partially specified.
- List fields: null when nothing is reported, except exact_boolean_queries which is [].
- snowballing is always true or false, never null.
- A fact counts as reported only if it describes THIS review's own methods.

# TARGET FIELDS
{TARGET_FIELD_INSTRUCTIONS}
{REJECTED_BLOCK}
# OUTPUT FORMAT
Return only this JSON object, with exactly these keys in this order and nothing else.

{
{SCHEMA_BLOCK}
}

# INPUT TEXT
{TEXT}

# BEFORE YOU ANSWER
- verbatim_source must be a contiguous exact substring of the input above.
- Null is the expected answer for a field the paper does not report. Do not stretch.
- A keyword list is not a Boolean query. An objective is not a research question.
- Output the JSON object only. No prose, no fences, no comments.

# OUTPUT
"""

_REJECTED_TEMPLATE = """
# PREVIOUSLY REJECTED (do not propose these again)
An earlier pass proposed the following, and verification rejected them. Either find
different, genuinely supported evidence, or return null.
{REJECTED_LINES}
"""


def _trim(text: Any, limit: int = 200) -> str:
    s = " ".join(str(text).split())
    return s if len(s) <= limit else s[:limit] + " ..."


def render_rejections(rejections: Dict[str, Dict[str, Any]], fields: List[str]) -> str:
    """
    rejections: field -> {"value": ..., "verbatim_source": ..., "reason": ...}
    Only fields being re-asked are shown.
    """
    lines = []
    for f in fields:
        r = rejections.get(f)
        if not r:
            continue
        parts = [f"- {f}: rejected value {_trim(r.get('value'), 120)!r}"]
        if r.get("reason"):
            parts.append(f"  reason: {r['reason']}")
        if r.get("verbatim_source"):
            parts.append(f"  rejected span: \"{_trim(r['verbatim_source'])}\"")
        lines.append("\n".join(parts))
    if not lines:
        return ""
    return _REJECTED_TEMPLATE.replace("{REJECTED_LINES}", "\n".join(lines))


def get_repair_user_prompt(
    doc_text: str,
    missing_keys: List[str],
    rejections: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[str]:
    """
    Build a repair prompt for the requested fields only.

    Returns None when no requested key is repairable, so the caller skips the
    document instead of triggering a full re-extraction.
    """
    keys = [k for k in REPAIRABLE_FIELDS if k in set(missing_keys)]  # canonical order
    if not keys:
        return None

    targets = "\n".join(f"## {k}\n{FIELD_RULES[k]}" for k in keys)
    schema_block = ",\n".join("  " + SCHEMA_LINES[k] for k in keys)
    rejected_block = render_rejections(rejections or {}, keys)

    return (
        BASE_INSTRUCTIONS
        .replace("{TARGET_FIELD_INSTRUCTIONS}", targets)
        .replace("{REJECTED_BLOCK}", rejected_block)
        .replace("{SCHEMA_BLOCK}", schema_block)
        .replace("{TEXT}", doc_text)
    )


def subset_schema(fields: List[str]) -> Dict[str, Any]:
    """
    JSON schema for the requested fields only, derived from ReviewExtraction so the
    two cannot drift. Pass as vLLM guided_json / structured outputs.
    """
    from extraction_v2.schema import ReviewExtraction

    full = ReviewExtraction.model_json_schema()
    keys = [k for k in REPAIRABLE_FIELDS if k in set(fields)]
    if not keys:
        raise ValueError("subset_schema called with no repairable fields")
    return {
        "type": "object",
        "properties": {k: full["properties"][k] for k in keys},
        "required": keys,
        "additionalProperties": False,
        # $refs in the copied properties point into $defs, so it must come along.
        "$defs": full.get("$defs", {}),
    }