#!/usr/bin/env python3
"""Generate the example package: 3 dummy systematic-review PDFs + tasks.jsonl."""
import json
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

OUT = Path(__file__).resolve().parent.parent / "example"
(OUT / "pdfs").mkdir(parents=True, exist_ok=True)

DOCS = [
    {
        "doc_id": "W0000000001",
        "title": "Blue ocean leadership in organizations: a systematic literature review",
        "paras": [
            ("Abstract", "Leadership research has expanded rapidly. Therefore, for the reader to "
             "better understand it, there is a need to conduct a Systematic Literature Review on "
             "this topic. The objective of this paper is to review the dimension and influence of "
             "blue ocean leadership in different perspectives in Malaysia by using SLR."),
            ("1. Introduction", "Blue ocean leadership (BOL) has received growing attention. This "
             "systematic literature review was performed to address the following questions: "
             "(a) What is the dimension of BOL? (b) What is the influence of BOL in different "
             "perspectives in Malaysia?"),
            ("2. Methodology", "We searched Scopus, Google Scholar and Dimensions. The search string "
             "used in Scopus was TITLE-ABS-KEY (\"blue ocean leadership\" OR \"blue ocean strategy "
             "in leadership\") restricted to articles published between 2014 and 2020 in English. "
             "Only research articles and conference proceedings with empirical data were included."),
            ("3. Study selection", "The findings showed 45 articles collected from the selected "
             "databases. Next, we removed 12 duplicated articles. After screening for field of "
             "study and eligibility, a total of 10 papers went through to the final stage."),
            ("4. Results", "The included studies show that blue ocean leadership practices vary "
             "considerably across organizational contexts and sectors."),
        ],
        "fields": [
            {"name": "objective",
             "value": "to review the dimension and influence of blue ocean leadership in different perspectives in Malaysia by using SLR",
             "evidence_span": "The objective of this paper is to review the dimension and influence of blue ocean leadership in different perspectives in Malaysia by using SLR."},
            {"name": "research_questions",
             "value": ["What is the dimension of BOL?", "What is the influence of BOL in different perspectives in Malaysia?"],
             "evidence_span": "(a) What is the dimension of BOL? (b) What is the influence of BOL in different perspectives in Malaysia?"},
            {"name": "boolean_query",
             "value": "TITLE-ABS-KEY (\"blue ocean leadership\" OR \"blue ocean strategy in leadership\")",
             "evidence_span": "TITLE-ABS-KEY (\"blue ocean leadership\" OR \"blue ocean strategy in leadership\")"},
            {"name": "n_studies_initial", "value": 45,
             "evidence_span": "The findings showed 45 articles collected from the selected databases."},
            {"name": "n_studies_final", "value": 10,
             "evidence_span": "a total of 10 papers went through to the final stage"},
            {"name": "year_range", "value": "2014-2020",
             "evidence_span": "articles published between 2014 and 2020"},
        ],
        # databases_used IS reported in the PDF -> a real "missed" case for recall testing
        "null_fields": ["databases_used", "exclusion_criteria", "snowballing"],
    },
    {
        "doc_id": "W0000000002",
        "title": "Workplace interventions for burnout: a systematic review and meta-analysis",
        "paras": [
            ("Abstract", "Burnout is a growing occupational health concern. This systematic review "
             "aimed to synthesize evidence on organizational interventions for reducing burnout "
             "among adult employees."),
            ("2. Methods", "We searched PubMed, PsycINFO and Scopus from January 2000 to March 2020. "
             "The search combined the terms (burnout OR \"emotional exhaustion\") AND (intervention "
             "OR program OR trial) AND (workplace OR organizational). We included randomized "
             "controlled trials and quasi-experimental studies of adult employees. Studies without "
             "a control group were excluded, as were dissertations and conference abstracts."),
            ("3. Results", "Of 3,412 records identified, 58 studies met the inclusion criteria and "
             "were included in the synthesis."),
        ],
        "fields": [
            {"name": "objective",
             "value": "to synthesize evidence on organizational interventions for reducing burnout among adult employees",
             "evidence_span": "This systematic review aimed to synthesize evidence on organizational interventions for reducing burnout among adult employees."},
            # deliberately truncated query -> a real "partial/incorrect" case for testing
            {"name": "boolean_query",
             "value": "(burnout OR \"emotional exhaustion\") AND (intervention OR program)",
             "evidence_span": "(burnout OR \"emotional exhaustion\") AND (intervention OR program OR trial) AND (workplace OR organizational)"},
            {"name": "databases_used", "value": ["PubMed", "PsycINFO", "Scopus"],
             "evidence_span": "We searched PubMed, PsycINFO and Scopus"},
            {"name": "inclusion_criteria",
             "value": ["randomized controlled trials and quasi-experimental studies", "adult employees"],
             "evidence_span": "We included randomized controlled trials and quasi-experimental studies of adult employees."},
            {"name": "n_studies_initial", "value": 3412,
             "evidence_span": "Of 3,412 records identified"},
            {"name": "n_studies_final", "value": 58,
             "evidence_span": "58 studies met the inclusion criteria and were included in the synthesis"},
        ],
        # exclusion criteria ARE in the PDF -> missed; research questions are NOT -> correctly null
        "null_fields": ["exclusion_criteria", "research_questions", "year_range"],
    },
    {
        "doc_id": "W0000000003",
        "title": "Machine learning for code smell detection: a systematic review",
        "paras": [
            ("Abstract", "Code smells indicate potential design problems in software. The goal of "
             "this systematic review is to characterize machine learning approaches for automated "
             "code smell detection and to identify open research challenges."),
            ("3. Review protocol", "Following Kitchenham's guidelines, we defined the search string "
             "(\"code smell\" OR \"bad smell\" OR anti-pattern) AND (\"machine learning\" OR "
             "classification OR \"deep learning\") and executed it on IEEE Xplore, the ACM Digital "
             "Library and Scopus. We additionally performed backward snowballing on the reference "
             "lists of all included papers. Peer-reviewed studies published between 2005 and 2023 "
             "were eligible; grey literature was excluded."),
            ("4. Results", "The search retrieved 1,286 candidate papers, of which 74 primary "
             "studies were retained after full-text screening."),
        ],
        "fields": [
            {"name": "objective",
             "value": "to characterize machine learning approaches for automated code smell detection and to identify open research challenges",
             "evidence_span": "The goal of this systematic review is to characterize machine learning approaches for automated code smell detection and to identify open research challenges."},
            {"name": "boolean_query",
             "value": "(\"code smell\" OR \"bad smell\" OR anti-pattern) AND (\"machine learning\" OR classification OR \"deep learning\")",
             "evidence_span": "(\"code smell\" OR \"bad smell\" OR anti-pattern) AND (\"machine learning\" OR classification OR \"deep learning\")"},
            {"name": "databases_used", "value": ["IEEE Xplore", "ACM Digital Library", "Scopus"],
             "evidence_span": "executed it on IEEE Xplore, the ACM Digital Library and Scopus"},
            {"name": "snowballing", "value": True,
             "evidence_span": "We additionally performed backward snowballing on the reference lists of all included papers."},
            {"name": "year_range", "value": "2005-2023",
             "evidence_span": "Peer-reviewed studies published between 2005 and 2023 were eligible"},
            # deliberately wrong value -> "incorrect" case for testing
            {"name": "n_studies_final", "value": 84,
             "evidence_span": "74 primary studies were retained after full-text screening"},
        ],
        "null_fields": ["research_questions", "inclusion_criteria"],
    },
]


def build_pdf(doc):
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=10.5, leading=15, spaceAfter=8)
    head = ParagraphStyle("head", parent=styles["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4)
    title = ParagraphStyle("title", parent=styles["Title"], fontSize=15, spaceAfter=14)
    els = [Paragraph(doc["title"], title)]
    for h, p in doc["paras"]:
        els += [Paragraph(h, head), Paragraph(p, body), Spacer(1, 4)]
    SimpleDocTemplate(str(OUT / "pdfs" / f"{doc['doc_id']}.pdf"), pagesize=A4).build(els)


with open(OUT / "tasks.jsonl", "w", encoding="utf-8") as fh:
    for i, doc in enumerate(DOCS):
        build_pdf(doc)
        fh.write(json.dumps({
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "pdf": f"pdfs/{doc['doc_id']}.pdf",
            "shared": i < 2,  # first two marked shared for demo purposes
            "fields": doc["fields"],
            "null_fields": doc["null_fields"],
        }, ensure_ascii=False) + "\n")

print(f"example package written to {OUT}")
