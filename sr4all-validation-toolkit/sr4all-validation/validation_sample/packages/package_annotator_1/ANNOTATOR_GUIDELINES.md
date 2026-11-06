# SR4ALL Annotator Guidelines

## About this task

The SR4ALL Dataset uses an automated system to extract information about how systematic reviews were conducted. Your task is to compare these extracted fields with the content of the original paper. This will show which information was extracted correctly and which information was missed or extracted incorrectly.

The annotation tool displays one paper at a time. For each field, it shows either an extracted value with its evidence passage or an empty field for which no value was extracted.

## What you need to do

For each paper:

1. Read the field name, extracted value, and evidence passage shown in the tool.
2. Find and read the relevant passage in the PDF.
3. Choose the annotation option that best describes the extraction.
4. Add a comment when the extraction is partial, incorrect, or unclear.
5. If an empty field was missed, enter the supporting text and PDF page number.

Review every field before moving to the next paper. Export your annotations regularly so that you have a backup.

## Annotation options

### Fields with an extracted value

- **Correct:** The extracted value is fully supported by the paper, and the evidence passage supports that value.
- **Partial:** The extracted value is generally correct, but a small part is missing or inaccurate.
- **Incorrect:** The extracted value is wrong or unsupported, or the evidence passage does not support it.

### Empty fields

An empty field means that the system did not extract a value.

- **Not in PDF:** The paper does not report the requested information.
- **Present in PDF (missed):** The paper reports the requested information, but the system did not extract it.

For **Present in PDF (missed)**, enter the supporting text and PDF page number.

## General annotation rules

- Use the source PDF only.
- Judge each field independently.
- Check both the extracted value and its evidence passage. A correct value with unrelated or insufficient evidence is not fully correct.
- Do not infer information that the paper does not state.
- The same passage may support more than one field.
- Use **Search** and **Locate in PDF** to find text, but always check the passage yourself.
- Ignore harmless OCR, spacing, punctuation, and line-break differences.
- Add a short comment whenever your decision may not be obvious to another annotator.

## Rules for specific fields

### Research questions

- A research question does not need a question mark.
- It must be presented as a question or clearly identified as a research question.
- An objective is not automatically a research question.
- Example: "The aim was to assess treatment X" is an objective.
- Example: "We asked whether treatment X was effective" is a research question.

### Inclusion and exclusion criteria

- Judge the two fields separately.
- The same passage may support both fields.
- Do not create an exclusion criterion by simply negating an inclusion criterion.
- Example: "English-language studies were included" does not by itself prove that non-English studies were excluded.

### Boolean queries

Check each query separately. The terms, operators, parentheses, order, and database name must match the PDF. A list of keywords is not a Boolean query.

### Snowballing

When snowballing is extracted as "false," the tool may not show an evidence passage. Check the PDF to see whether snowballing was used. If it was used, mark the extraction as **Incorrect** and add a comment with the supporting text and page number. If it was not used, mark the extraction as **Correct**. No comment is required.

## Shared documents

Some papers are marked as shared. The same paper is assigned to every annotator. Shared documents are used to measure how consistently different annotators apply these guidelines.

Annotate shared documents independently. Do not compare or discuss your annotations with the other annotators until everyone has submitted their work.
