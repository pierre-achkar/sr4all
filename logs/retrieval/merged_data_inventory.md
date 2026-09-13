# Merged Retrieval Data Inventory

Generated: 2026-08-28 08:34:59 UTC

## Inputs
- JSONL: `data/retrieval/merged/oax_slim.jsonl`
- Parsed Markdown directory: `data/retrieval/merged/mds`
- PDF directory: `data/retrieval/merged/pdfs`

## Record Coverage

| Measure | Count | Percentage of JSONL entries |
| --- | ---: | ---: |
| Total valid JSONL entries | 334,893 | 100.00% |
| Works with parsed Markdown full text | 108,167 | 32.30% |
| Works without parsed Markdown full text | 226,726 | 67.70% |

## Artifact Counts

- Parsed Markdown files found: 108,167
- PDF files found: 88,686

## JSONL Field Coverage

A field is counted when its value is not null, not an empty string, and not an empty list or object.

| Field | Populated entries | Coverage |
| --- | ---: | ---: |
| `abstract` | 232,628 | 69.46% |
| `authors` | 334,350 | 99.84% |
| `cited_by_count` | 334,893 | 100.00% |
| `concepts` | 326,629 | 97.53% |
| `doi` | 331,824 | 99.08% |
| `field` | 331,479 | 98.98% |
| `id` | 334,893 | 100.00% |
| `is_oa` | 334,893 | 100.00% |
| `keywords` | 326,629 | 97.53% |
| `language` | 334,893 | 100.00% |
| `pdf_url` | 222,368 | 66.40% |
| `referenced_works` | 334,893 | 100.00% |
| `referenced_works_count` | 334,893 | 100.00% |
| `source` | 327,151 | 97.69% |
| `subfield` | 331,479 | 98.98% |
| `title` | 334,893 | 100.00% |
| `topics` | 331,479 | 98.98% |
| `type` | 334,893 | 100.00% |
| `year` | 334,891 | 100.00% |

## Primary Field Distribution

Percentages use records with a populated `field` value as the denominator.

| Primary field | Entries | Percentage |
| --- | ---: | ---: |
| Agricultural and Biological Sciences | 3,696 | 1.12% |
| Arts and Humanities | 797 | 0.24% |
| Biochemistry, Genetics and Molecular Biology | 13,591 | 4.10% |
| Business, Management and Accounting | 3,049 | 0.92% |
| Chemical Engineering | 36 | 0.01% |
| Chemistry | 247 | 0.07% |
| Computer Science | 4,921 | 1.48% |
| Decision Sciences | 1,484 | 0.45% |
| Dentistry | 8,935 | 2.70% |
| Earth and Planetary Sciences | 197 | 0.06% |
| Economics, Econometrics and Finance | 2,695 | 0.81% |
| Energy | 199 | 0.06% |
| Engineering | 5,642 | 1.70% |
| Environmental Science | 6,225 | 1.88% |
| Health Professions | 16,391 | 4.94% |
| Immunology and Microbiology | 4,670 | 1.41% |
| Materials Science | 485 | 0.15% |
| Mathematics | 526 | 0.16% |
| Medicine | 210,997 | 63.65% |
| Neuroscience | 8,979 | 2.71% |
| Nursing | 3,571 | 1.08% |
| Pharmacology, Toxicology and Pharmaceutics | 1,019 | 0.31% |
| Physics and Astronomy | 180 | 0.05% |
| Psychology | 20,906 | 6.31% |
| Social Sciences | 11,526 | 3.48% |
| Veterinary | 515 | 0.16% |

## Data Quality Notes

- Malformed or non-object JSONL lines skipped: 0
- JSONL entries without an identifiable OpenAlex work ID: 0
- Full-text work coverage is determined from Markdown files using the final `W<digits>` token in each Markdown path.
