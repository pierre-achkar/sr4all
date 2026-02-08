"""
This script generates a bar plot to visualize the number of documents at each stage of the SR4All corpus construction and filtering process. 
It uses the Seaborn library to create a visually appealing barchart, with custom styling and annotations to highlight the document counts at each step. 
"""

import seaborn as sns
import matplotlib.pyplot as plt

steps = [
    "Initial retrieval",
    "Deduplication",
    "Ref. list available",
    "English (metadata)",
    "DOI available",
    "Resolvable full-text URL",
    "Title heuristics",
    "PDF acquisition",
    "English (full text)",
    "Context-length filter"
]

counts = [
    485_446,
    465_103,
    377_461,
    374_415,
    370_914,
    224_233,
    137_621,
    65_215,
    64_699,
    63_977
]

sns.set_theme(style="whitegrid")

plt.figure(figsize=(11, 5))
palette = ["#4C72B0"] * len(steps)
ax = sns.barplot(x=steps, y=counts, palette=palette)
ax.set_ylabel("Number of documents")
ax.set_xlabel("")
ax.set_title("SR4ALL corpus construction and filtering stages")
plt.xticks(rotation=30, ha="right")

_orig_text = ax.text
def _bold_text(*args, **kwargs):
    kwargs.pop("fontsize", None)
    kwargs.setdefault("fontsize", 11)
    kwargs.setdefault("fontweight", "bold")
    return _orig_text(*args, **kwargs)
ax.text = _bold_text

for i, v in enumerate(counts):
    ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("/home/fhg/pie65738/projects/sr4all/data/sr4all/sr4all_corpus_filtering_figure.png", dpi=300)
plt.close()
