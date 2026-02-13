"""
Slimming OpenAlex Records for Systematic Reviews
- Takes the filtered OpenAlex records that have valid PDFs
- Extracts only the essential fields needed for LLM processing and alignment
- Reconstructs the abstract text from the inverted index and cleans it up
- Simplifies the authorship information to just names and affiliations
- Saves the slimmed records to a new JSON file for downstream processing (alignment, extraction, etc.)
- Logs progress and any issues encountered during slimming
"""
import json
import logging
import os
import re
from typing import Dict, Any, List, Optional
from tqdm import tqdm

# =========================
# CONFIG
# =========================
INPUT_JSON  = "./data/filtered/ft_subset/oax_sr_refs_title_doi_pdf_downloaded_filtered.json"
OUTPUT_JSON = "./data/filtered/ft_subset/oax_sr_slim.json"
LOG_FILE    = "./logs/retrieval/oax_slim_ft.log"

# =========================
# Setup
# =========================
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def reconstruct_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
    """
    Reconstructs the abstract and cleans up the 'Abstract' prefix artifact.
    """
    if not inverted_index:
        return ""
    
    # 1. Find max index
    max_index = 0
    for positions in inverted_index.values():
        if positions:
            max_index = max(max_index, max(positions))
            
    # 2. Rebuild text
    text_list = [""] * (max_index + 1)
    for word, positions in inverted_index.items():
        for pos in positions:
            text_list[pos] = word
            
    raw_text = " ".join(text_list)

    # 3. Clean up leading "Abstract" 
    # ^ = start of string, \s+ = one or more spaces
    # flags=re.IGNORECASE handles "Abstract", "abstract", "ABSTRACT"
    clean_text = re.sub(r"^Abstract\s+", "", raw_text, flags=re.IGNORECASE)

    return clean_text.strip()

def simplify_authors(authorships: List[Dict]) -> List[Dict[str, Any]]:
    """
    Returns a list of authors where affiliations are just a simple list of strings.
    """
    simple_authors = []
    
    for auth in authorships:
        a_profile = auth.get("author") or {}
        
        # EXTRACT: Simple list of institution names (Strings only)
        inst_names = []
        for inst in auth.get("institutions", []):
            if inst.get("display_name"):
                inst_names.append(inst.get("display_name"))
        
        # Fallback: If no structured institutions, check raw string
        if not inst_names:
            raw_affs = auth.get("raw_affiliation_strings") or []
            inst_names = raw_affs

        simple_authors.append({
            "id": a_profile.get("id"),
            "name": a_profile.get("display_name"),
            "affiliations": inst_names 
        })
        
    return simple_authors

def extract_pdf_link(rec: Dict) -> Optional[str]:
    """Tries to find the best direct PDF link."""
    pl = rec.get("primary_location") or {}
    if pl.get("pdf_url"): return pl.get("pdf_url")
    boa = rec.get("best_oa_location") or {}
    if boa.get("pdf_url"): return boa.get("pdf_url")
    return None

def process_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Maps raw OpenAlex record to Slim Schema."""
    
    # 1. Topics processing
    primary_topic = rec.get("primary_topic") or {}
    all_topics = [t.get("display_name") for t in rec.get("topics", [])]
    
    # 2. Field/Subfield
    field = primary_topic.get("field", {}).get("display_name")
    subfield = primary_topic.get("subfield", {}).get("display_name")
    
    # 3. Keywords/Concepts (Top 15 merged)
    keywords = [k.get("display_name") for k in rec.get("keywords", [])]
    concepts = [c.get("display_name") for c in rec.get("concepts", [])]
    combined_tags = list(set(keywords + concepts))[:15]

    # 4. Source (Journal)
    source = rec.get("primary_location", {}).get("source", {}).get("display_name")

    # 5. Extract Abstract (Cleaned)
    abstract_text = reconstruct_abstract(rec.get("abstract_inverted_index"))

    return {
        "id": rec.get("id"),
        "title": rec.get("display_name") or rec.get("title"),
        "doi": rec.get("doi"),
        "abstract": abstract_text,
        "year": rec.get("publication_year"),
        # "date": Removed as requested
        "type": rec.get("type"),
        "source": source,
        "cited_by_count": rec.get("cited_by_count"),
        
        # --- GROUND TRUTH ---
        "referenced_works_count": rec.get("referenced_works_count"),
        "referenced_works": rec.get("referenced_works", []), 
        # --------------------

        "pdf_url": extract_pdf_link(rec),
        "language": rec.get("language"),
        
        # Taxonomy
        "field": field,
        "subfield": subfield,
        "topics": all_topics,
        "keywords": combined_tags,
        
        # People (Simple affiliations list)
        "authors": simplify_authors(rec.get("authorships", []))
    }

# =========================
# Main
# =========================
def main():
    print(f"Reading from: {INPUT_JSON}")
    
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    slim_data = []
    
    for rec in tqdm(data, desc="Slimming Records"):
        try:
            slim_rec = process_record(rec)
            slim_data.append(slim_rec)
        except Exception as e:
            logging.error(f"Error processing {rec.get('id')}: {e}")
            
    print(f"Saving {len(slim_data)} records to: {OUTPUT_JSON}")
    
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(slim_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()