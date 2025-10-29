import csv
import random
from collections import defaultdict
from typing import Dict, List

from keybert import KeyBERT  # pip install keybert sentence-transformers
# moving keybert usage here to only extract keywords during dataset prep


# Config

IN_CSV = "data/train.csv"
OUT_CSV = "data/train_prompted.csv"
PROMPT_TXT = "promptstructures.txt"  # or "data/promptstructures.txt"
RANDOM_SEED = 42          # set to none for completely random behavior, but this keeps it deterministic 
KEYBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # from sentence-transformers
KEYBERT_TOP_N = 10
KEYBERT_USE_MMR = True # maximum marginal relevance to avoid redudancy but maintain relevance 
KEYBERT_DIVERSITY = 0.6 # rendundance penalty for mmr

# IMPORTANT: match to actual CSV column names exactly (case / spacing)
fields = [
    "Title",
    "Category",
    "Culture",
    "Country",
    "Dates",
    "Material and Techniques",
    "Inscription",
    "Era of extraction", 
    "Museum",
]

if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)


def load_prompt_map(path: str) -> Dict[str, List[str]]:
    mp = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "::" not in line:
                continue
            key, tmpl = line.split("::", 1)
            key, tmpl = key.strip(), tmpl.strip()
            if tmpl:
                mp[key].append(tmpl)
    return mp

prompts_map = load_prompt_map(PROMPT_TXT)

# normalization per specific fields 
def transform_value(field: str, val: str) -> str:
    if val is None:
        return ""
    s = ' '.join(str(val).split()) # normalize whitespace 
    if field in {"Category", "Material and Techniques"}:
        s = s.lower()
    return s

# keybert on the title only 
kw_model = KeyBERT(model=KEYBERT_MODEL)

def compress_title_with_keybert(title_text: str) -> str:
    if not title_text or not str(title_text).strip():
        return ""
    try:
        keywords = kw_model.extract_keywords(
            str(title_text),
            keyphrase_ngram_range=(1, 3) #1-3 word phrases,
            stop_words="english",
            use_mmr=KEYBERT_USE_MMR,
            diversity=KEYBERT_DIVERSITY,
            top_n=KEYBERT_TOP_N,
        )
        # keep only keyword strings; join for a compact Title
        return "; ".join([kw for kw, _ in keywords]) or str(title_text)
    except Exception:
        # fallback: return original on any error
        return str(title_text)

# rendering 
def render_field(field: str, value: str) -> str:
    if value is None or str(value).strip() == "":
        return ""
    templates = prompts_map.get(field)
    if not templates:
        return str(value)
    template = random.choice(templates)
    return template.format(value=transform_value(field, value))

def transform_row(row: dict) -> dict:
    new_row = dict(row)

    # 1) Pre-step: first convert title to KeyBERT keywords
    if "Title" in new_row:
        new_row["Title"] = compress_title_with_keybert(new_row["Title"])

    # 2) For each field, replace with a random per-field prompt
    for field in fields:
        if field in new_row:
            new_row[field] = render_field(field, new_row[field])

    return new_row


# write new csv
def write_prompted_csv(in_csv: str, out_csv: str):
    with open(in_csv, "r", newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames
        assert fieldnames is not None, "CSV appears empty or malformed."
        rows_out = [transform_row(row) for row in reader]

    with open(out_csv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

if __name__ == "__main__":
    write_prompted_csv(IN_CSV, OUT_CSV)
    print(f"Wrote {OUT_CSV}")