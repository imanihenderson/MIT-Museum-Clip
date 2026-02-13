import csv
import random
from collections import defaultdict
from typing import Dict, List


# Config

IN_CSV = "data/train.csv"
OUT_CSV = "data/train_prompted.csv"
PROMPT_TXT = "promptstructures.txt"  # or "data/promptstructures.txt"
RANDOM_SEED = 42          # set to none for completely random behavior, but this keeps it deterministic 
ROW_KEY = "ROW"
OUT_PROMPT_COL = "prompt"


fields = [
    "id",
    "country",
    "itemlat",
    "itemlon",
    "murdock_name",
    "Acq.date",
    "museum",
    "categorization",
    "is_fragment",
    "materials",
    "Image1", "Image2", "Image3", "Image4", "Image5"
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
    if field in {"categorization", "materials"}:
        s = s.lower()
    return s

def render_row_prompt(row: dict) -> str:
    templates = prompts_map.get(ROW_KEY)
    if not templates:
        return ""
    template = random.choice(templates)

    safe = defaultdict(str)
    for k, v in row.items():
        safe[k] = transform_value(k, v)

    try:
        return template.format_map(safe)
    
    except Exception as e:
        raise ValueError(f"bad row template: {template}") from e
    
def transform_row(row: dict) -> dict:
    new_row = dict(row)
    new_row[OUT_PROMPT_COL] = render_row_prompt(row)
    return new_row


# write new csv
def write_prompted_csv(in_csv: str, out_csv: str):
    with open(in_csv, "r", newline="", encoding="utf-8") as fin, \
         open(out_csv, "w", newline="", encoding="utf-8") as fout:
        
        
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames
        assert fieldnames is not None, "CSV appears empty or malformed."
        if OUT_PROMPT_COL not in fieldnames:
            fieldnames = fieldnames + [OUT_PROMPT_COL]

        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            writer.writerow(transform_row(row))

if __name__ == "__main__":
    write_prompted_csv(IN_CSV, OUT_CSV)
    print(f"Wrote {OUT_CSV}")