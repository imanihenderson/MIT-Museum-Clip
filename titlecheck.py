import csv

field = ["museum"]
targets = {"Louvre Museum"}

def find_matching_rows_with_numbers(csv_path, limit=None):
    results = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if "museum" not in reader.fieldnames:
            raise ValueError(f"'museum' column not found. Columns: {reader.fieldnames}")

        for row_num, row in enumerate(reader, start=2):  # omits header 
            museum_val = (row.get("museum") or "").strip()

            if museum_val in targets:
                results.append((row_num, row))  # row is a dict of all columns
                if limit and len(results) >= limit:
                    break

    return results


matches = find_matching_rows_with_numbers("CLIP_items.csv")

if matches:
    for row_num, row in matches:
        print(row_num, row["museum"])
else:
    print(f"No matching 'museum' values found for targets: {targets}")






