import csv
import hashlib
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

INPUT_CSVS = [
    os.path.join(PROJECT_ROOT, 'data/labelled/regex_fp_reclassified_hq.csv'),
    os.path.join(PROJECT_ROOT, 'data/labelled/regex_fn_reclassified.csv'),
]

AI_DIR = os.path.join(PROJECT_ROOT, 'data/extracted_descriptions/ai')
NON_AI_DIR = os.path.join(PROJECT_ROOT, 'data/extracted_descriptions/non_ai')

os.makedirs(AI_DIR, exist_ok=True)
os.makedirs(NON_AI_DIR, exist_ok=True)

DIR_MAP = {'ai': AI_DIR, 'non_ai': NON_AI_DIR}

written = 0
skipped_existing = 0
skipped_invalid = 0

for csv_path in INPUT_CSVS:
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            descrizione = row.get('descrizione', '').strip()
            label = row.get('predicted_label', '').strip()

            if not descrizione or label not in DIR_MAP:
                skipped_invalid += 1
                continue

            hash8 = hashlib.md5(descrizione.encode()).hexdigest()[:8]
            filename = f"{label}_{hash8}.txt"
            filepath = os.path.join(DIR_MAP[label], filename)

            if os.path.exists(filepath):
                skipped_existing += 1
                continue

            with open(filepath, 'w', encoding='utf-8') as out:
                out.write(descrizione)
            written += 1

total = written + skipped_existing + skipped_invalid
print(f"Totale righe processate : {total}")
print(f"File scritti            : {written}")
print(f"File saltati (esistenti): {skipped_existing}")
print(f"Righe skippate (invalide): {skipped_invalid}")
