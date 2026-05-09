import csv
import os

input_files = [
    ('data/distilled/ai.txt', 'ai'),
    ('data/distilled/non_ai.txt', 'non_ai')
]
output_file = 'data/distilled/merged_descriptions.csv'

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, 'w', encoding='utf-8', newline='') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(['Description', 'Label'])
    
    for file_path, label in input_files:
        with open(file_path, 'r', encoding='utf-8') as f_in:
            reader = csv.DictReader(f_in, delimiter=';')
            for row in reader:
                description = row.get('Descrizione', '')
                if description:
                    writer.writerow([description, label])

print(f"File CSV '{output_file}' creato con successo.")
