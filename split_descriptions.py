import csv
import os

input_csv = 'data/distilled/merged_descriptions.csv'
output_base_dir = 'data/extracted_descriptions'

ai_dir = os.path.join(output_base_dir, 'ai')
non_ai_dir = os.path.join(output_base_dir, 'non_ai')

# Ensure directories exist
os.makedirs(ai_dir, exist_ok=True)
os.makedirs(non_ai_dir, exist_ok=True)

with open(input_csv, 'r', encoding='utf-8') as f_in:
    reader = csv.DictReader(f_in)
    
    # Keep track of file counts to generate unique names
    ai_count = 1
    non_ai_count = 1
    
    for row in reader:
        description = row.get('Description')
        label = row.get('Label')
        
        if not description or not label:
            continue
            
        if label == 'ai':
            file_name = f"ai_desc_{ai_count:03d}.txt"
            file_path = os.path.join(ai_dir, file_name)
            ai_count += 1
        elif label == 'non_ai':
            file_name = f"non_ai_desc_{non_ai_count:03d}.txt"
            file_path = os.path.join(non_ai_dir, file_name)
            non_ai_count += 1
        else:
            continue
            
        with open(file_path, 'w', encoding='utf-8') as f_out:
            f_out.write(description)

print(f"File estratti con successo: {ai_count - 1} in '{ai_dir}' e {non_ai_count - 1} in '{non_ai_dir}'.")
