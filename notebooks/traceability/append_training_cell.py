import json
import os

nb_path = '/home/gabs/Documenti/Università/AI nelle Imprese/open-data-analytics/notebooks/traceability/traceability_extraction.ipynb'

with open(nb_path, 'r') as f:
    nb = json.load(f)

source_code = """import glob
import os
import pandas as pd

# Percorsi
input_dir = '../../data/traceability'
output_dir = os.path.join(input_dir, 'training')
os.makedirs(output_dir, exist_ok=True)

# Trova i file
file_pattern = os.path.join(input_dir, 'traceability_aiuti_*.csv')
csv_files = glob.glob(file_pattern)

print(f"Trovati {len(csv_files)} file da aggregare.")

# Carica e concatena
dfs = []
for file in csv_files:
    try:
        df = pd.read_csv(file, usecols=['TITOLO_PROGETTO', 'DESCRIZIONE_PROGETTO'])
        dfs.append(df)
    except Exception as e:
        print(f"Errore nella lettura di {file}: {e}")

if dfs:
    final_df = pd.concat(dfs, ignore_index=True)
    # Aggiungi colonna Tracciabilita
    final_df['Tracciabilita'] = True
    
    # Salva il file
    output_path = os.path.join(output_dir, 'traceability_training_dataset.csv')
    final_df.to_csv(output_path, index=False)
    print(f"Dataset di training salvato in: {output_path}")
    print(f"Totale righe: {len(final_df)}")
else:
    print("Nessun dato trovato da aggregare.")
"""

new_cell = {
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [line + "\n" if i < len(source_code.split('\n')) - 1 else line for i, line in enumerate(source_code.split('\n'))]
}

nb['cells'].append(new_cell)

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)
