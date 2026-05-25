import os
import glob
import pandas as pd
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

DATA_DIR = '../data'
OUTPUT_DIR = os.path.join(DATA_DIR, 'raw')
CHUNK_SIZE = 100000

# Regex pattern as defined in the notebook
AI_REGEX_PATTERN = r'(?i:\b(intelligenza artificiale|robotica collaborativa|visione artificiale|algoritmi predittivi|manutenzione preventiva|predictive analytics|artificial intelligence|machine learning|deep learning|reti neurali|computer vision|ai generativa|elaborazione del linguaggio naturale|nlp|llm|virtual assistant|virtual assistants|guida autonoma|chatbot|classificazione semantica)\b)|\b(IA|tecnologie AI|tecnologie IA|modelli di AI|algoritmi di AI|algoritmi di IA|LLM)\b'
ai_regex = re.compile(AI_REGEX_PATTERN)

MULTICLASS_COLS = ['CLASSIFICAZIONE_MULTICLASS', 'CLASSIFICAZIONE_MULTICLASS_CONFIDENZA']

def get_manual_ai_set():
    manual_labels_path = os.path.join(DATA_DIR, 'labelled', 'regex_fn_reclassified.csv')
    if os.path.exists(manual_labels_path):
        df_manual = pd.read_csv(manual_labels_path)
        ai_manual = df_manual[df_manual['predicted_label'].str.lower().str.strip() == 'ai']
        return set(ai_manual['descrizione'].fillna('').str.strip())
    return set()

MANUAL_AI_SET = get_manual_ai_set()

def process_reclassification_file(filepath):
    filename = os.path.basename(filepath)
    new_filename = filename.replace('classified_', 'reclassified_')
    out_filepath = os.path.join(OUTPUT_DIR, new_filename)
    
    chunk_iter = pd.read_csv(filepath, chunksize=CHUNK_SIZE, dtype=str)
    
    first_chunk = True
    for chunk in chunk_iter:
        desc = chunk['DESCRIZIONE_PROGETTO'].fillna('')
        
        regex_preds = desc.str.contains(ai_regex, regex=True).fillna(False)
        in_manual_set = desc.str.strip().isin(MANUAL_AI_SET)
        
        is_ai_current = (chunk['CLASSIFICAZIONE'].fillna('').str.upper() == 'AI')
        keep_ai_mask = (regex_preds | in_manual_set) & is_ai_current
        
        chunk.loc[~keep_ai_mask, 'CLASSIFICAZIONE'] = 'NON_AI'
        
        for col in MULTICLASS_COLS:
            if col in chunk.columns:
                chunk.loc[~keep_ai_mask, col] = pd.NA
                
        mode = 'w' if first_chunk else 'a'
        chunk.to_csv(out_filepath, index=False, mode=mode, header=first_chunk, quoting=1)
        first_chunk = False
        
    return f"{filename} -> {new_filename}"

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Caricate {len(MANUAL_AI_SET)} descrizioni ri-classificate manualmente come 'AI' dal merge.")
    
    files = glob.glob(os.path.join(DATA_DIR, 'classified_multiclass_aiuti_*.csv'))
    print(f"\nInizio elaborazione MULTIPROCESSING di {len(files)} file per la ri-classificazione annuale...")

    MAX_RECLASS_WORKERS = min(16, len(files))

    with ProcessPoolExecutor(max_workers=MAX_RECLASS_WORKERS) as executor:
        futures = [executor.submit(process_reclassification_file, f) for f in files]
        
        for future in as_completed(futures):
            try:
                res = future.result()
                print(f"Completato: {res}")
            except Exception as e:
                print(f"Errore durante l'elaborazione di un file: {e}")

    print(f"\nTutti i dataset annuali riclassificati sono stati estratti in: {OUTPUT_DIR}")