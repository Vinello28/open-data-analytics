import pandas as pd
import os
import re

CHUNK_SIZE = 100_000

def get_mask(series):
    p1 = r'\b(?:tracciabilit[aà]|rintracciabilit[aà]|tracciamento|autenticazione|blockchain|registri distribuiti|distributed ledger technology|provenienza del prodotto|origine certificata|qr code|codice qr|codice a barre bidirezionale|filiera|caten[ae] del valore|(?:catena|sistema) di approvvigionamento|supply chain|internet of things|sicurezza del prodotto|smart logistics)\b'
    m1 = series.str.contains(p1, regex=True)
    
    m2 = series.str.contains(r'\bmonitoraggio\b', regex=True) & series.str.contains(r'\bend-to-end\b', regex=True)
    m3 = series.str.contains(r'\blocalizzazione\b', regex=True) & series.str.contains(r'\b(?:prodotto|materia)\b', regex=True)
    m4 = series.str.contains(r'\bidentificazione\b', regex=True) & series.str.contains(r'\b(?:prodotto|materia)\b', regex=True)
    m5 = series.str.contains(r'\bindustria 4\.0\b', regex=True) & series.str.contains(r'\btracciabilit[aà]\b', regex=True)
    m6 = series.str.contains(r'\blogistica\b', regex=True) & series.str.contains(r'\bintelligente\b', regex=True)
    
    return m1 | m2 | m3 | m4 | m5 | m6

def process_chunk(chunk):
    titolo = chunk['TITOLO_PROGETTO'].fillna('').str.lower()
    descrizione = chunk['DESCRIZIONE_PROGETTO'].fillna('').str.lower()
    
    mask_titolo = get_mask(titolo)
    mask_descrizione = get_mask(descrizione)
    
    final_mask = mask_titolo | mask_descrizione
    return chunk[final_mask]

def process_file(args):
    file_path, output_dir = args
    filename = os.path.basename(file_path)
    match = re.search(r'aiuti_(\d{4})\.csv', filename)
    if not match:
        return filename, 0
    year = match.group(1)
    output_path = os.path.join(output_dir, f'traceability_aiuti_{year}.csv')
    
    print(f"[{filename}] Inizio elaborazione...")
    try:
        chunk_iterator = pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=False)
    except Exception as e:
        print(f"[{filename}] Errore apertura: {e}")
        return filename, 0
        
    first = True
    total_matches = 0
    
    for chunk in chunk_iterator:
        filtered_chunk = process_chunk(chunk)
        if not filtered_chunk.empty:
            total_matches += len(filtered_chunk)
            filtered_chunk.to_csv(output_path, mode='a', index=False, header=first)
            first = False
            
    print(f"[{filename}] Fine elaborazione. Match trovati: {total_matches}")
    return filename, total_matches
