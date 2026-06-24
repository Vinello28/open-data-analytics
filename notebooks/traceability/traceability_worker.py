import pandas as pd
import os
import re

CHUNK_SIZE = 100_000

def get_mask(series):
    # 1. Termini forti inequivocabili (bastano da soli)
    p_core = r'\b(?:(?:rin)?tracciabilit[aà]|provenienza del prodotto|origine certificata|passaporto digitale)\b'
    m_core = series.str.contains(p_core, regex=True)
    
    # 2. Azione + Oggetto (es. "tracciamento" ma solo se associato a "prodotti", "filiera", ecc.)
    p_action = r'\b(?:tracciamento|tracciatura|identificazione|monitoraggio|autenticazione)\b'
    p_target = r'\b(?:prodotto|prodotti|materia|materie|filiera|supply chain|merce|merci|lotto|lotti|end-to-end)\b'
    m_action_target = series.str.contains(p_action, regex=True) & series.str.contains(p_target, regex=True)
    
    # 3. Tecnologie + Contesto (es. "Internet of Things" o "Blockchain" ma solo se si cita filiera, provenienza o logistica)
    p_tech = r'\b(?:blockchain|registri distribuiti|distributed ledger|qr code|codice qr|rfid|nfc|internet of things|iot)\b'
    p_tech_target = r'\b(?:filiera|supply chain|provenienza|logistica|anticontraffazione)\b'
    m_tech_target = series.str.contains(p_tech, regex=True) & series.str.contains(p_tech_target, regex=True)
    
    # 4. Filiera/Supply chain + Attributi di trasparenza/controllo
    p_chain = r'\b(?:filiera|caten[ae] del valore|supply chain)\b'
    p_chain_mod = r'\b(?:trasparente|trasparenza|sicurezza alimentare|certificata|garantita)\b'
    m_chain = series.str.contains(p_chain, regex=True) & series.str.contains(p_chain_mod, regex=True)
    
    # 5. Manteniamo alcune delle vecchie combinazioni sensate rendendole più robuste
    m5 = series.str.contains(r'\bindustria 4\.0\b', regex=True) & series.str.contains(r'\b(?:rin)?tracciabilit[aà]\b', regex=True)
    m6 = series.str.contains(r'\blogistica\b', regex=True) & series.str.contains(r'\bintelligente\b', regex=True) & series.str.contains(r'\b(?:merci|prodotti|tracciamento)\b', regex=True)

    return m_core | m_action_target | m_tech_target | m_chain | m5 | m6

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
