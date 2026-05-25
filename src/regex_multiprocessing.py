import os
import re
import pandas as pd
from multiprocessing import Pool, cpu_count
import glob

# Compile regexes globally so workers don't recompile them
REGEX_FORMAZIONE = re.compile(r'\b(formazione|corso|training|tirocinio|aggiornamento|docenza)\b', re.IGNORECASE)
REGEX_IMPLEMENTAZIONE = re.compile(r'\b(sviluppo|implementazione|creazione|produzione|realizzazione)\b', re.IGNORECASE)

def process_chunk(chunk):
    """
    Processa un chunk di DataFrame, assegnando le etichette in base alle regex.
    Ritorna un subset del chunk contenente solo i record etichettati, 
    ed eliminando copie duplicate per la colonna DESCRIZIONE_PROGETTO.
    """
    if "DESCRIZIONE_PROGETTO" not in chunk.columns:
        return pd.DataFrame()
    
    # Drop rows without description
    chunk = chunk.dropna(subset=["DESCRIZIONE_PROGETTO"]).copy()
    chunk["DESCRIZIONE_PROGETTO_CLEAN"] = chunk["DESCRIZIONE_PROGETTO"].astype(str).str.strip()
    
    # Remove duplicates within the chunk to speed up
    chunk = chunk.drop_duplicates(subset=["DESCRIZIONE_PROGETTO_CLEAN"])
    
    def classify_text(text):
        if pd.isna(text):
            return None
        text_str = str(text)
        if REGEX_FORMAZIONE.search(text_str):
            return "formazione"
        elif REGEX_IMPLEMENTAZIONE.search(text_str):
            return "implementazione"
        return None

    chunk["label"] = chunk["DESCRIZIONE_PROGETTO_CLEAN"].apply(classify_text)
    
    # Filter only classified records
    classified = chunk[chunk["label"].notna()]
    
    return classified[["DESCRIZIONE_PROGETTO_CLEAN", "label"]].rename(
        columns={"DESCRIZIONE_PROGETTO_CLEAN": "DESCRIZIONE_PROGETTO"}
    )

def process_file(filepath):
    """
    Processa un singolo file CSV in chunk per ridurre l'impatto in memoria.
    """
    results = []
    try:
        # Use simple try-except to handle possibly huge files 
        # Read in chunks
        chunk_iter = pd.read_csv(filepath, chunksize=50000, 
                                 usecols=lambda c: c == "DESCRIZIONE_PROGETTO", 
                                 low_memory=False, 
                                 on_bad_lines='skip')
        for chunk in chunk_iter:
            classified_chunk = process_chunk(chunk)
            if not classified_chunk.empty:
                results.append(classified_chunk)
                
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        
    if results:
        res_df = pd.concat(results, ignore_index=True)
        res_df = res_df.drop_duplicates(subset=["DESCRIZIONE_PROGETTO"])
        return res_df
    return pd.DataFrame(columns=["DESCRIZIONE_PROGETTO", "label"])

def run_extraction(data_dir, target_per_class=5000):
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    print(f"Trovati {len(files)} file CSV da processare.")
    
    final_df = pd.DataFrame(columns=["DESCRIZIONE_PROGETTO", "label"])
    
    num_workers = min(cpu_count(), len(files))
    if num_workers == 0:
        return final_df
        
    with Pool(num_workers) as pool:
        for file_res in pool.imap_unordered(process_file, files):
            if not file_res.empty:
                final_df = pd.concat([final_df, file_res], ignore_index=True)
                final_df = final_df.drop_duplicates(subset=["DESCRIZIONE_PROGETTO"])
                
                count_formazione = (final_df["label"] == "formazione").sum()
                count_implementazione = (final_df["label"] == "implementazione").sum()
                
                print(f"Campioni aggregati - Formazione: {count_formazione}, Implementazione: {count_implementazione}")
                if count_formazione >= target_per_class and count_implementazione >= target_per_class:
                    print("Raggiunto il target desiderato per entrambe le classi!")
                    # Ferma il processo early (i worker attivi finiranno, ma non importa)
                    pool.terminate()
                    break
                    
    # Assicurati di non avere duplicati e bilancia / riduci ai soli sample voluti
    final_df = final_df.drop_duplicates(subset=["DESCRIZIONE_PROGETTO"])
    
    df_formazione = final_df[final_df["label"] == "formazione"]
    df_implementazione = final_df[final_df["label"] == "implementazione"]
    
    # Subsampling
    if len(df_formazione) > target_per_class:
        df_formazione = df_formazione.sample(n=target_per_class, random_state=42)
    if len(df_implementazione) > target_per_class:
        df_implementazione = df_implementazione.sample(n=target_per_class, random_state=42)
        
    final_labeled = pd.concat([df_formazione, df_implementazione]).sample(frac=1, random_state=42).reset_index(drop=True)
    return final_labeled
