import pandas as pd
import os

def get_unique_projects(filepath):
    try:
        # Carica solo le colonne necessarie per ridurre al minimo l'uso della RAM
        df = pd.read_csv(filepath, usecols=['TITOLO_PROGETTO', 'DESCRIZIONE_PROGETTO'])
        df = df.dropna(subset=['DESCRIZIONE_PROGETTO'])
        # Rimuove subito i duplicati locali basandosi solo sulla descrizione (come da analisi)
        df = df.drop_duplicates(subset=['DESCRIZIONE_PROGETTO'])
        return df
    except Exception as e:
        print(f"Errore nella lettura di {filepath}: {e}")
        return None
