"""Estrae in multiprocessing tutte le righe CLASSIFICAZIONE == 'AI' dai file
data/raw/reclassified_multiclass_aiuti_*.csv in un unico CSV solo-AI.

Riusa il pattern di src/regex_multiprocessing.py (Pool su file + lettura a chunk).
Eseguibile da CLI o importabile (run()).
"""
import os
import glob
import time
from multiprocessing import Pool, cpu_count

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw")
OUTPUT = os.path.join(REPO_ROOT, "data", "distilled", "ai_subset.csv")

RAW_GLOB = "reclassified_multiclass_aiuti_*.csv"
CHUNK_SIZE = 100_000

# Colonne tenute nel sottoinsieme. CAR/COR/ANNO servono come chiave per la futura
# propagazione in-place; DESCRIZIONE_PROGETTO e' il testo da analizzare.
KEEP_COLS = [
    "ANNO", "CAR", "COR", "TITOLO_PROGETTO", "DESCRIZIONE_PROGETTO",
    "CLASSIFICAZIONE_MULTICLASS", "TIPO_AI",
]
READ_COLS = set(KEEP_COLS) | {"CLASSIFICAZIONE"}


def process_file(filepath):
    """Legge un file a chunk e ritorna (filepath, DataFrame righe AI, errore|None)."""
    results = []
    try:
        chunk_iter = pd.read_csv(
            filepath,
            chunksize=CHUNK_SIZE,
            usecols=lambda c: c in READ_COLS,
            dtype=str,
            low_memory=False,
            on_bad_lines="skip",
        )
        for chunk in chunk_iter:
            if "CLASSIFICAZIONE" not in chunk.columns or "DESCRIZIONE_PROGETTO" not in chunk.columns:
                continue
            mask = chunk["CLASSIFICAZIONE"].astype(str).str.strip().str.upper() == "AI"
            sub = chunk[mask].dropna(subset=["DESCRIZIONE_PROGETTO"]).copy()
            sub = sub[sub["DESCRIZIONE_PROGETTO"].astype(str).str.strip() != ""]
            if not sub.empty:
                results.append(sub)
    except Exception as e:  # file enorme/corrotto: non bloccare gli altri
        return filepath, pd.DataFrame(), str(e)

    if results:
        df = pd.concat(results, ignore_index=True)
        for col in KEEP_COLS:
            if col not in df.columns:
                df[col] = pd.NA
        return filepath, df[KEEP_COLS], None
    return filepath, pd.DataFrame(columns=KEEP_COLS), None


def run(raw_dir=RAW_DIR, output=OUTPUT, verbose=True):
    files = sorted(glob.glob(os.path.join(raw_dir, RAW_GLOB)))
    if verbose:
        print(f"Trovati {len(files)} file in {raw_dir}")
    if not files:
        return pd.DataFrame(columns=KEEP_COLS)

    n_workers = min(cpu_count(), len(files))
    frames = []
    t0 = time.time()
    with Pool(n_workers) as pool:
        for filepath, df, err in pool.imap_unordered(process_file, files):
            name = os.path.basename(filepath)
            if err:
                print(f"  [ERRORE] {name}: {err}")
            elif verbose:
                print(f"  {name}: {len(df)} righe AI")
            if not df.empty:
                frames.append(df)

    final = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=KEEP_COLS)
    final = final.drop_duplicates().reset_index(drop=True)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    final.to_csv(output, index=False)
    if verbose:
        print(f"\nTotale righe AI: {len(final)} -> {output}")
        print(f"Tempo: {time.time() - t0:.1f}s")
        if "ANNO" in final.columns:
            print("Per anno:\n", final["ANNO"].value_counts().sort_index().to_string())
    return final


if __name__ == "__main__":
    run()
