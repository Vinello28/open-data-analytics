"""Genera un dataset etichettato aggiungendo la colonna TECNOLOGIE_AI ai file
reclassified_multiclass_aiuti_*.csv, SENZA toccare gli originali.

Per ogni file: legge a chunk, calcola le tecnologie sulle righe CLASSIFICAZIONE=='AI'
(vuoto per le NON_AI), aggiunge la colonna in coda e scrive una copia in
data/technology_mapping/ mantenendo il formato dei raw (quoting=csv.QUOTE_NONE,
escapechar='\\\\', come src/reclassify_annual.py).
"""
import os
import csv
import glob
import time
from functools import partial
from multiprocessing import Pool, cpu_count

import pandas as pd

from .methods import regex_gazetteer as rg

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw")
OUT_DIR = os.path.join(REPO_ROOT, "data", "technology_mapping")
RAW_GLOB = "reclassified_multiclass_aiuti_*.csv"
COL = "TECNOLOGIE_AI"
CHUNK_SIZE = 100_000


def _clean(val):
    # Coerente con la pulizia dei raw: niente virgole/newline (romperebbero QUOTE_NONE).
    return val.replace(",", " ").replace("\n", " ").replace("\r", " ")


def process_file(filepath, out_dir):
    out_path = os.path.join(out_dir, os.path.basename(filepath))
    t0 = time.time()
    n_ai = 0
    first = True
    try:
        for chunk in pd.read_csv(filepath, dtype=str, chunksize=CHUNK_SIZE,
                                 low_memory=False, on_bad_lines="skip"):
            chunk = chunk.fillna("")
            tech = pd.Series([""] * len(chunk), index=chunk.index)
            if "CLASSIFICAZIONE" in chunk.columns and "DESCRIZIONE_PROGETTO" in chunk.columns:
                mask = chunk["CLASSIFICAZIONE"].str.strip().str.upper() == "AI"
                n_ai += int(mask.sum())
                for idx in chunk.index[mask]:
                    labels = rg.extract(chunk.at[idx, "DESCRIZIONE_PROGETTO"])
                    tech.at[idx] = _clean("|".join(labels))
            chunk[COL] = tech
            chunk.to_csv(out_path, index=False, mode="w" if first else "a",
                         header=first, quoting=csv.QUOTE_NONE, escapechar="\\")
            first = False
        return f"{os.path.basename(filepath)}: AI={n_ai}, {time.time() - t0:.1f}s -> {os.path.basename(out_path)}"
    except Exception as e:
        if os.path.exists(out_path):
            os.remove(out_path)
        return f"[ERRORE] {os.path.basename(filepath)}: {e}"


def run(raw_dir=RAW_DIR, out_dir=OUT_DIR, dry_run=False):
    files = sorted(glob.glob(os.path.join(raw_dir, RAW_GLOB)))
    print(f"{len(files)} file da etichettare in {raw_dir} -> {out_dir}")
    if dry_run:
        print("DRY-RUN: nessuna scrittura. Richiamare run(dry_run=False) per generare.")
        for f in files:
            print("  -", os.path.basename(f))
        return
    os.makedirs(out_dir, exist_ok=True)
    n_workers = min(cpu_count(), len(files))
    t0 = time.time()
    with Pool(n_workers) as pool:
        for res in pool.imap_unordered(partial(process_file, out_dir=out_dir), files):
            print(res)
    print(f"Completato in {time.time() - t0:.1f}s. Dataset etichettato in {out_dir}")


if __name__ == "__main__":
    run(dry_run=False)
