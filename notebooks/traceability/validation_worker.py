"""Worker picklabile per la validazione modello-vs-regex.

Caricato come modulo esterno (non in __main__) per evitare l'errore di pickling
di `multiprocessing` dentro Jupyter, esattamente come `traceability_worker.py`.

Per ogni file classificato:
  - legge a chunk solo le colonne necessarie,
  - deduplica sulle DESCRIZIONE_PROGETTO (come ha fatto il modello in fase di
    classificazione: AI_LABEL e' costante per descrizione),
  - ricalcola la label regex con la STESSA `get_mask` usata per il training,
    applicata a TITOLO_PROGETTO OR DESCRIZIONE_PROGETTO (lowercase).

Restituisce un DataFrame compatto con: DESCRIZIONE_PROGETTO, TITOLO_PROGETTO,
ANNO, AI_LABEL, AI_CONFIDENCE, REGEX_LABEL.
"""
import os

import pandas as pd

# Riusa la regex IDENTICA a quella con cui e' stato costruito il training set.
from traceability_worker import get_mask

CHUNK_SIZE = 200_000

USECOLS = [
    "TITOLO_PROGETTO",
    "DESCRIZIONE_PROGETTO",
    "ANNO",
    "AI_LABEL",
    "AI_CONFIDENCE",
]

# Mappatura booleano regex -> label testuale coerente con AI_LABEL del modello.
POS = "tracciabilita"
NEG = "altro"


def _regex_label(df):
    """Ritorna una Series di label regex ('tracciabilita'/'altro') per il chunk.

    La regex del training viene valutata sia sul titolo sia sulla descrizione
    (OR), replicando `process_chunk` di traceability_worker.
    """
    titolo = df["TITOLO_PROGETTO"].fillna("").astype(str).str.lower()
    descr = df["DESCRIZIONE_PROGETTO"].fillna("").astype(str).str.lower()
    mask = get_mask(titolo) | get_mask(descr)
    return pd.Series([POS if m else NEG for m in mask], index=df.index)


def process_file(file_path):
    """Elabora un singolo CSV classificato e ritorna (filename, DataFrame compatto).

    Il DataFrame e' gia' deduplicato per DESCRIZIONE_PROGETTO a livello di file;
    la deduplica globale viene fatta dal chiamante dopo il concat.
    """
    filename = os.path.basename(file_path)
    print(f"[{filename}] Inizio elaborazione...", flush=True)

    parts = []
    try:
        chunk_iter = pd.read_csv(
            file_path,
            usecols=USECOLS,
            chunksize=CHUNK_SIZE,
            low_memory=False,
        )
    except Exception as e:  # pragma: no cover - diagnostica runtime
        print(f"[{filename}] Errore apertura: {e}", flush=True)
        return filename, pd.DataFrame(columns=USECOLS + ["REGEX_LABEL"])

    seen = set()  # dedup incrementale su descrizione, cross-chunk
    for chunk in chunk_iter:
        chunk = chunk.dropna(subset=["DESCRIZIONE_PROGETTO"])
        if chunk.empty:
            continue
        chunk = chunk.drop_duplicates(subset=["DESCRIZIONE_PROGETTO"])
        # Rimuove descrizioni gia' viste nei chunk precedenti dello stesso file.
        mask_new = ~chunk["DESCRIZIONE_PROGETTO"].isin(seen)
        chunk = chunk[mask_new]
        if chunk.empty:
            continue
        seen.update(chunk["DESCRIZIONE_PROGETTO"].tolist())
        chunk = chunk.copy()
        chunk["REGEX_LABEL"] = _regex_label(chunk)
        parts.append(chunk)

    if parts:
        result = pd.concat(parts, ignore_index=True)
    else:
        result = pd.DataFrame(columns=USECOLS + ["REGEX_LABEL"])

    n_nan = int(result["AI_LABEL"].isna().sum()) if len(result) else 0
    print(
        f"[{filename}] Fine. Descrizioni uniche: {len(result)} "
        f"(AI_LABEL NaN: {n_nan})",
        flush=True,
    )
    return filename, result
