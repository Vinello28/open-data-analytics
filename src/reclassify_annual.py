"""Riclassificazione annuale: applica il gate AI e (storicamente) mutila il testo per STATA.

Il gate AI
----------
    CLASSIFICAZIONE = 'AI'  <=>  (AI_REGEX su DESCRIZIONE_PROGETTO | descrizione in MANUAL_AI_SET)
                                 AND  il modello aveva gia' detto 'AI'
E' monotono-sottrattivo: puo' solo DECLASSARE AI -> NON_AI, mai promuovere. La colonna
CLASSIFICAZIONE arriva gia' calcolata nell'input (modello esterno): qui non si fa inferenza.

La mutilazione per STATA (`stata_mutilate`)
-------------------------------------------
Storicamente questo script scriveva con csv.QUOTE_NONE, e per non rompere il formato toglieva
le virgolette e sostituiva OGNI virgola con uno spazio. E' un danno permanente al testo:
6,7 milioni di descrizioni (28,1%) hanno perso le virgole, e SETTORI_ATTIVITA - che e' una lista
comma-separated - ha perso i separatori. La causa vera era un default sbagliato di STATA
(`bindquote(loose)`): un CSV quotato si importa con
    import delimited using file.csv, bindquote(strict) maxquotedrows(unlimited) clear
oppure, meglio, si consegna un .dta e STATA non deve parsare niente (vedi src/export_stata.py).

Qui la mutilazione e' stata ISOLATA in una funzione a parte e resa opzionale, senza cambiare il
comportamento di default: `process_reclassification_file` mutila ancora, cosi' i chiamanti storici
producono byte identici. Il nuovo `src/rebuild_corpus.py` chiama `apply_ai_gate` e NON mutila.
"""
import csv
import glob
import os
import re
import warnings

import pandas as pd

# Path ASSOLUTO, non relativo alla cwd: prima era '../data', quindi importando questo modulo da
# una directory diversa da src/ il MANUAL_AI_SET si caricava VUOTO IN SILENZIO, perdendo 144 dei
# 7.022 record AI senza alcun errore.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(REPO_ROOT, 'data')
OUTPUT_DIR = os.path.join(DATA_DIR, 'raw')
CHUNK_SIZE = 100000

AI_REGEX_PATTERN = r'(?i:\b(intelligenza artificiale|robotica collaborativa|visione artificiale|algoritmi predittivi|manutenzione preventiva|predictive analytics|artificial intelligence|machine learning|deep learning|reti neurali|computer vision|ai generativa|elaborazione del linguaggio naturale|nlp|llm|virtual assistant|virtual assistants|guida autonoma|chatbot|classificazione semantica)\b)|\b(IA|tecnologie AI|tecnologie IA|modelli di AI|algoritmi di AI|algoritmi di IA|LLM)\b'
ai_regex = re.compile(AI_REGEX_PATTERN)

MULTICLASS_COLS = ['CLASSIFICAZIONE_MULTICLASS', 'CLASSIFICAZIONE_MULTICLASS_CONFIDENZA']

# 177 RIGHE in regex_fn_reclassified.csv hanno predicted_label='ai', ma le descrizioni DISTINTE
# sono 128: il set deduplica. E' 128 il numero che deve restare stabile, non 177.
N_MANUAL_AI_EXPECTED = 128     # se cambia, il gate non riproduce piu' i 7.022 pubblicati


def load_manual_ai_set(data_dir: str = DATA_DIR) -> set:
    """Le descrizioni riclassificate a mano come 'AI' (falsi negativi della regex recuperati)."""
    path = os.path.join(data_dir, 'labelled', 'regex_fn_reclassified.csv')
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    ai = df[df['predicted_label'].str.lower().str.strip() == 'ai']
    return set(ai['descrizione'].fillna('').str.strip())


MANUAL_AI_SET = load_manual_ai_set()
get_manual_ai_set = load_manual_ai_set          # alias storico


def ai_gate_mask(chunk: pd.DataFrame, manual_ai_set: set = None) -> pd.Series:
    """Le righe che restano 'AI'.

    ATTENZIONE ALL'ORDINE: va calcolata sul testo ORIGINALE. 12 delle 177 chiavi di
    MANUAL_AI_SET contengono il mojibake '¿': se si ripara la descrizione prima di questo
    confronto, quelle 12 chiavi non matchano piu' e il gate non riproduce i 7.022 pubblicati.
    """
    manual = MANUAL_AI_SET if manual_ai_set is None else manual_ai_set

    # .astype(object) NON e' cosmesi: da pandas 3 le colonne stringa sono Arrow-backed e
    # .str.contains passa per RE2, dove \b e' ASCII-only. Il corpus pubblicato e' stato prodotto
    # con il motore `re` di Python, dove \b e' Unicode-aware. Su testo mojibake come
    # "liintelligenza artificiale" (l'apostrofo corrotto in 'i' accentata) i due motori
    # DIVERGONO: RE2 vede un confine di parola dove Python non lo vede, e classificherebbe AI
    # un record che il pubblicato ha lasciato NON_AI. Senza questa riga la ricostruzione
    # aggiungerebbe record AI che il paper non ha, per un aggiornamento di dipendenza.
    desc = chunk['DESCRIZIONE_PROGETTO'].fillna('').astype(object)

    with warnings.catch_warnings():
        # AI_REGEX_PATTERN ha gruppi di cattura, ma qui serve solo il booleano: il warning
        # di pandas ("use str.extract") e' rumore, non un difetto.
        warnings.simplefilter('ignore', UserWarning)
        regex_preds = desc.str.contains(ai_regex, regex=True).fillna(False)
    in_manual_set = desc.str.strip().isin(manual)
    is_ai_current = (chunk['CLASSIFICAZIONE'].fillna('').str.upper() == 'AI')
    return (regex_preds | in_manual_set) & is_ai_current


def apply_ai_gate(chunk: pd.DataFrame, manual_ai_set: set = None) -> pd.DataFrame:
    """Declassa a NON_AI cio' che non passa il gate e azzera le colonne multiclasse."""
    keep = ai_gate_mask(chunk, manual_ai_set)
    chunk.loc[~keep, 'CLASSIFICAZIONE'] = 'NON_AI'
    for col in MULTICLASS_COLS:
        if col in chunk.columns:
            chunk.loc[~keep, col] = pd.NA
    return chunk


def stata_mutilate(chunk: pd.DataFrame) -> pd.DataFrame:
    """Il danno storico, ora isolato: via le virgolette, ogni virgola diventa uno spazio.

    Serviva solo a non rompere csv.QUOTE_NONE. Non chiamarla su codice nuovo: e' irreversibile.
    """
    for col in chunk.columns:
        chunk[col] = chunk[col].astype(str)\
            .str.replace('"', '', regex=False)\
            .str.replace(',', ' ', regex=False)\
            .str.replace('\n', ' ', regex=False)\
            .str.replace('\r', ' ', regex=False)
        chunk.loc[chunk[col].isin(['nan', '<NA>', 'None']), col] = ''
    return chunk


def process_reclassification_file(filepath, mutilate: bool = True):
    """Comportamento storico invariato (mutilate=True): produce byte identici a prima."""
    filename = os.path.basename(filepath)
    new_filename = filename.replace('classified_', 'reclassified_')
    out_filepath = os.path.join(OUTPUT_DIR, new_filename)

    first_chunk = True
    for chunk in pd.read_csv(filepath, chunksize=CHUNK_SIZE, dtype=str):
        chunk = apply_ai_gate(chunk)
        if mutilate:
            chunk = stata_mutilate(chunk)
        chunk.to_csv(out_filepath, index=False, mode='w' if first_chunk else 'a',
                     header=first_chunk,
                     quoting=csv.QUOTE_NONE if mutilate else csv.QUOTE_MINIMAL,
                     escapechar='\\' if mutilate else None)
        first_chunk = False

    return f"{filename} -> {new_filename}"


if __name__ == '__main__':
    from concurrent.futures import ProcessPoolExecutor, as_completed

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Caricate {len(MANUAL_AI_SET)} descrizioni riclassificate manualmente come 'AI'.")

    files = glob.glob(os.path.join(DATA_DIR, 'classified_multiclass_aiuti_*.csv'))
    print(f"\nRiclassificazione di {len(files)} file annuali...")

    with ProcessPoolExecutor(max_workers=min(16, len(files))) as executor:
        futures = [executor.submit(process_reclassification_file, f) for f in files]
        for future in as_completed(futures):
            try:
                print(f'Completato: {future.result()}')
            except Exception as e:
                print(f'Errore: {e}')

    print(f'\nDataset annuali riclassificati in: {OUTPUT_DIR}')
