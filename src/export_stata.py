"""Esporta il corpus ricostruito in .dta, cosi' STATA non deve parsare niente.

Perche' esiste
--------------
La mutilazione del testo (ogni virgola -> spazio) nasce da un CSV che "dava problemi" a STATA.
La causa vera era un default: `import delimited` usa bindquote(loose) e su prosa italiana piena
di virgole e virgolette sbanda. Il CSV quotato si importa benissimo con

    import delimited using aiuti_2023.csv, bindquote(strict) maxquotedrows(unlimited) clear

ma la strada senza rischi e' non far parsare niente a STATA: un .dta si apre con `use` e basta.

Tre trappole di to_stata, tutte verificate su questi dati
--------------------------------------------------------
1. version=118 e' OBBLIGATORIO. E' l'unico formato UTF-8 (il 117 e' latin-1: dopo aver riparato
   il mojibake, riscriverlo in latin-1 lo ricreerebbe) ed e' l'unico con strL, che serve perche'
   SETTORI_ATTIVITA arriva a 5.517 caratteri (oltre il limite str2045).
2. CLASSIFICAZIONE_MULTICLASS_CONFIDENZA e' lunga 37 caratteri. STATA ammette 32, e pandas
   TRONCA IN SILENZIO (solo un warning). Va rinominata a mano, col nome vero salvato come label.
3. to_stata NON e' streaming: tiene l'anno intero in RAM. 2023 e 2024 sono da ~6,5M record:
   parallelizzarli insieme fa OOM. Gli anni grandi vanno in sequenza.

    ./.venv/bin/python src/export_stata.py --anni 2015     # prova con round-trip
    ./.venv/bin/python src/export_stata.py                 # tutto + il file dei soli AI
"""
import argparse
import glob
import multiprocessing as mp
import os
import sys
import time

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
V2_DIR = os.path.join(REPO_ROOT, 'data', 'technology_mapping_v2')
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'stata')

# STATA: massimo 32 caratteri per nome variabile. Solo questa sfora (37).
RINOMINA = {'CLASSIFICAZIONE_MULTICLASS_CONFIDENZA': 'CLASSIF_MULTICLASS_CONFIDENZA'}
ETICHETTE_VAR = {'CLASSIF_MULTICLASS_CONFIDENZA': 'CLASSIFICAZIONE_MULTICLASS_CONFIDENZA'}

NUMERICHE = ['IMPORTO_NOMINALE_TOTALE', 'ELEMENTO_DI_AIUTO_TOTALE',
             'CLASSIFICAZIONE_CONFIDENZA', 'CLASSIF_MULTICLASS_CONFIDENZA']
INTERE = ['ANNO', 'NUM_COMPONENTI', 'NUM_STRUMENTI']

# DATA_CONCESSIONE ha formato '2017-10-30+01:00' (data + offset di fuso, senza ora):
# nessun parser di date la digerisce senza inventare. Resta stringa, ed e' la scelta onesta.
ANNI_GRANDI = {2020, 2021, 2023, 2024}          # > 1M righe: to_stata li tiene in RAM interi


def _prepara(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=RINOMINA)
    for c in NUMERICHE:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float64')
    for c in INTERE:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('Int32')
    # STATA non ha un missing per le stringhe: NaN -> campo vuoto.
    # .astype(object) perche' lo string dtype di pandas 3 fa litigare to_stata.
    for c in df.columns:
        if c not in NUMERICHE and c not in INTERE:
            df[c] = df[c].fillna('').astype(object)
    return df


def _scrivi(df: pd.DataFrame, path: str):
    df.to_stata(path, version=118, write_index=False,
                variable_labels={k: v for k, v in ETICHETTE_VAR.items() if k in df.columns})


def export_year(anno: int) -> str:
    src = os.path.join(V2_DIR, f'reclassified_multiclass_aiuti_{anno}.csv')
    out = os.path.join(OUT_DIR, f'aiuti_{anno}.dta')
    t0 = time.time()
    df = _prepara(pd.read_csv(src, dtype=str, low_memory=False))
    _scrivi(df, out)
    mb = os.path.getsize(out) / 1e6
    return f'  aiuti_{anno}.dta {len(df):>10,} righe  {mb:>8.0f} MB  {time.time() - t0:>5.0f}s'


def round_trip(anno: int):
    """Riapre il .dta e lo confronta cella per cella col CSV: l'unica prova che regge."""
    csv_p = os.path.join(V2_DIR, f'reclassified_multiclass_aiuti_{anno}.csv')
    dta_p = os.path.join(OUT_DIR, f'aiuti_{anno}.dta')
    a = _prepara(pd.read_csv(csv_p, dtype=str, low_memory=False))
    b = pd.read_stata(dta_p)

    print(f'\nROUND-TRIP {anno}: CSV {a.shape} vs .dta riletto {b.shape}')
    if list(a.columns) != list(b.columns):
        sys.exit(f'colonne diverse!\n  csv: {list(a.columns)}\n  dta: {list(b.columns)}')

    rotte = []
    for c in a.columns:
        if c in NUMERICHE or c in INTERE:
            uguale = ((a[c].isna() & b[c].isna()) | (a[c] - b[c]).abs().fillna(9).lt(1e-6)).all()
        else:
            uguale = (a[c].astype(str) == b[c].astype(str)).all()
        if not uguale:
            rotte.append(c)
    if rotte:
        sys.exit(f'round-trip FALLITO su: {rotte}')

    # e la prova che conta davvero: le virgole sono sopravvissute al giro in STATA?
    virgole = b['DESCRIZIONE_PROGETTO'].astype(str).str.contains(',').mean()
    print(f'  tutte le {len(a.columns)} colonne identiche dopo il round-trip')
    print(f'  descrizioni con virgola nel .dta: {virgole:.1%}  (nel corpus mutilato: 0,0%)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--anni', nargs='*', type=int)
    ap.add_argument('--no-ai-file', action='store_true')
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    anni = args.anni or sorted(
        int(os.path.basename(f).split('_')[-1][:4])
        for f in glob.glob(os.path.join(V2_DIR, 'reclassified_multiclass_aiuti_*.csv')))

    piccoli = [a for a in anni if a not in ANNI_GRANDI]
    grandi = [a for a in anni if a in ANNI_GRANDI]

    print(f'Export STATA (.dta v118) -> {OUT_DIR}\n')
    if piccoli:
        with mp.get_context('fork').Pool(min(8, len(piccoli))) as pool:
            for r in pool.imap_unordered(export_year, piccoli):
                print(r)
    for a in grandi:                       # in sequenza: to_stata non e' streaming
        print(export_year(a))

    if len(anni) == 1:
        round_trip(anni[0])

    if not args.no_ai_file and not args.anni:
        print('\nFile dei soli record AI...')
        parts = []
        for a in anni:
            d = pd.read_csv(os.path.join(V2_DIR, f'reclassified_multiclass_aiuti_{a}.csv'),
                            dtype=str, low_memory=False)
            parts.append(d[d['CLASSIFICAZIONE'].fillna('').str.upper() == 'AI'])
        ai = _prepara(pd.concat(parts, ignore_index=True))
        _scrivi(ai, os.path.join(OUT_DIR, 'aiuti_AI.dta'))
        ai.to_csv(os.path.join(OUT_DIR, 'aiuti_AI.csv'), index=False)
        print(f'  aiuti_AI.dta  {len(ai):,} record AI')

    print('\nIl team apre cosi\', senza configurare niente:')
    print('    use "aiuti_2023.dta", clear')


if __name__ == '__main__':
    main()
