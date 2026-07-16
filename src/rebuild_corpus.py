"""Ricostruisce il corpus con il TESTO INTEGRO, senza cambiare una sola etichetta.

Perche'
-------
Il corpus pubblicato (data/technology_mapping/) e' stato scritto con csv.QUOTE_NONE, e per non
rompere quel formato reclassify_annual.py ha tolto tutte le virgolette e sostituito OGNI virgola
con uno spazio: 6,7 milioni di descrizioni (28,1%) mutilate, e SETTORI_ATTIVITA - che e' una
lista comma-separated - senza piu' separatori. Irreversibile su quel file.

Ma il testo integro esiste ancora: data/classified_multiclass_aiuti_*.csv e' l'INPUT di
reclassify_annual.py, e ha virgole e virgolette. Contiene gia' CLASSIFICAZIONE (modello esterno,
pesi non su questa macchina): non si rifa' inferenza, si riusa la colonna.

Invariante da rispettare
------------------------
Il paper e' gia' uscito. Le etichette NON devono cambiare: 7.022 record AI, stessi TIPO_AI,
stessi TECNOLOGIE_AI. Cambia SOLO il testo. src/verify_corpus.py lo dimostra, e fallisce
rumorosamente se non e' vero.

Tre trappole, tutte misurate
----------------------------
1. Il gate AI gira sul testo ORIGINALE, la riparazione del mojibake viene DOPO. 12 delle 177
   chiavi di MANUAL_AI_SET contengono '¿': ripararle prima del confronto perderebbe 12 match.
2. COR NON e' una chiave (duplicato 2.275 volte nel 2024). Allineamento POSIZIONALE col
   carryover; COR serve solo come assertion. Un merge(on='COR') farebbe fan-out.
3. regex_gazetteer.extract_batch apre Pool(cpu_count()) al suo interno: chiamarlo dentro un
   worker annida Pool dentro Pool. Si usa extract().

    ./.venv/bin/python src/rebuild_corpus.py              # tutti gli anni
    ./.venv/bin/python src/rebuild_corpus.py --anni 2015  # prova
"""
import argparse
import csv
import glob
import multiprocessing as mp
import os
import sys
import time
from collections import Counter

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import text_repair                                            # noqa: E402
from ai_tech.methods import regex_gazetteer as rg             # noqa: E402
from reclassify_annual import (MANUAL_AI_SET, N_MANUAL_AI_EXPECTED,  # noqa: E402
                               apply_ai_gate)
from regex_multiprocessing import classify_obiettivo          # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(HERE, '..'))
SRC_DIR = os.path.join(REPO_ROOT, 'data')
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'technology_mapping_v2')
CARRYOVER = os.path.join(OUT_DIR, '_carryover_tipo_ai.csv')
CHUNK_SIZE = 100_000
N_PROC = 10

# Le colonne che contengono prosa e possono avere il mojibake.
COLONNE_TESTO = ['TITOLO_MISURA', 'DES_TIPO_MISURA', 'TITOLO_PROGETTO', 'DESCRIZIONE_PROGETTO',
                 'DENOMINAZIONE_BENEFICIARIO', 'DES_TIPO_BENEFICIARIO', 'REGIONE_BENEFICIARIO',
                 'SETTORI_ATTIVITA', 'OBIETTIVO']

_VOCAB = None


def _vocab():
    """Il vocabolario del repair: caricato una volta per worker (12MB), non per chunk."""
    global _VOCAB
    if _VOCAB is None:
        _VOCAB = text_repair._load()
    return _VOCAB


def _repara(s: pd.Series) -> pd.Series:
    """Ripara solo le celle che contengono '¿': il 99% salta del tutto la funzione."""
    if s.isna().all():
        return s
    rotte = s.fillna('').str.contains('¿', regex=False)
    if not rotte.any():
        return s
    v = _vocab()
    out = s.copy()
    out.loc[rotte] = s.loc[rotte].map(lambda t: text_repair.repair(t, v))
    return out


def rebuild_year(anno: int) -> dict:
    src = os.path.join(SRC_DIR, f'classified_multiclass_aiuti_{anno}.csv')
    out = os.path.join(OUT_DIR, f'reclassified_multiclass_aiuti_{anno}.csv')
    t0 = time.time()

    # carryover: (ROW_POS) -> (COR, TIPO_AI ereditato dall'API). Solo per questo anno.
    co = pd.read_csv(CARRYOVER, dtype={'COR': str})
    co = co[co['ANNO'] == anno]
    api = co[co['FONTE'] == 'api_20260526'].set_index('ROW_POS')
    cor_atteso = co.set_index('ROW_POS')['COR']

    # chiavi seminate: un Counter non le materializza se non vengono mai incrementate,
    # e un anno senza 'doubt' romperebbe il report
    stat = Counter({'righe': 0, 'ai': 0, 'celle_riparate': 0, 'doubt_reintegrati': 0})
    offset, first = 0, True

    for chunk in pd.read_csv(src, chunksize=CHUNK_SIZE, dtype=str, on_bad_lines='error',
                             low_memory=False):
        chunk = chunk.reset_index(drop=True)
        pos = pd.Index(range(offset, offset + len(chunk)))     # posizione assoluta nell'anno

        # 1. GATE sul testo ORIGINALE (prima di qualsiasi riparazione: vedi trappola 1)
        chunk = apply_ai_gate(chunk, MANUAL_AI_SET)
        ai = chunk['CLASSIFICAZIONE'].fillna('').str.upper() == 'AI'
        stat['ai'] += int(ai.sum())

        # 2. RIPARAZIONE del mojibake
        for col in COLONNE_TESTO:
            if col in chunk.columns:
                prima = chunk[col]
                chunk[col] = _repara(prima)
                stat['celle_riparate'] += int((prima.fillna('') != chunk[col].fillna('')).sum())

        # 3. TIPO_AI sull'OBIETTIVO riparato; i 'doubt' li reintegra il carryover
        tipo = pd.Series('', index=chunk.index, dtype=object)
        fonte = pd.Series('', index=chunk.index, dtype=object)
        if ai.any():
            idx_ai = chunk.index[ai]
            regex_lab = chunk.loc[idx_ai, 'OBIETTIVO'].map(classify_obiettivo)
            tipo.loc[idx_ai] = regex_lab
            fonte.loc[idx_ai] = 'regex'

            for i in idx_ai:
                p = int(pos[i])
                # assertion di allineamento: la posizione deve puntare allo stesso record
                if cor_atteso.get(p) != chunk.at[i, 'COR']:
                    raise SystemExit(
                        f'{anno}: disallineamento a riga {p}: sorgente COR={chunk.at[i, "COR"]}, '
                        f'carryover COR={cor_atteso.get(p)}. Ordine delle righe non conservato.')
                if regex_lab[i] == 'doubt' and p in api.index:
                    tipo.at[i] = api.at[p, 'TIPO_AI_PUBBLICATO']
                    fonte.at[i] = 'api_20260526'
                    stat['doubt_reintegrati'] += 1

        # 4. TECNOLOGIE_AI dal gazetteer sulla descrizione riparata (extract, NON extract_batch)
        tech = pd.Series('', index=chunk.index, dtype=object)
        for i in chunk.index[ai]:
            tech.at[i] = '|'.join(rg.extract(chunk.at[i, 'DESCRIZIONE_PROGETTO']))

        chunk['TIPO_AI'] = tipo.replace('', pd.NA)
        chunk['TECNOLOGIE_AI'] = tech
        chunk['TIPO_AI_FONTE'] = fonte.replace('', pd.NA)

        # 5. Scrittura con quoting VERO: le virgole restano virgole
        chunk.to_csv(out, index=False, mode='w' if first else 'a', header=first,
                     quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        first = False
        offset += len(chunk)
        stat['righe'] += len(chunk)

    return {'anno': anno, 'secondi': time.time() - t0, **stat}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--anni', nargs='*', type=int)
    args = ap.parse_args()

    if not os.path.exists(CARRYOVER):
        sys.exit('carryover assente: lancia prima src/freeze_tipo_ai_carryover.py')
    assert len(MANUAL_AI_SET) == N_MANUAL_AI_EXPECTED, (
        f'MANUAL_AI_SET ha {len(MANUAL_AI_SET)} chiavi invece di {N_MANUAL_AI_EXPECTED}: '
        'il gate non riprodurrebbe i 7.022 pubblicati')

    anni = args.anni or sorted(
        int(os.path.basename(f).split('_')[-1][:4])
        for f in glob.glob(os.path.join(SRC_DIR, 'classified_multiclass_aiuti_*.csv')))
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f'Ricostruzione di {len(anni)} anni -> {OUT_DIR}\n')
    print(f"{'anno':<6}{'righe':>12}{'AI':>8}{'celle riparate':>16}{'doubt reint.':>14}{'sec':>8}")
    tot = Counter()
    with mp.get_context('fork').Pool(N_PROC) as pool:
        for r in sorted(pool.imap_unordered(rebuild_year, anni), key=lambda x: x['anno']):
            print(f"{r['anno']:<6}{r['righe']:>12,}{r['ai']:>8,}{r['celle_riparate']:>16,}"
                  f"{r['doubt_reintegrati']:>14,}{r['secondi']:>8.0f}")
            for k in ('righe', 'ai', 'celle_riparate', 'doubt_reintegrati'):
                tot[k] += r[k]

    print(f"\n{'TOT':<6}{tot['righe']:>12,}{tot['ai']:>8,}{tot['celle_riparate']:>16,}"
          f"{tot['doubt_reintegrati']:>14,}")
    print('\nOra: ./.venv/bin/python src/verify_corpus.py')


if __name__ == '__main__':
    main()
