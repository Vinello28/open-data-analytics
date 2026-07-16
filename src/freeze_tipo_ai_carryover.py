"""Congela i TIPO_AI che non sono riproducibili da codice, prima che si perdano per sempre.

Il problema
-----------
TIPO_AI (formazione | implementazione) e' assegnato da una regex sull'OBIETTIVO. Ma quando la
regex non decide, restituisce 'doubt': quei casi furono risolti UNA VOLTA da un classificatore
servito su http://localhost:8080, oggi non piu' disponibile. Sono 1.807 record su 7.022 (25,7%).

Ricostruire il corpus da zero li perderebbe. Rilanciare l'API non e' possibile, e comunque non
sarebbe riproducibile. Quindi si estraggono UNA VOLTA SOLA dal corpus pubblicato e si versionano
in una lookup table: cosi' l'etichetta sopravvive, ed e' AUDITABILE - il lettore vede quali
etichette vengono dal codice (FONTE='regex') e quali sono ereditate (FONTE='api_20260526').

L'alternativa - rigenerarle a ogni build chiamando un servizio - non e' riproducibile;
lasciarle a 'doubt' butterebbe un quarto delle etichette e romperebbe la continuita' col paper.

Allineamento
------------
COR NON e' una chiave: e' duplicato (2.275 volte nel 2024). Sorgente e corpus pubblicato hanno
lo stesso numero di righe NELLO STESSO ORDINE (verificato), quindi si allinea per POSIZIONE e
si usa COR solo come assertion. Un merge(on='COR') farebbe fan-out e cambierebbe il conteggio.

    ./.venv/bin/python src/freeze_tipo_ai_carryover.py
"""
import glob
import multiprocessing as mp
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regex_multiprocessing import classify_obiettivo  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(REPO_ROOT, 'data')
PUB_DIR = os.path.join(REPO_ROOT, 'data', 'technology_mapping')
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'technology_mapping_v2')
OUT = os.path.join(OUT_DIR, '_carryover_tipo_ai.csv')
N_PROC = 10


def freeze_year(anno: int) -> pd.DataFrame:
    src = os.path.join(SRC_DIR, f'classified_multiclass_aiuti_{anno}.csv')
    pub = os.path.join(PUB_DIR, f'reclassified_multiclass_aiuti_{anno}.csv')

    s = pd.read_csv(src, usecols=['COR', 'OBIETTIVO'], dtype=str, low_memory=False)
    p = pd.read_csv(pub, usecols=['COR', 'CLASSIFICAZIONE', 'TIPO_AI'], dtype=str,
                    low_memory=False)

    if len(s) != len(p):
        raise SystemExit(f'{anno}: sorgente {len(s):,} righe, pubblicato {len(p):,}. '
                         'Allineamento posizionale impossibile.')
    if not (s['COR'].values == p['COR'].values).all():
        raise SystemExit(f'{anno}: la sequenza dei COR differisce fra sorgente e pubblicato.')

    ai = p['CLASSIFICAZIONE'].fillna('').str.strip().str.upper() == 'AI'
    if not ai.any():
        return pd.DataFrame()

    out = pd.DataFrame({
        'ANNO': anno,
        'ROW_POS': p.index[ai],
        'COR': p.loc[ai, 'COR'].values,
        'TIPO_AI_PUBBLICATO': p.loc[ai, 'TIPO_AI'].values,
        # la regex gira sull'OBIETTIVO della SORGENTE (testo integro), non su quello mutilato
        'TIPO_AI_REGEX': [classify_obiettivo(t) for t in s.loc[ai, 'OBIETTIVO'].values],
    })
    out['FONTE'] = out['TIPO_AI_REGEX'].where(
        out['TIPO_AI_REGEX'] == 'doubt', 'regex').replace('doubt', 'api_20260526')

    # dove la regex DECIDE, deve coincidere col pubblicato: se no, la regex e' cambiata
    decisa = out['TIPO_AI_REGEX'] != 'doubt'
    conflitti = (out.loc[decisa, 'TIPO_AI_REGEX'] != out.loc[decisa, 'TIPO_AI_PUBBLICATO']).sum()
    if conflitti:
        raise SystemExit(f'{anno}: {conflitti} conflitti fra regex ricalcolata e pubblicato. '
                         'La regex TIPO_AI non e\' piu\' quella che ha prodotto il corpus.')
    return out


def main():
    anni = sorted(int(os.path.basename(f).split('_')[-1][:4])
                  for f in glob.glob(os.path.join(PUB_DIR, 'reclassified_multiclass_aiuti_*.csv')))
    with mp.get_context('fork').Pool(N_PROC) as pool:
        parts = [d for d in pool.map(freeze_year, anni) if len(d)]

    df = pd.concat(parts, ignore_index=True).sort_values(['ANNO', 'ROW_POS'])
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(OUT, index=False)

    n_api = int((df['FONTE'] == 'api_20260526').sum())
    print(f'record AI                    : {len(df):>6,}')
    print(f'  TIPO_AI da regex (codice)  : {len(df) - n_api:>6,}  riproducibili')
    print(f'  TIPO_AI da API (ereditati) : {n_api:>6,}  ({n_api / len(df):.1%})  <- congelati qui')
    print(f'\n{OUT}')
    print('Questo file va COMMITTATO: e\' l\'unica copia rimasta di quelle etichette.')


if __name__ == '__main__':
    main()
