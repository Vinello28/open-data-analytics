"""La riparazione del mojibake cambia le etichette della tracciabilita'?

La regex e' girata su testo corrotto: "tracciabilit¿", "attivit¿", "l¿azienda". Alcune parole
non hanno potuto fare match. Qui si misura l'effetto, invece di ipotizzarlo: stessa regex,
stesso codice, due corpora (originale e riparato), e si guarda CHI cambia stato.

Non basta contare i positivi: il totale potrebbe restare uguale mentre entrano ed escono
record diversi. Quindi si confrontano gli INSIEMI di COR, non le cardinalita'.
"""
import glob
import multiprocessing as mp
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traceability_patterns import get_mask, rule_labels  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
ORIG = os.path.join(ROOT, 'data', 'technology_mapping')
FIXED = os.path.join(ROOT, 'data', 'technology_mapping_repaired')
OUT = os.path.join(ROOT, 'data', 'traceability', 'validation', 'repair_delta.csv')
CHUNK, N_PROC = 100_000, 10

FIELDS = ['TITOLO_MISURA', 'TITOLO_PROGETTO', 'DESCRIZIONE_PROGETTO']


def _mask(chunk):
    """Positivo se una qualsiasi delle tre colonne fa match (come traceability_worker)."""
    out = pd.Series(False, index=chunk.index)
    for col in FIELDS:
        if col in chunk:
            out |= get_mask(chunk[col])
    return out


def scan(name):
    """Ritorna, per un anno, i COR positivi nei due corpora + il testo dei record che cambiano."""
    pos_o, pos_f, campioni = set(), set(), []
    it_o = pd.read_csv(os.path.join(ORIG, name), chunksize=CHUNK, low_memory=False)
    it_f = pd.read_csv(os.path.join(FIXED, name), chunksize=CHUNK, low_memory=False)

    for co, cf in zip(it_o, it_f):
        mo, mf = _mask(co), _mask(cf)
        pos_o |= set(co.loc[mo, 'COR'])
        pos_f |= set(cf.loc[mf, 'COR'])

        entrati = mf & ~mo           # non matchavano da corrotti, matchano da riparati
        if entrati.any():
            g = cf.loc[entrati, ['COR', 'TITOLO_PROGETTO', 'DESCRIZIONE_PROGETTO']].copy()
            g['REGOLA'] = rule_labels(cf.loc[entrati, 'DESCRIZIONE_PROGETTO'])
            g['ANNO'] = name
            campioni.append(g)
        usciti = mo & ~mf            # matchavano solo grazie alla corruzione: sospetti
        if usciti.any():
            g = co.loc[usciti, ['COR', 'TITOLO_PROGETTO', 'DESCRIZIONE_PROGETTO']].copy()
            g['REGOLA'] = 'USCITO'
            g['ANNO'] = name
            campioni.append(g)

    return pos_o, pos_f, (pd.concat(campioni) if campioni else pd.DataFrame())


def main():
    names = sorted(os.path.basename(f) for f in glob.glob(os.path.join(FIXED, '*aiuti_*.csv')))
    with mp.get_context('fork').Pool(N_PROC) as pool:
        parts = pool.map(scan, names)

    O = set().union(*[p[0] for p in parts])
    F = set().union(*[p[1] for p in parts])
    delta = pd.concat([p[2] for p in parts if len(p[2])], ignore_index=True) \
        if any(len(p[2]) for p in parts) else pd.DataFrame()

    print('=' * 74)
    print('POSITIVI TRACCIABILITA\': corpus corrotto vs corpus riparato')
    print('=' * 74)
    print(f'  corpus originale : {len(O):>7,}')
    print(f'  corpus riparato  : {len(F):>7,}   ({len(F) - len(O):+,})')
    print(f'\n  entrati (match solo dopo la riparazione) : {len(F - O):>6,}')
    print(f'  usciti  (matchavano solo da corrotti)    : {len(O - F):>6,}')
    print(f'  invariati                                : {len(O & F):>6,}')

    if len(delta):
        delta.to_csv(OUT, index=False)
        print(f'\n  dettaglio -> {OUT}')
        entrati = delta[delta['REGOLA'] != 'USCITO']
        if len(entrati):
            print('\n  Per quale regola sono entrati:')
            for r, n in entrati['REGOLA'].value_counts().head(8).items():
                print(f'    {n:>5,}  {r}')
            print('\n  Esempi di record recuperati dalla riparazione:')
            for _, r in entrati.head(5).iterrows():
                print(f"    [{r['REGOLA']}] {str(r['TITOLO_PROGETTO'])[:78]}")
    print()


if __name__ == '__main__':
    main()
