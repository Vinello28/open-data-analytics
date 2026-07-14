"""Campiona i candidati all'aggiunta in v2.1, per decidere sui dati e non a tavolino.

Ogni candidato viene cercato SOLO tra i record che la v2 attualmente scarta:
se il termine e' gia' coperto non ci interessa; ci interessa cosa AGGIUNGE.

Pool(10) = 12 core fisici - 2 liberi, contesto `fork` (Python 3.14 usa forkserver di default).
"""
import glob
import multiprocessing as mp
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traceability_patterns import get_mask as mask_v2  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(HERE, '..', '..', 'data', 'technology_mapping')

N_PROC = 10
CHUNK_SIZE = 100_000
N = 8  # campioni per termine, per file

CANDIDATI = {
    'approvvigionamento':     r'\b(?:caten[ae]|sistem[ai]) di approvvigionamento\b',
    'smart_logistics':        r'\bsmart logistics\b',
    'logistica_intelligente': r'\blogistica intelligente\b',
    'sicurezza_prodotto':     r'\bsicurezza (?:del |dei )?prodott\w*\b',
    'catena_valore':          r'\bcaten[ae] del valore\b',
}


def scan(file_path):
    """Ritorna {termine: (n_orfani, [campioni])} per un singolo file."""
    out = {k: [0, []] for k in CANDIDATI}
    try:
        it = pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=False)
    except Exception as e:
        print(f"errore {file_path}: {e}", flush=True)
        return out

    for chunk in it:
        t = chunk.get('TITOLO_PROGETTO', pd.Series('', index=chunk.index))
        d = chunk.get('DESCRIZIONE_PROGETTO', pd.Series('', index=chunk.index))
        m = chunk.get('TITOLO_MISURA', pd.Series('', index=chunk.index))
        v2 = mask_v2(t) | mask_v2(d) | mask_v2(m)

        tn = t.fillna('').astype(str).str.lower().astype(object)
        dn = d.fillna('').astype(str).str.lower().astype(object)

        for name, pat in CANDIDATI.items():
            orph = (tn.str.contains(pat, regex=True) | dn.str.contains(pat, regex=True)) & ~v2
            n = int(orph.sum())
            if not n:
                continue
            out[name][0] += n
            if len(out[name][1]) < N:
                for _, r in chunk.loc[orph].head(N).iterrows():
                    out[name][1].append(
                        f"{r.get('TITOLO_PROGETTO', '')} || {r.get('DESCRIZIONE_PROGETTO', '')}"[:190]
                    )
    return out


def main():
    files = sorted(glob.glob(os.path.join(CORPUS_DIR, 'reclassified_multiclass_aiuti_*.csv')))
    ctx = mp.get_context('fork')
    with ctx.Pool(processes=N_PROC) as pool:
        results = pool.map(scan, files)

    for name in CANDIDATI:
        tot = sum(r[name][0] for r in results)
        samples = [s for r in results for s in r[name][1]]
        print("\n" + "=" * 88)
        print(f"CANDIDATO: {name}  — aggiungerebbe {tot:,} record alla v2")
        print("=" * 88)
        for x in list(dict.fromkeys(samples))[:10]:
            print(f"  - {x}")


if __name__ == '__main__':
    main()
