"""Campiona i record ORFANI (contengono un termine della specifica, la v2 li scarta).

Serve a decidere, guardando i dati e non a tavolino, se il termine vada REINTEGRATO
o se la v2 fa bene a escluderlo. Un solo passaggio sul primo file utile per velocita'.
"""
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traceability_patterns import get_mask as mask_v2  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(HERE, '..', '..', 'data', 'technology_mapping')

PROBE = {
    'tracciamento (standalone)': r'\btracciament\w*\b',
    'blockchain (standalone)':   r'\bblockchain\b',
    'autenticazione':            r'\bautenticazion\w*\b',
    'qr/barcode (standalone)':   r'\b(?:qr[- ]?code|codice qr|codic[ei] a barre|barcode)\b',
    'identificazione+prodotto':  r'\bidentificazion\w*\b.{0,60}?\b(?:prodott\w+|materi\w+)\b|\b(?:prodott\w+|materi\w+)\b.{0,60}?\bidentificazion\w*\b',
}

N_PER_TERM = 8


def main():
    files = sorted(glob.glob(os.path.join(CORPUS_DIR, 'reclassified_multiclass_aiuti_*.csv')))
    found = {k: [] for k in PROBE}

    for fp in files[::-1]:                      # dagli anni piu' recenti (piu' densi)
        for chunk in pd.read_csv(fp, chunksize=100_000, low_memory=False):
            t = chunk.get('TITOLO_PROGETTO', pd.Series('', index=chunk.index))
            d = chunk.get('DESCRIZIONE_PROGETTO', pd.Series('', index=chunk.index))
            m = chunk.get('TITOLO_MISURA', pd.Series('', index=chunk.index))
            v2 = mask_v2(t) | mask_v2(d) | mask_v2(m)

            tn = t.fillna('').astype(str).str.lower().astype(object)
            dn = d.fillna('').astype(str).str.lower().astype(object)

            for name, pat in PROBE.items():
                if len(found[name]) >= N_PER_TERM:
                    continue
                hit = (tn.str.contains(pat, regex=True) | dn.str.contains(pat, regex=True)) & ~v2
                if hit.any():
                    for _, r in chunk.loc[hit].head(N_PER_TERM).iterrows():
                        txt = f"{r.get('TITOLO_PROGETTO', '')} || {r.get('DESCRIZIONE_PROGETTO', '')}"
                        found[name].append(str(txt)[:230])
            if all(len(v) >= N_PER_TERM for v in found.values()):
                break
        if all(len(v) >= N_PER_TERM for v in found.values()):
            break

    for name, samples in found.items():
        print("\n" + "=" * 90)
        print(f"ORFANI: {name}   (la specifica li vorrebbe, la v2 li scarta)")
        print("=" * 90)
        for s in dict.fromkeys(samples):
            print(f"  - {s}")


if __name__ == '__main__':
    main()
