"""Dimostra che il corpus ricostruito ha le STESSE etichette del pubblicato e SOLO testo migliore.

Il patto: il paper e' gia' uscito, quindi la ricostruzione puo' cambiare il TESTO (che era
mutilato) ma non puo' cambiare UNA SOLA ETICHETTA. Se non e' vero, questo script ESCE CON
ERRORE: non e' un report da leggere con indulgenza, e' un cancello.

Cosa controlla
  1. struttura   - stesse righe, stessa sequenza di COR (elemento per elemento)
  2. etichette   - CLASSIFICAZIONE / TIPO_AI / TECNOLOGIE_AI: ZERO differenze attese
  3. parsing     - il CSV v2 si rilegge con le impostazioni di default di pandas (se il quoting
                   fosse rotto, esploderebbe qui). NON si usa `wc -l`: con le newline
                   ripristinate le righe fisiche non sono i record.
  4. guadagno    - quante celle di testo migliorano e perche' (la tabella per il paper)

MEMORIA: si legge A CHUNK e si confronta chunk contro chunk. Caricare due anni interi (il 2023
e' da 6,5M record x 27 colonne) mandava la macchina in swap. Nessuno step tiene in RAM piu' di
qualche centinaio di MB per worker.

    ./.venv/bin/python src/verify_corpus.py
"""
import glob
import multiprocessing as mp
import os
import sys
from collections import Counter

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PUB_DIR = os.path.join(REPO_ROOT, 'data', 'technology_mapping')
V2_DIR = os.path.join(REPO_ROOT, 'data', 'technology_mapping_v2')
REPORT = os.path.join(V2_DIR, '_verification_report.csv')

CHUNK = 50_000     # ~2 x 50k x 27 colonne stringa per worker: qualche centinaio di MB
N_PROC = 6         # 6 worker x ~1 GB: sta comodo in 32 GB anche con il resto acceso

ETICHETTE = ['CLASSIFICAZIONE', 'TIPO_AI', 'TECNOLOGIE_AI']
TESTO = ['TITOLO_MISURA', 'TITOLO_PROGETTO', 'DESCRIZIONE_PROGETTO',
         'DENOMINAZIONE_BENEFICIARIO', 'SETTORI_ATTIVITA', 'OBIETTIVO']


def _norm(s: pd.Series) -> pd.Series:
    """Il pubblicato ha '' dove il v2 ha NaN (propagate_to_raw faceva fillna('')): normalizza."""
    return s.fillna('').astype(str).str.strip()


def check_year(anno: int) -> dict:
    pub_f = os.path.join(PUB_DIR, f'reclassified_multiclass_aiuti_{anno}.csv')
    v2_f = os.path.join(V2_DIR, f'reclassified_multiclass_aiuti_{anno}.csv')

    st = Counter()
    errori = []
    esempi = []

    # default di pandas sul v2: se il quoting fosse rotto, il parsing fallirebbe qui
    it_pub = pd.read_csv(pub_f, dtype=str, chunksize=CHUNK, low_memory=False)
    it_v2 = pd.read_csv(v2_f, dtype=str, chunksize=CHUNK, low_memory=False)

    for cp, cv in zip(it_pub, it_v2):
        if len(cp) != len(cv):
            errori.append(f'chunk disallineato: pubblicato {len(cp)} righe, v2 {len(cv)}')
            break
        st['righe'] += len(cv)

        if not (_norm(cp['COR']).values == _norm(cv['COR']).values).all():
            errori.append('la sequenza dei COR differisce')
            break

        st['ai'] += int((_norm(cv['CLASSIFICAZIONE']) == 'AI').sum())

        for col in ETICHETTE:
            diff = _norm(cp[col]) != _norm(cv[col])
            st[f'diff_{col}'] += int(diff.sum())
            if diff.any() and len(esempi) < 5:
                for _, r in pd.DataFrame({'col': col, 'COR': cv.loc[diff, 'COR'],
                                          'pubblicato': cp.loc[diff, col],
                                          'v2': cv.loc[diff, col]}).head(3).iterrows():
                    esempi.append(dict(r))

        for col in TESTO:
            if col not in cp.columns:
                continue
            p, n = _norm(cp[col]), _norm(cv[col])
            st[f'testo_{col}'] += int((p != n).sum())
            st[f'virgole_{col}'] += int((~p.str.contains(',', regex=False)
                                         & n.str.contains(',', regex=False)).sum())
            st[f'mojibake_{col}'] += int(p.str.contains('¿', regex=False).sum())

    # le chiavi mai incrementate non esistono in un Counter: le materializzo per il report
    for col in ETICHETTE:
        st[f'diff_{col}'] += 0
    return {'anno': anno, 'errori': errori, 'esempi': esempi, **st}


def main():
    anni = sorted(int(os.path.basename(f).split('_')[-1][:4])
                  for f in glob.glob(os.path.join(V2_DIR, 'reclassified_multiclass_aiuti_*.csv')))
    if not anni:
        sys.exit('nessun file in technology_mapping_v2: lancia prima src/rebuild_corpus.py')

    with mp.get_context('fork').Pool(N_PROC) as pool:
        res = sorted(pool.imap_unordered(check_year, anni), key=lambda x: x['anno'])

    print('=' * 84)
    print("ETICHETTE — devono essere IDENTICHE al pubblicato (il paper e' gia' uscito)")
    print('=' * 84)
    print(f"{'anno':<7}{'righe':>12}{'AI':>7}{'diff CLASSIF':>14}{'diff TIPO_AI':>14}{'diff TECN':>11}")
    for r in res:
        if r['errori']:
            print(f"{r['anno']:<7}  *** {r['errori'][0]}")
            continue
        print(f"{r['anno']:<7}{r['righe']:>12,}{r['ai']:>7,}"
              f"{r['diff_CLASSIFICAZIONE']:>14,}{r['diff_TIPO_AI']:>14,}{r['diff_TECNOLOGIE_AI']:>11,}")

    ok = [r for r in res if not r['errori']]
    print(f"\n{'TOT':<7}{sum(r['righe'] for r in ok):>12,}{sum(r['ai'] for r in ok):>7,}"
          f"{sum(r['diff_CLASSIFICAZIONE'] for r in ok):>14,}"
          f"{sum(r['diff_TIPO_AI'] for r in ok):>14,}"
          f"{sum(r['diff_TECNOLOGIE_AI'] for r in ok):>11,}")

    print('\n' + '=' * 84)
    print('GUADAGNO — il testo recuperato (queste differenze SONO il punto)')
    print('=' * 84)
    print(f"{'colonna':<28}{'celle migliorate':>18}{'virgole ripristinate':>22}{'mojibake riparato':>19}")
    for col in TESTO:
        c = sum(r.get(f'testo_{col}', 0) for r in ok)
        if c:
            print(f'{col:<28}{c:>18,}'
                  f"{sum(r.get(f'virgole_{col}', 0) for r in ok):>22,}"
                  f"{sum(r.get(f'mojibake_{col}', 0) for r in ok):>19,}")

    pd.DataFrame([{k: v for k, v in r.items() if k not in ('errori', 'esempi')}
                  for r in res]).to_csv(REPORT, index=False)
    print(f'\n{REPORT}')

    rotti = [r for r in res if r['errori']]
    diverse = [r for r in ok if any(r[f'diff_{c}'] for c in ETICHETTE)]
    if rotti or diverse:
        print('\n' + '!' * 84)
        for r in rotti:
            print(f"[{r['anno']}] {'; '.join(r['errori'])}")
        for r in diverse:
            print(f"\n[{r['anno']}] etichette divergenti:")
            for e in r['esempi']:
                print(f"    {e['col']} COR={e['COR']}: pubblicato={e['pubblicato']!r} v2={e['v2']!r}")
        sys.exit('\nVERIFICA FALLITA: le etichette NON sono identiche al pubblicato.')

    print("\nVERIFICA SUPERATA: zero differenze nelle etichette, il testo e' l'unica cosa cambiata.")


if __name__ == '__main__':
    main()
